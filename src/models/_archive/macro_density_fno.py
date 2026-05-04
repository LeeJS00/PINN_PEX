# src/models/macro_density_fno.py
# DS-PINN Stream A: Macro Density FNO for BEOL screening context.
# Scatters metal volume fraction (scalar density) to per-layer 2D grids,
# applies FNO-2D shared across layers, bilinear-interpolates back to each cuboid.

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.gino_enricher import FNO2DBlock


class MacroDensityFNO(nn.Module):
    """
    Macro PDN screening context via metal density FNO (DS-PINN Stream A).

    Scatter metal volume fraction to a per-layer 2D grid, apply FNO-2D shared
    across layers, bilinear-interpolate back to each cuboid. Output Z_macro
    (B, N, d_macro) conditions both GND and CPL heads in NeuralFluxRouter —
    encodes "I'm near a VSS rail / isolated / in a dense PDN" without blurring
    the near-field geometry handled by NeuralFluxRouter (Stream B).

    Soft top-2 z-bucketing: each cuboid contributes to its two nearest z-anchors
    with linear distance weights, smoothing the per-layer density field for
    cuboids whose z falls between anchors. Cuboids exactly at an anchor degrade
    to hard assignment (top-1 weight = 1.0).

    Two-channel grid: {power_density, signal_density}. Power rails (VSS/VDD,
    cuboid channel 9 > 0.5) drive PDN screening; signal density adds context
    about routing congestion. Falls back to width-proxy (w > 2 μm) when channel
    9 is missing. The previously-tried eps channel was dropped — it carries no
    spatial signal (diag_eps_channel_utility.py).

    Single-phase training: FNO blocks train from step 0. proj_out is zero-init
    so initial Z_macro ≈ 0 (neutral screening signal); the downstream heads see
    no disruption while FNO learns useful spatial patterns.

    Edge-midpoint sampling: sample_field_at_edge_midpoints() exposes the FNO
    field for sparse-edge sampling — used by NeuralFluxRouter to evaluate
    Z_macro at (src_xy + dst_xy)/2 of each CPL edge, capturing "what's between
    coupling wires" directly rather than only at endpoints.
    """

    def __init__(self, grid_size: int = 16, d_macro: int = 32,
                 fno_modes: int = 8, n_fno_blocks: int = 2,
                 sigma_xy: float = 0.3, window: float = 4.0,
                 layer_map=None):
        super().__init__()
        self.grid_size = grid_size
        self.d_macro   = d_macro
        self.sigma_xy  = sigma_xy
        self.window    = window

        # z-anchors: metal ('C') and via ('V') only — dielectric shares z_pos
        # with adjacent metals and adds no spatial info while doubling FNO ops.
        CONDUCTOR_TYPES = {'C', 'V'}
        if layer_map is not None and len(layer_map) > 0:
            zs = sorted({float(info.get('z_pos', 0.0)) for info in layer_map.values()
                         if info.get('type', '').upper() in CONDUCTOR_TYPES})
            if not zs:  # fallback if 'type' field absent
                zs = sorted({float(info.get('z_pos', 0.0)) for info in layer_map.values()})
            unique_z: list[float] = []
            for z in zs:
                if len(unique_z) == 0 or abs(z - unique_z[-1]) > 0.05:
                    unique_z.append(z)
            z_anchors = torch.tensor(unique_z, dtype=torch.float32)
        else:
            z_anchors = torch.linspace(0.0, 7.0, 8)
        self.register_buffer('z_anchors', z_anchors)
        self.n_layers = int(z_anchors.numel())

        # FNO-2D shared across L layers: 2 input channels {power_density,
        # signal_density} → d_macro.
        blocks = [FNO2DBlock(2, d_macro, fno_modes)]
        for _ in range(n_fno_blocks - 1):
            blocks.append(FNO2DBlock(d_macro, d_macro, fno_modes))
        self.fno = nn.ModuleList(blocks)

        # Zero-init proj_out → Z_macro ≈ 0 at start (neutral screening signal).
        # Acts as a soft warm-up: FNO trains from step 0 but downstream heads see
        # no disruption until proj_out picks up non-zero weight.
        self.proj_out = nn.Linear(d_macro, d_macro)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

        # DS-PINN D1: auxiliary head that predicts log1p(per-net total cap)
        # from mean-pooled Z_macro. Provides a direct gradient signal so
        # MacroDensityFNO doesn't depend solely on the indirect path through
        # gnd_mlp/cpl_mlp losses (which can plateau before Z_macro learns).
        # Small (d_macro → 16 → 1) — purely a bootstrap regularizer.
        self.aux_cap_head = nn.Sequential(
            nn.LayerNorm(d_macro),
            nn.Linear(d_macro, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, cuboids: torch.Tensor, padding_mask: torch.Tensor):
        """
        cuboids:      (B, N, 9+)  ch: 0=x_rel, 1=y_rel, 2=z_abs, 3=w, 4=h, 5=d,
                                  9=net_type (0=signal, 0.67=VDD, 1.0=VSS)
        padding_mask: (B, N) bool  True = padding
        Returns: (z_macro, fno_field)
            z_macro:   (B, N, d_macro)  per-cuboid macro screening context.
            fno_field: (B, L, d_macro, G, G)  pre-proj_out FNO output, exposed
                       so NeuralFluxRouter can sample at edge midpoints.
        """
        B, N, C = cuboids.shape
        G  = self.grid_size
        L  = self.n_layers
        device = cuboids.device

        # Run scatter + log1p + FFT in float32. cuboids may be bf16 under autocast;
        # bf16 + log1p with small densities loses ~3 bits of precision per op.
        coords_xy = cuboids[..., :2].float()                                # (B, N, 2)
        z_abs     = cuboids[..., 2].float()                                 # (B, N)
        w         = cuboids[..., 3].float()
        h         = cuboids[..., 4].float()
        d_c       = cuboids[..., 5].float()
        valid     = (~padding_mask).float()                                  # (B, N)

        # Power vs signal split (channel 9 = net_type; 0.67=VDD, 1.0=VSS).
        # Fallback for older 9-channel datasets: w > 2 μm wide-conductor proxy.
        if C >= 10:
            is_power = (cuboids[..., 9].float() > 0.5).float() * valid       # (B, N)
        else:
            is_power = (w > 2.0).float() * valid
        is_signal = (1.0 - is_power) * valid                                  # (B, N)

        # --- Thickness-aware soft top-K z-bucket assignment (F1) ---
        # Each cuboid is a slab of thickness d_c centered at z_abs. Compute
        # Gaussian weights per anchor with sigma proportional to d_c — thick
        # cuboids (vias spanning two metals, fat upper-metal straps) auto-
        # spread their density across multiple anchors. Thin wires (d_c→0)
        # collapse to top-1 hard assignment, identical to the old behavior.
        sigma_z = (d_c * 0.5).clamp(min=0.05)                                # (B, N) μm; floor at 50nm
        dist_z  = torch.abs(z_abs.unsqueeze(-1) - self.z_anchors)            # (B, N, L)
        w_z     = torch.exp(-(dist_z ** 2) / (2.0 * sigma_z.unsqueeze(-1) ** 2))  # (B, N, L)
        K = min(2, L)
        top_w, top_idx = torch.topk(w_z, k=K, dim=-1, largest=True)          # (B, N, K)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)      # normalize → Σ=1
        z_assign = torch.zeros(B, N, L, device=device, dtype=torch.float32)
        z_assign.scatter_(-1, top_idx, top_w)
        z_assign = z_assign * valid.unsqueeze(-1)                            # mute padding

        # Per-cuboid mass and Gaussian footprint on the xy grid.
        vol = (w * h * d_c).clamp(min=0.0) * valid                          # (B, N) μm³
        lin    = torch.linspace(0.0, self.window, G, device=device, dtype=torch.float32)
        gx, gy = torch.meshgrid(lin, lin, indexing='ij')
        grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)         # (G², 2)
        xy      = coords_xy + self.window / 2                               # (B, N, 2) in [0, window]
        dxy     = xy.unsqueeze(2) - grid_xy                                 # (B, N, G², 2)
        w_gauss = torch.exp(
            -dxy.pow(2).sum(-1) / (2.0 * self.sigma_xy ** 2)
        ) * valid.unsqueeze(-1)                                              # (B, N, G²)

        # --- Vectorized P2G: (B, L, 2, G²) via two bmms (one per channel) ---
        # density[b,l,g] = Σ_n z_assign[b,n,l] * vol_ch[b,n] * w_gauss[b,n,g]
        # Unnormalized "smoothed sum" — represents total {power, signal} volume
        # in cell g on layer l (rather than weighted avg). log1p compresses range.
        vol_power      = vol * is_power                                      # (B, N)
        vol_signal     = vol * is_signal                                     # (B, N)
        pw_T = (vol_power.unsqueeze(-1)  * z_assign).transpose(1, 2)        # (B, L, N)
        sg_T = (vol_signal.unsqueeze(-1) * z_assign).transpose(1, 2)        # (B, L, N)
        power_grid  = torch.bmm(pw_T, w_gauss)                              # (B, L, G²)
        signal_grid = torch.bmm(sg_T, w_gauss)                              # (B, L, G²)
        density_2ch = torch.stack([power_grid, signal_grid], dim=2)         # (B, L, 2, G²)
        density_grid = torch.log1p(density_2ch).reshape(B * L, 2, G, G)

        # --- FNO-2D (shared weights across L layers) ---
        x = density_grid
        for blk in self.fno:
            x = blk(x)                                                       # (B*L, d_macro, G, G)
        fno_field = x.reshape(B, L, self.d_macro, G, G)                     # exposed for edge sampling

        # --- Vectorized G2P: single grid_sample over (B*L, d_macro, G, G) ---
        # Replicate xy_norm L times so each (b, l) feature map is sampled at the
        # same N positions, then combine across L using soft z_assign weights.
        xy_norm     = ((xy / self.window) * 2.0 - 1.0)                      # (B, N, 2) in [-1, 1]
        xy_norm_rep = xy_norm.unsqueeze(1).expand(B, L, N, 2).reshape(B * L, N, 2)
        gs_grid     = xy_norm_rep.unsqueeze(1)                              # (B*L, 1, N, 2)
        sampled     = F.grid_sample(x, gs_grid, mode='bilinear', align_corners=True)
        # sampled: (B*L, d_macro, 1, N) → (B, L, d_macro, N) → (B, N, L, d_macro)
        sampled = sampled.squeeze(2).reshape(B, L, self.d_macro, N).permute(0, 3, 1, 2)
        # Combine across L using soft z_assign weights (each cuboid pulls from its
        # top-2 anchors, weighted by distance — same weights used in P2G).
        z_macro = (sampled * z_assign.unsqueeze(-1)).sum(dim=2)              # (B, N, d_macro)

        # proj_out (zero-init) keeps Z_macro ≈ 0 at start; cast back to cuboids dtype.
        z_macro = self.proj_out(z_macro).to(cuboids.dtype)
        # Mask padded rows so any non-zero proj_out bias drift cannot leak into
        # downstream pad slots (defense in depth — flux_router also pad-masks).
        z_macro = z_macro * valid.unsqueeze(-1).to(z_macro.dtype)
        return z_macro, fno_field

    def _soft_top_k_z(self, z_abs: torch.Tensor, K: int = 2) -> torch.Tensor:
        """Soft top-K z-bucket assignment — same logic used inside forward().
        z_abs: (..., ) absolute z values. Returns (..., L) weights summing to 1.
        """
        flat = z_abs.reshape(-1)
        L = self.n_layers
        K = min(K, L)
        dist_z = torch.abs(flat.unsqueeze(-1) - self.z_anchors)              # (E, L)
        top_dist, top_idx = torch.topk(dist_z, k=K, dim=-1, largest=False)   # (E, K)
        sum_d = top_dist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        w_top = (sum_d - top_dist) / sum_d
        w_top = w_top / w_top.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        z_assign = torch.zeros(flat.shape[0], L, device=flat.device, dtype=torch.float32)
        z_assign.scatter_(-1, top_idx, w_top)
        return z_assign.reshape(*z_abs.shape, L)

    def sample_field_at_edge_midpoints(
        self,
        fno_field: torch.Tensor,
        b_idx: torch.Tensor,
        xy_rel: torch.Tensor,
        z_abs: torch.Tensor,
        chunk_size: int = 4096,
    ) -> torch.Tensor:
        """Sample the FNO field at sparse-edge points (e.g. CPL edge midpoints).

        fno_field: (B, L, d_macro, G, G)
        b_idx:     (E,) batch index for each edge
        xy_rel:    (E, 2) xy in tile-relative coords (matches cuboids[..., :2])
        z_abs:     (E,) absolute z height (μm)
        chunk_size: process at most this many edges at a time. Bilinear corner
                    materialization scales as O(chunk * L * d_macro); chunking
                    bounds peak memory regardless of total edge count.
        Returns:   (E, d_macro)  Z_macro_edge after proj_out (matches per-cuboid path).
        """
        E = b_idx.shape[0]
        if E == 0:
            return torch.zeros(0, self.d_macro, device=fno_field.device,
                               dtype=fno_field.dtype)
        B, L, D, G, _ = fno_field.shape

        # Soft top-2 z-assignment for the midpoints (same kernel as forward).
        z_assign_e = self._soft_top_k_z(z_abs.float(), K=2)                  # (E, L)

        # xy in tile-relative → [0, window] → [-1, 1]
        xy_world = xy_rel.float() + self.window / 2.0                        # (E, 2)
        xy_norm  = ((xy_world / self.window) * 2.0 - 1.0).clamp(-1.0, 1.0)   # (E, 2)

        xg = (xy_norm[:, 0] + 1.0) * 0.5 * (G - 1)                           # (E,)
        yg = (xy_norm[:, 1] + 1.0) * 0.5 * (G - 1)
        x0 = xg.floor().long().clamp(0, G - 1)
        y0 = yg.floor().long().clamp(0, G - 1)
        x1 = (x0 + 1).clamp(max=G - 1)
        y1 = (y0 + 1).clamp(max=G - 1)
        fx_all = (xg - x0.float()).view(-1, 1, 1)                            # (E, 1, 1)
        fy_all = (yg - y0.float()).view(-1, 1, 1)

        # Chunked bilinear interpolation to bound peak memory at ~chunk*L*D*4*4
        # bytes (worst case ~50 MB at chunk=4096, L=25, D=32). Without chunking
        # E=50k would materialize 4 corner tensors of ~640 MB → OOM at AL bs=4.
        chunks: list[torch.Tensor] = []
        for s in range(0, E, chunk_size):
            e = slice(s, s + chunk_size)
            b_c   = b_idx[e]
            x0_c, x1_c = x0[e], x1[e]
            y0_c, y1_c = y0[e], y1[e]
            fx, fy = fx_all[e], fy_all[e]
            # Advanced indexing: (E_c,) batch + two `:` slices (L, D) + two
            # advanced (x, y) → result (E_c, L, D).
            f00 = fno_field[b_c, :, :, x0_c, y0_c]
            f01 = fno_field[b_c, :, :, x0_c, y1_c]
            f10 = fno_field[b_c, :, :, x1_c, y0_c]
            f11 = fno_field[b_c, :, :, x1_c, y1_c]
            sampled_c = ((1 - fx) * (1 - fy) * f00 + (1 - fx) * fy * f01 +
                         fx * (1 - fy) * f10 + fx * fy * f11)                # (E_c, L, D)
            # Reduce L dim immediately so the chunk's intermediate footprint
            # collapses to (E_c, D) before storing for autograd replay.
            zm_c = (sampled_c * z_assign_e[e].unsqueeze(-1).to(sampled_c.dtype)).sum(dim=1)
            chunks.append(zm_c)
        z_macro_edge_pre = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        return self.proj_out(z_macro_edge_pre).to(fno_field.dtype)
