"""
GINO Runtime Benchmark vs. Current DeepPEX
===========================================
Measures wall-clock inference time for:
  A. Current DeepPEX (CuboidEncoder + NeuralFluxRouter)
  B. 2.5D GINO skeleton (P2G → FNO-2D per layer → Z-MLP → G2P → cap head)
  C. Voxel FNO (full 3D-equivalent at 128x128 and 256x256)

Benchmark scenarios:
  - Single tile inference (latency)
  - Batch inference  (batch size 4, 8, 16, 32)
  - Full chip extrapolation (using tile count from manifest)

Also reports:
  - Parameter count
  - GPU memory usage
  - Breakdown: P2G / FNO / G2P / head

Output: output_intel22/diag_gino_runtime.txt
"""

import sys, gzip, pickle, math, argparse, time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import configs.config as cfg

import torch
import torch.nn as nn
import torch.nn.functional as F

PROCESSED_DIR = Path(cfg.PROCESSED_DIR)
MANIFEST_PATH = PROCESSED_DIR / "dataset_manifest.csv"
N_LAYERS = 8
LAYER_BOUNDARIES = [0.0, 0.65, 0.85, 1.00, 1.20, 1.40, 1.80, 4.80, 9.0]


# ─────────────────────────────────────────────
# Current DeepPEX Skeleton (faithful to production)
# ─────────────────────────────────────────────

class DeepPEX_Skeleton(nn.Module):
    """
    Stripped-down DeepPEX: encoder + simplified flux router.
    Approximates production timing without needing the full checkpoint.
    """
    def __init__(self, in_dim=10, d_model=256, max_cuboids=1024):
        super().__init__()
        self.d_model = d_model
        self.max_cuboids = max_cuboids

        # CuboidEncoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

        # GND head: Z_dim=16 + 4 features
        Z_DIM = 16
        self.gnd_basis = nn.Sequential(
            nn.Linear(d_model, Z_DIM), nn.GELU())
        self.gnd_mlp = nn.Sequential(
            nn.Linear(Z_DIM + 4, 64), nn.GELU(), nn.Linear(64, 1))

        # CPL head: edge features -> coupling
        CPL_IN = d_model * 3 + 10
        self.cpl_mlp = nn.Sequential(
            nn.Linear(CPL_IN, 128), nn.GELU(), nn.Linear(128, 1))

        # Context aggregation norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cub: torch.Tensor, padding_mask: torch.Tensor) -> dict:
        B, N, C = cub.shape
        device = cub.device

        # 1. Encode
        x = cub.clone()
        x[..., :3] /= 2.5
        x[..., 3:6] = torch.log1p(x[..., 3:6].clamp(0)) / 1.5
        x[..., 8:9] = torch.log(x[..., 8:9].clamp(min=1.0)) / 2.0
        feats = self.encoder(x)  # (B, N, d_model)

        # 2. 1-hop context aggregation (BMM approximation)
        # In production this uses a sparse distance mask; here we use dense attention
        mask_f = (~padding_mask).float().unsqueeze(-1)  # (B, N, 1)
        ctx = (feats * mask_f).sum(1, keepdim=True) / (mask_f.sum(1, keepdim=True) + 1e-6)
        feats_ctx = self.norm(feats + ctx.expand_as(feats))

        # 3. GND prediction
        Z = self.gnd_basis(feats_ctx)
        is_target = (cub[..., 7] == 1.0).float().unsqueeze(-1)
        area = (cub[..., 3] * cub[..., 4]).unsqueeze(-1)
        z_abs = cub[..., 2:3]
        eps   = cub[..., 8:9]
        wide  = (cub[..., 3] > 2.0).float().unsqueeze(-1)
        gnd_in = torch.cat([Z, torch.log1p(area), torch.log(eps.clamp(1.0)), z_abs/10, wide], -1)
        c_gnd = F.softplus(self.gnd_mlp(gnd_in)) * is_target  # (B, N, 1)

        # 4. CPL prediction (sparse — only nearby pairs)
        # Approximate with all-pairs for timing (overestimates slightly)
        src = feats_ctx.unsqueeze(2).expand(B, N, N, -1)
        dst = feats_ctx.unsqueeze(1).expand(B, N, N, -1)
        # Simplified edge geometry (10 channels, zeros for bench)
        edge_geom = torch.zeros(B, N, N, 10, device=device)
        cpl_in = torch.cat([src, dst, src - dst, edge_geom], -1).reshape(B * N * N, -1)
        c_cpl  = F.softplus(self.cpl_mlp(cpl_in)).reshape(B, N, N)

        return {'c_gnd': c_gnd.squeeze(-1), 'c_cpl': c_cpl}


# ─────────────────────────────────────────────
# 2.5D GINO Skeleton
# ─────────────────────────────────────────────

class SpectralConv2d(nn.Module):
    def __init__(self, w, m):
        super().__init__()
        self.m = m
        self.W = nn.Parameter((1/(w*w)) * torch.randn(w, w, m, m, dtype=torch.cfloat))

    def forward(self, x):
        B, C, H, W = x.shape
        m = self.m
        xf = torch.fft.rfft2(x, norm='ortho')
        out = torch.zeros_like(xf)
        out[:, :, :m, :m] = torch.einsum('bixk,ioxk->boxk', xf[:, :, :m, :m], self.W)
        return torch.fft.irfft2(out, s=(H, W), norm='ortho')


class FNOBlock2D(nn.Module):
    def __init__(self, w, m):
        super().__init__()
        self.spec = SpectralConv2d(w, m)
        self.bypass = nn.Conv2d(w, w, 1)
        self.norm = nn.InstanceNorm2d(w)

    def forward(self, x):
        return F.gelu(self.norm(self.spec(x) + self.bypass(x)))


class BEOL_GINO(nn.Module):
    """
    2.5D Layerwise GINO for BEOL capacitance extraction.

    Architecture:
      1. CuboidEncoder (per-point MLP)
      2. P2G: Gaussian scatter to (B, L, G, G, d_model)
      3. FNO-2D per layer independently
      4. Z-MLP: fuse across layers (local Z neighborhood)
      5. G2P: interpolate latent back to each cuboid
      6. Cap head: per-target-cuboid → c_gnd; edge MLP → c_cpl
    """
    def __init__(self, in_dim=10, d_enc=128, d_fno=64, fno_modes=16,
                 n_fno_blocks=4, grid_size=64, n_layers=N_LAYERS,
                 sigma_xy=0.25, sigma_z=0.15, window=8.0):
        super().__init__()
        self.G    = grid_size
        self.L    = n_layers
        self.d_enc = d_enc
        self.d_fno = d_fno
        self.sigma_xy = sigma_xy
        self.sigma_z  = sigma_z
        self.window   = window

        # 1. Encoder
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, d_enc), nn.LayerNorm(d_enc), nn.GELU(),
            nn.Linear(d_enc, d_enc), nn.LayerNorm(d_enc))

        # 2. Lift to FNO width
        self.lift = nn.Linear(d_enc, d_fno)

        # 3. FNO-2D blocks (shared weights across all L layers)
        self.fno_blocks = nn.ModuleList([FNOBlock2D(d_fno, fno_modes)
                                         for _ in range(n_fno_blocks)])

        # 4. Z-MLP: (d_fno × 3) → d_fno  [processes layer triplets]
        self.z_mlp = nn.Sequential(
            nn.Linear(d_fno * 3, d_fno), nn.GELU(),
            nn.Linear(d_fno, d_fno))

        # 5. Project back to d_enc
        self.proj = nn.Linear(d_fno, d_enc)

        # 6a. GND head
        self.gnd_head = nn.Sequential(
            nn.Linear(d_enc + 3, 64), nn.GELU(),   # +3: area,eps,z
            nn.Linear(64, 1))

        # 6b. CPL head
        self.cpl_head = nn.Sequential(
            nn.Linear(d_enc * 2 + 6, 64), nn.GELU(),  # +6: geometry
            nn.Linear(64, 1))

        # Fixed Gaussian kernel (precomputed at first call)
        self._kernel_cache = {}

        # Layer z-centres
        self.register_buffer('z_centres', torch.tensor(
            [(LAYER_BOUNDARIES[i]+LAYER_BOUNDARIES[i+1])/2 for i in range(n_layers)]))

    def _gaussian_kernel(self, G, device):
        """Precompute per-cell Gaussian weights on the grid (G×G)."""
        key = (G, str(device))
        if key in self._kernel_cache:
            return self._kernel_cache[key]
        half = self.window / 2.0
        xs = torch.linspace(-half + half/G, half - half/G, G, device=device)
        gx, gy = torch.meshgrid(xs, xs, indexing='ij')  # (G, G)
        self._kernel_cache[key] = (gx, gy)
        return gx, gy

    def p2g(self, feats: torch.Tensor, cuboids: torch.Tensor) -> torch.Tensor:
        """
        Gaussian P2G scatter.
        feats:   (B, N, d_enc)
        cuboids: (B, N, C)   coords in cuboids[:,:,:3]
        Returns: (B, L, G, G, d_fno)  after lifting
        """
        B, N, D = feats.shape
        G, L = self.G, self.L
        device = feats.device
        gx, gy = self._gaussian_kernel(G, device)  # (G, G)

        # Lift features
        feats_lift = self.lift(feats)  # (B, N, d_fno)

        grid_feat = torch.zeros(B, L, G, G, self.d_fno, device=device)
        grid_wt   = torch.zeros(B, L, G, G, 1, device=device)

        x_c  = cuboids[:, :, 0]  # (B, N)
        y_c  = cuboids[:, :, 1]
        z_c  = cuboids[:, :, 2]

        # Compute layer assignments
        z_t = z_c.unsqueeze(-1)  # (B, N, 1)
        z_e = self.z_centres.view(1, 1, L)  # (1, 1, L)
        w_z = torch.exp(-((z_t - z_e)**2) / (2 * self.sigma_z**2))  # (B, N, L)

        # Scatter per sample (vectorise over N via broadcasting)
        # For each cuboid i, compute w_xy(i) over all grid cells: (G, G)
        # Then accumulate to grid. Use batch-matmul trick:
        # reshape as: scatter_add over grid cells.

        # Simplified: flatten grid → (G²,) and scatter_add
        G2 = G * G
        gx_flat = gx.reshape(G2)  # (G²,)
        gy_flat = gy.reshape(G2)

        for b in range(B):
            for i in range(N):
                xi, yi = x_c[b, i], y_c[b, i]
                w_xy = torch.exp(-((gx - xi)**2 + (gy - yi)**2)
                                 / (2 * self.sigma_xy**2))  # (G, G)
                for li in range(L):
                    w = w_xy * w_z[b, i, li]  # (G, G)
                    w = w.unsqueeze(-1)  # (G, G, 1)
                    grid_feat[b, li] += w * feats_lift[b, i]  # broadcast
                    grid_wt[b, li]   += w

        grid_feat = grid_feat / (grid_wt + 1e-8)
        return grid_feat  # (B, L, G, G, d_fno)

    def g2p(self, grid_feat: torch.Tensor, cuboids: torch.Tensor) -> torch.Tensor:
        """
        Gaussian G2P interpolation.
        grid_feat: (B, L, G, G, d_fno)
        cuboids:   (B, N, C)
        Returns:   (B, N, d_fno)
        """
        B, L, G, _, D = grid_feat.shape
        N = cuboids.shape[1]
        device = cuboids.device
        gx, gy = self._gaussian_kernel(G, device)

        x_c = cuboids[:, :, 0]  # (B, N)
        y_c = cuboids[:, :, 1]
        z_c = cuboids[:, :, 2]

        z_t = z_c.unsqueeze(-1)
        z_e = self.z_centres.view(1, 1, L)
        w_z = torch.exp(-((z_t - z_e)**2) / (2 * self.sigma_z**2))  # (B, N, L)

        out = torch.zeros(B, N, D, device=device)
        wt  = torch.zeros(B, N, 1, device=device)

        for b in range(B):
            for i in range(N):
                xi, yi = x_c[b, i], y_c[b, i]
                w_xy = torch.exp(-((gx - xi)**2 + (gy - yi)**2)
                                 / (2 * self.sigma_xy**2))  # (G, G)
                for li in range(L):
                    w = w_xy * w_z[b, i, li]  # (G, G)
                    # Weighted sum of grid features
                    out[b, i] += (grid_feat[b, li] * w.unsqueeze(-1)).sum(0).sum(0)
                    wt[b, i]  += w.sum()

        return out / (wt + 1e-8)

    def fno_per_layer(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Apply FNO-2D to each layer.
        grid: (B, L, G, G, d_fno)
        Returns: (B, L, G, G, d_fno)
        """
        B, L, G, _, D = grid.shape
        out = []
        for li in range(L):
            x = grid[:, li].permute(0, 3, 1, 2)  # (B, D, G, G)
            for blk in self.fno_blocks:
                x = blk(x)
            out.append(x.permute(0, 2, 3, 1))  # (B, G, G, D)
        return torch.stack(out, dim=1)  # (B, L, G, G, D)

    def z_fusion(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Z-MLP: fuse adjacent layers [l-1, l, l+1] for inter-layer coupling.
        grid: (B, L, G, G, d_fno)
        Returns: (B, L, G, G, d_fno)
        """
        B, L, G, _, D = grid.shape
        padded = F.pad(grid.permute(0,4,1,2,3), (0,0,0,0,1,1)).permute(0,2,3,4,1)
        # padded: (B, L+2, G, G, D)
        out = []
        for li in range(L):
            triplet = padded[:, li:li+3].reshape(B, G, G, 3*D)  # (B,G,G,3D)
            fused = self.z_mlp(triplet)  # (B, G, G, D)
            out.append(fused)
        return torch.stack(out, dim=1)  # (B, L, G, G, D)

    def forward(self, cuboids: torch.Tensor, padding_mask: torch.Tensor) -> dict:
        B, N, C = cuboids.shape
        device = cuboids.device

        # 1. Encode
        x = cuboids.clone()
        x[..., :3] /= 2.5
        x[..., 3:6] = torch.log1p(x[..., 3:6].clamp(0)) / 1.5
        x[..., 8:9] = torch.log(x[..., 8:9].clamp(min=1.0)) / 2.0
        feats = self.encoder(x)  # (B, N, d_enc)

        # 2. P2G
        grid = self.p2g(feats, cuboids)  # (B, L, G, G, d_fno)

        # 3. FNO per layer
        grid = self.fno_per_layer(grid)

        # 4. Z fusion
        grid = self.z_fusion(grid)

        # 5. G2P
        feats_rich = self.g2p(grid, cuboids)        # (B, N, d_fno)
        feats_rich = self.proj(feats_rich)           # (B, N, d_enc)

        # 6a. GND cap
        is_target = (cuboids[..., 7] == 1.0).float().unsqueeze(-1)
        area = (cuboids[..., 3] * cuboids[..., 4]).log1p().unsqueeze(-1)
        eps  = cuboids[..., 8:9].clamp(min=1.0).log()
        z    = cuboids[..., 2:3] / 10.0
        gnd_in = torch.cat([feats_rich, area, eps, z], -1)
        c_gnd = F.softplus(self.gnd_head(gnd_in)) * is_target

        # 6b. CPL cap (approximate: only predict for adjacent pairs)
        # Full implementation would use sparse edge selection
        src = feats_rich[:, :, :]
        # Return placeholder CPL for timing purposes
        c_cpl = torch.zeros(B, N, N, device=device)

        return {'c_gnd': c_gnd.squeeze(-1), 'c_cpl': c_cpl}


# ─────────────────────────────────────────────
# Voxel FNO baseline (for comparison)
# ─────────────────────────────────────────────

class VoxelFNO(nn.Module):
    """Standard voxel FNO: rasterize → FNO-2D → global pool → cap."""
    def __init__(self, in_ch=32, d_fno=64, fno_modes=16, n_blocks=4, G=128):
        super().__init__()
        self.G = G
        self.lift = nn.Conv2d(in_ch, d_fno, 1)
        self.blocks = nn.ModuleList([FNOBlock2D(d_fno, fno_modes) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Sequential(nn.Linear(d_fno * 16, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        x = self.lift(voxel)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.pool(x).flatten(1))


# ─────────────────────────────────────────────
# Benchmark utilities
# ─────────────────────────────────────────────

def make_synthetic_batch(B: int, N: int = 200, C: int = 10, G: int = 128,
                         n_layers: int = N_LAYERS, device=None):
    """Create a realistic synthetic input batch."""
    cub = torch.zeros(B, N, C, device=device)
    cub[..., 0] = torch.rand(B, N) * 8.0 - 4.0   # x_rel
    cub[..., 1] = torch.rand(B, N) * 8.0 - 4.0   # y_rel
    cub[..., 2] = torch.rand(B, N) * 4.0 + 0.5   # z_abs
    cub[..., 3] = torch.rand(B, N) * 0.5 + 0.044  # w
    cub[..., 4] = cub[..., 3]                      # h ≈ w
    cub[..., 5] = torch.rand(B, N) * 0.3 + 0.05   # d (thickness)
    cub[..., 7] = (torch.rand(B, N) > 0.9).float() # is_target
    cub[..., 8] = torch.ones(B, N) * 3.9           # eps
    pad = torch.zeros(B, N, dtype=torch.bool, device=device)
    # voxel (for VoxelFNO)
    vox = torch.rand(B, n_layers * 4, G, G, device=device)
    return cub, pad, vox


def time_model(model, inputs, n_warmup=5, n_runs=20, label=""):
    """Measure median inference time."""
    device = next(model.parameters()).device

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            if isinstance(inputs, tuple) and len(inputs) == 2 and not isinstance(inputs[0], torch.Tensor):
                _ = model(*inputs)
            else:
                _ = model(*inputs)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(*inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    med = np.median(times) * 1000  # ms
    p95 = np.percentile(times, 95) * 1000
    return med, p95


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--grid_size', type=int, default=64,
                        help='GINO latent grid size (G×G per layer)')
    parser.add_argument('--n_cuboids', type=int, default=200,
                        help='Cuboids per tile (realistic: 50-300)')
    parser.add_argument('--full_chip_tiles', type=int, default=0,
                        help='If >0, override full-chip tile count for extrapolation')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== GINO Runtime Benchmark ===")
    print(f"Device: {device} | Grid: {args.grid_size}×{args.grid_size}×{N_LAYERS}")
    print(f"Cuboids/tile: {args.n_cuboids}")

    # Estimate full-chip tile count from manifest
    if args.full_chip_tiles > 0:
        chip_tiles = args.full_chip_tiles
    else:
        try:
            df = pd.read_csv(MANIFEST_PATH)
            chip_tiles = len(df[df['split'] == 'test'].groupby('net_name'))
            print(f"  Full-chip tile estimate: {chip_tiles:,} tiles (from manifest)")
        except Exception:
            chip_tiles = 10000
            print(f"  Full-chip tile estimate: {chip_tiles:,} tiles (default)")

    # ── Instantiate models ────────────────────────────────────
    N = args.n_cuboids
    G = args.grid_size

    pex_model  = DeepPEX_Skeleton(in_dim=10, d_model=256, max_cuboids=1024).to(device).eval()
    gino_model = BEOL_GINO(in_dim=10, d_enc=128, d_fno=64, fno_modes=16,
                           n_fno_blocks=4, grid_size=G, n_layers=N_LAYERS,
                           sigma_xy=0.25, sigma_z=0.15, window=8.0).to(device).eval()
    vfno_128   = VoxelFNO(in_ch=N_LAYERS*4, d_fno=64, fno_modes=16,
                          n_blocks=4, G=128).to(device).eval()
    vfno_256   = VoxelFNO(in_ch=N_LAYERS*4, d_fno=64, fno_modes=16,
                          n_blocks=4, G=256).to(device).eval()

    print(f"\n  Parameters:")
    print(f"    DeepPEX Skeleton   : {count_params(pex_model):>10,}")
    print(f"    BEOL-GINO (G={G:>3})  : {count_params(gino_model):>10,}")
    print(f"    Voxel-FNO (128)    : {count_params(vfno_128):>10,}")
    print(f"    Voxel-FNO (256)    : {count_params(vfno_256):>10,}")

    if device.type == 'cuda':
        mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"  GPU: {torch.cuda.get_device_name(device)} ({mem:.1f} GB)")

    # ── Benchmark: single tile latency ────────────────────────
    print(f"\n── Single tile latency (B=1, N={N}) ──────────────────────")
    batch_sizes = [1, 4, 8, 16]
    results = {}

    for B in batch_sizes:
        cub, pad, vox = make_synthetic_batch(B, N, G=128, device=device)
        cub256, _, vox256 = make_synthetic_batch(B, N, G=256, device=device)

        row = {'B': B}

        # DeepPEX: only feasible with small N×N CPL (approximation)
        # Use smaller N for CPL-heavy pex to avoid OOM
        N_pex = min(N, 64)  # CPL is O(N²) — limit for timing
        cub_p, pad_p, _ = make_synthetic_batch(B, N_pex, device=device)
        med, p95 = time_model(pex_model, (cub_p, pad_p), label='DeepPEX')
        row['pex_ms']  = med; row['pex_p95'] = p95

        # GINO
        med, p95 = time_model(gino_model, (cub, pad), label='GINO')
        row['gino_ms'] = med; row['gino_p95'] = p95

        # VoxelFNO 128
        med, p95 = time_model(vfno_128, (vox,), label='VoxFNO-128')
        row['vfno128_ms'] = med; row['vfno128_p95'] = p95

        # VoxelFNO 256
        med, p95 = time_model(vfno_256, (vox256,), label='VoxFNO-256')
        row['vfno256_ms'] = med; row['vfno256_p95'] = p95

        results[B] = row
        print(f"  B={B:<3}  PEX={row['pex_ms']:>7.1f}ms  "
              f"GINO={row['gino_ms']:>7.1f}ms  "
              f"FNO-128={row['vfno128_ms']:>7.1f}ms  "
              f"FNO-256={row['vfno256_ms']:>7.1f}ms")

    # ── GINO component breakdown (B=1) ────────────────────────
    print(f"\n── GINO Component Timing Breakdown (B=1, N={N}) ──────────")
    cub1, pad1, _ = make_synthetic_batch(1, N, G=G, device=device)
    comp_times = {}

    with torch.no_grad():
        # Encoder
        t0 = time.perf_counter()
        for _ in range(20):
            x = cub1.clone()
            x[..., :3] /= 2.5
            x[..., 3:6] = torch.log1p(x[..., 3:6].clamp(0)) / 1.5
            x[..., 8:9] = torch.log(x[..., 8:9].clamp(min=1.0)) / 2.0
            feats = gino_model.encoder(x)
        if device.type == 'cuda': torch.cuda.synchronize()
        comp_times['Encoder'] = (time.perf_counter() - t0) / 20 * 1000

        # P2G
        t0 = time.perf_counter()
        for _ in range(20):
            grid = gino_model.p2g(feats, cub1)
        if device.type == 'cuda': torch.cuda.synchronize()
        comp_times['P2G'] = (time.perf_counter() - t0) / 20 * 1000

        # FNO per layer
        t0 = time.perf_counter()
        for _ in range(20):
            grid_f = gino_model.fno_per_layer(grid)
        if device.type == 'cuda': torch.cuda.synchronize()
        comp_times['FNO (L layers)'] = (time.perf_counter() - t0) / 20 * 1000

        # Z fusion
        t0 = time.perf_counter()
        for _ in range(20):
            grid_z = gino_model.z_fusion(grid_f)
        if device.type == 'cuda': torch.cuda.synchronize()
        comp_times['Z-MLP'] = (time.perf_counter() - t0) / 20 * 1000

        # G2P
        t0 = time.perf_counter()
        for _ in range(20):
            fr = gino_model.g2p(grid_z, cub1)
        if device.type == 'cuda': torch.cuda.synchronize()
        comp_times['G2P'] = (time.perf_counter() - t0) / 20 * 1000

    total_comp = sum(comp_times.values())
    for comp, ms in comp_times.items():
        print(f"  {comp:<20} : {ms:>8.2f} ms  ({ms/total_comp*100:>5.1f}%)")
    print(f"  {'Total (no CPL head)':<20} : {total_comp:>8.2f} ms")

    # ── Full chip extrapolation ────────────────────────────────
    print(f"\n── Full-Chip Latency Extrapolation ({chip_tiles:,} tiles) ─────")
    # Assume B=8 throughput (batch inference)
    B_inf = 8
    tiles_per_sec_pex  = B_inf / (results[B_inf]['pex_ms']  / 1000) if B_inf in results else None
    tiles_per_sec_gino = B_inf / (results[B_inf]['gino_ms'] / 1000) if B_inf in results else None
    tiles_per_sec_fno  = B_inf / (results[B_inf]['vfno128_ms'] / 1000) if B_inf in results else None

    models_ext = [
        ("DeepPEX",     tiles_per_sec_pex),
        ("BEOL-GINO",   tiles_per_sec_gino),
        ("VoxelFNO-128",tiles_per_sec_fno),
        ("StarRC",      None),
    ]
    print(f"  {'Model':<16}  {'Tiles/sec':>12}  {'Full-chip (min)':>17}")
    print("  " + "─" * 50)
    for name, tps in models_ext:
        if tps is None and name == 'StarRC':
            print(f"  {name:<16}  {'N/A':>12}  {'30-180 min (ref)':>17}")
        elif tps is not None:
            eta_min = chip_tiles / tps / 60
            print(f"  {name:<16}  {tps:>12.1f}  {eta_min:>16.1f} min")

    # ── GPU memory ─────────────────────────────────────────────
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        cub8, pad8, _ = make_synthetic_batch(8, N, G=G, device=device)
        with torch.no_grad():
            _ = gino_model(cub8, pad8)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"\n── GPU Memory (GINO, B=8, N={N}) ─────────────────────────")
        print(f"  Peak allocated: {peak_mb:.0f} MB")
        print(f"  Latent grid ({8}×{N_LAYERS}×{G}×{G}×{gino_model.d_fno}): "
              f"{8*N_LAYERS*G*G*gino_model.d_fno*4/1e6:.0f} MB")

    # ── Save report ────────────────────────────────────────────
    out = Path('output_intel22') / 'diag_gino_runtime.txt'
    out.parent.mkdir(exist_ok=True)
    with open(out, 'w') as fp:
        fp.write("=== GINO Runtime Benchmark ===\n")
        fp.write(f"Device: {device}, Grid: {G}x{G}x{N_LAYERS}, N={N}\n\n")
        fp.write(f"Parameters:\n")
        fp.write(f"  DeepPEX Skeleton  : {count_params(pex_model):,}\n")
        fp.write(f"  BEOL-GINO (G={G}) : {count_params(gino_model):,}\n")
        fp.write(f"  VoxelFNO-128      : {count_params(vfno_128):,}\n\n")
        fp.write(f"Latency (ms, median over 20 runs):\n")
        fp.write(f"{'B':>4}  {'PEX':>8}  {'GINO':>8}  {'FNO-128':>8}  {'FNO-256':>8}\n")
        for B, row in results.items():
            fp.write(f"{B:>4}  {row['pex_ms']:>8.1f}  {row['gino_ms']:>8.1f}  "
                     f"{row['vfno128_ms']:>8.1f}  {row['vfno256_ms']:>8.1f}\n")
        fp.write(f"\nComponent breakdown (B=1):\n")
        for comp, ms in comp_times.items():
            fp.write(f"  {comp:<20} : {ms:.2f} ms\n")
        fp.write(f"\nFull-chip extrapolation ({chip_tiles} tiles, B={B_inf}):\n")
        for name, tps in models_ext:
            if tps is not None:
                fp.write(f"  {name}: {tps:.1f} tiles/s → {chip_tiles/tps/60:.1f} min\n")
    print(f"\nResult saved → {out}")


if __name__ == '__main__':
    main()
