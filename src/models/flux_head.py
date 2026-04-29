# ==========================================================
# FILE: src/models/flux_head.py (Rigorous Surface PINN Version)
# ==========================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.compute_sheilding import compute_sparse_shield_factor

class NeuralFluxRouter(nn.Module):
    def __init__(self, model_dim=128, context_radius=2, cutoff_radius=2, layer_map=None,
                 use_rail_coupling=False):
        super().__init__()
        self.context_r = context_radius
        self.cutoff_r = cutoff_radius
        self.use_rail_coupling = use_rail_coupling

        Z_dim = model_dim + 4
        self.norm = nn.LayerNorm(Z_dim)

        # ---------------------------------------------------------
        # [GND MLP] GND 추론: Z_prime + area + eps + z + wide_density + log_n + endpoint_prox
        # Input: Z_prime(Z_dim) + log1p(area)(1) + log1p(eps)(1) + z_abs(1) + wide_density(1) + log_n(1) + ep(1)
        # ---------------------------------------------------------
        self.gnd_mlp = nn.Sequential(
            nn.LayerNorm(Z_dim + 6),
            nn.Linear(Z_dim + 6, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # ---------------------------------------------------------
        # [CPL MLP] Surface physics → coupling modifier + residual.
        # [FIX I] Pre-project Z-features to 32-dim edge space, then fuse with
        # 9-channel normalized edge_geom (8 physics + 1 rank). Input dim drops
        # from 788→105 (12× squeeze → 2× squeeze); hidden widened 64→256.
        # [FIX I] Fresh cpl_mlp params; legacy ckpt cpl_mlp.* shapes mismatch
        # and are filtered by run_active_learning.py's shape-filtered loader.
        # ---------------------------------------------------------
        self.cpl_edge_proj = nn.Linear(Z_dim, 32)
        # v10b: edge_geom 12 channels (lateral/broadside decomposition + d_lateral + P_over).
        CPL_IN = 32 * 3 + 12
        self.cpl_mlp = nn.Sequential(
            nn.LayerNorm(CPL_IN),
            nn.Linear(CPL_IN, 256),
            nn.GELU(),
            nn.Linear(256, 2)  # [0]: logit_mult, [1]: logit_add
        )

        # [FIX H] Z-anchors indexed by metal layer positions from layer_map,
        # not a 61-bin dense grid. Rare bins got no gradient and drifted; the
        # compact (K,) parameter gets a gradient every batch. Fallback to legacy
        # 61-bin grid if layer_map is missing so the class stays robust.
        if layer_map is not None and len(layer_map) > 0:
            zs = sorted({float(info.get('z_pos', 0.0)) for info in layer_map.values()})
            unique_z = []
            for z in zs:
                if len(unique_z) == 0 or abs(z - unique_z[-1]) > 0.05:
                    unique_z.append(z)
            metal_z = torch.tensor(unique_z, dtype=torch.float32)
        else:
            metal_z = torch.linspace(-10.0, 50.0, 61)
        self.register_buffer('metal_z_anchors', metal_z)
        self.num_anchors = int(self.metal_z_anchors.numel())
        self.layer_scale_phys_gnd = nn.Parameter(self._make_gnd_cap_density_init())
        # v10b: K×K symmetric layer-pair coupling scale replaces scalar phys_scale_cpl(K).
        # Diagonal = same-layer correction; off-diagonal = cross-layer coupling bias.
        # init=0 → softplus(0)=log(2)≈0.693; symmetry enforced at lookup time.
        self.cpl_layer_pair_log_scale = nn.Parameter(torch.zeros(self.num_anchors, self.num_anchors))

        # [Fix 2] Per-layer fringe/sidewall scale: learns how much sidewall area
        # contributes to GND cap. Thick upper metals (M7 2μm, M8 4.5μm) have
        # sidewall >> bottom face → this corrects the bottom-face-only bias.
        # v9: physics-based init (M1≈0.05, M7≈0.70, M8≈0.90) instead of uniform 0.12.
        self.gnd_fringe_scale = nn.Parameter(self._make_gnd_fringe_scale_init())

        # v9-c: Per-layer scale for explicit VSS/GND rail coupling (Sakurai-Tamaru geometry).
        # softplus(-3) ≈ 0.049 — starts near-zero; training ramps it to the correct level.
        self.vss_gnd_scale = nn.Parameter(torch.full((self.num_anchors,), -3.0))

        # [Fix 5 / v8b] Learnable scale for explicit VSS rail coupling term.
        # Only active when use_rail_coupling=True. softplus(-3) ≈ 0.049 initially.
        # -3.0 (not -1.0): rail_proximity after log1p is O(0-5), so softplus(-1)*5≈1.57 fF
        # per cuboid was 100-1000× too large; -3.0 → 0.049*5≈0.25 fF, much safer.
        if self.use_rail_coupling:
            self.rail_scale = nn.Parameter(torch.tensor(-3.0))

        # edge_geom: 12 channels (lateral/broadside physics + direction features); normalize
        # before MLP fusion to give the first Linear O(1) inputs.
        self.edge_geom_norm = nn.LayerNorm(12)

        # SSL BEM pretraining head: features → (q_raw, log_equiv_r).
        # q_raw: per-cuboid BEM charge (signed); equiv_radius: min self-distance (prevents V→∞).
        # Only used in train_ssl.py; idle in AL/eval (compute_coupling=True path skips it).
        self.charge_basis_mlp = nn.Sequential(
            nn.LayerNorm(Z_dim),
            nn.Linear(Z_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2)  # [0]: q_raw, [1]: log_equiv_r
        )

        # [FIX B] Initialize heads near identity of the physics base:
        #   gnd_modifier = exp(0) = 1.0                    (bias = 0)
        #   cpl_modifier = sigmoid(-2.3026)*9.9 + 0.1 ≈ 1  (logit_mult bias)
        #   cpl_residual = softplus(-4.0)*base*0.1 ≈ 0     (logit_add bias)
        for mlp in [self.cpl_mlp, self.gnd_mlp, self.charge_basis_mlp]:
            nn.init.normal_(mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.gnd_mlp[-1].bias)
        nn.init.zeros_(self.charge_basis_mlp[-1].bias)
        with torch.no_grad():
            self.cpl_mlp[-1].bias.copy_(torch.tensor([-2.3026, -4.0]))

    def _make_gnd_cap_density_init(self) -> torch.Tensor:
        import math
        z_anchors = self.metal_z_anchors.tolist()
        def density_for_z(z: float) -> float:
            # v6 re-calibration: chip_gnd=0.20 observed with effective=0.25 fF/um2.
            # gnd_modifier converges to ~0.5 during training; raise density 5x so
            # effective at convergence (density * 0.5) matches target ~1.25 fF/um2.
            # All values 5x their v5b counterparts.
            if z < 0.40:   return 2.50   # pre-M1  (was 0.50)
            elif z < 0.60: return 2.50   # M1      (was 0.50)
            elif z < 0.75: return 3.00   # M2      (was 0.60)
            elif z < 0.90: return 3.00   # M3      (was 0.60)
            elif z < 1.05: return 2.75   # M4      (was 0.55)
            elif z < 1.20: return 2.75   # M5      (was 0.55)
            elif z < 1.45: return 2.50   # M6      (was 0.50)
            elif z < 4.50: return 1.50   # upper   (was 0.30)
            elif z < 6.00: return 4.00   # top     (was 0.80)
            else:           return 2.00  # others  (was 0.40)
        inits = []
        for z in z_anchors:
            d = density_for_z(z)
            x = math.log(math.exp(d) - 1.0)
            inits.append(x)
        return torch.tensor(inits, dtype=torch.float32)

    def _make_gnd_fringe_scale_init(self) -> torch.Tensor:
        """Physics-based per-layer fringe_frac initialization.
        sigmoid(x) = fringe_frac. Values derived from conductor thickness ratio:
        fringe_frac ≈ sidewall_area / (sidewall_area + bottom_area).
        M1 (thin) → small fringe; M8 (4.5μm thick) → sidewall dominates.
        """
        def init_for_z(z: float) -> float:
            if   z < 0.55: return -2.94   # M1   fringe_frac ≈ 0.05
            elif z < 0.70: return -2.44   # M2   fringe_frac ≈ 0.08
            elif z < 0.84: return -2.20   # M3   fringe_frac ≈ 0.10
            elif z < 0.99: return -1.73   # M4   fringe_frac ≈ 0.15
            elif z < 1.14: return -1.39   # M5   fringe_frac ≈ 0.20
            elif z < 1.45: return -0.85   # M6   fringe_frac ≈ 0.30
            elif z < 4.60: return -1.73   # M7   fringe_frac ≈ 0.15 (lateral → explicit VSS edge)
            else:           return -1.39  # M8   fringe_frac ≈ 0.20 (lateral → explicit VSS edge)
        return torch.tensor([init_for_z(z) for z in self.metal_z_anchors.tolist()],
                             dtype=torch.float32)

    @torch.compiler.disable
    def forward(self, features, cuboids, is_target, is_aggr, padding_mask, compute_coupling=True, frw_ratio_matrix=None, n_tiles=None, endpoint_prox=None):
        B, N, D = features.shape
        device = features.device
        EPS_0 = 0.008854 
        
        centers = cuboids[..., :3].clone()
        centers[padding_mask] = 1e5
        sizes = cuboids[..., 3:6]
        epsilons = torch.clamp(cuboids[..., 8:9], min=1.0)
        z_abs = cuboids[..., 2:3]

        # v9: channel 9 = net_type (0.0=signal, 0.67=VDD, 1.0=VSS).
        # Falls back to all-zero (signal) if cuboids has only 9 channels.
        if cuboids.shape[-1] >= 10:
            net_type = cuboids[..., 9]
        else:
            net_type = torch.zeros(B, N, device=cuboids.device)

        # [기하학의 뼈대] Min/Max BBox Coordinates
        mins = centers - sizes / 2.0
        maxs = centers + sizes / 2.0

        w, h, d = sizes[..., 0], sizes[..., 1], sizes[..., 2]
        is_aggr, is_target = is_aggr.float(), is_target.float()

        # [FIX F] Pad rows have cuboid[...,7]==0 → is_aggr=True; that leaks pad
        # into is_aggr-weighted bmms (ambient_potential) and into edge detection
        # at line 117. Zero w/h/d/eps at pad positions too so area_face and
        # context bmms carry no pad mass. centers are already pushed to 1e5 above.
        valid = (~padding_mask).float()
        is_aggr   = is_aggr   * valid
        is_target = is_target * valid
        w = w * valid
        h = h * valid
        d = d * valid
        epsilons = epsilons * valid.unsqueeze(-1)

        # v9: power rail aggressors (VSS/VDD) must not generate CPL edges; they
        # are context-only for GND estimation via wide_density.
        is_power = (net_type > 0.5) * valid   # VSS (1.0) or VDD (0.67) cuboids
        is_vss   = (net_type > 0.9) * valid   # VSS-only (1.0) — for explicit GND edges
        is_signal_aggr = is_aggr * (1.0 - is_power)  # aggressors excluding power rails

        # 1. 노드 수준(Node-level) 기하학 및 Z_prime 임베딩
        # 상관계수 0.72를 기록했던 '절대 표면적' 연산
        area_face = torch.clamp(2.0 * (w*h + h*d + d*w).unsqueeze(-1), min=1e-6)
        
        # 거시적 밀도 및 맥락 인지를 위한 Graph Context (유지)
        dist_vec = torch.clamp(torch.max(mins.unsqueeze(2) - maxs.unsqueeze(1), mins.unsqueeze(1) - maxs.unsqueeze(2)), min=0.0) 
        closest_dist = torch.norm(dist_vec, dim=-1)
        adj_mask = (closest_dist < self.context_r) & ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        inv_dist_ctx = 1.0 / torch.clamp(closest_dist, min=0.1)
        adj_matrix = (1.0 / torch.clamp(closest_dist**2, min=0.01)) * adj_mask.float() 
        
        analytical_density = torch.log1p(torch.bmm(adj_matrix, area_face) * 0.01)
        ambient_potential = torch.log1p(torch.bmm(inv_dist_ctx * adj_mask.float(), is_aggr.unsqueeze(-1) * area_face) * 0.01)
        target_charge_proxy = is_target.unsqueeze(-1) * area_face
        segment_ratio = area_face / (torch.sum(target_charge_proxy, dim=1, keepdim=True) + 1e-6)
        self_potential = torch.log1p(torch.bmm(inv_dist_ctx * adj_mask.float(), target_charge_proxy) * 0.01)

        # [Fix 4] Wide aggressor (power rail proxy) density for each target cuboid.
        # Wide conductor (w > 2μm on any layer) ≈ VSS/VDD stripe; its proximity
        # correlates with rail-coupling GND cap missed by the bottom-face formula.
        # v9: use is_power (VSS/VDD channel) as power rail signal; fall back to
        # w > 2μm width proxy when USE_VSS_AGGRESSORS=False (9-channel tiles).
        # .any().item() converts to Python bool to avoid tensor-in-if autograd hazard.
        power_signal = is_power if is_power.any().item() else (w > 2.0).float() * is_aggr
        wide_aggr_area = (w * h * power_signal).unsqueeze(-1)   # (B, N, 1)
        wide_density = torch.log1p(
            torch.bmm(inv_dist_ctx * adj_mask.float(), wide_aggr_area) * 0.1
        )  # (B, N, 1) — log-scaled for MLP stability

        # Node Embedding 완성
        Z_prime = self.norm(torch.cat([features, analytical_density, ambient_potential, self_potential, segment_ratio], dim=-1))

        # BEM charge basis — only computed in SSL forward (compute_coupling=False).
        # q_raw: signed charge per target cuboid; equiv_radius: self-distance floor (μm).
        if not compute_coupling:
            _cb = self.charge_basis_mlp(Z_prime)       # (B, N, 2)
            q_raw = _cb[..., 0] * is_target.float() * valid
            equiv_radius = F.softplus(_cb[..., 1]) * 0.1 + 0.005  # [0.005, ~0.5] μm

        # ---------------------------------------------------------
        # 2. [GND Physics] Ground Capacitance 예측
        # ---------------------------------------------------------
        # BEOL ground coupling은 기판(z_abs 거리)이 아니라 로컬 power rail/via에
        # 지배됨 (유효거리 ~0.05 μm). 기존 EPS_0*A/z_abs는 ~275× 과소추정.
        # 새 공식: w_gnd_base = cap_density_layer × area_face
        # cap_density_layer (fF/μm²)는 레이어별 학습 가능한 empirical 상수.
        # 초기값은 StarRC golden 캘리브레이션 (M1~7.02, M2~7.97, M3~8.29, ...).
        z_flat = z_abs.squeeze(-1)
        z_idx_gnd = torch.argmin(
            torch.abs(z_flat.unsqueeze(-1) - self.metal_z_anchors), dim=-1
        )
        layer_cap_density = F.softplus(self.layer_scale_phys_gnd[z_idx_gnd])  # (B, N) fF/μm²
        # [Fix 2] Fringe-aware GND area: bottom face + learned sidewall fraction.
        # M7 (2.1 μm thick) and M8 (4.5 μm thick) have sidewall >> bottom face;
        # bottom-face-only formula was missing ≥9× sidewall contribution for CTS nets.
        bottom_area = (w * h).clamp(min=1e-6)                                 # (B, N) μm²
        sidewall_area = 2.0 * (w + h) * d                                     # (B, N) μm²
        fringe_frac = torch.sigmoid(self.gnd_fringe_scale[z_idx_gnd])         # (B, N) ∈ (0,1)
        gnd_area = bottom_area + fringe_frac * sidewall_area                   # (B, N) μm²
        w_gnd_base = gnd_area * layer_cap_density                              # (B, N) fF

        # n_tiles conditioning: log-scale tile count (wire length proxy) + endpoint proximity.
        # Defaults to zeros (neutral) when n_tiles/endpoint_prox not supplied (SSL, selector).
        # Cast to features.dtype so cat with Z_prime doesn't fail under bfloat16 autocast.
        feat_dtype = features.dtype
        if n_tiles is not None:
            log_n = torch.log1p(n_tiles.to(feat_dtype)).view(B, 1, 1).expand(B, N, 1)
        else:
            log_n = torch.zeros(B, N, 1, device=device, dtype=feat_dtype)
        if endpoint_prox is not None:
            ep = endpoint_prox.to(feat_dtype).view(B, 1, 1).expand(B, N, 1)
        else:
            ep = torch.zeros(B, N, 1, device=device, dtype=feat_dtype)

        gnd_feats = torch.cat([
            Z_prime,
            torch.log1p(area_face),
            torch.log1p(epsilons),
            torch.clamp(z_abs, min=0.0)/10.0,
            wide_density,   # [Fix 4] power rail proximity proxy (log-scaled)
            log_n,          # wire-length proxy (log n_tiles)
            ep,             # endpoint proximity (0=endpoint tile, 1=middle tile)
        ], dim=-1)

        gnd_modifier = torch.exp(torch.clamp(self.gnd_mlp(gnd_feats).squeeze(-1), min=-3.0, max=3.0))

        c_gnd_seg = w_gnd_base * gnd_modifier * is_target.float() * (~padding_mask).float()

        # [Fix 5 / v8b] Explicit VSS rail coupling addend.
        # Wide aggressors (w > 2 μm) approximate VDD/VSS stripes; their proximity
        # drives rail coupling GND cap that the substrate-density formula cannot learn.
        # Only active when use_rail_coupling=True (v8: False, v8b: True).
        # log1p scaling: matches wide_density scale (O(0-5), dimensionless).
        # Avoids raw rail_proximity O(10^4 μm) × EPS_0 blow-up at init.
        if self.use_rail_coupling:
            rail_proximity = torch.log1p(
                torch.bmm(inv_dist_ctx * adj_mask.float(), wide_aggr_area).squeeze(-1) * 0.1
            )  # (B, N) log-scaled, O(0-5), same range as wide_density
            c_rail = F.softplus(self.rail_scale) * rail_proximity * is_target.float() * valid
            c_gnd_seg = c_gnd_seg + c_rail

        # v9-c: Explicit VSS→target sparse edges with Sakurai-Tamaru geometry.
        # Uses is_vss (net_type > 0.9, VSS-only) to avoid conflating VDD coupling
        # with GND cap. wide_density in gnd_feats proxies the same signal but has
        # no directionality; this term adds facing-area and distance geometry.
        vss_edges = (
            (is_target > 0.5).unsqueeze(2) &
            (is_vss > 0.5).unsqueeze(1) &
            (closest_dist < self.cutoff_r)
        ).nonzero(as_tuple=False)

        if len(vss_edges) > 0:
            bv, sv, dv = vss_edges[:, 0], vss_edges[:, 1], vss_edges[:, 2]

            src_mins_v, dst_mins_v = mins[bv, sv], mins[bv, dv]
            src_maxs_v, dst_maxs_v = maxs[bv, sv], maxs[bv, dv]

            dx_v = torch.clamp(torch.maximum(src_mins_v[:, 0], dst_mins_v[:, 0]) - torch.minimum(src_maxs_v[:, 0], dst_maxs_v[:, 0]), min=0.0)
            dy_v = torch.clamp(torch.maximum(src_mins_v[:, 1], dst_mins_v[:, 1]) - torch.minimum(src_maxs_v[:, 1], dst_maxs_v[:, 1]), min=0.0)
            dz_v = torch.clamp(torch.maximum(src_mins_v[:, 2], dst_mins_v[:, 2]) - torch.minimum(src_maxs_v[:, 2], dst_maxs_v[:, 2]), min=0.0)
            D_v = torch.clamp(torch.sqrt(dx_v**2 + dy_v**2 + dz_v**2), min=1e-4) + 0.005

            ox_v = torch.clamp(torch.minimum(src_maxs_v[:, 0], dst_maxs_v[:, 0]) - torch.maximum(src_mins_v[:, 0], dst_mins_v[:, 0]), min=0.0)
            oy_v = torch.clamp(torch.minimum(src_maxs_v[:, 1], dst_maxs_v[:, 1]) - torch.maximum(src_mins_v[:, 1], dst_mins_v[:, 1]), min=0.0)
            oz_v = torch.clamp(torch.minimum(src_maxs_v[:, 2], dst_maxs_v[:, 2]) - torch.maximum(src_mins_v[:, 2], dst_mins_v[:, 2]), min=0.0)
            A_v = torch.maximum(torch.maximum(ox_v * oy_v, oy_v * oz_v), oz_v * ox_v)
            L_v = torch.max(torch.max(ox_v, oy_v), oz_v)
            T_v = torch.minimum(sizes[bv, sv, 2], sizes[bv, dv, 2])

            eps_v = (epsilons[bv, sv].view(-1) + epsilons[bv, dv].view(-1)) / 2.0
            c_vss_base = EPS_0 * eps_v * (A_v / D_v + L_v * torch.log1p(T_v / D_v) * (2.0 / math.pi))

            z_src_v = z_abs[bv, sv, 0]
            z_idx_v = torch.argmin(torch.abs(z_src_v.unsqueeze(-1) - self.metal_z_anchors), dim=-1)
            c_vss = c_vss_base * F.softplus(self.vss_gnd_scale[z_idx_v])

            flat_sv = bv * N + sv
            vss_contribution = torch.zeros(B * N, device=device, dtype=c_vss.dtype).index_add(0, flat_sv, c_vss)
            c_gnd_seg = c_gnd_seg + vss_contribution.view(B, N) * is_target.float() * valid

        if not compute_coupling:
            return {
                'c_total_phys': c_gnd_seg,
                'c_gnd_seg': c_gnd_seg,
                'q_raw': q_raw,
                'equiv_radius': equiv_radius,
            }
            
        # ---------------------------------------------------------
        # 3. [CPL Physics] Sparse Coupling 예측 (Neural Shader 도입)
        # ---------------------------------------------------------
        # v9: exclude power rail aggressors (VSS/VDD) from CPL edges; they
        # contribute to GND cap, not inter-wire coupling.
        edges = ((is_target > 0.5).unsqueeze(2) & (is_signal_aggr > 0.5).unsqueeze(1) & (closest_dist < self.cutoff_r)).nonzero(as_tuple=False)
        sparse_cpl = {k: list() for k in ['b_idx', 'src_idx', 'dst_idx', 'c_cpl', 'w_cpl', 'cpl_modifier', 'cpl_residual', 'overlap_area', 'soft_dist', 'shield_E', 'L_eff']}

        sum_c_cpl_flat = torch.zeros(B * N, device=device)
        
        if len(edges) > 0:
            b_e, src_e, dst_e = edges[:, 0], edges[:, 1], edges[:, 2]
            
            # [기하학의 정수] 표면 대 표면 거리 연산 (Surface-to-Surface)
            src_mins, dst_mins = mins[b_e, src_e], mins[b_e, dst_e]
            src_maxs, dst_maxs = maxs[b_e, src_e], maxs[b_e, dst_e]
            
            dx_gap = torch.clamp(torch.maximum(src_mins[:, 0], dst_mins[:, 0]) - torch.minimum(src_maxs[:, 0], dst_maxs[:, 0]), min=0.0)
            dy_gap = torch.clamp(torch.maximum(src_mins[:, 1], dst_mins[:, 1]) - torch.minimum(src_maxs[:, 1], dst_maxs[:, 1]), min=0.0)
            dz_gap = torch.clamp(torch.maximum(src_mins[:, 2], dst_mins[:, 2]) - torch.minimum(src_maxs[:, 2], dst_maxs[:, 2]), min=0.0)
            D_surf = torch.clamp(torch.sqrt(dx_gap**2 + dy_gap**2 + dz_gap**2), min=1e-4)
            
            # [기하학의 정수] 마주보는 3축 투영 면적 (Projected Overlap Area)
            ox = torch.clamp(torch.minimum(src_maxs[:, 0], dst_maxs[:, 0]) - torch.maximum(src_mins[:, 0], dst_mins[:, 0]), min=0.0)
            oy = torch.clamp(torch.minimum(src_maxs[:, 1], dst_maxs[:, 1]) - torch.maximum(src_mins[:, 1], dst_mins[:, 1]), min=0.0)
            oz = torch.clamp(torch.minimum(src_maxs[:, 2], dst_maxs[:, 2]) - torch.maximum(src_mins[:, 2], dst_mins[:, 2]), min=0.0)
            # [FIX E] Dominant-plane area retained for diagnostic dict.
            A_over = torch.maximum(torch.maximum(ox * oy, oy * oz), oz * ox)
            P_over = 2.0 * (ox + oy + oz)
            L_eff  = torch.max(torch.max(ox, oy), oz)

            # v10b: directional area decomposition (lateral vs broadside).
            A_xy = ox * oy                      # broadside: XY face overlap (inter-layer)
            A_xz = ox * oz                      # xz lateral
            A_yz = oy * oz                      # yz lateral
            A_lateral = torch.maximum(A_xz, A_yz)
            # In-plane lateral gap: L2 norm of X,Y components — correct for diagonal pairs.
            d_lateral = torch.clamp(torch.sqrt(dx_gap**2 + dy_gap**2), min=1e-4)

            # Direction-weighted ε_eff: harmonic mean for vertical (series dielectric),
            # arithmetic mean for horizontal (parallel). cos²θ blends the two.
            # Same-layer (dz_gap=0) → arithmetic. Broadside (dz_gap=D_surf) → harmonic.
            eps_src_e    = epsilons[b_e, src_e].view(-1).clamp(min=1.0)
            eps_dst_e    = epsilons[b_e, dst_e].view(-1).clamp(min=1.0)
            eps_arith    = (eps_src_e + eps_dst_e) / 2.0
            eps_harmonic = 2.0 / (1.0 / eps_src_e + 1.0 / eps_dst_e)
            cos2_theta   = (dz_gap / D_surf.clamp(min=1e-6)).clamp(0.0, 1.0) ** 2
            eps_pair     = eps_harmonic * cos2_theta + eps_arith * (1.0 - cos2_theta)

            # Raycast Shielding (Ray-tracing 물리 엔진 결과값은 Raw Feature로 활용)
            shield_E = compute_sparse_shield_factor(cuboids, b_e, src_e, dst_e).view(-1)
            D_eff = D_surf + 0.005

            # T_eff = min conductor thickness of the pair (needed for Sakurai-Tamaru fringe)
            T_eff = torch.minimum(sizes[b_e, src_e, 2], sizes[b_e, dst_e, 2])

            # v10b: lateral + broadside physics decomposition.
            # Lateral (same-layer sidewall): arithmetic ε, in-plane L2 distance.
            # Guard fringe: skip when A_lateral≈0 to prevent d_lateral→1e-4 blow-up.
            lateral_guard = (A_lateral > 1e-6).float()
            base_lateral = EPS_0 * eps_arith * (
                A_lateral / (d_lateral + 0.005) +
                lateral_guard * L_eff * torch.log1p(T_eff / (d_lateral + 0.005)) * (2.0 / math.pi)
            )
            # Broadside (inter-layer face-to-face): harmonic ε, vertical gap.
            base_broadside = EPS_0 * eps_harmonic * A_xy / (dz_gap + 0.005)
            w_cpl_base = (base_lateral + base_broadside) * torch.clamp(shield_E, min=0.01)

            # Stage 3: 물리적으로 무의미한 edge 제거 (small net coupling 과대추정 방지)
            sig = w_cpl_base > 1e-5
            if sig.sum() < sig.shape[0]:
                b_e, src_e, dst_e = b_e[sig], src_e[sig], dst_e[sig]
                D_surf, A_over, P_over, L_eff = D_surf[sig], A_over[sig], P_over[sig], L_eff[sig]
                dz_gap, eps_pair = dz_gap[sig], eps_pair[sig]
                eps_arith, eps_harmonic = eps_arith[sig], eps_harmonic[sig]
                shield_E, w_cpl_base, D_eff, T_eff = shield_E[sig], w_cpl_base[sig], D_eff[sig], T_eff[sig]
                A_xy, A_lateral, d_lateral = A_xy[sig], A_lateral[sig], d_lateral[sig]
                base_lateral, base_broadside = base_lateral[sig], base_broadside[sig]

            # 2D 차원 맞추기
            E_edges = b_e.shape[0]
            def force_2d(t):
                if not isinstance(t, torch.Tensor): t = torch.tensor(t, device=device, dtype=torch.float32)
                if t.dim() == 0: t = t.view(1).expand(E_edges)
                if t.dim() == 1: return t.expand(E_edges) if t.shape[0] == 1 and E_edges > 1 else t.unsqueeze(-1)
                return t

            # ---------------------------------------------------------
            # 🧠 Neural Shader (MLP Input)
            # 순수하고 강력한 8가지 물리/기하학 텐서 집합
            # ---------------------------------------------------------
            # [NEW — occlusion rank] Probe showed F41/F45/C5 (rank-discounted
            # features) cluster at top-tier; current base has no rank prior.
            # For each edge (src→dst), rank = #aggressors closer to src than dst.
            # O(E·N); non-differentiable (argmin-like) — fed into the MLP as
            # context, no gradient needs to flow back through rank itself.
            cd_src_all = closest_dist[b_e, src_e]                                 # (E, N)
            cd_src_dst = cd_src_all.gather(1, dst_e.unsqueeze(1)).squeeze(1)      # (E,)
            # v9: use is_signal_aggr so wide VSS stripes don't inflate occlusion
            # rank for every signal edge (VSS physically present but not a shielding
            # signal wire in the traditional sense).
            is_signal_aggr_per_edge = is_signal_aggr[b_e]                         # (E, N)
            occ_rank = (
                (cd_src_all < cd_src_dst.unsqueeze(1)) &
                (is_signal_aggr_per_edge > 0.5)
            ).sum(dim=1).float()                                                   # (E,)

            edge_geom = torch.cat([
                force_2d(D_surf),                       # 1. surface-to-surface distance
                force_2d(torch.log1p(A_lateral)),       # 2. lateral area (max xz/yz)
                force_2d(torch.log1p(A_xy)),            # 3. broadside area (xy overlap)
                force_2d(dz_gap),                       # 4. vertical gap
                force_2d(eps_pair),                     # 5. direction-weighted ε_eff
                force_2d(torch.log1p(base_lateral)),    # 6. lateral physics base
                force_2d(torch.log1p(base_broadside)),  # 7. broadside physics base
                force_2d(shield_E),                     # 8. raycast shielding
                force_2d(torch.log1p(occ_rank)),        # 9. occlusion rank
                force_2d(torch.log1p(T_eff)),           # 10. min conductor thickness
                force_2d(d_lateral),                    # 11. in-plane lateral gap
                force_2d(torch.log1p(P_over)),          # 12. perimeter (residual context)
            ], dim=-1)
            # Normalize mixed-scale channels before MLP fusion.
            edge_geom = self.edge_geom_norm(edge_geom)

            Z_i, Z_j = Z_prime[b_e, src_e], Z_prime[b_e, dst_e]
            # Pre-project to 32-dim edge space; cpl_mlp input is
            # 32·3 + 12 = 108.
            Z_i_e, Z_j_e = self.cpl_edge_proj(Z_i), self.cpl_edge_proj(Z_j)

            cpl_logits = self.cpl_mlp(
                torch.cat([Z_i_e + Z_j_e, Z_i_e * Z_j_e, torch.abs(Z_i_e - Z_j_e), edge_geom], dim=-1)
            )
            logit_mult = cpl_logits[:, 0]
            logit_add = cpl_logits[:, 1]

            # v10b: symmetric K×K layer-pair coupling scale.
            # Diagonal = same-layer; off-diagonal = cross-layer biases (M1-M4 etc.).
            # Symmetry enforced via (S + S.T)/2 so training cannot encode asymmetry.
            z_idx_src_cpl = torch.argmin(torch.abs(z_abs[b_e, src_e, 0].unsqueeze(-1) - self.metal_z_anchors), dim=-1)
            z_idx_dst_cpl = torch.argmin(torch.abs(z_abs[b_e, dst_e, 0].unsqueeze(-1) - self.metal_z_anchors), dim=-1)
            sym_lp = (self.cpl_layer_pair_log_scale + self.cpl_layer_pair_log_scale.T) / 2.0
            phys_scale_cpl = F.softplus(sym_lp[z_idx_src_cpl, z_idx_dst_cpl])
            
            # Modifier (상한선을 풀어주어 다이나믹 레인지 확보)
            cpl_modifier = torch.sigmoid(logit_mult) * 9.9 + 0.1
            
            # [FIX C] Gate residual by the physics baseline so it cannot dominate
            # on long-perimeter nets (the prior P_over·0.1 gate made residual grow
            # with wire length independent of w_cpl_base, which softplus gradients
            # exploit → long-net MAPE blowup). Residual is now ≤ ~0.3·w_cpl_base
            # across softplus's useful range.
            cpl_residual = F.softplus(logit_add) * torch.clamp(w_cpl_base, min=1e-9) * 0.1
            
            # [최종 커플링 도출]
            c_cpl_E = (w_cpl_base * phys_scale_cpl * cpl_modifier) + cpl_residual

            # [FIX A] KCL: accumulate per-edge coupling into the target (src) node.
            # dst is always an aggressor here (is_target==0); the `* is_target`
            # mask at the c_total_phys line nulls any dst contribution anyway,
            # so scattering to src alone is correct and cheapest.
            flat_src = b_e * N + src_e
            sum_c_cpl_flat = sum_c_cpl_flat.index_add(0, flat_src, c_cpl_E)

            for key, val in zip(
                ['b_idx', 'src_idx', 'dst_idx', 'c_cpl', 'w_cpl', 'cpl_modifier', 'cpl_residual', 'overlap_area', 'soft_dist', 'shield_E', 'L_eff'], 
                [b_e, src_e, dst_e, c_cpl_E, w_cpl_base, cpl_modifier, cpl_residual, A_over, D_surf, shield_E, L_eff]
            ):
                sparse_cpl[key].append(val.view(-1))

        # ---------------------------------------------------------
        # 4. KCL 기반 최종 Total Capacitance 병합
        # ---------------------------------------------------------
        c_total_phys = c_gnd_seg + sum_c_cpl_flat.view(B, N) * is_target.float() * (~padding_mask).float()
        Z_safe = torch.clamp(c_total_phys, min=1e-9)
        
        for k in sparse_cpl.keys():
            if len(sparse_cpl[k]) > 0:
                sparse_cpl[k] = torch.cat(sparse_cpl[k], dim=0)
            else:
                sparse_cpl[k] = torch.tensor([], device=device)
                
        if len(sparse_cpl['b_idx']) > 0:
            sparse_cpl['p_cpl'] = sparse_cpl['c_cpl'] / Z_safe[sparse_cpl['b_idx'].long(), sparse_cpl['src_idx'].long()]
        else:
            sparse_cpl['p_cpl'] = torch.tensor([], device=device)

        return {
            'c_total_phys': c_total_phys, 
            'c_gnd_seg': c_gnd_seg,
            'sparse_cpl': sparse_cpl, 
            'p_gnd': c_gnd_seg / Z_safe
        }