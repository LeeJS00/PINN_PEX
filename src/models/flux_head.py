# ==========================================================
# FILE: src/models/flux_head.py (Rigorous Surface PINN Version)
# ==========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.compute_sheilding import compute_sparse_shield_factor

class NeuralFluxRouter(nn.Module):
    def __init__(self, model_dim=128, context_radius=2, cutoff_radius=2, layer_map=None):
        super().__init__()
        self.context_r = context_radius
        self.cutoff_r = cutoff_radius
        
        Z_dim = model_dim + 4
        self.norm = nn.LayerNorm(Z_dim)
        
        # ---------------------------------------------------------
        # [GND MLP] 총 면적(Total Area)과 유전율(Eps)을 바탕으로 GND를 추론
        # Input: Z_prime(Z_dim) + Area_total(1) + Eps(1) + Z_abs(1)
        # ---------------------------------------------------------
        self.gnd_mlp = nn.Sequential(
            nn.LayerNorm(Z_dim + 3), 
            nn.Linear(Z_dim + 3, 64), 
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
        CPL_IN = 32 * 3 + 9
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
        self.layer_scale_phys_cpl = nn.Parameter(torch.ones(self.num_anchors))

        # [FIX G] edge_geom now has 9 channels (8 physics + 1 rank); normalize
        # before MLP fusion to give the first Linear O(1) inputs.
        self.edge_geom_norm = nn.LayerNorm(9)

        # [FIX B] Initialize heads near identity of the physics base:
        #   gnd_modifier = exp(0) = 1.0                    (bias = 0)
        #   cpl_modifier = sigmoid(-2.3026)*9.9 + 0.1 ≈ 1  (logit_mult bias)
        #   cpl_residual = softplus(-4.0)*base*0.1 ≈ 0     (logit_add bias)
        for mlp in [self.cpl_mlp, self.gnd_mlp]:
            nn.init.normal_(mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.gnd_mlp[-1].bias)
        with torch.no_grad():
            self.cpl_mlp[-1].bias.copy_(torch.tensor([-2.3026, -4.0]))

    def _make_gnd_cap_density_init(self) -> torch.Tensor:
        import math
        z_anchors = self.metal_z_anchors.tolist()
        def density_for_z(z: float) -> float:
            if z < 0.40:   return 6.0
            elif z < 0.60: return 7.02
            elif z < 0.75: return 7.97
            elif z < 0.90: return 8.29
            elif z < 1.05: return 7.76
            elif z < 1.20: return 7.59
            elif z < 1.45: return 7.46
            elif z < 4.50: return 3.71
            elif z < 6.00: return 13.69
            else:           return 5.0
        inits = []
        for z in z_anchors:
            d = density_for_z(z)
            x = math.log(math.exp(d) - 1.0)
            inits.append(x)
        return torch.tensor(inits, dtype=torch.float32)

    @torch.compiler.disable
    def forward(self, features, cuboids, is_target, is_aggr, padding_mask, compute_coupling=True, frw_ratio_matrix=None):
        B, N, D = features.shape
        device = features.device
        EPS_0 = 0.008854 
        
        centers = cuboids[..., :3].clone()
        centers[padding_mask] = 1e5 
        sizes = cuboids[..., 3:6]
        epsilons = torch.clamp(cuboids[..., 8:9], min=1.0)
        z_abs = cuboids[..., 2:3]
        
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
        
        # Node Embedding 완성
        Z_prime = self.norm(torch.cat([features, analytical_density, ambient_potential, self_potential, segment_ratio], dim=-1))
        
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
        # Bottom face (XY projection) = w*h; cap_density calibrated vs this area.
        # Total surface area (area_face) is ~5-6× larger → wrong init magnitude.
        gnd_area = (w * h).clamp(min=1e-6)                                    # (B, N) μm²
        w_gnd_base = gnd_area * layer_cap_density                              # (B, N) fF

        gnd_feats = torch.cat([
            Z_prime,
            torch.log1p(area_face),
            torch.log1p(epsilons),
            torch.clamp(z_abs, min=0.0)/10.0
        ], dim=-1)

        gnd_modifier = torch.exp(torch.clamp(self.gnd_mlp(gnd_feats).squeeze(-1), min=-3.0, max=3.0))

        c_gnd_seg = w_gnd_base * gnd_modifier * is_target.float() * (~padding_mask).float()

        if not compute_coupling:
            return {
                'c_total_phys': c_gnd_seg, 
                'c_gnd_seg': c_gnd_seg,
            }
            
        # ---------------------------------------------------------
        # 3. [CPL Physics] Sparse Coupling 예측 (Neural Shader 도입)
        # ---------------------------------------------------------
        edges = ((is_target > 0.5).unsqueeze(2) & (is_aggr > 0.5).unsqueeze(1) & (closest_dist < self.cutoff_r)).nonzero(as_tuple=False) 
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
            # [FIX E] True facing area: dominant-plane projection, not sum of
            # three products. For side-by-side wires only one face is the actual
            # facing plane; summing triple-counts and inflates w_cpl_base.
            A_over = torch.maximum(torch.maximum(ox * oy, oy * oz), oz * ox)
            P_over = 2.0 * (ox + oy + oz)
            L_eff  = torch.max(torch.max(ox, oy), oz)
            
            # [물질의 정수] 쌍방 평균 유전율
            eps_pair = (epsilons[b_e, src_e].view(-1) + epsilons[b_e, dst_e].view(-1)) / 2.0
            
            # Raycast Shielding (Ray-tracing 물리 엔진 결과값은 Raw Feature로 활용)
            shield_E = compute_sparse_shield_factor(cuboids, b_e, src_e, dst_e).view(-1)
            D_eff = D_surf + 0.005 # 1e-4의 극단적 클램프 대신 물리적 최소 틈새(5nm) 강제 보장
            
            # [물리 엔진] Base Capacitance 연산 (폭발하지 않는 안전한 베이스)
            base_parallel = EPS_0 * eps_pair * (A_over / D_eff)
            base_fringe = EPS_0 * eps_pair * (P_over / torch.log1p(D_eff * 10.0))
            w_cpl_base = (base_parallel + base_fringe) * torch.clamp(shield_E, min=0.01)

            # Stage 3: 물리적으로 무의미한 edge 제거 (small net coupling 과대추정 방지)
            # w_cpl_base < 1e-5 fF 인 edge는 golden 최소 coupling 문턱 이하 → noise만 추가
            sig = w_cpl_base > 1e-5
            if sig.sum() < sig.shape[0]:
                b_e, src_e, dst_e = b_e[sig], src_e[sig], dst_e[sig]
                D_surf, A_over, P_over = D_surf[sig], A_over[sig], P_over[sig]
                L_eff, dz_gap, eps_pair = L_eff[sig], dz_gap[sig], eps_pair[sig]
                shield_E, w_cpl_base = shield_E[sig], w_cpl_base[sig]
                D_eff, base_parallel, base_fringe = D_eff[sig], base_parallel[sig], base_fringe[sig]

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
            is_aggr_per_edge = is_aggr[b_e]                                       # (E, N)
            occ_rank = (
                (cd_src_all < cd_src_dst.unsqueeze(1)) &
                (is_aggr_per_edge > 0.5)
            ).sum(dim=1).float()                                                   # (E,)

            edge_geom = torch.cat([
                force_2d(D_surf),                  # 1. 표면 거리
                force_2d(torch.log1p(A_over)),     # 2. 마주보는 면적 (Fix E: dominant plane)
                force_2d(torch.log1p(P_over)),     # 3. 투영 둘레
                force_2d(dz_gap),                  # 4. Z축 단차
                force_2d(eps_pair),                # 5. 쌍방 유전율
                force_2d(torch.log1p(base_parallel)), # 6. 평행판 뼈대
                force_2d(torch.log1p(base_fringe)),   # 7. 프린징 뼈대
                force_2d(shield_E),                # 8. Raycast 차폐율
                force_2d(torch.log1p(occ_rank)),   # 9. Occlusion rank prior
            ], dim=-1)
            # [FIX G] Normalize mixed-scale channels before MLP fusion.
            edge_geom = self.edge_geom_norm(edge_geom)

            Z_i, Z_j = Z_prime[b_e, src_e], Z_prime[b_e, dst_e]
            # [FIX I] Pre-project to 32-dim edge space; cpl_mlp input is
            # 32·3 + 9 = 105 instead of 788 (12× → 2× bottleneck).
            Z_i_e, Z_j_e = self.cpl_edge_proj(Z_i), self.cpl_edge_proj(Z_j)

            cpl_logits = self.cpl_mlp(
                torch.cat([Z_i_e + Z_j_e, Z_i_e * Z_j_e, torch.abs(Z_i_e - Z_j_e), edge_geom], dim=-1)
            )
            logit_mult = cpl_logits[:, 0]
            logit_add = cpl_logits[:, 1]

            # [FIX H] CPL z-index also via compact per-layer anchors.
            z_mid = (centers[b_e, src_e, 2] + centers[b_e, dst_e, 2]) / 2.0
            z_idx_cpl = torch.argmin(
                torch.abs(z_mid.unsqueeze(-1) - self.metal_z_anchors), dim=-1
            )
            phys_scale_cpl = F.softplus(self.layer_scale_phys_cpl[z_idx_cpl])
            
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