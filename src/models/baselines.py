# src/models/baselines.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class GNN_Cap(nn.Module):
    """
    [Baseline C] GNN-Cap: Chip-Scale Interconnect Capacitance Extraction Using Graph Neural Network
    - Reference: IEEE TCAD 2024 (Liu et al.)
    - Preprocessing Strictly Implemented: 
      1) AABB Extension Overlap for Edge Construction
      2) Directional 7D Edge Features ("node with smaller x,y,z as node A")
      3) Linear Scaling (No Log1p)
    """
    def __init__(self, config):
        super().__init__()
        self.H = getattr(config, 'MODEL_DIM', 128)
        self.xy_scale = getattr(config, 'SCALE_FACTOR', 2.5)
        
        self.d_v = 3 # Node feature: w, h, d
        self.d_e = 7 # Edge feature: dAB, dx1, dx2, dy1, dy2, dz1, dz2
        
        # Node Input Projector
        self.node_emb = nn.Sequential(nn.Linear(self.d_v, 64), nn.ReLU(), nn.Linear(64, self.H), nn.ReLU())

        # [Layer 1] Message Passing (42, 83, 71, 128)
        self.l1_msg = nn.Sequential(nn.Linear(self.H + self.d_e, 83), nn.ReLU(), nn.Linear(83, self.H), nn.ReLU())
        self.l1_node_up = nn.Sequential(nn.Linear(self.H * 2, 71), nn.ReLU(), nn.Linear(71, self.H), nn.ReLU())
        self.l1_edge_tmp = nn.Sequential(nn.Linear(self.d_e, 42), nn.ReLU(), nn.Linear(42, 64), nn.ReLU())
        self.l1_edge_up = nn.Sequential(nn.Linear(self.H * 2 + 64, 128), nn.ReLU(), nn.Linear(128, self.H), nn.ReLU())

        #[Layer 2] Message Passing (112, 184, 80, 264)
        self.l2_msg = nn.Sequential(nn.Linear(self.H + self.H, 184), nn.ReLU(), nn.Linear(184, self.H), nn.ReLU())
        self.l2_node_up = nn.Sequential(nn.Linear(self.H * 2, 80), nn.ReLU(), nn.Linear(80, self.H), nn.ReLU())
        self.l2_edge_tmp = nn.Sequential(nn.Linear(self.H, 112), nn.ReLU(), nn.Linear(112, 128), nn.ReLU())
        self.l2_edge_up = nn.Sequential(nn.Linear(self.H * 2 + 128, 264), nn.ReLU(), nn.Linear(264, self.H), nn.ReLU())

        # Virtual Edge Processing (135, 368)
        self.vir_emb = nn.Sequential(
            nn.Linear(self.d_e, 135), nn.ReLU(), 
            nn.Linear(135, 368), nn.ReLU(), 
            nn.Linear(368, self.H), nn.ReLU()
        )

        # Independent Predictors (No KCL Router)
        self.tot_head = nn.Sequential(nn.Linear(self.H, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gnd_head = nn.Sequential(nn.Linear(self.H, 64), nn.ReLU(), nn.Linear(64, 1))
        self.cpl_head = nn.Sequential(nn.Linear(self.H, 64), nn.ReLU(), nn.Linear(64, 1))
        self.vir_cpl_head = nn.Sequential(nn.Linear(self.H * 3, 64), nn.ReLU(), nn.Linear(64, 1))

    def _chunked_edge_forward(self, mlp, feat_src, feat_dst, feat_edge, chunk_size=200000):
        """
        VRAM 폭발을 막기 위해 간선 텐서를 청크(Chunk) 단위로 나누어 순차적으로 연산합니다.
        chunk_size는 GPU 메모리 상태에 따라 조절할 수 있습니다. (기본 20만 개)
        """
        num_edges = feat_src.shape[0]
        if num_edges == 0:
            return torch.empty(0, mlp[-2].out_features, device=feat_src.device, dtype=feat_src.dtype)
        
        out_list = []
        for i in range(0, num_edges, chunk_size):
            c_src = feat_src[i:i+chunk_size]
            c_dst = feat_dst[i:i+chunk_size]
            c_edge = feat_edge[i:i+chunk_size]
            
            c_cat = torch.cat([c_src, c_dst, c_edge], dim=-1)
            out_list.append(mlp(c_cat))
            
            # 명시적 메모리 해제
            del c_src, c_dst, c_edge, c_cat
            
        return torch.cat(out_list, dim=0)

    def _get_edge_features(self, centers, sizes, flat_src, flat_dst):
        """
        [PREPROCESSING 2] 7D Edge Features 
        Always choose the node with a smaller coordinate as Node A for each axis.
        """
        c1, c2 = centers[flat_src], centers[flat_dst]
        s1, s2 = sizes[flat_src], sizes[flat_dst]
        
        dAB = torch.norm(c1 - c2, dim=-1, keepdim=True)
        
        edge_feats = [dAB]
        for dim in range(3):
            # 마스크: Node 2의 좌표가 더 크면 1, 아니면 0
            mask = (c2[:, dim] >= c1[:, dim]).float().unsqueeze(-1)
            
            # Node A는 항상 중심 좌표가 작은 노드, Node B는 중심 좌표가 큰 노드
            cA = mask * c1[:, dim:dim+1] + (1 - mask) * c2[:, dim:dim+1]
            cB = (1 - mask) * c1[:, dim:dim+1] + mask * c2[:, dim:dim+1]
            sA = mask * s1[:, dim:dim+1] + (1 - mask) * s2[:, dim:dim+1]
            sB = (1 - mask) * s1[:, dim:dim+1] + mask * s2[:, dim:dim+1]
            
            # 논문의 Fig 4 기준: d1 = minB - minA, d2 = maxB - maxA
            minA, maxA = cA - sA / 2.0, cA + sA / 2.0
            minB, maxB = cB - sB / 2.0, cB + sB / 2.0
            
            d1 = minB - minA
            d2 = maxB - maxA
            
            edge_feats.extend([d1, d2])
            
        # Neural Network 안정화를 위해 Linear Scaling (xy_scale) 적용
        return torch.cat(edge_feats, dim=-1) / self.xy_scale

    @torch.compiler.disable
    def forward(self, cuboids, padding_mask, compute_coupling=True):
        B, N, _ = cuboids.shape
        device = cuboids.device
        
        # [PREPROCESSING 3] Linear Scaling (No Log1p trick here)
        sizes = cuboids[..., 3:6].clone()
        H_v = self.node_emb(sizes.view(B * N, 3) / self.xy_scale) # (B*N, H)
        
        centers = cuboids[..., :3].clone()
        mins = centers - sizes / 2.0
        maxs = centers + sizes / 2.0
        
        # 마스킹된 노드 우주 밖으로 추방
        mins[padding_mask] = 1e5
        maxs[padding_mask] = 1e5
        
        # -------------------------------------------------------------
        # [PREPROCESSING 1] Edge Construction by AABB Extension Overlap
        # -------------------------------------------------------------
        # 1. Normal Edges (d_e = 1.0um -> 팽창 0.5um)
        de = 1.0
        ext_mins_n = mins.clone()
        ext_maxs_n = maxs.clone()
        ext_mins_n[..., :2] -= de / 2.0
        ext_maxs_n[..., :2] += de / 2.0
        ext_mins_n[..., 2] -= 0.5  # Z축 팽창 (Adjacent Layer 커버)
        ext_maxs_n[..., 2] += 0.5
        
        overlap_n = (ext_mins_n.unsqueeze(2) <= ext_maxs_n.unsqueeze(1)) & (ext_maxs_n.unsqueeze(2) >= ext_mins_n.unsqueeze(1))
        normal_adj = overlap_n.all(dim=-1) & ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
        
        b_e, src_e, dst_e = normal_adj.nonzero(as_tuple=True)
        flat_src, flat_dst = b_e * N + src_e, b_e * N + dst_e
        
        H_e = self._get_edge_features(centers.view(B * N, 3), sizes.view(B * N, 3), flat_src, flat_dst)
        
        # -------------------------------------------------------------
        # Message Passing Layer 1
        # -------------------------------------------------------------
        # n_ij = self.l1_msg(torch.cat([H_v[flat_dst], H_e], dim=-1))
        n_ij = self._chunked_mlp_forward(self.l1_msg, H_v[flat_dst], H_e)
        # Make same src.dtype
        # n_i = torch.zeros(B * N, self.H, device=device).scatter_add_(0, flat_src.unsqueeze(-1).expand(-1, self.H), n_ij)
        # degree = torch.zeros(B * N, device=device).scatter_add_(0, flat_src, torch.ones_like(flat_src, dtype=torch.float))
        n_i = torch.zeros(B * N, self.H, dtype=n_ij.dtype, device=device).scatter_add_(0, flat_src.unsqueeze(-1).expand(-1, self.H), n_ij)
        degree = torch.zeros(B * N, dtype=n_ij.dtype, device=device).scatter_add_(0, flat_src, torch.ones_like(flat_src, dtype=n_ij.dtype))
        n_i = n_i / torch.clamp(degree.unsqueeze(-1), min=1.0)
        
        H_v_1 = self.l1_node_up(torch.cat([H_v, n_i], dim=-1))
        H_e_tmp = self.l1_edge_tmp(H_e)
        # H_e_1 = self._chunked_edge_forward(self.l1_edge_up, H_v_1[flat_src], H_v_1[flat_dst], H_e_tmp, chunk_size=100000)
        H_e_1 = self._chunked_mlp_forward(self.l1_edge_up, H_v_1[flat_src], H_v_1[flat_dst], H_e_tmp)
        # -------------------------------------------------------------
        # Message Passing Layer 2
        # -------------------------------------------------------------
        # n_ij_2 = self.l2_msg(torch.cat([H_v_1[flat_dst], H_e_1], dim=-1))
        n_ij_2 = self._chunked_mlp_forward(self.l2_msg, H_v_1[flat_dst], H_e_1)
        # n_i_2 = torch.zeros(B * N, self.H, device=device).scatter_add_(0, flat_src.unsqueeze(-1).expand(-1, self.H), n_ij_2)
        n_i_2 = torch.zeros(B * N, self.H, dtype=n_ij_2.dtype, device=device).scatter_add_(0, flat_src.unsqueeze(-1).expand(-1, self.H), n_ij_2)
        n_i_2 = n_i_2 / torch.clamp(degree.unsqueeze(-1), min=1.0)
        
        H_v_2 = self.l2_node_up(torch.cat([H_v_1, n_i_2], dim=-1))
        H_e_tmp_2 = self.l2_edge_tmp(H_e_1)
        # H_e_2 = self._chunked_edge_forward(self.l2_edge_up, H_v_2[flat_src], H_v_2[flat_dst], H_e_tmp_2, chunk_size=100000)
        H_e_2 = self._chunked_mlp_forward(self.l2_edge_up, H_v_2[flat_src], H_v_2[flat_dst], H_e_tmp_2)
        # -------------------------------------------------------------
        # Independent Node Predictions (KCL Not Enforced)
        # -------------------------------------------------------------
        c_total = F.softplus(self.tot_head(H_v_2)).view(B, N).squeeze(-1) * (~padding_mask).float()
        c_gnd = F.softplus(self.gnd_head(H_v_2)).view(B, N).squeeze(-1) * (~padding_mask).float()
        
        sparse_cpl = {'b_idx': torch.empty(0, dtype=torch.long, device=device), 
                      'src_idx': torch.empty(0, dtype=torch.long, device=device), 
                      'dst_idx': torch.empty(0, dtype=torch.long, device=device), 
                      'c_cpl': torch.empty(0, dtype=H_v.dtype, device=device)}
        
        if compute_coupling:
            is_tgt = (cuboids[..., 7] == 1.0).view(B * N)
            is_agg = (cuboids[..., 7] == 0.0).view(B * N)
            
            # (1) Normal Edges Coupling (Target -> Aggressor)
            tgt_agg_mask = is_tgt[flat_src] & is_agg[flat_dst]
            c_cpl_n = F.softplus(self.cpl_head(H_e_2[tgt_agg_mask])).squeeze(-1)
            b_n, src_n, dst_n = b_e[tgt_agg_mask], src_e[tgt_agg_mask], dst_e[tgt_agg_mask]
            
            # (2) Virtual Edges (d_evir = 4.0um -> 팽창 2.0um)
            devir = 4.0
            ext_mins_v = mins.clone()
            ext_maxs_v = maxs.clone()
            ext_mins_v[..., :2] -= devir / 2.0
            ext_maxs_v[..., :2] += devir / 2.0
            ext_mins_v[..., 2] -= 0.5
            ext_maxs_v[..., 2] += 0.5
            
            overlap_v = (ext_mins_v.unsqueeze(2) <= ext_maxs_v.unsqueeze(1)) & (ext_maxs_v.unsqueeze(2) >= ext_mins_v.unsqueeze(1))
            vir_adj = overlap_v.all(dim=-1) & ~normal_adj & ~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
            
            b_v, src_v, dst_v = vir_adj.nonzero(as_tuple=True)
            flat_src_v_all, flat_dst_v_all = b_v * N + src_v, b_v * N + dst_v
            tgt_agg_mask_v = is_tgt[flat_src_v_all] & is_agg[flat_dst_v_all]
            
            b_v, src_v, dst_v = b_v[tgt_agg_mask_v], src_v[tgt_agg_mask_v], dst_v[tgt_agg_mask_v]
            flat_src_v, flat_dst_v = flat_src_v_all[tgt_agg_mask_v], flat_dst_v_all[tgt_agg_mask_v]
            
            if len(b_v) > 0:
                H_e_vir = self._get_edge_features(centers.view(B * N, 3), sizes.view(B * N, 3), flat_src_v, flat_dst_v)
                H_ev_emb = self.vir_emb(H_e_vir)
                pair_feat_v = torch.cat([H_v_2[flat_src_v], H_v_2[flat_dst_v], H_ev_emb], dim=-1)
                c_cpl_v = self._chunked_edge_forward(self.vir_cpl_head, H_v_2[flat_src_v], H_v_2[flat_dst_v], H_ev_emb)
                c_cpl_v = F.softplus(c_cpl_v).squeeze(-1)
            else:
                c_cpl_v = torch.empty(0, dtype=H_v.dtype, device=device)
                
            # Merge Normal and Virtual Edges
            sparse_cpl['b_idx'] = torch.cat([b_n, b_v])
            sparse_cpl['src_idx'] = torch.cat([src_n, src_v])
            sparse_cpl['dst_idx'] = torch.cat([dst_n, dst_v])
            sparse_cpl['c_cpl'] = torch.cat([c_cpl_n, c_cpl_v])
            
        # return {'c_total_phys': c_total, 'c_gnd_seg': c_gnd, 'sparse_cpl': sparse_cpl}
        is_target = (cuboids[..., 7] == 1.0)
        is_aggr = (cuboids[..., 7] == 0.0)
        
        c_gnd_flat = c_gnd.view(B * N)
        sum_cpl_flat = torch.zeros_like(c_gnd_flat)
        
        if len(sparse_cpl['b_idx']) > 0:
            flat_src_all = sparse_cpl['b_idx'] * N + sparse_cpl['src_idx']
            sum_cpl_flat.scatter_add_(0, flat_src_all, sparse_cpl['c_cpl'])
            
        Z_pseudo = torch.clamp(c_gnd_flat + sum_cpl_flat, min=1e-6)
        p_gnd = (c_gnd_flat / Z_pseudo).view(B, N)
        
        if len(sparse_cpl['b_idx']) > 0:
            sparse_cpl['p_cpl'] = sparse_cpl['c_cpl'] / Z_pseudo[flat_src_all]
        else:
            sparse_cpl['p_cpl'] = torch.empty(0, dtype=H_v.dtype, device=device)

        return {
            'c_total_phys': c_total, 
            'c_gnd_seg': c_gnd, 
            'sparse_cpl': sparse_cpl,
            'is_target': is_target,
            'is_aggr': is_aggr,
            'p_gnd': p_gnd
        }
    
    def _chunked_mlp_forward(self, mlp, *tensors, chunk_size=32768):
        """
        [VRAM 킬러 방어] Gradient Checkpointing이 적용된 Chunked Forward.
        학습 시 엄청난 메모리를 잡아먹는 MLP의 중간 계산 그래프를 버리고 Backward 시 재계산합니다.
        """
        num_edges = tensors[0].shape[0]
        if num_edges == 0:
            out_dim = mlp[-2].out_features if hasattr(mlp[-2], 'out_features') else mlp[-1].out_features
            return torch.empty(0, out_dim, device=tensors[0].device, dtype=tensors[0].dtype)
        
        out_list = []
        for i in range(0, num_edges, chunk_size):
            # 각 텐서를 chunk_size만큼 슬라이싱
            chunked_tensors = tuple(t[i:i+chunk_size] for t in tensors)
            
            if self.training and torch.is_grad_enabled():
                # Checkpoint 내부에서 실행될 래퍼(Wrapper) 함수
                def wrapper(*args):
                    return mlp(torch.cat(args, dim=-1))
                
                # Checkpointing은 입력 중 하나라도 requires_grad=True여야 동작함
                if any(t.requires_grad for t in chunked_tensors):
                    out = checkpoint.checkpoint(wrapper, *chunked_tensors, use_reentrant=False)
                else:
                    out = mlp(torch.cat(chunked_tensors, dim=-1))
            else:
                c_cat = torch.cat(chunked_tensors, dim=-1)
                out = mlp(c_cat)
                
            out_list.append(out)
            
        return torch.cat(out_list, dim=0)