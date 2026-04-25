# src/models/neural_field.py
import torch
import torch.nn as nn

from src.models.flux_head import NeuralFluxRouter
from src.preprocessing.layer_parser import LayerInfoParser


class CuboidEncoder(nn.Module):
    """
    (N, 9) Raw Cuboid Tensor -> High-dim Feature
    """
    def __init__(self, input_dim=9, model_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim)
        )
    def forward(self, x):
        return self.net(x)

class DeepPEX_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = getattr(config, 'INPUT_DIM', 9)
        # self.scale_factor = getattr(config, 'SCALE_FACTOR', 2.5)
        self.xy_scale = getattr(config, 'SCALE_FACTOR', 2.5)
        # self.z_scale = getattr(config, 'Z_SCALE', 1.0)

        self.encoder = CuboidEncoder(input_dim, config.MODEL_DIM)
        layer_map = LayerInfoParser(config.LAYERS_INFO_PATH).parse()
        # [NEW] 단일 통합 라우터 (Attention, PhysicsHead, CapHead 모두 완벽 대체)
        self.flux_router = NeuralFluxRouter(
            model_dim=config.MODEL_DIM,
            context_radius=4,   # 1-Hop 통신 반경
            cutoff_radius=2.5,  # Coupling 탐색 반경: 4.0→2.5 μm (에너지 82.5% 유지, noise edge 제거)
            layer_map=layer_map
        )
        self._step_counter = 0

    def freeze_ssl_layers(self):
        # [FIX B supplement] Re-seed cpl/gnd head biases to identity AFTER SSL
        # ckpt load — the shape-filtered loader preserves bias values from
        # legacy checkpoints (old bias=0.1 → cpl_modifier≈5.3× at step 0).
        # Done first so it runs regardless of any downstream attribute issue.
        cpl_bias = self.flux_router.cpl_mlp[-1].bias
        gnd_bias = self.flux_router.gnd_mlp[-1].bias
        with torch.no_grad():
            cpl_bias.copy_(torch.tensor([-2.3026, -4.0], device=cpl_bias.device))
            gnd_bias.zero_()
            # Re-seed to empirical cap_density; prevents stale ≈1.0 values from
            # SSL/AL checkpoints (loaded with strict=False) from overwriting the new init.
            init_vals = self.flux_router._make_gnd_cap_density_init().to(
                self.flux_router.layer_scale_phys_gnd.device)
            self.flux_router.layer_scale_phys_gnd.copy_(init_vals)

        for param in self.encoder.parameters(): param.requires_grad = False
        for param in self.flux_router.norm.parameters(): param.requires_grad = False
        if hasattr(self.flux_router, 'charge_basis_mlp'):
            for param in self.flux_router.charge_basis_mlp.parameters(): param.requires_grad = True
        for param in self.flux_router.gnd_mlp.parameters(): param.requires_grad = True
        for param in self.flux_router.cpl_mlp.parameters(): param.requires_grad = True
        self.flux_router.layer_scale_phys_gnd.requires_grad_(True)
        self.flux_router.layer_scale_phys_cpl.requires_grad_(True)

        print("🔒 SSL Layers Frozen: Only 'charge_basis_mlp', 'gnd_mlp', 'cpl_mlp' are trainable.")

    def forward(self, cuboids, padding_mask, compute_coupling=True, frw_ratio_matrix=None):
        self._step_counter += 1
        
        # -------------------------------------------------------------
        # [PROFILING 1] 원본 피처 스케일 모니터링 (100 Step 마다 출력)
        # -------------------------------------------------------------
        # if self.training and self._step_counter % 100 == 0:
        #     valid_c = cuboids[~padding_mask]
        #     print(f"\n[Tensor Profiler - Step {self._step_counter}]")
        #     print(f"  - X/Y (rel) : Min {valid_c[:,:2].min().item():.2f}, Max {valid_c[:,:2].max().item():.2f}, Std {valid_c[:,:2].std().item():.2f}")
        #     print(f"  - Z (abs)   : Min {valid_c[:,2].min().item():.2f}, Max {valid_c[:,2].max().item():.2f}, Std {valid_c[:,2].std().item():.2f}")
        #     print(f"  - W/H/D     : Min {valid_c[:,3:6].min().item():.2f}, Max {valid_c[:,3:6].max().item():.2f}, Std {valid_c[:,3:6].std().item():.2f}")

        # 1. Feature Extraction (고립된 큐보이드 피처)
        neural_cuboids = cuboids.clone()
        # X, Y (상대 좌표) 및 Z (Layer 높이): 선형 스케일 다운
        neural_cuboids[..., :3] /= self.xy_scale 
        neural_cuboids[..., 3:6] = torch.log1p(torch.clamp(cuboids[..., 3:6], min=0.0)) / 1.5 
        # [예외 처리] Epsilon이 0.0(패딩)일 경우 log(0) = -inf 방지를 위해 1.0(진공)으로 클램핑
        neural_cuboids[..., 8:9] = torch.log(torch.clamp(cuboids[..., 8:9], min=1.0)) / 2.0
        feats = self.encoder(neural_cuboids)

        is_target = (cuboids[..., 7] == 1.0)
        is_aggr = (cuboids[..., 7] == 0.0)
        # 2. Physics-Informed Neural Flux Routing
        # 모든 KCL, 1-Hop 문맥, Sparse 차폐/거리가 이 함수 안에서 C++ 속도로 처리됨.
        # preds = self.flux_router(feats, norm_cuboids, is_target, is_aggr, padding_mask, compute_coupling)
        preds = self.flux_router(feats, cuboids, is_target, is_aggr, padding_mask, compute_coupling, frw_ratio_matrix=frw_ratio_matrix)
        preds['is_target'] = is_target
        preds['is_aggr'] = is_aggr
        
        return preds

