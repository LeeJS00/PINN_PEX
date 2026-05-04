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
        self.xy_scale = getattr(config, 'SCALE_FACTOR', 2.5)

        self.encoder = CuboidEncoder(input_dim, config.MODEL_DIM)
        layer_map = LayerInfoParser(config.LAYERS_INFO_PATH).parse()
        cutoff_r = getattr(config, 'CUTOFF_RADIUS', 4.0)
        self.flux_router = NeuralFluxRouter(
            model_dim=config.MODEL_DIM,
            context_radius=4,   # 1-Hop 통신 반경
            cutoff_radius=cutoff_r,
            layer_map=layer_map,
            use_rail_coupling=getattr(config, 'USE_RAIL_COUPLING', False),
        )
        self._step_counter = 0

    def freeze_ssl_layers(self):
        # Re-seed cpl/gnd head biases to identity AFTER SSL ckpt load — the
        # shape-filtered loader preserves bias values from legacy checkpoints
        # (old bias=0.1 → cpl_modifier≈5.3× at step 0). Done first so it runs
        # regardless of any downstream attribute issue.
        cpl_bias = self.flux_router.cpl_mlp[-1].bias
        gnd_bias = self.flux_router.gnd_mlp[-1].bias
        with torch.no_grad():
            # P4 init: cpl_modifier = exp(0) = 1.0; cpl_residual = softplus(-4)*0.01 ≈ 1.8e-4
            cpl_bias.copy_(torch.tensor([0.0, -4.0], device=cpl_bias.device))
            gnd_bias.zero_()
            # Re-seed GND density: prevents stale values from SSL/AL checkpoints
            # (loaded with strict=False) from overwriting the canonical init.
            init_vals = self.flux_router._make_gnd_cap_density_init().to(
                self.flux_router.layer_scale_phys_gnd.device)
            self.flux_router.layer_scale_phys_gnd.copy_(init_vals)
            # Re-seed CPL pair scale matrix to canonical zero-init.
            if hasattr(self.flux_router, 'cpl_layer_pair_log_scale'):
                self.flux_router.cpl_layer_pair_log_scale.data.zero_()
            # Re-seed fringe scale with physics-based per-layer values
            # (M1≈0.05 to M8≈0.90) instead of uniform sigmoid(-2)≈0.12.
            fringe_init = self.flux_router._make_gnd_fringe_scale_init().to(
                self.flux_router.gnd_fringe_scale.device)
            self.flux_router.gnd_fringe_scale.data.copy_(fringe_init)

        for param in self.encoder.parameters(): param.requires_grad = False
        for param in self.flux_router.norm.parameters(): param.requires_grad = False
        if hasattr(self.flux_router, 'charge_basis_mlp'):
            for param in self.flux_router.charge_basis_mlp.parameters(): param.requires_grad = True
        for param in self.flux_router.gnd_mlp.parameters(): param.requires_grad = True
        for param in self.flux_router.cpl_mlp.parameters(): param.requires_grad = True
        self.flux_router.layer_scale_phys_gnd.requires_grad_(True)
        if hasattr(self.flux_router, 'cpl_layer_pair_log_scale'):
            self.flux_router.cpl_layer_pair_log_scale.requires_grad_(True)
        self.flux_router.gnd_fringe_scale.requires_grad_(True)
        if hasattr(self.flux_router, 'rail_scale'):
            self.flux_router.rail_scale.data.fill_(-3.0)  # softplus(-3)≈0.049
            self.flux_router.rail_scale.requires_grad_(True)
        if hasattr(self.flux_router, 'vss_gnd_scale'):
            self.flux_router.vss_gnd_scale.data.fill_(-3.0)  # softplus(-3)≈0.049
            self.flux_router.vss_gnd_scale.requires_grad_(True)

        print("SSL layers frozen: encoder + flux_router.norm frozen. "
              "Trainable: charge_basis/gnd/cpl mlps + per-layer scales "
              "(layer_scale_phys_gnd, gnd_fringe_scale, cpl_layer_pair_log_scale, "
              "vss_gnd_scale, rail_scale if enabled).")

    def forward(self, cuboids, padding_mask, compute_coupling=True, frw_ratio_matrix=None, n_tiles=None, endpoint_prox=None):
        self._step_counter += 1

        # 1. Feature Extraction (고립된 큐보이드 피처)
        neural_cuboids = cuboids.clone()
        # X, Y (상대 좌표) 및 Z (Layer 높이): 선형 스케일 다운
        neural_cuboids[..., :3] /= self.xy_scale
        neural_cuboids[..., 3:6] = torch.log1p(torch.clamp(cuboids[..., 3:6], min=0.0)) / 1.5
        # [예외 처리] Epsilon이 0.0(패딩)일 경우 log(0) = -inf 방지를 위해 1.0(진공)으로 클램핑
        neural_cuboids[..., 8:9] = torch.log(torch.clamp(cuboids[..., 8:9], min=1.0)) / 2.0
        # channel 9 = net_type (0.0–1.0), already in [0,1] range — no extra normalization.
        feats = self.encoder(neural_cuboids)

        is_target = (cuboids[..., 7] == 1.0)
        is_aggr = (cuboids[..., 7] == 0.0)
        # 2. Physics-Informed Neural Flux Routing (1-hop GNN + surface physics)
        preds = self.flux_router(
            feats, cuboids, is_target, is_aggr, padding_mask,
            compute_coupling,
            frw_ratio_matrix=frw_ratio_matrix,
            n_tiles=n_tiles,
            endpoint_prox=endpoint_prox,
        )
        preds['is_target'] = is_target
        preds['is_aggr'] = is_aggr

        return preds
