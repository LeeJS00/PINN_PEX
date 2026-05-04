# src/models/gino_enricher.py
# 2.5D BEOL GINO Enricher: P2G → FNO-2D → G2P
# Sits between CuboidEncoder and NeuralFluxRouter as a drop-in feature enricher.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """FNO-2D spectral convolution. Supports d_in != d_out."""

    def __init__(self, d_in: int, d_out: int, modes: int):
        super().__init__()
        self.d_in  = d_in
        self.d_out = d_out
        self.modes = modes
        # Store real/imag as last dim=2; view_as_complex requires contiguous tensor
        scale = 1.0 / (d_in * d_out)
        self.weight = nn.Parameter(
            scale * torch.rand(d_in, d_out, modes, modes // 2 + 1, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_in, H, W)
        B, C, H, W = x.shape
        # rfft2 does not support bfloat16 — upcast for FFT, then restore dtype
        orig_dtype = x.dtype
        x_f32 = x.float()
        x_ft = torch.fft.rfft2(x_f32, norm='ortho')              # (B, d_in, H, W//2+1) complex
        # Trim to modes (avoid indexing past rfft2 output size)
        mx = min(self.modes, H)
        my = min(self.modes // 2 + 1, W // 2 + 1)
        x_trim = x_ft[:, :, :mx, :my]                            # (B, d_in, mx, my)
        w = torch.view_as_complex(self.weight.contiguous())       # (d_in, d_out, modes, modes//2+1)
        w_trim = w[:, :, :mx, :my]                               # (d_in, d_out, mx, my)
        out_ft = torch.einsum('bimn,iomn->bomn', x_trim, w_trim) # (B, d_out, mx, my)
        result = torch.zeros(B, self.d_out, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        result[:, :, :mx, :my] = out_ft
        out = torch.fft.irfft2(result, s=(H, W), norm='ortho')    # (B, d_out, H, W) float32
        return out.to(orig_dtype)                                   # restore bfloat16 if needed


class FNO2DBlock(nn.Module):
    """FNO-2D block: spectral conv + pointwise skip + GELU + GroupNorm."""

    def __init__(self, d_in: int, d_out: int, modes: int):
        super().__init__()
        self.spec = SpectralConv2d(d_in, d_out, modes)
        self.skip = nn.Conv2d(d_in, d_out, kernel_size=1)
        self.norm = nn.GroupNorm(1, d_out)   # equivalent to spatial LayerNorm
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.spec(x) + self.skip(x)))


class BEOL_GINO_Enricher(nn.Module):
    """
    Gaussian P2G → FNO-2D → G2P feature enricher for BEOL parasitic cap.

    Inserts global FNO spatial context between CuboidEncoder and NeuralFluxRouter.
    Input/output shape: (B, N, enc_dim) — NeuralFluxRouter receives the same
    tensor shape as before, but now globally context-enriched via FNO.

    2D design: all cuboids scatter to one (B, D_fno, G, G) grid;
    z_abs is encoded as an extra input channel to FNO block 0.
    """

    def __init__(self, enc_dim: int = 128, d_fno: int = 32, fno_modes: int = 8,
                 n_fno_blocks: int = 2, grid_size: int = 16,
                 sigma_xy: float = 0.4, window: float = 4.0):
        super().__init__()
        self.enc_dim   = enc_dim
        self.d_fno     = d_fno
        self.grid_size = grid_size
        self.sigma_xy  = sigma_xy
        self.window    = window

        self.proj_in  = nn.Linear(enc_dim, d_fno)

        # Block 0: (d_fno + 1) → d_fno  (extra z-abs channel)
        # Block k>0: d_fno → d_fno
        blocks = [FNO2DBlock(d_fno + 1, d_fno, fno_modes)]
        for _ in range(n_fno_blocks - 1):
            blocks.append(FNO2DBlock(d_fno, d_fno, fno_modes))
        self.fno = nn.ModuleList(blocks)

        self.proj_out = nn.Linear(d_fno, enc_dim)
        # Zero-init so initial output ≡ feats (stable residual from start)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def _p2g_weights(self, coords_xy: torch.Tensor,
                     padding_mask: torch.Tensor):
        """Compute Gaussian scatter weights (shared for P2G and G2P).

        coords_xy: (B, N, 2)  x_rel, y_rel in approximately [-2, 2] μm
        padding_mask: (B, N)  True = padding

        Returns
        -------
        w_p2g: (B, N, G²)  normalized per grid cell  (for P2G scatter)
        w_g2p: (B, N, G²)  normalized per cuboid     (for G2P interpolation)
        """
        B, N, _ = coords_xy.shape
        G = self.grid_size
        device = coords_xy.device

        lin = torch.linspace(0.0, self.window, G, device=device)
        gx, gy = torch.meshgrid(lin, lin, indexing='ij')
        grid_xy = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (G², 2)

        # Shift from [-window/2, window/2] → [0, window]
        xy = coords_xy + self.window / 2                              # (B, N, 2)
        dxy = xy.unsqueeze(2) - grid_xy                               # (B, N, G², 2)
        w = torch.exp(-dxy.pow(2).sum(-1) / (2.0 * self.sigma_xy ** 2))  # (B, N, G²)

        # Zero padding cuboids
        valid = (~padding_mask).float().unsqueeze(-1)                 # (B, N, 1)
        w = w * valid                                                  # (B, N, G²)

        # P2G: normalize per grid cell (sum over N dim)
        w_p2g = w / w.sum(dim=1, keepdim=True).clamp(min=1e-8)       # (B, N, G²)
        # G2P: normalize per cuboid (sum over G² dim)
        w_g2p = w / w.sum(dim=2, keepdim=True).clamp(min=1e-8)       # (B, N, G²)

        return w_p2g, w_g2p

    def _p2g(self, feats_d: torch.Tensor, z_abs: torch.Tensor,
             w_p2g: torch.Tensor) -> torch.Tensor:
        """Scatter cuboid features to 2D grid.

        feats_d: (B, N, D_fno)
        z_abs:   (B, N)
        w_p2g:   (B, N, G²)
        Returns: (B, D_fno+1, G, G)
        """
        B, N, D = feats_d.shape
        G = self.grid_size

        feat_grid = torch.einsum('bng,bnd->bgd', w_p2g, feats_d)     # (B, G², D_fno)
        feat_grid = feat_grid.reshape(B, G, G, D).permute(0, 3, 1, 2) # (B, D_fno, G, G)

        z_grid = torch.einsum('bng,bn->bg', w_p2g, z_abs / 5.0)      # (B, G²)
        z_grid = z_grid.reshape(B, 1, G, G)                            # (B, 1, G, G)

        return torch.cat([feat_grid, z_grid], dim=1)                   # (B, D_fno+1, G, G)

    def _g2p(self, fno_out: torch.Tensor, w_g2p: torch.Tensor) -> torch.Tensor:
        """Interpolate grid features back to cuboids.

        fno_out: (B, D_fno, G, G)
        w_g2p:   (B, N, G²)
        Returns: (B, N, D_fno)
        """
        B, D, G, _ = fno_out.shape
        grid_flat = fno_out.permute(0, 2, 3, 1).reshape(B, G * G, D) # (B, G², D_fno)
        return torch.einsum('bng,bgd->bnd', w_g2p, grid_flat)          # (B, N, D_fno)

    def forward(self, feats: torch.Tensor, cuboids: torch.Tensor,
                padding_mask: torch.Tensor) -> torch.Tensor:
        """
        feats:        (B, N, enc_dim)
        cuboids:      (B, N, 9+)  channels: 0-1=x_rel,y_rel; 2=z_abs
        padding_mask: (B, N) bool
        Returns:      (B, N, enc_dim)  enriched feats + residual
        """
        coords_xy = cuboids[..., :2]   # (B, N, 2)
        z_abs     = cuboids[..., 2]    # (B, N)

        w_p2g, w_g2p = self._p2g_weights(coords_xy, padding_mask)

        x     = self.proj_in(feats)                        # (B, N, D_fno)
        grid  = self._p2g(x, z_abs, w_p2g)                # (B, D_fno+1, G, G)
        for blk in self.fno:
            grid = blk(grid)                               # (B, D_fno, G, G)
        enriched = self._g2p(grid, w_g2p)                  # (B, N, D_fno)

        return self.proj_out(enriched) + feats             # (B, N, enc_dim) + residual
