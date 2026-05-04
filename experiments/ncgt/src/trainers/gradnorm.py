"""
GradNorm multi-task balancing (ParaFormer / Chen et al. 2018).

Balances gradient norms of K composite tasks by adjusting per-task loss weights.
Per Plan v4 §4: K=6 composite tasks (gnd, cpl_total, cpl_classifier, net_reg, kcl, zero).

Initial 100 steps use hand-tuned weights (warmup); after that GradNorm takes over.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class GradNorm:
    """Stateful GradNorm balancer.

    Usage:
        gn = GradNorm(model, K=6, alpha=1.5, init_weights=[3, 3, 1, 0.5, 0.1, 0.1])
        ...
        for step in range(N):
            losses = compute_losses(...)  # list of K tensors
            total = gn.combine(losses, model, step)
            total.backward()
            opt.step()
            gn.update(losses, model, step)
    """

    def __init__(
        self,
        K: int,
        alpha: float = 1.5,
        init_weights: Optional[List[float]] = None,
        warmup_steps: int = 100,
        lr_w: float = 0.025,
        device: Optional[torch.device] = None,
    ):
        self.K = K
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.lr_w = lr_w
        device = device or torch.device("cpu")
        if init_weights is None:
            init_weights = [1.0] * K
        assert len(init_weights) == K
        self.weights = torch.tensor(init_weights, dtype=torch.float32, device=device, requires_grad=True)
        self.loss_init: Optional[torch.Tensor] = None  # L_i(0) snapshot

    def combine(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """Returns weighted-sum loss with current weights detached from optimizer."""
        w = self.weights.detach()
        return sum(w[i] * losses[i] for i in range(self.K))

    @torch.no_grad()
    def _record_init(self, losses: List[torch.Tensor]) -> None:
        if self.loss_init is None:
            self.loss_init = torch.stack([L.detach() for L in losses]).clamp(min=1e-9)

    def update(
        self,
        losses: List[torch.Tensor],
        shared_params: List[nn.Parameter],
        step: int,
    ) -> None:
        """GradNorm step. Skip during warmup. Updates self.weights in-place."""
        self._record_init(losses)
        if step < self.warmup_steps:
            return
        # Compute G_i = ||∇_W (w_i · L_i)||_2 for each task, where W is shared params.
        # We approximate by computing gradient of (w_i * L_i) wrt the LAST shared
        # layer's weight (cheaper than full param set).
        if not shared_params:
            return
        target_param = shared_params[-1]

        # G_i = ||∇_W (w_i * L_i)||_2 — gradient norm wrt the chosen shared param,
        # graph retained so we can backprop through G_i back to self.weights.
        G_i_list = []
        for i, L in enumerate(losses):
            if L.requires_grad and L.detach().item() != 0.0:
                gi = torch.autograd.grad(
                    self.weights[i] * L,
                    target_param,
                    retain_graph=True,
                    create_graph=True,
                )[0]
                G_i_list.append(gi.norm())
            else:
                G_i_list.append(torch.zeros((), device=L.device))
        G_i = torch.stack(G_i_list)
        G_avg = G_i.mean().detach().clamp(min=1e-9)

        # Relative inverse training rate r_i (no grad needed).
        tilde_L = torch.stack([L.detach() for L in losses]).clamp(min=1e-9) / self.loss_init
        r_i = (tilde_L / tilde_L.mean().clamp(min=1e-9)).detach()

        # Target gradient norm.
        G_target = (G_avg * r_i.pow(self.alpha)).detach()

        # GradNorm loss (L1 between G_i and target).
        L_grad = (G_i - G_target).abs().sum()

        # Update weights via gradient descent on L_grad.
        grad_w = torch.autograd.grad(L_grad, self.weights, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            self.weights.data -= self.lr_w * grad_w
            self.weights.data.clamp_(min=1e-3)
            # Renormalize: weights sum = K (preserves total scale).
            self.weights.data *= self.K / self.weights.data.sum().clamp(min=1e-3)


# ---------------------------------------------------------------------------
# Smoke test.
# ---------------------------------------------------------------------------
def _smoke_test() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    gn = GradNorm(K=4, init_weights=[1.0, 1.0, 1.0, 1.0], warmup_steps=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    x = torch.randn(32, 8)
    targets = [torch.randn(32, 1) for _ in range(4)]

    for step in range(10):
        opt.zero_grad()
        out = model(x)
        losses = [(out[:, i:i+1] - targets[i]).abs().mean() for i in range(4)]
        total = gn.combine(losses)
        total.backward(retain_graph=True)
        gn.update(losses, list(model.parameters()), step)
        opt.step()
        if step in (0, 5, 9):
            print(f"step {step}: weights={gn.weights.detach().tolist()}, total={total.item():.4f}")
    print("[gradnorm smoke] OK")


if __name__ == "__main__":
    _smoke_test()
