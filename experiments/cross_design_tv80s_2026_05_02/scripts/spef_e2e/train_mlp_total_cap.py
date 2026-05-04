"""Train MLP on v3 features for total_cap (5 seeds, GPU).

Designed to add diversity to the LGBM+CatBoost ensemble.
Targets ~8.5% individual MAPE (matching ResMLP-v3-nova from prior work).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols


class ResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 384, depth: int = 6, dropout: float = 0.10):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        ) for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        h = self.input(x)
        for blk in self.blocks:
            h = h + blk(h)
        h = self.norm(h)
        return self.head(h).squeeze(-1) + self.bias


def hybrid_loss(pred_log, y_lin, eps=1e-3):
    pred = torch.exp(pred_log)
    y = y_lin.clamp(min=eps)
    ape = torch.abs(pred - y) / y
    return ape.mean() + 0.05 * F.mse_loss(pred_log, torch.log(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, seed: {args.seed}")

    cache = cfg.CACHE_DIR / "features_v3"
    train = pd.concat([pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in cfg.TRAIN_DESIGNS], ignore_index=True)
    val = pd.read_parquet(cache / "intel22_nova_f3.parquet").assign(design_name="intel22_nova_f3")
    test = pd.read_parquet(cache / "intel22_tv80s_f3.parquet").assign(design_name="intel22_tv80s_f3")
    fcols = _select_feature_cols(train)
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(np.float32); y_tr = train["total_cap_fF"].to_numpy(np.float32)
    X_va = val[fcols].to_numpy(np.float32);   y_va = val["total_cap_fF"].to_numpy(np.float32)
    X_te = test[fcols].to_numpy(np.float32);  y_te = test["total_cap_fF"].to_numpy(np.float32)

    # Standardize
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    X_te = (X_te - mu) / sd

    model = ResMLP(in_dim=len(fcols), hidden=args.hidden, depth=args.depth, dropout=0.10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_steps = (len(X_tr) // args.batch_size + 1) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    # Convert to tensors
    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)
    X_te_t = torch.from_numpy(X_te).to(device)

    best_val_mape = float("inf"); best_state = None
    rng = np.random.default_rng(args.seed)

    print(f"Training {args.epochs} epochs, batch_size={args.batch_size}, depth={args.depth}, hidden={args.hidden}")
    for ep in range(args.epochs):
        model.train()
        idx = rng.permutation(len(X_tr))
        for i in range(0, len(idx), args.batch_size):
            b = idx[i:i+args.batch_size]
            xb = X_tr_t[b]; yb = y_tr_t[b]
            pred = model(xb)
            loss = hybrid_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()

        # Eval
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = torch.exp(model(X_va_t)).cpu().numpy()
            ape = 100 * np.abs(pv - y_va) / np.maximum(y_va, 1e-3)
            mape_val = ape.mean()
            if mape_val < best_val_mape:
                best_val_mape = mape_val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
                print(f"  ep {ep+1:>3d}/{args.epochs}: val_mape={mape_val:.3f}% (best {best_val_mape:.3f}%)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.exp(model(X_va_t)).cpu().numpy()
        pt = torch.exp(model(X_te_t)).cpu().numpy()
    ape_t = 100 * np.abs(pt - y_te) / np.maximum(y_te, 1e-3)
    print(f"Final test MAPE: {ape_t.mean():.3f}%")

    out_dir = cfg.OUTPUT_DIR / "spef_e2e" / "total_cap_mlp"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save
    torch.save({"model": best_state, "feature_cols": fcols, "mu": mu, "sd": sd,
                "hidden": args.hidden, "depth": args.depth, "dropout": 0.10},
               out_dir / f"seed{args.seed}.pt")
    pd.DataFrame({"design_name": val.design_name.values,
                  "net_name": val.net_name.values,
                  "y_true": y_va, "y_pred": pv}).to_csv(out_dir / f"seed{args.seed}__val.csv", index=False)
    pd.DataFrame({"design_name": test.design_name.values,
                  "net_name": test.net_name.values,
                  "y_true": y_te, "y_pred": pt}).to_csv(out_dir / f"seed{args.seed}__test.csv", index=False)
    print(f"saved {out_dir}/seed{args.seed}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f}s")
