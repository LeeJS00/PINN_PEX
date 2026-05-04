"""Improved MLP with residual connections, deeper, bigger.

Trains on log(total_cap_fF). Tracks both mean and median MAPE.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import _select_feature_cols, report_mape
from src.derived_feats import add_derived


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.fc1(x))
        h = self.drop(h)
        h = self.fc2(h)
        return self.norm(x + h)


class ResMLP(nn.Module):
    def __init__(self, in_dim, hidden=384, n_blocks=6, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x = F.gelu(self.proj(x))
        for b in self.blocks:
            x = b(x)
        return self.out(x).squeeze(-1) + self.bias


def hybrid_loss(pred_log, y_lin, eps=1e-3):
    pred = torch.exp(pred_log)
    y = y_lin.clamp(min=eps)
    ape = torch.abs(pred - y) / y
    return ape.mean() + 0.1 * F.mse_loss(pred_log, torch.log(y))


def assemble_split(cache: Path, add_derived_feats: bool):
    avail = {p.stem for p in cache.glob("*.parquet")}
    train_pool = [d for d in cfg.TRAIN_DESIGNS if d in avail]
    val_pool = [d for d in cfg.VAL_DESIGNS if d in avail]
    if not val_pool:
        if "intel22_ibex_core_f3" in avail:
            val_pool = ["intel22_ibex_core_f3"]
        else:
            val_pool = [train_pool[-1]]
        for v in val_pool:
            if v in train_pool:
                train_pool.remove(v)
    print(f"train: {train_pool}\nval: {val_pool}")
    train = pd.concat([
        pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in train_pool
    ], ignore_index=True)
    val   = pd.concat([
        pd.read_parquet(cache / f"{d}.parquet").assign(design_name=d) for d in val_pool
    ], ignore_index=True)
    test  = pd.read_parquet(cache / "intel22_tv80s_f3.parquet").assign(design_name="intel22_tv80s_f3")
    if add_derived_feats:
        train = add_derived(train); val = add_derived(val); test = add_derived(test)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="features_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--out", default="resmlp_v2")
    ap.add_argument("--hidden", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--add-derived", action="store_true")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    cache = cfg.CACHE_DIR / args.cache
    train, val, test = assemble_split(cache, args.add_derived)
    fcols = _select_feature_cols(train)
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val  [fcols].to_numpy(np.float32)
    X_te = test [fcols].to_numpy(np.float32)
    y_tr_lin = train["total_cap_fF"].to_numpy(np.float32)
    y_va_lin = val  ["total_cap_fF"].to_numpy(np.float32)
    y_te_lin = test ["total_cap_fF"].to_numpy(np.float32)

    mu = np.nanmean(X_tr, axis=0); sd = np.nanstd(X_tr, axis=0); sd[sd == 0] = 1.0
    X_tr = ((X_tr - mu) / sd).clip(-8, 8)
    X_va = ((X_va - mu) / sd).clip(-8, 8)
    X_te = ((X_te - mu) / sd).clip(-8, 8)

    X_tr_t = torch.from_numpy(np.nan_to_num(X_tr)).to(device)
    X_va_t = torch.from_numpy(np.nan_to_num(X_va)).to(device)
    X_te_t = torch.from_numpy(np.nan_to_num(X_te)).to(device)
    y_tr_t = torch.from_numpy(y_tr_lin).to(device)
    y_va_t = torch.from_numpy(y_va_lin).to(device)

    model = ResMLP(in_dim=len(fcols), hidden=args.hidden, n_blocks=args.blocks).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs)

    best_val = float("inf")
    best_state = None
    n = X_tr_t.shape[0]
    n_batches = max(n // args.batch_size, 1)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(n_batches):
            idx = perm[i*args.batch_size:(i+1)*args.batch_size]
            opt.zero_grad()
            pred = model(X_tr_t[idx])
            loss = hybrid_loss(pred, y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.exp(model(X_va_t)).cpu().numpy()
        ape = np.abs(pv - y_va_lin) / np.maximum(y_va_lin, 1e-3)
        med = float(np.median(ape * 100))
        if med < best_val:
            best_val = med
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0 or ep == args.epochs - 1:
            mean = float(np.mean(ape * 100))
            print(f"  ep{ep}: val mean={mean:.3f}% med={med:.3f}% (best_med={best_val:.3f}%)")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.exp(model(X_va_t)).cpu().numpy()
        pt = torch.exp(model(X_te_t)).cpu().numpy()
    print()
    vm = report_mape(y_va_lin, pv, "BEST val")
    tm = report_mape(y_te_lin, pt, "BEST test")

    out_dir = cfg.OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"design_name": test["design_name"].values, "net_name": test["net_name"].values,
                  "y_true": y_te_lin, "y_pred": pt}).to_csv(out_dir / f"seed{args.seed}__test.csv", index=False)
    pd.DataFrame({"design_name": val["design_name"].values, "net_name": val["net_name"].values,
                  "y_true": y_va_lin, "y_pred": pv}).to_csv(out_dir / f"seed{args.seed}__val.csv", index=False)
    with open(out_dir / f"seed{args.seed}_summary.json", "w") as f:
        json.dump({"seed": args.seed, "val": vm, "test": tm}, f, indent=2)


if __name__ == "__main__":
    main()
