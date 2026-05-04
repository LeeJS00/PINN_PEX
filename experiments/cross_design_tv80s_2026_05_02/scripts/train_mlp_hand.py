"""
Hand-feature MLP — pure tabular neural net, no cuboid set loading.

Trains an MLP on the parquet features only. Runs on GPU (fast). Used as
an alternative to GBDT in the ensemble.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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


class HandMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, depth: int = 4, dropout: float = 0.05):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1) + self.bias


def hybrid_loss(pred_log: torch.Tensor, y_lin: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    pred = torch.exp(pred_log)
    y = y_lin.clamp(min=eps)
    ape = torch.abs(pred - y) / y
    return ape.mean() + 0.05 * F.mse_loss(pred_log, torch.log(y))


def _load(d: str, cache: Path) -> pd.DataFrame:
    df = pd.read_parquet(cache / f"{d}.parquet")
    df["design_name"] = d
    return df


def assemble_split(cache: Path):
    avail = {p.stem for p in cache.glob("*.parquet")}
    train_pool = [d for d in cfg.TRAIN_DESIGNS if d in avail]
    val_pool = [d for d in cfg.VAL_DESIGNS if d in avail]
    if not val_pool:
        if "intel22_ldpc_decoder_802_3an_f3" in avail:
            val_pool = ["intel22_ldpc_decoder_802_3an_f3"]
        else:
            val_pool = ["intel22_ibex_core_f3"] if "intel22_ibex_core_f3" in avail else [train_pool[0]]
        for v in val_pool:
            if v in train_pool:
                train_pool.remove(v)
    train = pd.concat([_load(d, cache) for d in train_pool], ignore_index=True)
    val   = pd.concat([_load(d, cache) for d in val_pool],   ignore_index=True)
    test  = _load("intel22_tv80s_f3", cache)
    return train, val, test, train_pool, val_pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="features_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--out", default="mlp_hand")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = cfg.CACHE_DIR / args.cache
    train, val, test, tr_pool, va_pool = assemble_split(cache_dir)
    fcols = _select_feature_cols(train)
    print(f"train designs: {tr_pool}")
    print(f"val designs:   {va_pool}")
    print(f"features: {len(fcols)}, train: {len(train):,}, val: {len(val):,}, test: {len(test):,}")

    X_tr = train[fcols].to_numpy(np.float32)
    X_va = val  [fcols].to_numpy(np.float32)
    X_te = test [fcols].to_numpy(np.float32)
    y_tr_lin = train["total_cap_fF"].to_numpy(np.float32)
    y_va_lin = val  ["total_cap_fF"].to_numpy(np.float32)
    y_te_lin = test ["total_cap_fF"].to_numpy(np.float32)

    # Normalize features (use train statistics)
    mu = np.nanmean(X_tr, axis=0); sd = np.nanstd(X_tr, axis=0); sd[sd == 0] = 1.0
    X_tr = ((X_tr - mu) / sd).clip(-8, 8)
    X_va = ((X_va - mu) / sd).clip(-8, 8)
    X_te = ((X_te - mu) / sd).clip(-8, 8)

    X_tr_t = torch.from_numpy(np.nan_to_num(X_tr)).to(device)
    X_va_t = torch.from_numpy(np.nan_to_num(X_va)).to(device)
    X_te_t = torch.from_numpy(np.nan_to_num(X_te)).to(device)
    y_tr_t = torch.from_numpy(y_tr_lin).to(device)
    y_va_t = torch.from_numpy(y_va_lin).to(device)

    model = HandMLP(in_dim=len(fcols), hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    best_state = None
    out_dir = cfg.OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    n = X_tr_t.shape[0]
    n_batches = max(n // args.batch_size, 1)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        losses = []
        for i in range(n_batches):
            idx = perm[i*args.batch_size:(i+1)*args.batch_size]
            opt.zero_grad()
            pred = model(X_tr_t[idx])  # log scale
            loss = hybrid_loss(pred, y_tr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.exp(model(X_va_t)).cpu().numpy()
        if ep % 10 == 0 or ep == args.epochs - 1:
            vm = report_mape(y_va_lin, pv, f"ep{ep}")
            if vm["mape_median"] < best_val:
                best_val = vm["mape_median"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            ape = np.abs(pv - y_va_lin) / np.maximum(y_va_lin, 1e-3)
            med = float(np.median(ape * 100))
            if med < best_val:
                best_val = med
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = torch.exp(model(X_va_t)).cpu().numpy()
        pt = torch.exp(model(X_te_t)).cpu().numpy()
    print()
    vm = report_mape(y_va_lin, pv, "BEST val")
    tm = report_mape(y_te_lin, pt, "BEST test")

    pd.DataFrame({
        "design_name": test["design_name"].values,
        "net_name":    test["net_name"].values,
        "y_true":      y_te_lin,
        "y_pred":      pt,
    }).to_csv(out_dir / f"seed{args.seed}__test.csv", index=False)
    pd.DataFrame({
        "design_name": val["design_name"].values,
        "net_name":    val["net_name"].values,
        "y_true":      y_va_lin,
        "y_pred":      pv,
    }).to_csv(out_dir / f"seed{args.seed}__val.csv", index=False)

    with open(out_dir / f"seed{args.seed}_summary.json", "w") as f:
        json.dump({"seed": args.seed, "val": vm, "test": tm}, f, indent=2)
    torch.save({"model": best_state, "feature_cols": fcols, "mu": mu, "sd": sd},
               out_dir / f"seed{args.seed}.pt")


if __name__ == "__main__":
    main()
