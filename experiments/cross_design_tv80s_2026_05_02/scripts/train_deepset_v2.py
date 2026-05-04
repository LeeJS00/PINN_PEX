"""
Train 3-stream DeepSet over cached cuboid arrays + hand-feature MLP.

Reads cache/cuboid_arr/<design>.npz for set inputs and cache/features_v3/<design>.parquet
for hand features. Per-net targets from the parquets.

3-stream encoder pools target / aggressor / power cuboid sets via {mean, max, sum, std}.
Hand-feature branch fuses with the set embeddings. Trained with hybrid (MAPE + log-MSE) loss.
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _scale_cuboid_input(x: torch.Tensor) -> torch.Tensor:
    """x: (B, N, 10). Apply same scaling logic as the original neural_field."""
    x = x.clone()
    SCALE = 4.0
    x[..., 0] = x[..., 0] / SCALE
    x[..., 1] = x[..., 1] / SCALE
    x[..., 2] = (x[..., 2] - 1.5)
    x[..., 3] = torch.log1p(x[..., 3])
    x[..., 4] = torch.log1p(x[..., 4])
    x[..., 5] = torch.log1p(x[..., 5])
    x[..., 8] = (x[..., 8] - 3.5)
    return x


class CuboidEncoder(nn.Module):
    def __init__(self, in_dim=10, hidden=128, out=128, dropout=0.05):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.fc(_scale_cuboid_input(x))


def masked_pool(emb, mask):
    """emb (B, K, D), mask (B, K) bool. Returns mean+max+sum+std → (B, 4D).
    If all elements are masked, returns zeros (max would otherwise be -inf).
    """
    m = mask.unsqueeze(-1).to(emb.dtype)
    denom = m.sum(dim=1).clamp(min=1.0)
    s = (emb * m).sum(dim=1)
    mean = s / denom
    # Max with safe fallback: use a large finite negative, then clamp result if all-masked
    NEG = -1e6
    e_max = emb.masked_fill(~mask.unsqueeze(-1), NEG).max(dim=1).values
    has_any = mask.any(dim=1, keepdim=True).to(emb.dtype)
    e_max = e_max * has_any  # zero out rows where no valid elements
    mu = mean.unsqueeze(1)
    var = (((emb - mu) ** 2) * m).sum(dim=1) / denom
    std = torch.sqrt(var + 1e-8)
    return torch.cat([mean, e_max, s, std], dim=-1)


class DeepSetModel(nn.Module):
    def __init__(self, hand_dim, cub_hidden=128, cub_out=128, hand_hidden=192,
                 trunk_hidden=384, dropout=0.1):
        super().__init__()
        self.enc_t = CuboidEncoder(10, cub_hidden, cub_out, dropout)
        self.enc_a = CuboidEncoder(10, cub_hidden, cub_out, dropout)
        self.enc_p = CuboidEncoder(10, cub_hidden, cub_out, dropout)
        pool_d = 4 * cub_out
        self.proj_t = nn.Sequential(nn.Linear(pool_d, trunk_hidden), nn.GELU())
        self.proj_a = nn.Sequential(nn.Linear(pool_d, trunk_hidden), nn.GELU())
        self.proj_p = nn.Sequential(nn.Linear(pool_d, trunk_hidden), nn.GELU())
        self.hand = nn.Sequential(
            nn.Linear(hand_dim, hand_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hand_hidden, hand_hidden), nn.GELU(),
        )
        in_trunk = 3 * trunk_hidden + hand_hidden
        self.trunk = nn.Sequential(
            nn.Linear(in_trunk, trunk_hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(trunk_hidden, trunk_hidden), nn.GELU(),
            nn.Linear(trunk_hidden, 1),
        )
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _stream(self, enc, x, mask, proj):
        return proj(masked_pool(enc(x), mask))

    def forward(self, T, A, P, mT, mA, mP, hand):
        zt = self._stream(self.enc_t, T, mT, self.proj_t)
        za = self._stream(self.enc_a, A, mA, self.proj_a)
        zp = self._stream(self.enc_p, P, mP, self.proj_p)
        zh = self.hand(hand)
        z = torch.cat([zt, za, zp, zh], dim=-1)
        return self.trunk(z).squeeze(-1) + self.bias


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def hybrid_loss(pred_log, y_lin, eps=1e-3):
    pred = torch.exp(pred_log)
    y = y_lin.clamp(min=eps)
    ape = torch.abs(pred - y) / y
    return ape.mean() + 0.05 * F.mse_loss(pred_log, torch.log(y))


def load_design_arrays(design: str, parquet_path: Path, cuboid_path: Path,
                        hand_mu, hand_sd, fcols):
    """Load aligned arrays for a design.
    Returns: T, A, P, n_t, n_a, n_p, hand, y, design_names, net_names
    """
    arr = np.load(cuboid_path, allow_pickle=True)
    df = pd.read_parquet(parquet_path)
    df["design_name"] = design

    # align by net_name
    cuboid_names = arr["net_names"]
    name_to_idx = {n: i for i, n in enumerate(cuboid_names)}
    keep_mask = df["net_name"].isin(name_to_idx).to_numpy()
    df = df[keep_mask].reset_index(drop=True)

    arr_idx = np.array([name_to_idx[n] for n in df["net_name"].values])
    T = arr["target"][arr_idx]
    A = arr["aggressor"][arr_idx]
    P = arr["power"][arr_idx]
    n_t = arr["n_target"][arr_idx]
    n_a = arr["n_agg"][arr_idx]
    n_p = arr["n_pwr"][arr_idx]

    hand = df[fcols].to_numpy(np.float32)
    hand = ((hand - hand_mu) / (hand_sd + 1e-6)).clip(-8, 8)
    hand = np.nan_to_num(hand)

    y = df["total_cap_fF"].to_numpy(np.float32)
    return T, A, P, n_t, n_a, n_p, hand, y, df["design_name"].values, df["net_name"].values


def collect_split(parquet_dir: Path, cuboid_dir: Path, designs, hand_mu=None, hand_sd=None,
                   fcols=None):
    """Returns concatenated arrays + (mu, sd) if not provided."""
    parquet_first = pd.read_parquet(parquet_dir / f"{designs[0]}.parquet")
    if fcols is None:
        fcols = _select_feature_cols(parquet_first)
    if hand_mu is None:
        # compute mu/sd from designs[0]
        Xs = []
        for d in designs:
            df = pd.read_parquet(parquet_dir / f"{d}.parquet")
            Xs.append(df[fcols].to_numpy(np.float32))
        Xall = np.vstack(Xs)
        hand_mu = np.nanmean(Xall, axis=0); hand_sd = np.nanstd(Xall, axis=0)
        hand_sd[hand_sd == 0] = 1.0

    T_l, A_l, P_l, nt_l, na_l, np_l, h_l, y_l, dn_l, nn_l = [], [], [], [], [], [], [], [], [], []
    for d in designs:
        T, A, P, n_t, n_a, n_p, h, y, dn, nn = load_design_arrays(
            d, parquet_dir / f"{d}.parquet", cuboid_dir / f"{d}.npz",
            hand_mu, hand_sd, fcols)
        T_l.append(T); A_l.append(A); P_l.append(P)
        nt_l.append(n_t); na_l.append(n_a); np_l.append(n_p)
        h_l.append(h); y_l.append(y); dn_l.append(dn); nn_l.append(nn)
    return (np.concatenate(T_l), np.concatenate(A_l), np.concatenate(P_l),
            np.concatenate(nt_l), np.concatenate(na_l), np.concatenate(np_l),
            np.concatenate(h_l), np.concatenate(y_l),
            np.concatenate(dn_l), np.concatenate(nn_l), fcols, hand_mu, hand_sd)


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--out", default="deepset_v2")
    ap.add_argument("--use-nova-val", action="store_true")
    args = ap.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    parquet_dir = cfg.CACHE_DIR / "features_v3"
    cuboid_dir = cfg.CACHE_DIR / "cuboid_arr"

    if args.use_nova_val:
        train_pool = list(cfg.TRAIN_DESIGNS)
        val_pool = ["intel22_nova_f3"]
    else:
        train_pool = [d for d in cfg.TRAIN_DESIGNS if d != "intel22_ibex_core_f3"]
        val_pool = ["intel22_ibex_core_f3"]
    test_pool = ["intel22_tv80s_f3"]
    print(f"train: {train_pool}\nval: {val_pool}")

    print("Loading train ...")
    Ttr, Atr, Ptr, nTtr, nAtr, nPtr, htr, ytr, dn_tr, nn_tr, fcols, mu, sd = \
        collect_split(parquet_dir, cuboid_dir, train_pool)
    print(f"  train shapes: T {Ttr.shape}, A {Atr.shape}, P {Ptr.shape}, hand {htr.shape}")
    print("Loading val ...")
    Tv, Av, Pv, nTv, nAv, nPv, hv, yv, dn_v, nn_v, _, _, _ = \
        collect_split(parquet_dir, cuboid_dir, val_pool, mu, sd, fcols)
    print("Loading test ...")
    Tt, At, Pt, nTt, nAt, nPt, ht, yt, dn_t, nn_t, _, _, _ = \
        collect_split(parquet_dir, cuboid_dir, test_pool, mu, sd, fcols)

    # Move to GPU
    def to_cuda(x): return torch.from_numpy(x).to(device)
    Ttr_t = to_cuda(Ttr); Atr_t = to_cuda(Atr); Ptr_t = to_cuda(Ptr)
    nTtr_t = to_cuda(nTtr.astype(np.int64))
    nAtr_t = to_cuda(nAtr.astype(np.int64))
    nPtr_t = to_cuda(nPtr.astype(np.int64))
    htr_t = to_cuda(htr); ytr_t = to_cuda(ytr)

    Tv_t = to_cuda(Tv); Av_t = to_cuda(Av); Pv_t = to_cuda(Pv)
    nTv_t = to_cuda(nTv.astype(np.int64)); nAv_t = to_cuda(nAv.astype(np.int64))
    nPv_t = to_cuda(nPv.astype(np.int64))
    hv_t = to_cuda(hv); yv_t = to_cuda(yv)

    Tt_t = to_cuda(Tt); At_t = to_cuda(At); Pt_t = to_cuda(Pt)
    nTt_t = to_cuda(nTt.astype(np.int64)); nAt_t = to_cuda(nAt.astype(np.int64))
    nPt_t = to_cuda(nPt.astype(np.int64))
    ht_t = to_cuda(ht)

    T_max, A_max, P_max = Ttr.shape[1], Atr.shape[1], Ptr.shape[1]
    rngT = torch.arange(T_max, device=device)
    rngA = torch.arange(A_max, device=device)
    rngP = torch.arange(P_max, device=device)

    def make_masks(nT, nA, nP):
        return rngT.unsqueeze(0) < nT.unsqueeze(1), rngA.unsqueeze(0) < nA.unsqueeze(1), rngP.unsqueeze(0) < nP.unsqueeze(1)

    model = DeepSetModel(hand_dim=len(fcols)).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs)

    def chunked_predict(model, T, A, P, nT, nA, nP, h, chunk=2048):
        """Chunked forward pass to avoid OOM on large val/test sets."""
        n = T.shape[0]
        out = []
        with torch.no_grad():
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                mT, mA, mP = make_masks(nT[s:e], nA[s:e], nP[s:e])
                p = torch.exp(model(T[s:e], A[s:e], P[s:e], mT, mA, mP, h[s:e]))
                out.append(p.cpu().numpy())
        return np.concatenate(out)

    best_val = float("inf"); best_state = None
    n = Ttr_t.shape[0]
    n_batches = max(n // args.batch_size, 1)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(n_batches):
            idx = perm[i*args.batch_size:(i+1)*args.batch_size]
            mT, mA, mP = make_masks(nTtr_t[idx], nAtr_t[idx], nPtr_t[idx])
            opt.zero_grad()
            pred = model(Ttr_t[idx], Atr_t[idx], Ptr_t[idx], mT, mA, mP, htr_t[idx])
            loss = hybrid_loss(pred, ytr_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        # Val (chunked)
        model.eval()
        pv = chunked_predict(model, Tv_t, Av_t, Pv_t, nTv_t, nAv_t, nPv_t, hv_t, chunk=2048)
        ape = np.abs(pv - yv) / np.maximum(yv, 1e-3) * 100
        med = float(np.median(ape)); mean = float(np.mean(ape))
        if med < best_val:
            best_val = med
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep}: val mean={mean:.3f}% med={med:.3f}% best_med={best_val:.3f}%")

    model.load_state_dict(best_state)
    model.eval()
    pt = chunked_predict(model, Tt_t, At_t, Pt_t, nTt_t, nAt_t, nPt_t, ht_t, chunk=2048)
    pv = chunked_predict(model, Tv_t, Av_t, Pv_t, nTv_t, nAv_t, nPv_t, hv_t, chunk=2048)

    print()
    vm = report_mape(yv, pv, "BEST val")
    tm = report_mape(yt, pt, "BEST test")

    out_dir = cfg.OUTPUT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"design_name": dn_t, "net_name": nn_t,
                  "y_true": yt, "y_pred": pt}).to_csv(out_dir / f"seed{args.seed}__test.csv", index=False)
    pd.DataFrame({"design_name": dn_v, "net_name": nn_v,
                  "y_true": yv, "y_pred": pv}).to_csv(out_dir / f"seed{args.seed}__val.csv", index=False)
    with open(out_dir / f"seed{args.seed}_summary.json", "w") as f:
        json.dump({"seed": args.seed, "val": vm, "test": tm}, f, indent=2)
    # Save model weights and feature norms for inference-time reuse
    torch.save({"model": best_state, "feature_cols": fcols,
                "mu": mu.tolist() if hasattr(mu, "tolist") else mu,
                "sd": sd.tolist() if hasattr(sd, "tolist") else sd,
                "args": vars(args)},
               out_dir / f"seed{args.seed}.pt")
    print(f"saved {out_dir / f'seed{args.seed}.pt'}")


if __name__ == "__main__":
    main()
