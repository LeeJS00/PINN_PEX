#!/usr/bin/env python3
"""
22_finetune_hybrid_perpair_smoke.py — Strike #2: Per-pair coupling smoke run.

HybridPexV3PerPair = Mesh + per-pair cpl head with explicit per-pair
supervision. K aggressors sampled per target per batch.

Loss:
    L = w_gnd × MAPE(gnd) + w_pair × MAPE(per-pair cpl) + w_total × MAPE(cpl_total estimate)

Pre-req:
    - 18_extract_per_net_cuboids.py done (per_net_cuboids/*.npz)
    - 21_extract_per_pair_golden.py done (per_pair_golden/*.parquet)
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.models.hybrid_v3_perpair import HybridPexV3PerPair  # noqa: E402
from src.models.residual_head_v3 import res_clamp_for_epoch  # noqa: E402
from src.data.cuboid_set_dataset import PerNetCuboidStore  # noqa: E402
from src.data.per_pair_dataset import (  # noqa: E402
    PerPairDataset, collate_per_pair_batch,
)
from src.trainers.finetune_hybrid_v3 import (  # noqa: E402
    split_by_manifest_column, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
)
from src.baselines.calibration_v3 import (  # noqa: E402
    fit_per_layer_calibration, apply_per_layer_calibration,
)
from src.utils.seeds import set_all_seeds  # noqa: E402
from src.utils.manifest_hash import write_provenance  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Strike #2 per-pair smoke")
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument(
        "--cuboid-dir", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids"),
    )
    p.add_argument(
        "--per-pair-dir", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/per_pair_golden"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "phase1_perpair_smoke",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)  # smaller because K aggressors per item
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--k-aggressors", type=int, default=5)
    p.add_argument("--w-gnd", type=float, default=1.0)
    p.add_argument("--w-pair", type=float, default=1.0)
    p.add_argument("--w-total", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--early-stop-patience", type=int, default=80)
    return p.parse_args()


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    valid_total_mape: list[float] = field(default_factory=list)
    valid_gnd_mape: list[float] = field(default_factory=list)
    valid_cpl_mape: list[float] = field(default_factory=list)
    valid_pair_mape: list[float] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)
    best_epoch: int = -1
    best_valid_total_mape: float = float("inf")
    best_valid_gnd_mape: float = float("inf")
    best_valid_cpl_mape: float = float("inf")


def evaluate(model, loader, device, eps_fF=1e-3) -> dict:
    model.eval()
    gnd_rels, cpl_rels, total_rels, pair_rels = [], [], [], []
    with torch.no_grad():
        for b in loader:
            tg_cb = b["target_cuboids"].to(device)
            tg_mk = b["target_mask"].to(device)
            ag_cb = b["aggr_cuboids"].to(device)
            ag_mk = b["aggr_mask"].to(device)
            ag_self = b["aggr_self_features"].to(device)
            target_self = b["target_self_features"].to(device)
            sm = b["sampled_mask"].to(device)
            analytic_gnd = b["target_analytic_gnd"].to(device)
            analytic_pair = b["analytic_pair_baseline"].to(device)
            n_aggr = b["n_aggr_total"].to(device)
            gnd_gold = b["target_golden_gnd"].to(device)
            cpl_gold = b["target_golden_cpl_total"].to(device)
            pair_gold = b["c_pair_golden"].to(device)

            tg_emb = model.encode_target(tg_cb, tg_mk)
            ag_emb = model.encode_aggressors(ag_cb, ag_mk)
            pg = model.predict_gnd(analytic_gnd, target_self, tg_emb)
            pp = model.predict_pair_cpl(analytic_pair, target_self, ag_self, tg_emb, ag_emb)
            cpl_pred = model.aggregate_cpl_total(pp, sm, n_aggr)

            gnd_rels.append(((pg - gnd_gold).abs() / gnd_gold.clamp(min=eps_fF)).cpu())
            cpl_rels.append(((cpl_pred - cpl_gold).abs() / cpl_gold.clamp(min=eps_fF)).cpu())
            total_rels.append(((pg + cpl_pred - gnd_gold - cpl_gold).abs() / (gnd_gold + cpl_gold).clamp(min=eps_fF)).cpu())
            # Per-pair MAPE: only on sampled (mask==1)
            valid_mask = sm.bool()
            if valid_mask.any():
                rel = ((pp - pair_gold).abs() / pair_gold.clamp(min=eps_fF))[valid_mask].cpu()
                pair_rels.append(rel)
    g = torch.cat(gnd_rels)
    c = torch.cat(cpl_rels)
    t = torch.cat(total_rels)
    p = torch.cat(pair_rels) if pair_rels else torch.tensor([0.0])
    return {
        "gnd_mape_median": float(g.median()),
        "cpl_mape_median": float(c.median()),
        "total_mape_median": float(t.median()),
        "pair_mape_median": float(p.median()),
        "n_targets": int(len(g)),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed, deterministic=True)
    print(f">>> Strike #2 per-pair smoke — seed {args.seed}")
    print(f">>> features:    {args.features_csv}")
    print(f">>> cuboids:     {args.cuboid_dir}")
    print(f">>> per-pair:    {args.per_pair_dir}")
    print(f">>> output:      {args.output_dir}")
    print(f">>> device:      {args.device}")
    print(f">>> k_aggressors: {args.k_aggressors}")
    print(f">>> loss weights: gnd={args.w_gnd} pair={args.w_pair} total={args.w_total}")

    # ---- Load + split + calibrate ----
    df = pd.read_csv(args.features_csv)
    train_df, valid_df, test_df = split_by_manifest_column(df)
    for d in (train_df, valid_df, test_df):
        d.drop(d[(d["c_gnd_fF"] + d["c_cpl_total_fF"]) <= 1e-4].index, inplace=True)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f">>> splits: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

    calib = fit_per_layer_calibration(train_df)
    train_df = apply_per_layer_calibration(train_df, calib)
    valid_df = apply_per_layer_calibration(valid_df, calib)
    test_df = apply_per_layer_calibration(test_df, calib)
    print(f">>> NNLS calibration applied")

    # ---- Load per-pair golden (concat all designs) ----
    print(">>> Loading per-pair golden parquets ...")
    train_designs = set(train_df["design_name"].unique())
    valid_designs = set(valid_df["design_name"].unique())
    test_designs = set(test_df["design_name"].unique())

    train_pairs = []
    valid_pairs = []
    test_pairs = []
    for parquet_path in sorted(args.per_pair_dir.glob("intel22_*.parquet")):
        design = parquet_path.stem
        if design not in train_designs and design not in test_designs:
            continue
        pp = pd.read_parquet(parquet_path)
        if design in train_designs:
            train_pairs.append(pp)
        if design in valid_designs:
            valid_pairs.append(pp)
        if design in test_designs:
            test_pairs.append(pp)
    train_pp = pd.concat(train_pairs, ignore_index=True) if train_pairs else pd.DataFrame()
    valid_pp = pd.concat(valid_pairs, ignore_index=True) if valid_pairs else pd.DataFrame()
    test_pp  = pd.concat(test_pairs,  ignore_index=True) if test_pairs  else pd.DataFrame()
    print(f">>> per-pair: train={len(train_pp):,}  valid={len(valid_pp):,}  test={len(test_pp):,}")

    # ---- Cuboid store ----
    print(">>> Loading cuboid store ...")
    store = PerNetCuboidStore(args.cuboid_dir)
    print(f">>> cuboid store entries: {len(store):,}")

    # ---- Datasets ----
    print(">>> Building per-pair datasets ...")
    train_ds = PerPairDataset(train_df, train_pp, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
                              k_aggressors=args.k_aggressors, rng_seed=args.seed)
    valid_ds = PerPairDataset(valid_df, valid_pp, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
                              k_aggressors=args.k_aggressors, rng_seed=args.seed + 1)
    test_ds  = PerPairDataset(test_df,  test_pp,  store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
                              k_aggressors=args.k_aggressors, rng_seed=args.seed + 2)
    print(f">>> datasets: train={len(train_ds):,}  valid={len(valid_ds):,}  test={len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_per_pair_batch,
                              pin_memory=("cuda" in args.device))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_per_pair_batch,
                              pin_memory=("cuda" in args.device))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_per_pair_batch,
                              pin_memory=("cuda" in args.device))

    # ---- Model ----
    torch.manual_seed(args.seed)
    model = HybridPexV3PerPair().to(args.device)
    pc = model.parameter_count()
    print(f">>> model: {pc}")

    # Provenance
    from configs import config_v3 as cfg
    snap = cfg.v3_snapshot()
    snap["task"] = "phase1_perpair_smoke"
    snap["k_aggressors"] = args.k_aggressors
    snap["model_params"] = pc["total"]
    write_provenance(args.output_dir, args.features_csv, snap, args.seed)

    # ---- Day-1 ----
    day1 = evaluate(model, valid_loader, args.device)
    print(f">>> day-1 valid: gnd={day1['gnd_mape_median']*100:.2f}%  "
          f"cpl(total)={day1['cpl_mape_median']*100:.2f}%  "
          f"per-pair={day1['pair_mape_median']*100:.2f}%  "
          f"total={day1['total_mape_median']*100:.2f}%")

    # ---- Train ----
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = History()
    epochs_no_improve = 0
    t0 = time.time()
    for epoch in range(args.n_epochs):
        clamp = res_clamp_for_epoch(epoch)
        model.set_clamp_bounds(clamp)
        model.train()
        running = 0.0
        n_batches = 0
        for b in train_loader:
            tg_cb = b["target_cuboids"].to(args.device)
            tg_mk = b["target_mask"].to(args.device)
            ag_cb = b["aggr_cuboids"].to(args.device)
            ag_mk = b["aggr_mask"].to(args.device)
            ag_self = b["aggr_self_features"].to(args.device)
            target_self = b["target_self_features"].to(args.device)
            sm = b["sampled_mask"].to(args.device)
            analytic_gnd = b["target_analytic_gnd"].to(args.device)
            analytic_pair = b["analytic_pair_baseline"].to(args.device)
            n_aggr = b["n_aggr_total"].to(args.device)
            gnd_gold = b["target_golden_gnd"].to(args.device)
            cpl_gold = b["target_golden_cpl_total"].to(args.device)
            pair_gold = b["c_pair_golden"].to(args.device)

            tg_emb = model.encode_target(tg_cb, tg_mk)
            ag_emb = model.encode_aggressors(ag_cb, ag_mk)
            pg = model.predict_gnd(analytic_gnd, target_self, tg_emb)
            pp = model.predict_pair_cpl(analytic_pair, target_self, ag_self, tg_emb, ag_emb)
            cpl_pred = model.aggregate_cpl_total(pp, sm, n_aggr)

            eps = 1e-3
            gnd_loss = ((pg - gnd_gold).abs() / gnd_gold.clamp(min=eps)).mean()
            valid_mask = sm.bool()
            if valid_mask.any():
                pair_loss = ((pp - pair_gold).abs() / pair_gold.clamp(min=eps))[valid_mask].mean()
            else:
                pair_loss = torch.tensor(0.0, device=args.device)
            total_loss_term = ((cpl_pred - cpl_gold).abs() / cpl_gold.clamp(min=eps)).mean()
            loss = args.w_gnd * gnd_loss + args.w_pair * pair_loss + args.w_total * total_loss_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        avg_loss = running / max(1, n_batches)
        history.train_loss.append(avg_loss)

        v = evaluate(model, valid_loader, args.device)
        history.valid_total_mape.append(v["total_mape_median"])
        history.valid_gnd_mape.append(v["gnd_mape_median"])
        history.valid_cpl_mape.append(v["cpl_mape_median"])
        history.valid_pair_mape.append(v["pair_mape_median"])
        history.epoch.append(epoch)
        elapsed = time.time() - t0
        print(f"  epoch {epoch}/{args.n_epochs}: clamp={clamp:.3f}  loss={avg_loss:.4f}  "
              f"valid: gnd={v['gnd_mape_median']*100:.2f}%  "
              f"cpl(tot)={v['cpl_mape_median']*100:.2f}%  "
              f"pair={v['pair_mape_median']*100:.2f}%  "
              f"total={v['total_mape_median']*100:.2f}%  ({elapsed:.0f}s)", flush=True)

        if v["total_mape_median"] < history.best_valid_total_mape:
            history.best_valid_total_mape = v["total_mape_median"]
            history.best_valid_gnd_mape = v["gnd_mape_median"]
            history.best_valid_cpl_mape = v["cpl_mape_median"]
            history.best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(f"  early stop at epoch {epoch}")
                break
    train_elapsed = time.time() - t0

    # ---- Final + test ----
    final_valid = evaluate(model, valid_loader, args.device)
    final_test = evaluate(model, test_loader, args.device)
    print()
    print(f">>> final valid: gnd={final_valid['gnd_mape_median']*100:.2f}%  "
          f"cpl(tot)={final_valid['cpl_mape_median']*100:.2f}%  "
          f"pair={final_valid['pair_mape_median']*100:.2f}%  "
          f"total={final_valid['total_mape_median']*100:.2f}%")
    print(f">>> final test : gnd={final_test['gnd_mape_median']*100:.2f}%  "
          f"cpl(tot)={final_test['cpl_mape_median']*100:.2f}%  "
          f"pair={final_test['pair_mape_median']*100:.2f}%  "
          f"total={final_test['total_mape_median']*100:.2f}%")

    val_total = final_valid["total_mape_median"]
    if val_total <= 0.06:
        verdict = f"🎯 ≤6%: BREAKTHROUGH — beats hand-feature ceiling"
    elif val_total <= 0.08:
        verdict = f"✅ ≤8%: paper-grade improvement on Mesh"
    else:
        verdict = f"⚠ >8%: marginal improvement; consider Strike #3 (loss redesign)"
    print()
    print("=" * 60)
    print(f">>> STRIKE #2 RESULT (final valid total = {val_total*100:.3f}%)")
    print(f"    {verdict}")
    print("=" * 60)

    torch.save(model.state_dict(), args.output_dir / "model.pt")
    summary = {
        "seed": args.seed,
        "n_epochs": args.n_epochs,
        "k_aggressors": args.k_aggressors,
        "model_params": pc,
        "elapsed_train_sec": train_elapsed,
        "day1_valid": day1,
        "final_valid": final_valid,
        "final_test": final_test,
        "best_epoch": history.best_epoch,
        "best_valid_total_mape": history.best_valid_total_mape,
        "best_valid_gnd_mape": history.best_valid_gnd_mape,
        "best_valid_cpl_mape": history.best_valid_cpl_mape,
        "verdict": verdict,
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(args.output_dir / "history.json", "w") as f:
        json.dump({
            "epoch": history.epoch,
            "train_loss": history.train_loss,
            "valid_total_mape": history.valid_total_mape,
            "valid_gnd_mape": history.valid_gnd_mape,
            "valid_cpl_mape": history.valid_cpl_mape,
            "valid_pair_mape": history.valid_pair_mape,
        }, f, indent=2)
    print(f"✅ smoke complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
