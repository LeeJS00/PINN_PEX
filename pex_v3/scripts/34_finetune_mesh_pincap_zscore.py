#!/usr/bin/env python3
"""
28_finetune_mesh_with_cell.py — Mesh PINN with cell-internal features.

Strike #8c (z-score per-design): extend self_features 16 → 29 by adding sister's v6 cell-internal
features (OBS, SIZE, signal pins). Hypothesis: cell-internal substrate cap
contribution will reduce gnd MAPE 19-22% ceiling.

Same as `19_finetune_hybrid_mesh_smoke.py` but:
- features-csv: all_designs_with_pincap_zscore.csv (62 cols, +7 Liberty pin-cap (z-score per-design) features)
- _SELF_FEATURE_COLS_EXT: 29 fields
- HybridPexV3Mesh(self_feature_dim=29)
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

from src.models.hybrid_v3_mesh import HybridPexV3Mesh  # noqa: E402
from src.models.hybrid_v3 import per_channel_mape_loss  # noqa: E402
from src.models.residual_head_v3 import res_clamp_for_epoch  # noqa: E402
from src.data.cuboid_set_dataset import (  # noqa: E402
    PerNetCuboidStore, CuboidAugmentedDataset, collate_cuboid_batch,
)
from src.trainers.finetune_hybrid_v3 import (  # noqa: E402
    split_by_manifest_column, _PAIR_FEATURE_COLS,
)
from src.baselines.calibration_v3 import (  # noqa: E402
    fit_per_layer_calibration, apply_per_layer_calibration,
    validate_calibration,
)
from src.utils.seeds import set_all_seeds  # noqa: E402
from src.utils.manifest_hash import write_provenance  # noqa: E402


# Extended self features: original 16 + 7 Liberty pin-cap features
_SELF_FEATURE_COLS_EXT = [
    # ---- original 16 (same as `_SELF_FEATURE_COLS`) ----
    "compact_gnd_estimate_fF",
    "total_wire_length_um",
    "total_metal_area_um2",
    "n_cuboids",
    "bbox_xy_um2",
    "bbox_z_um",
    "n_layers_present",
    "eps_mean",
    "vss_shield_M1_M3",
    "vss_shield_M4_M5",
    "vss_shield_M6_plus",
    "density_M1_M3",
    "density_M4_M5",
    "density_M6_plus",
    "fanout",
    "aspect_ratio",
    # ---- new 7 Liberty pin-cap features (z-score per-design for fF-valued, raw for counts) ----
    "pin_cap_total_zscore",
    "pin_cap_input_total_zscore",
    "pin_cap_max_zscore",
    "pin_cap_mean_zscore",
    "n_pins_lib_matched",
    "n_input_pins_lib",
    "n_output_pins_lib",
]
SELF_DIM_EXT = len(_SELF_FEATURE_COLS_EXT)  # 23


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs_with_pincap_zscore.csv"),
    )
    p.add_argument(
        "--cuboid-dir", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "phase1_mesh_pincap_zscore_smoke",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--early-stop-patience", type=int, default=80)
    return p.parse_args()


@dataclass
class Hist:
    train_loss: list[float] = field(default_factory=list)
    valid_total_mape: list[float] = field(default_factory=list)
    valid_gnd_mape: list[float] = field(default_factory=list)
    valid_cpl_mape: list[float] = field(default_factory=list)
    epoch: list[int] = field(default_factory=list)
    best_epoch: int = -1
    best_valid_total_mape: float = float("inf")
    best_valid_gnd_mape: float = float("inf")
    best_valid_cpl_mape: float = float("inf")


def evaluate(model, loader, device, eps=1e-3) -> dict:
    model.eval()
    g_rels, c_rels, t_rels = [], [], []
    with torch.no_grad():
        for b in loader:
            ag, ac = b["analytic_gnd"].to(device), b["analytic_cpl"].to(device)
            sf, pf = b["self_features"].to(device), b["pair_features"].to(device)
            cb, mk = b["cuboids"].to(device), b["padding_mask"].to(device)
            gg, gc = b["golden_gnd"].to(device), b["golden_cpl"].to(device)
            pg = model.predict_gnd(ag, sf, cb, mk)
            pc = model.predict_cpl(ac, pf, cb, mk)
            g_rels.append(((pg - gg).abs() / gg.clamp(min=eps)).cpu())
            c_rels.append(((pc - gc).abs() / gc.clamp(min=eps)).cpu())
            t_rels.append(((pg + pc - gg - gc).abs() / (gg + gc).clamp(min=eps)).cpu())
    g, c, t = torch.cat(g_rels), torch.cat(c_rels), torch.cat(t_rels)
    return {
        "gnd_mape_median": float(g.median()),
        "cpl_mape_median": float(c.median()),
        "total_mape_median": float(t.median()),
        "n_targets": int(len(g)),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed, deterministic=True)
    print(f">>> Strike #8c (z-score per-design) mesh + Liberty pin caps — seed {args.seed}")
    print(f">>> features:  {args.features_csv}")
    print(f">>> cuboids:   {args.cuboid_dir}")
    print(f">>> output:    {args.output_dir}")
    print(f">>> device:    {args.device}")
    print(f">>> self_dim:  {SELF_DIM_EXT} (was 16, +7 Liberty pin-cap (z-score per-design) features)")

    df = pd.read_csv(args.features_csv)
    train_df, valid_df, test_df = split_by_manifest_column(df)
    for d in (train_df, valid_df, test_df):
        d.drop(d[(d["c_gnd_fF"] + d["c_cpl_total_fF"]) <= 1e-4].index, inplace=True)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f">>> splits: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

    # Calibration
    calib = fit_per_layer_calibration(train_df)
    train_df = apply_per_layer_calibration(train_df, calib)
    valid_df = apply_per_layer_calibration(valid_df, calib)
    test_df = apply_per_layer_calibration(test_df, calib)

    print(">>> Loading cuboid store ...")
    store = PerNetCuboidStore(args.cuboid_dir)
    print(f"  store: {len(store):,}")

    train_ds = CuboidAugmentedDataset(train_df, store, _SELF_FEATURE_COLS_EXT, _PAIR_FEATURE_COLS,
                                       self_dim=SELF_DIM_EXT)
    valid_ds = CuboidAugmentedDataset(valid_df, store, _SELF_FEATURE_COLS_EXT, _PAIR_FEATURE_COLS,
                                       self_dim=SELF_DIM_EXT)
    test_ds  = CuboidAugmentedDataset(test_df,  store, _SELF_FEATURE_COLS_EXT, _PAIR_FEATURE_COLS,
                                       self_dim=SELF_DIM_EXT)
    print(f">>> datasets: train={len(train_ds):,}  valid={len(valid_ds):,}  test={len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
                              pin_memory=("cuda" in args.device))
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
                              pin_memory=("cuda" in args.device))
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
                              pin_memory=("cuda" in args.device))

    torch.manual_seed(args.seed)
    model = HybridPexV3Mesh(self_feature_dim=SELF_DIM_EXT).to(args.device)
    pc = model.parameter_count()
    print(f">>> model: {pc}")

    from configs import config_v3 as cfg
    snap = cfg.v3_snapshot()
    snap["task"] = "phase1_mesh_pincap_zscore_smoke"
    snap["self_dim"] = SELF_DIM_EXT
    write_provenance(args.output_dir, args.features_csv, snap, args.seed)

    day1 = evaluate(model, valid_loader, args.device)
    print(f">>> day-1 valid: gnd={day1['gnd_mape_median']*100:.2f}%  "
          f"cpl={day1['cpl_mape_median']*100:.2f}%  "
          f"total={day1['total_mape_median']*100:.2f}%")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = Hist()
    epochs_no_improve = 0
    t0 = time.time()
    for epoch in range(args.n_epochs):
        clamp = res_clamp_for_epoch(epoch)
        model.set_clamp_bounds(clamp)
        model.train()
        running = 0.0
        n = 0
        for b in train_loader:
            ag, ac = b["analytic_gnd"].to(args.device), b["analytic_cpl"].to(args.device)
            sf, pf = b["self_features"].to(args.device), b["pair_features"].to(args.device)
            cb, mk = b["cuboids"].to(args.device), b["padding_mask"].to(args.device)
            gg, gc = b["golden_gnd"].to(args.device), b["golden_cpl"].to(args.device)
            pg = model.predict_gnd(ag, sf, cb, mk)
            pc_ = model.predict_cpl(ac, pf, cb, mk)
            losses = per_channel_mape_loss(pg, gg, pc_, gc)
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            running += float(losses["total_loss"].item())
            n += 1
        avg_loss = running / max(1, n)
        history.train_loss.append(avg_loss)
        v = evaluate(model, valid_loader, args.device)
        history.valid_total_mape.append(v["total_mape_median"])
        history.valid_gnd_mape.append(v["gnd_mape_median"])
        history.valid_cpl_mape.append(v["cpl_mape_median"])
        history.epoch.append(epoch)
        elapsed = time.time() - t0
        print(f"  epoch {epoch}/{args.n_epochs}: clamp={clamp:.3f}  loss={avg_loss:.4f}  "
              f"valid: gnd={v['gnd_mape_median']*100:.2f}%  cpl={v['cpl_mape_median']*100:.2f}%  "
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

    final_valid = evaluate(model, valid_loader, args.device)
    final_test = evaluate(model, test_loader, args.device)
    print()
    print(f">>> final valid: gnd={final_valid['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_valid['cpl_mape_median']*100:.2f}%  "
          f"total={final_valid['total_mape_median']*100:.2f}%")
    print(f">>> final test : gnd={final_test['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_test['cpl_mape_median']*100:.2f}%  "
          f"total={final_test['total_mape_median']*100:.2f}%")

    print()
    print("=" * 60)
    print(f">>> LIBERTY PIN-CAP STRIKE RESULT")
    print(f"    valid total: {final_valid['total_mape_median']*100:.3f}%  (was 8.59%)")
    print(f"    valid gnd:   {final_valid['gnd_mape_median']*100:.3f}%  (was 19.09%)")
    print(f"    test  total: {final_test['total_mape_median']*100:.3f}%  (was 8.27%)")
    print(f"    test  gnd:   {final_test['gnd_mape_median']*100:.3f}%  (was 20.49%)")
    if final_valid["gnd_mape_median"] < 0.16:
        print(f"    🎯 BREAKTHROUGH: gnd <16% (was 19-20%)")
    elif final_valid["gnd_mape_median"] < 0.18:
        print(f"    ✅ improvement on gnd ceiling")
    else:
        print(f"    ⚠ marginal: gnd ceiling not broken")
    print("=" * 60)

    torch.save(model.state_dict(), args.output_dir / "model.pt")
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "self_dim": SELF_DIM_EXT,
            "self_feature_cols": _SELF_FEATURE_COLS_EXT,
            "model_params": pc,
            "elapsed_train_sec": train_elapsed,
            "day1_valid": day1,
            "final_valid": final_valid,
            "final_test": final_test,
            "best_epoch": history.best_epoch,
            "best_valid_total_mape": history.best_valid_total_mape,
            "best_valid_gnd_mape": history.best_valid_gnd_mape,
            "best_valid_cpl_mape": history.best_valid_cpl_mape,
        }, f, indent=2, default=str)
    with open(args.output_dir / "history.json", "w") as f:
        json.dump({
            "epoch": history.epoch,
            "train_loss": history.train_loss,
            "valid_total_mape": history.valid_total_mape,
            "valid_gnd_mape": history.valid_gnd_mape,
            "valid_cpl_mape": history.valid_cpl_mape,
        }, f, indent=2)
    print(f"✅ smoke complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
