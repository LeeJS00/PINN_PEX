#!/usr/bin/env python3
"""
19_finetune_hybrid_mesh_smoke.py — Phase 1 Path 2 single-seed smoke.

HybridPexV3Mesh = HybridPexV3 + per-net cuboid set encoder (DeepSet).
Same NNLS-calibrated analytic prior as Week 1, plus per-net cuboid
sequence as additional input to the bounded residual heads.

Codex Round 4 escalation criterion:
    valid total ≤ 8.0% → spatial info sufficient → 5-seed paper run
    8.0 < valid total ≤ 9.0% → borderline; check per-channel
    valid total > 9.0% → escalate to mesh_v3 full spec (BEM patches)

Pre-req: `pex_v3/scripts/18_extract_per_net_cuboids.py` must have run for
all designs in TRAIN_DEFS + TEST_DEFS.
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
    PerNetCuboidStore,
    CuboidAugmentedDataset,
    collate_cuboid_batch,
)
from src.trainers.finetune_hybrid_v3 import (  # noqa: E402
    split_by_manifest_column,
    _SELF_FEATURE_COLS,
    _PAIR_FEATURE_COLS,
)
from src.baselines.calibration_v3 import (  # noqa: E402
    fit_per_layer_calibration,
    apply_per_layer_calibration,
    validate_calibration,
)
from src.utils.seeds import set_all_seeds  # noqa: E402
from src.utils.manifest_hash import write_provenance  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 Path 2 — Mesh smoke")
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument(
        "--cuboid-dir", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids"),
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "phase1_finetune_mesh_smoke",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--max-cuboids-per-net", type=int, default=512)
    p.add_argument("--cuboid-hidden", type=int, default=64)
    p.add_argument("--cuboid-embed-dim", type=int, default=64)
    p.add_argument("--cuboid-n-layers", type=int, default=2)
    p.add_argument("--residual-hidden", type=int, default=64)
    p.add_argument("--residual-n-hidden", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--early-stop-patience", type=int, default=8)
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip NNLS calibration (use raw analytic prior).")
    return p.parse_args()


@dataclass
class MeshHistory:
    train_loss: list[float] = field(default_factory=list)
    valid_total_mape: list[float] = field(default_factory=list)
    valid_gnd_mape: list[float] = field(default_factory=list)
    valid_cpl_mape: list[float] = field(default_factory=list)
    epoch_complete: list[int] = field(default_factory=list)
    best_epoch: int = -1
    best_valid_total_mape: float = float("inf")
    best_valid_gnd_mape: float = float("inf")
    best_valid_cpl_mape: float = float("inf")


def evaluate_full_split(
    model: HybridPexV3Mesh,
    loader: DataLoader,
    device: str,
    eps_fF: float = 1e-3,
) -> dict:
    """Per-net gnd/cpl/total MAPE over the full split (streaming)."""
    model.eval()
    all_gnd_rel = []
    all_cpl_rel = []
    all_total_rel = []
    with torch.no_grad():
        for batch in loader:
            ag = batch["analytic_gnd"].to(device)
            ac = batch["analytic_cpl"].to(device)
            sf = batch["self_features"].to(device)
            pf = batch["pair_features"].to(device)
            cb = batch["cuboids"].to(device)
            mk = batch["padding_mask"].to(device)
            gg = batch["golden_gnd"].to(device)
            gc = batch["golden_cpl"].to(device)

            pg = model.predict_gnd(ag, sf, cb, mk)
            pc = model.predict_cpl(ac, pf, cb, mk)

            gnd_rel = (pg - gg).abs() / gg.clamp(min=eps_fF)
            cpl_rel = (pc - gc).abs() / gc.clamp(min=eps_fF)
            total_rel = (pg + pc - gg - gc).abs() / (gg + gc).clamp(min=eps_fF)

            all_gnd_rel.append(gnd_rel.cpu())
            all_cpl_rel.append(cpl_rel.cpu())
            all_total_rel.append(total_rel.cpu())
    gnd = torch.cat(all_gnd_rel)
    cpl = torch.cat(all_cpl_rel)
    tot = torch.cat(all_total_rel)
    return {
        "gnd_mape_median": float(gnd.median().item()),
        "gnd_mape_mean": float(gnd.mean().item()),
        "cpl_mape_median": float(cpl.median().item()),
        "cpl_mape_mean": float(cpl.mean().item()),
        "total_mape_median": float(tot.median().item()),
        "total_mape_mean": float(tot.mean().item()),
        "n_nets": int(len(gnd)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(args.seed, deterministic=True)

    print(f">>> Phase 1 Path 2 (Mesh) smoke — seed {args.seed}")
    print(f">>> features:  {args.features_csv}")
    print(f">>> cuboids:   {args.cuboid_dir}")
    print(f">>> output:    {args.output_dir}")
    print(f">>> device:    {args.device}")

    # 1. Load features + split
    df = pd.read_csv(args.features_csv)
    print(f">>> loaded features: {len(df):,} rows")
    train_df, valid_df, test_df = split_by_manifest_column(df)
    train_df = train_df[(train_df["c_gnd_fF"] + train_df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    valid_df = valid_df[(valid_df["c_gnd_fF"] + valid_df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    test_df  = test_df[ (test_df["c_gnd_fF"]  + test_df["c_cpl_total_fF"])  > 1e-4].reset_index(drop=True)
    print(f">>> splits: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

    # 2. NNLS calibration on train (Phase 1 Week 1 carryover)
    cal_summary = {"calibration": "per_layer" if not args.no_calibration else "none"}
    if not args.no_calibration:
        before_v = validate_calibration(valid_df)
        calib = fit_per_layer_calibration(train_df)
        train_df = apply_per_layer_calibration(train_df, calib)
        valid_df = apply_per_layer_calibration(valid_df, calib)
        test_df  = apply_per_layer_calibration(test_df,  calib)
        after_v = validate_calibration(valid_df)
        print(f">>> NNLS calibration: gnd ratio {before_v['median_ratio_gnd']:.3f} → {after_v['median_ratio_gnd']:.3f}, "
              f"cpl ratio {before_v['median_ratio_cpl']:.3f} → {after_v['median_ratio_cpl']:.3f}")
        cal_summary["before_valid"] = before_v
        cal_summary["after_valid"] = after_v

    # 3. Load cuboid store
    print()
    print(f">>> Loading cuboid store from {args.cuboid_dir}")
    store = PerNetCuboidStore(args.cuboid_dir)
    print(f">>> cuboid store entries: {len(store):,}")

    # 4. Build datasets
    train_ds = CuboidAugmentedDataset(
        train_df, store,
        self_feature_cols=_SELF_FEATURE_COLS,
        pair_feature_cols=_PAIR_FEATURE_COLS,
        max_cuboids_per_net=args.max_cuboids_per_net,
    )
    valid_ds = CuboidAugmentedDataset(
        valid_df, store,
        self_feature_cols=_SELF_FEATURE_COLS,
        pair_feature_cols=_PAIR_FEATURE_COLS,
        max_cuboids_per_net=args.max_cuboids_per_net,
    )
    test_ds = CuboidAugmentedDataset(
        test_df, store,
        self_feature_cols=_SELF_FEATURE_COLS,
        pair_feature_cols=_PAIR_FEATURE_COLS,
        max_cuboids_per_net=args.max_cuboids_per_net,
    )
    print(f">>> datasets: train={len(train_ds):,}  valid={len(valid_ds):,}  test={len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in args.device),
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in args.device),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in args.device),
    )

    # 5. Model
    torch.manual_seed(args.seed)
    model = HybridPexV3Mesh(
        cuboid_hidden=args.cuboid_hidden,
        cuboid_embed_dim=args.cuboid_embed_dim,
        cuboid_n_layers=args.cuboid_n_layers,
        residual_hidden=args.residual_hidden,
        residual_n_hidden=args.residual_n_hidden,
    ).to(args.device)
    pc = model.parameter_count()
    print(f">>> model params: {pc}")

    # Provenance
    from configs import config_v3 as cfg
    snap = cfg.v3_snapshot()
    snap["task"] = "phase1_finetune_mesh_smoke"
    snap["calibration"] = cal_summary["calibration"]
    snap["n_epochs"] = args.n_epochs
    snap["model_params"] = pc["total"]
    write_provenance(args.output_dir, args.features_csv, snap, args.seed)

    # 6. Day-1 eval (zero-init residual; output ≈ calibrated analytic since
    #    residual MLPs zero-init → multiplier = 1.0)
    print()
    print(">>> Day-1 evaluation ...")
    day1 = evaluate_full_split(model, valid_loader, args.device)
    print(f"  day-1 valid: gnd={day1['gnd_mape_median']*100:.2f}%  "
          f"cpl={day1['cpl_mape_median']*100:.2f}%  "
          f"total={day1['total_mape_median']*100:.2f}%")

    # 7. Train
    print()
    print(f">>> training {args.n_epochs} epochs ...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    history = MeshHistory()
    epochs_without_improvement = 0
    t0 = time.time()
    for epoch in range(args.n_epochs):
        clamp = res_clamp_for_epoch(epoch)
        model.set_clamp_bounds(clamp)
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            ag = batch["analytic_gnd"].to(args.device)
            ac = batch["analytic_cpl"].to(args.device)
            sf = batch["self_features"].to(args.device)
            pf = batch["pair_features"].to(args.device)
            cb = batch["cuboids"].to(args.device)
            mk = batch["padding_mask"].to(args.device)
            gg = batch["golden_gnd"].to(args.device)
            gc = batch["golden_cpl"].to(args.device)

            pg = model.predict_gnd(ag, sf, cb, mk)
            pc_ = model.predict_cpl(ac, pf, cb, mk)
            losses = per_channel_mape_loss(pg, gg, pc_, gc)
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            running_loss += float(losses["total_loss"].item())
            n_batches += 1
        avg_loss = running_loss / max(1, n_batches)
        history.train_loss.append(avg_loss)

        # Per-epoch eval
        v = evaluate_full_split(model, valid_loader, args.device)
        history.valid_total_mape.append(v["total_mape_median"])
        history.valid_gnd_mape.append(v["gnd_mape_median"])
        history.valid_cpl_mape.append(v["cpl_mape_median"])
        history.epoch_complete.append(epoch)
        elapsed = time.time() - t0
        print(
            f"  epoch {epoch}/{args.n_epochs}: clamp={clamp:.3f}  "
            f"train_loss={avg_loss:.4f}  "
            f"valid mape: gnd={v['gnd_mape_median']*100:.2f}%  "
            f"cpl={v['cpl_mape_median']*100:.2f}%  "
            f"total={v['total_mape_median']*100:.2f}%  ({elapsed:.0f}s)",
            flush=True,
        )

        if v["total_mape_median"] < history.best_valid_total_mape:
            history.best_valid_total_mape = v["total_mape_median"]
            history.best_valid_gnd_mape = v["gnd_mape_median"]
            history.best_valid_cpl_mape = v["cpl_mape_median"]
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"  early stop at epoch {epoch}")
                break
    train_elapsed = time.time() - t0

    # 8. Final
    final_valid = evaluate_full_split(model, valid_loader, args.device)
    final_test = evaluate_full_split(model, test_loader, args.device)
    print()
    print(f">>> final valid: gnd={final_valid['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_valid['cpl_mape_median']*100:.2f}%  "
          f"total={final_valid['total_mape_median']*100:.2f}%")
    print(f">>> final test : gnd={final_test['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_test['cpl_mape_median']*100:.2f}%  "
          f"total={final_test['total_mape_median']*100:.2f}%")

    # Codex Round 4 escalation
    val_total = final_valid["total_mape_median"]
    if val_total <= 0.08:
        verdict = "✅ ≤8.0%: spatial info SUFFICIENT — proceed to 5-seed Path 2"
    elif val_total <= 0.09:
        verdict = "⚠ 8-9%: BORDERLINE — review per-channel; consider Path 3"
    else:
        verdict = "❌ >9%: ESCALATE to mesh_v3 full spec (Path 3, 6 days)"
    print()
    print("=" * 60)
    print(f">>> CODEX ESCALATION (final valid total = {val_total*100:.3f}%)")
    print(f"    {verdict}")
    print("=" * 60)

    # Save
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "n_epochs": args.n_epochs,
            "model_params": pc,
            "elapsed_train_sec": train_elapsed,
            **cal_summary,
            "day1_valid": day1,
            "final_valid": final_valid,
            "final_test": final_test,
            "best_epoch": history.best_epoch,
            "best_valid_total_mape": history.best_valid_total_mape,
            "best_valid_gnd_mape": history.best_valid_gnd_mape,
            "best_valid_cpl_mape": history.best_valid_cpl_mape,
            "verdict": verdict,
        }, f, indent=2, default=str)
    with open(args.output_dir / "history.json", "w") as f:
        json.dump({
            "train_loss": history.train_loss,
            "valid_total_mape": history.valid_total_mape,
            "valid_gnd_mape": history.valid_gnd_mape,
            "valid_cpl_mape": history.valid_cpl_mape,
            "epoch_complete": history.epoch_complete,
        }, f, indent=2)
    print(f"✅ smoke complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
