#!/usr/bin/env python3
"""
38_finetune_mesh_input_subset_clamp_norm_smoke.py — Combined-stack smoke.

Variant: HybridPexV3MeshInputSubsetClampNorm — drop-in replacement for
HybridPexV3Mesh that stacks two orthogonal levers:
  - InputSubset (input-side, shared encoder, gnd zeros channels {6, 7, 9})
  - ClampNorm   (output-side, joint per-net (gnd, cpl) L2 norm-projection
                clamp on the residual logits, replacing element-wise clamp)

Hypothesis (full doc:
pex_v3/experiments/auto_optimize_2026_05_03/variants/input_subset_clamp_norm/DESIGN.md):
    Each lever attacks a different failure mode (input information vs
    output gradient flow). Composition predicted to combine InputSubset's
    gnd improvement with ClampNorm's last-step stability.

Decision gate (PASS if AT LEAST ONE):
  - test gnd ≤ 18.5%   (better than InputSubset alone 19.05%)
  - test cpl ≤ 14.7%   (better than ClampNorm alone 15.22%)
  - test total ≤ 6.8%  (better than both singles)
  - last_valid total ≤ 6.5%  (better than ClampNorm alone 6.66%)
AND no metric regresses by > 0.5 pp vs the better of the two singles.

Output:
  pex_v3/experiments/auto_optimize_2026_05_03/outputs/input_subset_clamp_norm/seed42/
      summary.json      — day-1, final_valid, final_test, best_epoch, transition log
      history.json      — train_loss + per-epoch valid metrics
      provenance.json   — manifest hash, git SHA, config snapshot
      model.pt          — state_dict
      eval_logger_valid.parquet
      eval_logger_test.parquet
      run.log           — full stdout
"""
from __future__ import annotations
import argparse
import json
import math
import os
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

from src.models.hybrid_v3_mesh_input_subset_clamp_norm import (  # noqa: E402
    HybridPexV3MeshInputSubsetClampNorm,
)
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
from src.utils.seeds import set_all_seeds, worker_init_fn  # noqa: E402
from src.utils.manifest_hash import write_provenance  # noqa: E402
from src.utils.eval_logger import collect_per_net_predictions, write_eval_parquet  # noqa: E402


CURRICULUM_TRANSITION_EPOCHS = (49, 50, 51, 52, 149, 150, 151, 152)


def parse_args():
    p = argparse.ArgumentParser(description="InputSubset+ClampNorm combined smoke (single seed)")
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
        default=_PROJECT_ROOT / "pex_v3" / "experiments"
        / "auto_optimize_2026_05_03" / "outputs" / "input_subset_clamp_norm" / "seed42",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-epochs", type=int, default=200)
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
    p.add_argument("--gpu", type=int, default=7,
                   help="CUDA device index (default 7; only free GPU during InputSubset 5-seed lock)")
    p.add_argument("--early-stop-patience", type=int, default=99999,
                   help="Disabled by default to match locked 5-seed run.")
    p.add_argument("--no-calibration", action="store_true",
                   help="Skip NNLS calibration (use raw analytic prior).")
    return p.parse_args()


@dataclass
class CCHistory:
    """Combined-stack history (gnd/cpl/total mape per epoch + transition log)."""
    train_loss: list[float] = field(default_factory=list)
    valid_total_mape: list[float] = field(default_factory=list)
    valid_gnd_mape: list[float] = field(default_factory=list)
    valid_cpl_mape: list[float] = field(default_factory=list)
    epoch_complete: list[int] = field(default_factory=list)
    best_epoch: int = -1
    best_valid_total_mape: float = float("inf")
    best_valid_gnd_mape: float = float("inf")
    best_valid_cpl_mape: float = float("inf")
    transition_log: dict[int, dict] = field(default_factory=dict)
    phase2_max_abs_delta: float = 0.0
    phase2_mean_abs_delta: float = 0.0


# ---------------------------------------------------------------------------
# Joint-clamp evaluator (ClampNorm-aware; uses _predict_joint).
# ---------------------------------------------------------------------------

def evaluate_joint(model, loader, device, eps_fF=1e-3):
    """Per-net gnd/cpl/total MAPE using `_predict_joint` for correct clamp."""
    model.eval()
    gnd_l, cpl_l, tot_l = [], [], []
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
            pg, pc = model._predict_joint(ag, ac, sf, pf, cb, mk)
            gnd_l.append(((pg - gg).abs() / gg.clamp(min=eps_fF)).cpu())
            cpl_l.append(((pc - gc).abs() / gc.clamp(min=eps_fF)).cpu())
            tot_l.append(((pg + pc - gg - gc).abs() / (gg + gc).clamp(min=eps_fF)).cpu())
    gnd = torch.cat(gnd_l); cpl = torch.cat(cpl_l); tot = torch.cat(tot_l)
    return {
        "gnd_mape_median": float(gnd.median().item()),
        "gnd_mape_mean":   float(gnd.mean().item()),
        "cpl_mape_median": float(cpl.median().item()),
        "cpl_mape_mean":   float(cpl.mean().item()),
        "total_mape_median": float(tot.median().item()),
        "total_mape_mean":   float(tot.mean().item()),
        "n_nets": int(len(gnd)),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout to run.log
    log_path = args.output_dir / "run.log"
    log_f = open(log_path, "w")
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams:
                st.write(s); st.flush()
        def flush(self):
            for st in self.streams: st.flush()
    sys.stdout = _Tee(sys.__stdout__, log_f)
    sys.stderr = _Tee(sys.__stderr__, log_f)

    # Determinism
    set_all_seeds(args.seed, deterministic=True)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    print(f">>> InputSubset+ClampNorm combined smoke — seed {args.seed}")
    print(f">>> features:  {args.features_csv}")
    print(f">>> cuboids:   {args.cuboid_dir}")
    print(f">>> output:    {args.output_dir}")
    print(f">>> device:    {device}")
    print(f">>> n_epochs:  {args.n_epochs}")

    # 1. Load features + split
    df = pd.read_csv(args.features_csv)
    print(f">>> loaded features: {len(df):,} rows")
    train_df, valid_df, test_df = split_by_manifest_column(df)
    train_df = train_df[(train_df["c_gnd_fF"] + train_df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    valid_df = valid_df[(valid_df["c_gnd_fF"] + valid_df["c_cpl_total_fF"]) > 1e-4].reset_index(drop=True)
    test_df  = test_df[ (test_df["c_gnd_fF"]  + test_df["c_cpl_total_fF"])  > 1e-4].reset_index(drop=True)
    print(f">>> splits: train={len(train_df):,}  valid={len(valid_df):,}  test={len(test_df):,}")

    # 2. NNLS calibration (matches baseline 5-seed)
    cal_summary = {"calibration": "per_layer" if not args.no_calibration else "none"}
    if not args.no_calibration:
        before_v = validate_calibration(valid_df)
        calib = fit_per_layer_calibration(train_df)
        train_df = apply_per_layer_calibration(train_df, calib)
        valid_df = apply_per_layer_calibration(valid_df, calib)
        test_df  = apply_per_layer_calibration(test_df,  calib)
        after_v = validate_calibration(valid_df)
        print(f">>> NNLS: gnd ratio {before_v['median_ratio_gnd']:.3f} → {after_v['median_ratio_gnd']:.3f}, "
              f"cpl ratio {before_v['median_ratio_cpl']:.3f} → {after_v['median_ratio_cpl']:.3f}")
        cal_summary["before_valid"] = before_v
        cal_summary["after_valid"] = after_v

    # 3. Cuboid store + datasets
    print(f">>> Loading cuboid store from {args.cuboid_dir}")
    store = PerNetCuboidStore(args.cuboid_dir)
    print(f">>> cuboid store entries: {len(store):,}")

    train_ds = CuboidAugmentedDataset(
        train_df, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
        max_cuboids_per_net=args.max_cuboids_per_net,
    )
    valid_ds = CuboidAugmentedDataset(
        valid_df, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
        max_cuboids_per_net=args.max_cuboids_per_net,
    )
    test_ds = CuboidAugmentedDataset(
        test_df,  store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
        max_cuboids_per_net=args.max_cuboids_per_net,
    )
    print(f">>> datasets: train={len(train_ds):,}  valid={len(valid_ds):,}  test={len(test_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in device), worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in device),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_cuboid_batch,
        pin_memory=("cuda" in device),
    )

    # 4. Model
    torch.manual_seed(args.seed)
    model = HybridPexV3MeshInputSubsetClampNorm(
        cuboid_hidden=args.cuboid_hidden,
        cuboid_embed_dim=args.cuboid_embed_dim,
        cuboid_n_layers=args.cuboid_n_layers,
        residual_hidden=args.residual_hidden,
        residual_n_hidden=args.residual_n_hidden,
    ).to(device)
    pc = model.parameter_count()
    print(f">>> model params: {pc}")
    assert pc["total"] == 44_738, f"param count mismatch: {pc['total']} != 44_738"

    # Provenance
    from configs import config_v3 as cfg
    snap = cfg.v3_snapshot()
    snap["task"] = "input_subset_clamp_norm_smoke"
    snap["calibration"] = cal_summary["calibration"]
    snap["n_epochs"] = args.n_epochs
    snap["model_params"] = pc["total"]
    snap["model_class"] = (
        "src.models.hybrid_v3_mesh_input_subset_clamp_norm."
        "HybridPexV3MeshInputSubsetClampNorm"
    )
    snap["seed"] = args.seed
    snap["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
    snap["gnd_interaction_channels"] = list(model.gnd_interaction_channels)
    write_provenance(args.output_dir, args.features_csv, snap, args.seed)

    # 5. Day-1 eval (with joint clamp evaluator)
    print()
    print(">>> Day-1 evaluation ...")
    day1 = evaluate_joint(model, valid_loader, device)
    print(f"  day-1 valid: gnd={day1['gnd_mape_median']*100:.2f}%  "
          f"cpl={day1['cpl_mape_median']*100:.2f}%  "
          f"total={day1['total_mape_median']*100:.2f}%")

    # 6. Train (joint forward + curriculum)
    print()
    print(f">>> training {args.n_epochs} epochs ...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    history = CCHistory()
    epochs_without_improvement = 0
    t0 = time.time()
    prev_valid_total = None
    phase2_diffs = []

    for epoch in range(args.n_epochs):
        clamp = res_clamp_for_epoch(epoch)
        model.set_clamp_bounds(clamp)
        model.train()
        running_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            ag = batch["analytic_gnd"].to(device)
            ac = batch["analytic_cpl"].to(device)
            sf = batch["self_features"].to(device)
            pf = batch["pair_features"].to(device)
            cb = batch["cuboids"].to(device)
            mk = batch["padding_mask"].to(device)
            gg = batch["golden_gnd"].to(device)
            gc = batch["golden_cpl"].to(device)

            # JOINT forward (the InputSubset + ClampNorm path)
            pg, pc_ = model._predict_joint(ag, ac, sf, pf, cb, mk)
            losses = per_channel_mape_loss(pg, gg, pc_, gc)
            optimizer.zero_grad()
            losses["total_loss"].backward()
            optimizer.step()
            running_loss += float(losses["total_loss"].item())
            n_batches += 1
        avg_loss = running_loss / max(1, n_batches)
        history.train_loss.append(avg_loss)

        v = evaluate_joint(model, valid_loader, device)
        history.valid_total_mape.append(v["total_mape_median"])
        history.valid_gnd_mape.append(v["gnd_mape_median"])
        history.valid_cpl_mape.append(v["cpl_mape_median"])
        history.epoch_complete.append(epoch)
        elapsed = time.time() - t0

        if epoch in CURRICULUM_TRANSITION_EPOCHS:
            history.transition_log[epoch] = {
                "valid_total_mape": v["total_mape_median"],
                "valid_gnd_mape":   v["gnd_mape_median"],
                "valid_cpl_mape":   v["cpl_mape_median"],
                "clamp": clamp,
            }

        if epoch >= 150 and prev_valid_total is not None:
            phase2_diffs.append(abs(v["total_mape_median"] - prev_valid_total))
        prev_valid_total = v["total_mape_median"]

        print(
            f"  epoch {epoch:3d}/{args.n_epochs}: clamp={clamp:.3f}  "
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

    if phase2_diffs:
        history.phase2_max_abs_delta = float(max(phase2_diffs))
        history.phase2_mean_abs_delta = float(sum(phase2_diffs) / len(phase2_diffs))
    else:
        history.phase2_max_abs_delta = 0.0
        history.phase2_mean_abs_delta = 0.0

    # 7. Final eval (joint clamp evaluator on both splits)
    final_valid = evaluate_joint(model, valid_loader, device)
    final_test  = evaluate_joint(model, test_loader,  device)
    print()
    print(f">>> final valid: gnd={final_valid['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_valid['cpl_mape_median']*100:.2f}%  "
          f"total={final_valid['total_mape_median']*100:.2f}%")
    print(f">>> final test : gnd={final_test['gnd_mape_median']*100:.2f}%  "
          f"cpl={final_test['cpl_mape_median']*100:.2f}%  "
          f"total={final_test['total_mape_median']*100:.2f}%")

    # 8. Decision gate (combined-stack thresholds; better-of-singles regression bound)
    SINGLE_BETTER_TEST = {"gnd": 0.1905, "cpl": 0.1502, "total": 0.0722}
    SINGLE_LASTV       = {"total": 0.0666}
    THRESH             = {"gnd": 0.185,  "cpl": 0.147,  "total": 0.068, "last_valid_total": 0.065}
    REG_TOL            = 0.005

    gnd_t = final_test["gnd_mape_median"]
    cpl_t = final_test["cpl_mape_median"]
    tot_t = final_test["total_mape_median"]
    last_v = final_valid["total_mape_median"]

    pass_any = (
        (gnd_t  <= THRESH["gnd"])   or
        (cpl_t  <= THRESH["cpl"])   or
        (tot_t  <= THRESH["total"]) or
        (last_v <= THRESH["last_valid_total"])
    )
    no_regression = (
        (gnd_t - SINGLE_BETTER_TEST["gnd"])   <= REG_TOL and
        (cpl_t - SINGLE_BETTER_TEST["cpl"])   <= REG_TOL and
        (tot_t - SINGLE_BETTER_TEST["total"]) <= REG_TOL
    )

    # Curriculum-transition gain check (preservation of ClampNorm's stability)
    tlog = history.transition_log
    transition_gain_50 = (tlog.get(49, {}).get("valid_total_mape", float("inf"))
                          - tlog.get(52, {}).get("valid_total_mape", float("inf")))
    transition_gain_150 = (tlog.get(149, {}).get("valid_total_mape", float("inf"))
                           - tlog.get(152, {}).get("valid_total_mape", float("inf")))
    transitions_preserved = (transition_gain_50 > 0) and (transition_gain_150 > 0)

    verdict = "PASS" if (pass_any and no_regression) else "FAIL"
    print()
    print("=" * 70)
    print(f">>> InputSubset+ClampNorm combined smoke verdict: {verdict}")
    print(f"    - test gnd:        {gnd_t*100:6.3f}%  (thresh ≤ {THRESH['gnd']*100:.2f}%)  "
          f"single-best {SINGLE_BETTER_TEST['gnd']*100:.2f}%")
    print(f"    - test cpl:        {cpl_t*100:6.3f}%  (thresh ≤ {THRESH['cpl']*100:.2f}%)  "
          f"single-best {SINGLE_BETTER_TEST['cpl']*100:.2f}%")
    print(f"    - test total:      {tot_t*100:6.3f}%  (thresh ≤ {THRESH['total']*100:.2f}%)  "
          f"single-best {SINGLE_BETTER_TEST['total']*100:.2f}%")
    print(f"    - last_valid total:{last_v*100:6.3f}%  (thresh ≤ {THRESH['last_valid_total']*100:.2f}%)  "
          f"CN-alone {SINGLE_LASTV['total']*100:.2f}%")
    print(f"    - pass_any={pass_any}  no_regression={no_regression}  "
          f"transitions_preserved={transitions_preserved}")
    print(f"    - transition gain Phase 0→1 (epoch 49→52): {transition_gain_50*100:+.3f} pp")
    print(f"    - transition gain Phase 1→2 (epoch 149→152): {transition_gain_150*100:+.3f} pp")
    print(f"    - Phase 2 max |Δvalid|:  {history.phase2_max_abs_delta*100:.3f} pp")
    print(f"    - Phase 2 mean |Δvalid|: {history.phase2_mean_abs_delta*100:.3f} pp")
    print("=" * 70)

    # 9. Save artifacts
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump({
            "seed": args.seed,
            "variant": "HybridPexV3MeshInputSubsetClampNorm",
            "n_epochs": args.n_epochs,
            "model_params": pc,
            "elapsed_train_sec": train_elapsed,
            "gnd_interaction_channels": list(model.gnd_interaction_channels),
            **cal_summary,
            "day1_valid": day1,
            "final_valid": final_valid,
            "final_test": final_test,
            "best_epoch": history.best_epoch,
            "best_valid_total_mape": history.best_valid_total_mape,
            "best_valid_gnd_mape": history.best_valid_gnd_mape,
            "best_valid_cpl_mape": history.best_valid_cpl_mape,
            "transition_log": history.transition_log,
            "transition_gain_50_pp":  transition_gain_50,
            "transition_gain_150_pp": transition_gain_150,
            "phase2_max_abs_delta":  history.phase2_max_abs_delta,
            "phase2_mean_abs_delta": history.phase2_mean_abs_delta,
            "verdict": verdict,
            "decision_gate": {
                "pass_any":  pass_any,
                "no_regression": no_regression,
                "transitions_preserved": transitions_preserved,
                "thresholds": THRESH,
                "single_best": SINGLE_BETTER_TEST,
                "single_clamp_norm_last_valid_total": SINGLE_LASTV["total"],
            },
        }, f, indent=2, default=str)
    with open(args.output_dir / "history.json", "w") as f:
        json.dump({
            "train_loss": history.train_loss,
            "valid_total_mape": history.valid_total_mape,
            "valid_gnd_mape": history.valid_gnd_mape,
            "valid_cpl_mape": history.valid_cpl_mape,
            "epoch_complete": history.epoch_complete,
        }, f, indent=2)

    # eval_logger uses standalone-API shims (logit_other = 0 fallback). Source
    # of truth for the headline summary.json metrics is `evaluate_joint` above.
    print(">>> writing eval_logger parquet ...")
    valid_pred_df = collect_per_net_predictions(model, valid_loader, device, valid_df)
    write_eval_parquet(valid_pred_df, args.output_dir / "eval_logger_valid.parquet")
    test_pred_df = collect_per_net_predictions(model, test_loader, device, test_df)
    write_eval_parquet(test_pred_df, args.output_dir / "eval_logger_test.parquet")

    print(f"smoke complete. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
