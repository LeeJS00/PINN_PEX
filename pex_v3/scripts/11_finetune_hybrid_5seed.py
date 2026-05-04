#!/usr/bin/env python3
"""
11_finetune_hybrid_5seed.py — Phase 1 5-seed multi-GPU fine-tune.

Runs HybridPexV3 with calibrated prior across 5 seeds in parallel on 5 GPUs.
Each seed produces metrics_row.csv compatible with seed_aggregator → MWU
+ Cohen's d vs B1 XGBoost.

Per benchmarking-statistician + A1 audit: ALL claims must come from this
5-seed protocol. No single-seed numbers in the paper.

Usage:
    python3 pex_v3/scripts/11_finetune_hybrid_5seed.py
    python3 pex_v3/scripts/11_finetune_hybrid_5seed.py \\
        --gpus 0 2 3 4 7 \\
        --n-epochs 80 \\
        --use-calibration per_layer
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 hybrid 5-seed multi-GPU")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--gpus", nargs="+", type=int, default=[0, 2, 3, 4, 7],
        help="GPU index per seed (length must match --seeds)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "phase1_hybrid_5seed",
    )
    p.add_argument(
        "--features-csv", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"),
    )
    p.add_argument("--n-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--use-calibration", choices=["scalar", "per_layer", "none"], default="per_layer",
        help="Apply calibration to compact_gnd/cpl prior",
    )
    p.add_argument(
        "--clamp-bound", type=float, default=None,
        help="If None, log(2.5) when calibrated, log(20) when uncalibrated",
    )
    p.add_argument(
        "--curriculum", action="store_true",
        help="Enable RES_CLAMP curriculum log(1.5)→log(2.5)→log(4.0)",
    )
    p.add_argument(
        "--early-stop-patience", type=int, default=20,
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def write_runner_script() -> Path:
    """Write a per-seed runner that the multi-GPU launcher subprocess'es."""
    script = _PROJECT_ROOT / "pex_v3" / "scripts" / "_finetune_hybrid_one_seed.py"
    script.write_text(
        """#!/usr/bin/env python3
'''Internal: run hybrid_v3 fine-tune for ONE seed; write metrics_row.csv.'''
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / 'pex_v3'))
import pandas as pd, torch, dataclasses
from src.models.hybrid_v3 import HybridPexV3
from src.trainers.finetune_hybrid_v3 import (
    df_to_tensors, split_by_manifest_column,
    finetune_hybrid, evaluate_per_channel, evaluate_beta_gate, FinetuneConfig,
)
from src.baselines.calibration_v3 import (
    fit_per_layer_calibration, apply_per_layer_calibration,
    fit_scalar_calibration, apply_scalar_calibration,
)
from src.evaluation.metrics import MetricsRow
from src.utils.seeds import set_all_seeds
from src.utils.manifest_hash import write_provenance
from configs import config_v3 as cfg

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, required=True)
p.add_argument('--features-csv', type=Path, required=True)
p.add_argument('--output-dir', type=Path, required=True)
p.add_argument('--n-epochs', type=int, default=80)
p.add_argument('--batch-size', type=int, default=256)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--clamp-bound', type=float, required=True)
p.add_argument('--use-calibration', choices=['scalar','per_layer','none'], default='per_layer')
p.add_argument('--curriculum', action='store_true')
p.add_argument('--early-stop-patience', type=int, default=20)
args = p.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
set_all_seeds(args.seed, deterministic=True)

# Provenance
snap = cfg.v3_snapshot()
snap['task'] = 'phase1_hybrid_5seed'
snap['n_epochs'] = args.n_epochs
snap['use_calibration'] = args.use_calibration
snap['clamp_bound'] = args.clamp_bound
snap['curriculum'] = args.curriculum
write_provenance(args.output_dir, args.features_csv, snap, args.seed)

# Load + split
df = pd.read_csv(args.features_csv)
train_df, valid_df, test_df = split_by_manifest_column(df)
train_df = train_df[(train_df['c_gnd_fF'] + train_df['c_cpl_total_fF']) > 1e-4].reset_index(drop=True)
valid_df = valid_df[(valid_df['c_gnd_fF'] + valid_df['c_cpl_total_fF']) > 1e-4].reset_index(drop=True)
test_df  = test_df[ (test_df['c_gnd_fF']  + test_df['c_cpl_total_fF'])  > 1e-4].reset_index(drop=True)

# Calibration on train only
if args.use_calibration == 'per_layer':
    calib = fit_per_layer_calibration(train_df, min_nets_per_layer=200)
    train_df = apply_per_layer_calibration(train_df, calib)
    valid_df = apply_per_layer_calibration(valid_df, calib)
    test_df  = apply_per_layer_calibration(test_df, calib)
    print(f'>>> calibration: per_layer, s_gnd_default={calib.s_gnd_default:.3f}, s_cpl_default={calib.s_cpl_default:.3f}')
elif args.use_calibration == 'scalar':
    calib = fit_scalar_calibration(train_df)
    train_df = apply_scalar_calibration(train_df, calib)
    valid_df = apply_scalar_calibration(valid_df, calib)
    test_df  = apply_scalar_calibration(test_df, calib)
    print(f'>>> calibration: scalar, s_gnd={calib.s_gnd:.3f}, s_cpl={calib.s_cpl:.3f}')
else:
    print('>>> calibration: none')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(args.seed)
model = HybridPexV3(clamp_bound=args.clamp_bound)
config = FinetuneConfig(
    n_epochs=args.n_epochs,
    batch_size=args.batch_size,
    lr=args.lr,
    seed=args.seed,
    log_every_n_steps=500,
    eval_every_n_epochs=2,
    curriculum_enabled=args.curriculum,
    early_stop_patience=args.early_stop_patience,
)

t0 = time.time()
history = finetune_hybrid(model, train_df, valid_df, config, device)
elapsed = time.time() - t0
print(f'>>> trained in {elapsed:.1f}s, best epoch {history.best_epoch}, best total {history.best_valid_total_mape*100:.2f}%')

# Final eval
final_v = evaluate_per_channel(model, df_to_tensors(valid_df), device)
final_t = evaluate_per_channel(model, df_to_tensors(test_df), device)
beta = evaluate_beta_gate(model, valid_df, config, device)

# Build MetricsRow (on validation)
row = MetricsRow(
    method='B5_hybrid_v3',
    seed=args.seed,
    cap_mape_median=final_v['total_mape_median'],
    cap_mape_mean=final_v['total_mape_mean'],
    cap_mape_p95=float('nan'),
    delay_err_median=float('nan'),
    delay_err_p95=float('nan'),
    power_err_median=float('nan'),
    rc_chip_ratio_p50=float('nan'),
    rc_chip_ratio_p95=float('nan'),
    n_valid_nets=len(valid_df),
)
pd.DataFrame([dataclasses.asdict(row)]).to_csv(args.output_dir / 'metrics_row.csv', index=False)

# Save extras
torch.save(model.state_dict(), args.output_dir / 'model.pt')
with open(args.output_dir / 'summary.json', 'w') as f:
    json.dump({
        'seed': args.seed,
        'elapsed_sec': elapsed,
        'best_epoch': history.best_epoch,
        'best_total': history.best_valid_total_mape,
        'best_gnd': history.best_valid_gnd_mape,
        'best_cpl': history.best_valid_cpl_mape,
        'final_valid': final_v,
        'final_test':  final_t,
        'beta_gate': {k: v for k, v in beta.items() if not isinstance(v, dict)},
    }, f, indent=2, default=str)

print(f'>>> seed {args.seed} done. valid total={final_v["total_mape_median"]*100:.2f}%, gnd={final_v["gnd_mape_median"]*100:.2f}%, cpl={final_v["cpl_mape_median"]*100:.2f}%')
"""
    )
    script.chmod(0o755)
    return script


def main():
    args = parse_args()
    if len(args.seeds) != len(args.gpus):
        raise SystemExit(f"--seeds (n={len(args.seeds)}) and --gpus (n={len(args.gpus)}) lengths must match")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import math
    if args.clamp_bound is None:
        args.clamp_bound = math.log(2.5) if args.use_calibration != "none" else math.log(20.0)

    print(f">>> Phase 1 hybrid_v3 5-seed multi-GPU")
    print(f">>> Output:    {args.output_dir}")
    print(f">>> Features:  {args.features_csv}")
    print(f">>> Calib:     {args.use_calibration}")
    print(f">>> Clamp:     log(e^{args.clamp_bound:.3f}) ≈ exp(±{args.clamp_bound:.3f})")
    print(f">>> Epochs:    {args.n_epochs}")
    print(f">>> seeds × GPUs:")
    for s, g in zip(args.seeds, args.gpus):
        print(f"      seed {s}  →  GPU {g}")

    if args.dry_run:
        print("⚠️  Dry-run.")
        return

    runner = write_runner_script()
    pybin = "/tool/etc/python/install/3.11.9/bin/python3"

    # Launch all 5 in parallel
    procs = []
    for seed, gpu in zip(args.seeds, args.gpus):
        seed_dir = args.output_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / f"multigpu_seed{seed}_gpu{gpu}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["TORCH_COMPILE_DISABLE"] = "1"

        cmd = [
            pybin, str(runner),
            "--seed", str(seed),
            "--features-csv", str(args.features_csv),
            "--output-dir", str(seed_dir),
            "--n-epochs", str(args.n_epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--clamp-bound", str(args.clamp_bound),
            "--use-calibration", args.use_calibration,
            "--early-stop-patience", str(args.early_stop_patience),
        ]
        if args.curriculum:
            cmd.append("--curriculum")

        log_f = open(log_path, "w")
        log_f.write(f">>> seed={seed} GPU={gpu}\n>>> started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.flush()

        p = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=str(_PROJECT_ROOT), env=env,
        )
        procs.append((seed, gpu, p, log_f, log_path))
        print(f"  launched seed {seed} on GPU {gpu} (pid {p.pid})")

    print(f">>> Launched {len(procs)} processes. Waiting ...")
    t0 = time.time()
    failed = []
    for seed, gpu, p, log_f, log_path in procs:
        rc = p.wait()
        log_f.close()
        elapsed = time.time() - t0
        status = "✅" if rc == 0 else "❌"
        print(f"  {status} seed {seed} (GPU {gpu}): rc={rc}  elapsed={elapsed/60:.1f}min")
        if rc != 0:
            failed.append((seed, gpu, rc))

    if failed:
        print(f"⚠️  {len(failed)} seed(s) failed.")
        sys.exit(1)

    # Aggregate
    print(">>> Aggregating ...")
    sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))
    from src.evaluation.seed_aggregator import collect_per_run_csvs, write_aggregation
    df = collect_per_run_csvs(args.output_dir)
    paths = write_aggregation(df, args.output_dir, metric_col="cap_mape_median")
    print(f">>> Wrote: {list(paths.keys())}")
    print("✅ phase1 hybrid_v3 5-seed complete.")


if __name__ == "__main__":
    main()
