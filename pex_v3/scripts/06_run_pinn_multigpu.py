#!/usr/bin/env python3
"""
06_run_pinn_multigpu.py — Phase B B3 PINN 5-seed parallel across GPUs.

User decision (2026-05-02): drop single-GPU assumption; run 5 seeds in
parallel on 5 GPUs. Each subprocess sets CUDA_VISIBLE_DEVICES=<gpu>, so
torch sees only that one GPU as cuda:0.

Per-seed work is independent (different model init, different DataLoader
shuffle), so parallel execution is safe. Cache files (predefined train/valid)
are read-only after first creation; first seed to run regenerates the
cache, subsequent seeds reuse — but with 5 parallel starts there's a small
race where each may regenerate. Acceptable: cache content is deterministic
(`random_state=42`).

Cost: 5 × ~4.5h = 4.5h wall-clock (vs 22.5h sequential). Final aggregation
runs after all subprocesses join.

Usage:
    python3 pex_v3/scripts/06_run_pinn_multigpu.py
    python3 pex_v3/scripts/06_run_pinn_multigpu.py --gpus 0 2 3 4 7
    python3 pex_v3/scripts/06_run_pinn_multigpu.py --seeds 0 1 2 --gpus 0 1 2
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
    p = argparse.ArgumentParser(
        description="B3 PINN baseline 5-seed multi-GPU runner"
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument(
        "--gpus", nargs="+", type=int, default=[0, 2, 3, 4, 7],
        help="GPU indices for each seed (length must match --seeds)",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "pex_v3" / "output" / "baselines" / "B3_pinn_real",
    )
    p.add_argument(
        "--method-spec", type=str,
        default="pex_v3/src/baselines/pinn_baseline.py:run_one_seed",
    )
    p.add_argument(
        "--metric-col", type=str, default="cap_mape_median",
    )
    p.add_argument(
        "--max-iters", type=int, default=1,
        help="Forwarded to pinn_baseline.run_one_seed via env (PEX_PINN_MAX_ITERS)",
    )
    p.add_argument(
        "--steps-per-iter", type=int, default=5000,
        help="Forwarded via env (PEX_PINN_STEPS_PER_ITER)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without launching subprocesses",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.seeds) != len(args.gpus):
        raise SystemExit(
            f"--seeds (n={len(args.seeds)}) and --gpus (n={len(args.gpus)}) "
            f"must have the same length"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> B3 PINN 5-seed multi-GPU launcher")
    print(f">>> Output:    {args.output_dir}")
    print(f">>> Method:    {args.method_spec}")
    print(f">>> seeds × GPUs:")
    for seed, gpu in zip(args.seeds, args.gpus):
        print(f"      seed {seed}  →  GPU {gpu}")
    print(f">>> max_iters={args.max_iters}  steps_per_iter={args.steps_per_iter}")

    if args.dry_run:
        print("⚠️  Dry-run only. Pass without --dry-run to launch.")
        return

    # ---- Launch ---------------------------------------------------------
    runner = _PROJECT_ROOT / "pex_v3" / "scripts" / "05_5seed_runner.py"
    pybin = "/tool/etc/python/install/3.11.9/bin/python3"

    procs = []
    for seed, gpu in zip(args.seeds, args.gpus):
        log_path = args.output_dir / f"multigpu_seed{seed}_gpu{gpu}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Forward max_iters and steps_per_iter to pinn_baseline via env
        env["PEX_PINN_MAX_ITERS"] = str(args.max_iters)
        env["PEX_PINN_STEPS_PER_ITER"] = str(args.steps_per_iter)
        env["TORCH_COMPILE_DISABLE"] = "1"  # Phase C A7 #12 fix: determinism

        cmd = [
            pybin,
            str(runner),
            "--method-spec", args.method_spec,
            "--output-dir", str(args.output_dir),
            "--seeds", str(seed),
            "--metric-col", args.metric_col,
            "--skip-existing",
        ]

        # Open log file for this subprocess
        log_f = open(log_path, "w")
        log_f.write(f">>> seed={seed} GPU={gpu} cmd={' '.join(cmd)}\n>>> started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_f.flush()

        p = subprocess.Popen(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(_PROJECT_ROOT),
            env=env,
        )
        procs.append((seed, gpu, p, log_f, log_path))
        print(f"  launched seed {seed} on GPU {gpu} (pid {p.pid}, log {log_path.name})")

    # ---- Wait -----------------------------------------------------------
    print(f">>> Launched {len(procs)} subprocesses. Waiting for all to complete ...")
    print(f"    (poll: tail -f {args.output_dir}/multigpu_seed*_gpu*.log)")
    t0 = time.time()
    failures = []
    for seed, gpu, p, log_f, log_path in procs:
        rc = p.wait()
        log_f.close()
        elapsed = time.time() - t0
        status = "✅" if rc == 0 else "❌"
        print(f"  {status} seed {seed} (GPU {gpu}): rc={rc}  elapsed={elapsed/60:.1f}min")
        if rc != 0:
            failures.append((seed, gpu, rc))

    # ---- Aggregate ------------------------------------------------------
    if failures:
        print(f"⚠️  {len(failures)} seed(s) failed:")
        for seed, gpu, rc in failures:
            print(f"      seed {seed} (GPU {gpu}): rc={rc}")
        print(f"    Inspect logs and re-run failed seeds with --skip-existing.")
        sys.exit(1)

    # All seeds done — re-run aggregation across ALL seed dirs
    print(">>> Aggregating across all 5 seeds ...")
    sys.path.insert(0, str(_PROJECT_ROOT / "pex_v3"))
    from src.evaluation.seed_aggregator import collect_per_run_csvs, write_aggregation
    df = collect_per_run_csvs(args.output_dir)
    paths = write_aggregation(df, args.output_dir, metric_col=args.metric_col)
    print(f">>> Wrote:")
    for k, p in paths.items():
        print(f"    {k}: {p}")
    print("✅ 06_run_pinn_multigpu.py complete.")


if __name__ == "__main__":
    main()
