"""5-seed measurement protocol for the v2 baseline (Step 1C of dspinn v5 plan).

Spawns 5 independent v2-recipe AL runs (one per seed, one per GPU), each
capped at a single AL iteration with a fixed step budget. Aggregates the
best net-level MAPE from each run's stdout log into mean / std / p10 / p90
so future single-seed comparisons can be judged against an actual noise
floor instead of the ±5% guess we used through v1-v4.

Recipe per seed:
  * --calib_path none   → disables the v4 NNLS ζ init (back to v3 hardcoded)
  * no --use_dspinn     → MacroDensityFNO / aux head off
  * --max_iters 1       → one AL acquisition + one training pass
  * --steps_per_iter N  → caps the training pass (default 5000, matches v2)

This is the closest "v2 recipe" we can reach without reverting the
flux_head ζ-hardcoded init introduced in v3 (architectural revert is
deferred to Step 2). Document the caveat when reading results.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_SCRIPT = REPO_ROOT / "run_active_learning.py"
CACHE_DIR = REPO_ROOT / "output_intel22" / "active_learning" / "cache"
VAL_CACHE = CACHE_DIR / "predefined_valid_subset.csv"
TRAIN_CACHE = CACHE_DIR / "predefined_train_subset.csv"
OUTPUT_ROOT = REPO_ROOT / "output_intel22" / "active_learning"
REPORT_DIR = OUTPUT_ROOT / "diag_phase_a"

NET_MAPE_RE = re.compile(r"Net-level MAPE\s*:\s*([0-9]+\.[0-9]+)%")


def build_cmd(model_name: str, gpu: int, seed: int, steps_per_iter: int,
              max_iters: int, extra: list[str]) -> list[str]:
    cmd = [
        sys.executable, "-u", str(RUN_SCRIPT),
        "--model_name", model_name,
        "--gpu", str(gpu),
        "--seed", str(seed),
        "--max_iters", str(max_iters),
        "--steps_per_iter", str(steps_per_iter),
        "--calib_path", "none",
    ]
    cmd.extend(extra)
    return cmd


def prewarm_cache(gpu: int, prefix: str, log_dir: Path) -> None:
    """Run a max_iters=0 job to populate the predefined cache once.

    Without this, all 5 parallel seeds race on the first cache-miss build
    and corrupt predefined_*.csv.
    """
    if VAL_CACHE.exists() and TRAIN_CACHE.exists():
        print(f"[prewarm] cache already populated at {CACHE_DIR}; skipping.")
        return

    print(f"[prewarm] building predefined cache on GPU {gpu} (max_iters=0) ...")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "prewarm.log"

    cmd = build_cmd(
        model_name=f"{prefix}_prewarm",
        gpu=gpu,
        seed=0,
        steps_per_iter=1,
        max_iters=0,
        extra=[],
    )
    print(f"[prewarm] $ {' '.join(shlex.quote(c) for c in cmd)}")
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(
            f"[prewarm] cache-build run failed (exit={proc.returncode}); see {log_path}"
        )
    if not (VAL_CACHE.exists() and TRAIN_CACHE.exists()):
        raise RuntimeError(
            f"[prewarm] returncode=0 but cache files missing under {CACHE_DIR}"
        )
    print(f"[prewarm] cache built. log: {log_path}")


def parse_best_mape(log_path: Path) -> Optional[float]:
    """Return the minimum 'Net-level MAPE: X.XX%' value seen in the stdout log."""
    if not log_path.exists():
        return None
    best: Optional[float] = None
    with open(log_path) as fh:
        for line in fh:
            m = NET_MAPE_RE.search(line)
            if not m:
                continue
            try:
                val = float(m.group(1))
            except ValueError:
                continue
            if best is None or val < best:
                best = val
    return best


def run_seeds(seeds: list[int], gpus: list[int], steps_per_iter: int,
              prefix: str, log_dir: Path, dry_run: bool) -> dict[int, dict]:
    """Spawn one process per seed/gpu pair; return per-seed metadata once all exit."""
    if len(seeds) != len(gpus):
        raise ValueError(
            f"seeds and gpus must have equal length (got {len(seeds)} seeds, {len(gpus)} gpus)"
        )

    log_dir.mkdir(parents=True, exist_ok=True)
    procs: list[tuple[int, int, subprocess.Popen, Path]] = []

    for seed, gpu in zip(seeds, gpus):
        model_name = f"{prefix}_seed{seed}"
        log_path = log_dir / f"seed{seed}_gpu{gpu}.log"
        cmd = build_cmd(
            model_name=model_name,
            gpu=gpu,
            seed=seed,
            steps_per_iter=steps_per_iter,
            max_iters=1,
            extra=[],
        )
        printable = " ".join(shlex.quote(c) for c in cmd)
        print(f"[seed {seed} → gpu {gpu}] $ {printable}")
        if dry_run:
            continue
        fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        procs.append((seed, gpu, proc, log_path))
        # tiny stagger so concurrent CUDA inits don't all hit at the same instant
        time.sleep(2)

    if dry_run:
        return {}

    print(f"[run] {len(procs)} seeds running. Polling every 60 s ...")
    pending = {p[2].pid: p for p in procs}
    while pending:
        time.sleep(60)
        done = []
        for pid, (seed, gpu, proc, log_path) in pending.items():
            ret = proc.poll()
            if ret is None:
                continue
            done.append(pid)
            tail = log_path.read_text().splitlines()[-3:]
            tail_s = " | ".join(t.strip() for t in tail)
            print(f"[seed {seed}] exit={ret} log={log_path.name} tail={tail_s!r}")
        for pid in done:
            pending.pop(pid)
        if pending:
            still = ", ".join(f"seed={s}/gpu={g}" for (s, g, _, _) in pending.values())
            print(f"[run] still running: {still}")

    # collect results
    results: dict[int, dict] = {}
    for seed, gpu, proc, log_path in procs:
        best = parse_best_mape(log_path)
        ret = proc.returncode
        model_dir = OUTPUT_ROOT / f"{prefix}_seed{seed}"
        results[seed] = {
            "gpu": gpu,
            "exit_code": ret,
            "best_net_mape": best,
            "log": str(log_path),
            "model_dir": str(model_dir),
            "best_ckpt": str(model_dir / "best_model.pth"),
        }
    return results


def write_report(results: dict[int, dict], steps_per_iter: int, prefix: str,
                 report_path: Path, started_at: str, finished_at: str) -> None:
    valid = [v["best_net_mape"] for v in results.values()
             if v.get("best_net_mape") is not None]

    if not valid:
        summary = "No seed produced a parseable Net-level MAPE; check logs."
        mean_s = std_s = p10_s = p90_s = "n/a"
    else:
        mean_v = statistics.fmean(valid)
        std_v = statistics.pstdev(valid) if len(valid) > 1 else 0.0
        sv = sorted(valid)
        # nearest-rank percentiles for small n
        def pct(p):
            if len(sv) == 1:
                return sv[0]
            k = max(0, min(len(sv) - 1, int(round(p / 100.0 * (len(sv) - 1)))))
            return sv[k]
        p10 = pct(10)
        p90 = pct(90)
        mean_s = f"{mean_v:.2f}%"
        std_s = f"±{std_v:.2f}%"
        p10_s = f"{p10:.2f}%"
        p90_s = f"{p90:.2f}%"
        summary = (
            f"v2 baseline (no β disable; ζ NNLS off; no DS-PINN) on 1500-net val set:"
            f" {mean_s} {std_s}  (p10={p10_s}, p90={p90_s}) across {len(valid)} seeds."
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# 5-Seed v2 Baseline Measurement (Step 1C/D)")
    lines.append("")
    lines.append(f"- Run prefix: `{prefix}`")
    lines.append(f"- Steps per iter: {steps_per_iter}")
    lines.append(f"- AL iterations: 1 (cap)")
    lines.append(f"- Calibration init: **disabled** (`--calib_path none`)")
    lines.append(f"- DS-PINN: **disabled** (no `--use_dspinn`)")
    lines.append(f"- Started: {started_at}")
    lines.append(f"- Finished: {finished_at}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(summary)
    lines.append("")
    lines.append("| Seed | GPU | Best Net MAPE | Exit | Log |")
    lines.append("|-----:|----:|--------------:|-----:|-----|")
    for seed in sorted(results):
        info = results[seed]
        mape = (f"{info['best_net_mape']:.2f}%"
                if info.get("best_net_mape") is not None else "n/a")
        lines.append(f"| {seed} | {info['gpu']} | {mape} | {info['exit_code']} | "
                     f"`{info['log']}` |")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Mean ± stdev: **{mean_s}** {std_s}")
    lines.append(f"- p10 / p90: {p10_s} / {p90_s}")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- The v3 hardcoded ζ init `(softplus_inv(8.0), softplus_inv(5.0))` ")
    lines.append("  remains active (`flux_head.py:130-132`). True v2 reproduction ")
    lines.append("  requires reverting that block, which is Step 2 architectural ")
    lines.append("  work.")
    lines.append("- The β term `loss_cpl_ratio * 2.0` is still summed into the ")
    lines.append("  loss (`finetuner.py:675`). Ditto — Step 2.")
    lines.append("- Therefore this measurement is the **noise floor of the current ")
    lines.append("  recipe minus NNLS**, not pure v2. Compare future runs against ")
    lines.append("  this floor, not against the historical 34.83%.")
    report_path.write_text("\n".join(lines))
    print(f"[report] wrote {report_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpus", type=str, default="0,1,2,3,4",
                   help="Comma-separated GPU ids, one per seed.")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4",
                   help="Comma-separated seed ints.")
    p.add_argument("--steps_per_iter", type=int, default=5000,
                   help="Training steps per AL iteration (matches v2's 5000).")
    p.add_argument("--prefix", type=str, default="v2_5seed_baseline",
                   help="Run-name prefix; per-seed model dirs are <prefix>_seed<N>.")
    p.add_argument("--log_dir", type=Path, default=None,
                   help="Where stdout logs go. Default: <repo>/output_intel22/active_learning/diag_phase_a/5seed_logs/.")
    p.add_argument("--prewarm_gpu", type=int, default=None,
                   help="GPU used for the cache-prewarm pass. Defaults to the first GPU in --gpus.")
    p.add_argument("--skip_prewarm", action="store_true",
                   help="Assume cache is already populated; do not run the prewarm pass.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without spawning subprocesses.")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]

    log_dir = args.log_dir or (REPORT_DIR / "5seed_logs")
    prewarm_gpu = args.prewarm_gpu if args.prewarm_gpu is not None else gpus[0]

    started_at = datetime.now().isoformat(timespec="seconds")

    if not args.skip_prewarm and not args.dry_run:
        prewarm_cache(gpu=prewarm_gpu, prefix=args.prefix, log_dir=log_dir)

    results = run_seeds(
        seeds=seeds,
        gpus=gpus,
        steps_per_iter=args.steps_per_iter,
        prefix=args.prefix,
        log_dir=log_dir,
        dry_run=args.dry_run,
    )

    finished_at = datetime.now().isoformat(timespec="seconds")

    if args.dry_run:
        print("[dry_run] no subprocesses spawned, no report written.")
        return 0

    json_path = log_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"[results] wrote {json_path}")

    report_path = REPORT_DIR / "report_5seed_v2.md"
    write_report(
        results=results,
        steps_per_iter=args.steps_per_iter,
        prefix=args.prefix,
        report_path=report_path,
        started_at=started_at,
        finished_at=finished_at,
    )

    bad = [s for s, info in results.items() if info["exit_code"] != 0]
    if bad:
        print(f"[warn] non-zero exit codes for seeds: {bad}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
