"""
Launch 5-seed v6_gamma protocol on free GPUs.

Re-uses the smoke-test seed 0 (already running on GPU 7 as v6_gamma_smoke
→ rename to m6_v6_gamma_seed0 by symlinking when complete) and launches
seeds 1-4 on free GPUs.

Mirrors run_5seed_remaining.py logic for queue management.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/tool/etc/python/install/3.11.9/bin/python3"
CAL_FULL = "/data/PINNPEX/data/processed/intel22/calibration_init.json"


def launch(seed: int, gpu: int) -> subprocess.Popen:
    log_path = ROOT / f"output_intel22/al_5seed_v6_gamma_seed{seed}.log"
    cmd = [
        PYTHON, "-u", str(ROOT / "run_active_learning.py"),
        "--model_name", f"m5_v6_gamma_seed{seed}",
        "--gpu", str(gpu),
        "--use_dspinn", "--use_gamma",
        "--calib_path", CAL_FULL,
        "--seed", str(seed),
        "--max_iters", "1",
        "--steps_per_iter", "5000",
    ]
    log_f = open(log_path, "w")
    print(f"  [LAUNCH] v6_gamma seed={seed} GPU={gpu} → {log_path.name}", flush=True)
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                         cwd=str(ROOT))
    p._log_f = log_f
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[2, 3, 4, 5])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 4])
    args = ap.parse_args()

    free_gpus = list(args.gpus)
    pending = list(args.seeds)
    active: list[tuple[subprocess.Popen, int, int]] = []

    print(f">>> Pending {len(pending)} jobs, free GPUs {free_gpus}")

    while True:
        while free_gpus and pending:
            seed = pending.pop(0)
            gpu = free_gpus.pop(0)
            p = launch(seed, gpu)
            active.append((p, gpu, seed))

        if not active:
            print(">>> All done.")
            break

        time.sleep(60)
        still: list[tuple[subprocess.Popen, int, int]] = []
        for p, gpu, seed in active:
            if p.poll() is None:
                still.append((p, gpu, seed))
            else:
                rc = p.returncode
                p._log_f.close()
                print(f"  [DONE] seed={seed} (GPU {gpu}, rc={rc})", flush=True)
                free_gpus.append(gpu)
        active = still

    return 0


if __name__ == '__main__':
    sys.exit(main())
