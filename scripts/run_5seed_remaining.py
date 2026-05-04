"""
Continue dispatching the 5-seed measurement jobs after the bash launcher
crashed. Skips jobs whose log already exists (already launched) and
dispatches the remaining queue across GPUs 1-4 with up to 4 in parallel.

Usage:
    python3 scripts/run_5seed_remaining.py --gpus 1 2 3 4
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PYTHON = "/tool/etc/python/install/3.11.9/bin/python3"
CAL_FULL = "/data/PINNPEX/data/processed/intel22/calibration_init.json"
CAL_GND_ONLY = "/data/PINNPEX/data/processed/intel22/calibration_init_gnd_only.json"

VARIANT_CALIB = {
    'v3_baseline':   'none',
    'v4_full_calib': CAL_FULL,
    'v5_gnd_only':   CAL_GND_ONLY,
}


def launch(variant: str, seed: int, gpu: int) -> subprocess.Popen:
    calib = VARIANT_CALIB[variant]
    log_path = ROOT / f"output_intel22/al_5seed_{variant}_seed{seed}.log"
    cmd = [
        PYTHON, "-u", str(ROOT / "run_active_learning.py"),
        "--model_name", f"m5_{variant}_seed{seed}",
        "--gpu", str(gpu),
        "--use_dspinn",
        "--calib_path", calib,
        "--seed", str(seed),
        "--max_iters", "1",
        "--steps_per_iter", "5000",
    ]
    log_f = open(log_path, "w")
    print(f"  [LAUNCH] {variant} seed={seed} GPU={gpu} → {log_path.name}", flush=True)
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                         cwd=str(ROOT))
    p._log_f = log_f  # type: ignore[attr-defined]
    return p


def already_running(variant: str, seed: int) -> bool:
    """Check ps for a matching live train process (parent only)."""
    label = f"m5_{variant}_seed{seed}"
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
    except subprocess.CalledProcessError:
        return False
    # Parent train process: ppid==1 (daemonized via nohup) AND args contain label
    # AND args contain run_active_learning.py.
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3: continue
        try:
            ppid = int(parts[1])
        except ValueError:
            continue
        args = parts[2]
        if 'run_active_learning.py' in args and label in args and ppid == 1:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', nargs='+', type=int, default=[1, 2, 3, 4])
    ap.add_argument('--variants', nargs='+',
                    default=['v3_baseline', 'v4_full_calib', 'v5_gnd_only'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    # Build full queue, then filter out anything already running.
    queue: list[tuple[str, int]] = []
    for variant in args.variants:
        for seed in args.seeds:
            queue.append((variant, seed))

    # Skip jobs already running.
    pending: list[tuple[str, int]] = []
    for variant, seed in queue:
        if already_running(variant, seed):
            print(f"  [SKIP] {variant} seed={seed} already running")
        else:
            pending.append((variant, seed))
    print(f"\n>>> Pending: {len(pending)} of {len(queue)} jobs")

    # GPU pool. Skip GPUs that already have a running job from this protocol.
    free_gpus = list(args.gpus)

    # Re-discover which GPUs are occupied by already-running jobs.
    # Parse ps to find each running m5_* parent, extract --gpu N from args.
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3: continue
            try:
                ppid = int(parts[1])
            except ValueError:
                continue
            args_str = parts[2]
            if 'run_active_learning.py' not in args_str: continue
            if 'm5_' not in args_str: continue
            # Extract --gpu N
            tokens = args_str.split()
            for i, t in enumerate(tokens):
                if t == '--gpu' and i + 1 < len(tokens):
                    try:
                        gpu_busy = int(tokens[i + 1])
                    except ValueError:
                        continue
                    if gpu_busy in free_gpus:
                        free_gpus.remove(gpu_busy)
                        print(f"  [OCCUPIED] GPU {gpu_busy} (already-running m5 job)")
                    break
    except subprocess.CalledProcessError:
        pass

    active: list[tuple[subprocess.Popen, int, str, int]] = []  # (p, gpu, variant, seed)
    occupied_gpus = [g for g in args.gpus if g not in free_gpus]
    print(f">>> Free GPUs for new launches: {free_gpus}, externally occupied: {occupied_gpus}")

    def gpu_externally_busy(gpu: int) -> bool:
        """Check if any non-active-tracked m5 job currently uses this GPU."""
        try:
            out = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
        except subprocess.CalledProcessError:
            return False
        active_pids = {p.pid for p, *_ in active}
        for line in out.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3: continue
            try:
                pid = int(parts[0]); ppid = int(parts[1])
            except ValueError:
                continue
            if pid in active_pids: continue
            if ppid != 1: continue
            args_str = parts[2]
            if 'run_active_learning.py' not in args_str: continue
            if 'm5_' not in args_str: continue
            tokens = args_str.split()
            for i, t in enumerate(tokens):
                if t == '--gpu' and i + 1 < len(tokens):
                    try:
                        if int(tokens[i + 1]) == gpu:
                            return True
                    except ValueError:
                        continue
                    break
        return False

    while True:
        # Fill free GPUs from pending queue.
        while free_gpus and pending:
            variant, seed = pending.pop(0)
            gpu = free_gpus.pop(0)
            p = launch(variant, seed, gpu)
            active.append((p, gpu, variant, seed))

        if not active and not pending:
            # Pending exhausted AND nothing tracked active. But external m5
            # jobs may still be running on `occupied_gpus`. Poll until they
            # finish too — only then is the protocol complete.
            externally_busy_now = [g for g in occupied_gpus if gpu_externally_busy(g)]
            if not externally_busy_now:
                print(">>> All done (no pending, no active, no external m5 jobs).")
                break
            print(f"  [WAIT] external m5 still on GPUs: {externally_busy_now}", flush=True)
            time.sleep(120)
            for g in occupied_gpus[:]:
                if not gpu_externally_busy(g):
                    occupied_gpus.remove(g)
                    free_gpus.append(g)
                    print(f"  [FREED] GPU {g} freed by external job completion", flush=True)
            continue

        # Reap finished active jobs.
        time.sleep(60)
        still: list[tuple[subprocess.Popen, int, str, int]] = []
        for p, gpu, variant, seed in active:
            if p.poll() is None:
                still.append((p, gpu, variant, seed))
            else:
                rc = p.returncode
                p._log_f.close()  # type: ignore[attr-defined]
                print(f"  [DONE] {variant} seed={seed} (GPU {gpu}, rc={rc})", flush=True)
                free_gpus.append(gpu)
        active = still

        # Also check externally-occupied GPUs.
        for g in occupied_gpus[:]:
            if not gpu_externally_busy(g):
                occupied_gpus.remove(g)
                free_gpus.append(g)
                print(f"  [FREED] GPU {g} freed by external job completion", flush=True)

    return 0


if __name__ == '__main__':
    sys.exit(main())
