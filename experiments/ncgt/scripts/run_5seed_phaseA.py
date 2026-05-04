#!/usr/bin/env python3
"""
Phase A: 5-seed validation across 3 configs (per Plan v4 protocol).

Configs:
  C1. Single-design vanilla (gcd_f3, 200 nets, 5K steps)
  C2. Multi-design vanilla (9 × 100 nets, 8K steps)
  C3. Multi-design + bins + aug (9 × 100, 15K steps)

For each config: 5 seeds. Reports mean ± std on best VALID MAPE, p95, r.

Output:
    experiments/ncgt/output_train/5seed_*.json (per config)
    experiments/ncgt/PHASE_A_RESULTS.md (consolidated)

Run from project root:
    python3 -m experiments.ncgt.scripts.run_5seed_phaseA
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PYTHON = "/tool/etc/python/install/3.11.9/bin/python3"


def run_config(name: str, args_list: list) -> dict:
    print(f"\n========== [Phase A] {name} ==========")
    print("  cmd:", " ".join(args_list))
    r = subprocess.run(args_list, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(f"[FAIL] stderr:\n{r.stderr[-2000:]}")
        return {"name": name, "status": "fail"}
    print(f"[OK] {name}")
    return {"name": name, "status": "ok"}


def main() -> int:
    base = [PYTHON, "-m", "experiments.ncgt.src.trainers.train_ncgt",
            "--mode", "train", "--gpu", "0", "--n_seeds", "5", "--seed", "0"]

    configs = [
        ("C1_single_vanilla", base + [
            "--design", "intel22_gcd_f3", "--n_nets", "200",
            "--steps", "5000", "--eval_interval", "500", "--lr", "5e-4",
        ]),
        ("C2_multi_vanilla", base + [
            "--multi_design", "--n_nets", "100",
            "--steps", "8000", "--eval_interval", "1000", "--lr", "5e-4",
        ]),
        ("C3_multi_bins_aug", base + [
            "--multi_design", "--n_nets", "100",
            "--steps", "15000", "--eval_interval", "1500", "--lr", "5e-4",
            "--use_bins", "--use_aug",
        ]),
    ]

    results = []
    for name, cmd in configs:
        results.append(run_config(name, cmd))

    print("\n========== PHASE A COMPLETE ==========")
    for r in results:
        print(f"  {r['name']}: {r['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
