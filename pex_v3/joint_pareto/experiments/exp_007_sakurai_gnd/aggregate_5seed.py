#!/usr/bin/env python3
"""aggregate_5seed.py — collect per-seed metrics + runtime into measurement.json.

Mirrors exp_006/aggregate_5seed.py with variant_id="v8_sakurai".
"""
from __future__ import annotations
import argparse
import json
import statistics
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-seeds", type=int, default=5)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    per_seed = []
    for s in range(args.n_seeds):
        m_path = args.in_dir / f"seed{s}_metrics.json"
        r_path = args.in_dir / f"seed{s}_runtime.json"
        if not m_path.exists() or not r_path.exists():
            print(f"[abort] missing seed {s}: metrics={m_path.exists()} runtime={r_path.exists()}",
                  file=sys.stderr)
            return 1
        m = json.loads(m_path.read_text())
        r = json.loads(r_path.read_text())
        per_seed.append({"seed": s, "metrics": m, "runtime": r})

    def _mean(values):
        return statistics.mean(values) if values else 0.0

    def _stdev(values):
        return statistics.stdev(values) if len(values) > 1 else 0.0

    wall_clocks = [p["runtime"]["wall_clock_s"] for p in per_seed]
    tot_means = [p["metrics"]["tot_mape_mean"] for p in per_seed]
    tot_medians = [p["metrics"]["tot_mape_median"] for p in per_seed]
    tot_p95s = [p["metrics"]["tot_mape_p95"] for p in per_seed]
    gnd_matched = [p["metrics"]["gnd_mape_matched_mean"] for p in per_seed]
    gnd_unmatched = [p["metrics"]["gnd_mape_unmatched_mean"] for p in per_seed]
    cpl_matched = [p["metrics"]["cpl_mape_matched_mean"] for p in per_seed]
    cpl_unmatched = [p["metrics"]["cpl_mape_unmatched_mean"] for p in per_seed]
    r2s = [p["metrics"]["r_squared_c"] for p in per_seed]

    out = {
        "_schema": "joint_pareto.measurement.v1",
        "variant_id": "v8_sakurai",
        "label": "Path-2 v8 Sakurai-Tamaru gnd allocator (16 workers)",
        "design": "intel22_tv80s_f3",
        "n_seeds": args.n_seeds,
        "wall_clock_s": _mean(wall_clocks),
        "wall_clock_s_stdev": _stdev(wall_clocks),
        "wall_clock_per_seed": wall_clocks,
        "mean_of_mean_mape": _mean(tot_means),
        "stdev_of_mean_mape": _stdev(tot_means),
        "mean_of_median_mape": _mean(tot_medians),
        "stdev_of_median_mape": _stdev(tot_medians),
        "mean_of_p95_mape": _mean(tot_p95s),
        "stdev_of_p95_mape": _stdev(tot_p95s),
        "gnd_matched_mean": _mean(gnd_matched),
        "gnd_matched_stdev": _stdev(gnd_matched),
        "gnd_unmatched_mean": _mean(gnd_unmatched),
        "gnd_unmatched_stdev": _stdev(gnd_unmatched),
        "cpl_matched_mean": _mean(cpl_matched),
        "cpl_matched_stdev": _stdev(cpl_matched),
        "cpl_unmatched_mean": _mean(cpl_unmatched),
        "cpl_unmatched_stdev": _stdev(cpl_unmatched),
        "r_squared_c": _mean(r2s),
        "r_squared_c_stdev": _stdev(r2s),
        "per_seed": per_seed,
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(
        {k: v for k, v in out.items() if k not in {"per_seed"}},
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
