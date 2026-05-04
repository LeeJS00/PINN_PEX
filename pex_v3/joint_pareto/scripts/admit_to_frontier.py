#!/usr/bin/env python3
"""admit_to_frontier.py — pareto-architect helper.

Reads a candidate variant's 5-seed measurement JSON, compares against the
current frontier in `results/leaderboard.json`, and either ADMITS or REJECTS.
On admission, updates the leaderboard and `PARETO.md`.

Usage:
    python3 admit_to_frontier.py \
        --measurement experiments/exp_003_v4_sakurai/five_seed_summary.json \
        --variant-id v4_sakurai \
        [--label "Path-2 v4 (Sakurai-Tamaru gnd)"] \
        [--dry-run]

Hard kill criteria (any reject the variant):
    - wall_clock_s     > 100
    - gnd_mape_matched > 35
    - cpl_mape_matched > 25
    - r2_c             < 0.98

Pareto admission tolerance ε:
    - wall_clock_s     +10 % from current best
    - total_mape_mean  +0.2 pp
    - gnd_mape_matched +1.0 pp
    - cpl_mape_matched +1.0 pp
    - r2_c             −0.005

A variant is ADMITTED if it strictly improves at least one axis without
regressing any other axis past ε.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LEADERBOARD = REPO / "pex_v3/joint_pareto/results/leaderboard.json"
PARETO_MD = REPO / "pex_v3/joint_pareto/PARETO.md"

EPS = {
    "wall_clock_s": 1.10,         # multiplicative +10 %
    "total_mape_mean": 0.2,       # additive
    "gnd_mape_matched": 1.0,
    "cpl_mape_matched": 1.0,
    "r2_c": -0.005,                # additive (more negative = bigger drop allowed)
}
KILL = {
    "wall_clock_s_max": 100.0,
    "gnd_mape_matched_max": 35.0,
    "cpl_mape_matched_max": 25.0,
    "r2_c_min": 0.98,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--measurement", type=Path, required=True)
    p.add_argument("--variant-id", required=True)
    p.add_argument("--label", default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_leaderboard() -> dict:
    return json.loads(LEADERBOARD.read_text())


def save_leaderboard(d: dict) -> None:
    LEADERBOARD.write_text(json.dumps(d, indent=2))


def is_killed(m: dict) -> tuple[bool, str]:
    if m.get("wall_clock_s", 0) > KILL["wall_clock_s_max"]:
        return True, f"wall_clock_s {m['wall_clock_s']} > {KILL['wall_clock_s_max']}"
    if m.get("gnd_mape_matched", 0) > KILL["gnd_mape_matched_max"]:
        return True, f"gnd_mape_matched {m['gnd_mape_matched']} > {KILL['gnd_mape_matched_max']}"
    if m.get("cpl_mape_matched", 0) > KILL["cpl_mape_matched_max"]:
        return True, f"cpl_mape_matched {m['cpl_mape_matched']} > {KILL['cpl_mape_matched_max']}"
    if m.get("r2_c", 1.0) < KILL["r2_c_min"]:
        return True, f"r2_c {m['r2_c']} < {KILL['r2_c_min']}"
    return False, ""


def is_dominated_or_within_epsilon(candidate: dict, frontier: list[dict]) -> tuple[bool, str]:
    """Return (admit, reason).

    Rule: ADMIT iff candidate does NOT regress any axis past ε vs the
    current best on that axis. This is stricter than "not dominated" —
    a candidate that improves some axis but regresses another past ε is
    rejected (the trade-off must lie inside ε).
    """
    if not frontier:
        return True, "empty frontier"
    frontier_active = [f for f in frontier if f.get("is_frontier")] or frontier

    # Per-axis best on the active frontier
    def best(axis, sign):
        if sign == "minimize":
            return min(f[axis] for f in frontier_active if axis in f and f[axis] is not None)
        return max(f[axis] for f in frontier_active if axis in f and f[axis] is not None)

    best_wall = best("wall_clock_s", "minimize")
    best_total_mean = best("total_mape_mean", "minimize")
    best_gnd = best("gnd_mape_matched", "minimize")
    best_cpl = best("cpl_mape_matched", "minimize")
    best_r2 = best("r2_c", "maximize")

    # Reject if regressed past ε on ANY axis
    if candidate["wall_clock_s"] > best_wall * EPS["wall_clock_s"]:
        return False, (f"wall_clock_s {candidate['wall_clock_s']:.2f} > "
                       f"{best_wall * EPS['wall_clock_s']:.2f} (best {best_wall:.2f} × {EPS['wall_clock_s']:.2f})")
    if candidate["total_mape_mean"] > best_total_mean + EPS["total_mape_mean"]:
        return False, (f"total_mape_mean {candidate['total_mape_mean']:.3f} > "
                       f"{best_total_mean + EPS['total_mape_mean']:.3f} (best {best_total_mean:.3f} + {EPS['total_mape_mean']})")
    if candidate["gnd_mape_matched"] > best_gnd + EPS["gnd_mape_matched"]:
        return False, (f"gnd_mape_matched {candidate['gnd_mape_matched']:.2f} > "
                       f"{best_gnd + EPS['gnd_mape_matched']:.2f} (best {best_gnd:.2f} + {EPS['gnd_mape_matched']})")
    if candidate["cpl_mape_matched"] > best_cpl + EPS["cpl_mape_matched"]:
        return False, (f"cpl_mape_matched {candidate['cpl_mape_matched']:.2f} > "
                       f"{best_cpl + EPS['cpl_mape_matched']:.2f} (best {best_cpl:.2f} + {EPS['cpl_mape_matched']})")
    if candidate["r2_c"] < best_r2 + EPS["r2_c"]:
        return False, (f"r2_c {candidate['r2_c']:.4f} < "
                       f"{best_r2 + EPS['r2_c']:.4f} (best {best_r2:.4f} + {EPS['r2_c']})")
    return True, "no axis regresses past ε vs best-of-frontier"


def main() -> int:
    args = parse_args()
    measurement = json.loads(args.measurement.read_text())
    candidate = {
        "id": args.variant_id,
        "label": args.label or args.variant_id,
        "wall_clock_s": measurement.get("wall_clock_s"),
        "total_mape_mean": measurement.get("mean_of_mean_mape"),
        "total_mape_median": measurement.get("mean_of_median_mape"),
        "total_mape_p95": measurement.get("mean_of_p95_mape"),
        "gnd_mape_matched": measurement.get("gnd_matched_mean"),
        "cpl_mape_matched": measurement.get("cpl_matched_mean"),
        "r2_c": measurement.get("r_squared_c"),
        "n_seeds": measurement.get("n_seeds", 5),
    }
    print(f">>> Candidate: {candidate['id']} — {candidate['label']}")
    for k, v in candidate.items():
        if isinstance(v, float):
            print(f"    {k:24s} {v:.4f}")
        else:
            print(f"    {k:24s} {v}")

    killed, reason = is_killed(candidate)
    if killed:
        print(f"\n❌ KILLED: {reason}")
        return 1

    lb = load_leaderboard()
    admit, reason = is_dominated_or_within_epsilon(candidate, lb["frontier"])
    print(f"\nVerdict: {'✅ ADMIT' if admit else '⛔ REJECT'} — {reason}")

    if admit and not args.dry_run:
        candidate["is_frontier"] = True
        candidate["dominated_by"] = []
        # Mark previously-frontier rows that are dominated by this candidate
        for f in lb["frontier"]:
            if f.get("is_frontier") and all([
                candidate["wall_clock_s"] <= f["wall_clock_s"],
                candidate["total_mape_mean"] <= f["total_mape_mean"],
                candidate["gnd_mape_matched"] <= f["gnd_mape_matched"],
                candidate["cpl_mape_matched"] <= f["cpl_mape_matched"],
                candidate["r2_c"] >= f["r2_c"],
            ]):
                f["is_frontier"] = False
                f.setdefault("dominated_by", []).append(candidate["id"])
        lb["frontier"].append(candidate)
        save_leaderboard(lb)
        print(f"\n✅ Frontier updated: {LEADERBOARD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
