#!/usr/bin/env python3
"""
21_extract_per_pair_golden.py — Extract per-pair golden cpl from StarRC SPEFs.

For each (design, target_net, aggressor_net), sum all `*CAP` coupling
entries `<id> <node_a> <node_b> <c_val>` where node_a is on target_net
and node_b is on aggressor_net (or vice versa).

Output: parquet with columns
    design_name, target_net, aggressor_net, c_pair_fF

Streaming SPEF parser (line-by-line) to handle 3GB nova SPEF.
Used by Strike #2 per-pair coupling supervision (HybridPexV3 cpl head).
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Per-pair golden cpl extractor")
    p.add_argument(
        "--spef-dir", type=Path,
        default=Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22"),
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/per_pair_golden"),
    )
    p.add_argument(
        "--designs", nargs="*", default=None,
        help="Limit to specific designs (default: all *.spef in --spef-dir)",
    )
    return p.parse_args()


def _normalize_node_to_net(node: str) -> str:
    """SPEF node '<net>:<n>' or '<inst>:<pin>' → net name only.

    StarRC convention:
      - Internal node:    `n_123:1` → net `n_123`
      - Instance pin:     `i_core/inst:Z` → net unknown (instance-driven, treat as net=instance)
                          Actually in `*CAP` context, node is `<net>:<index>` for routing nodes.
                          Pin nodes use full net name from earlier *CONN section.
      - Port:             `A[0]` → net `A[0]`

    For per-pair extraction we want NET name. The colon-suffix form is most
    common in coupling entries. Strip the trailing `:N` index.
    """
    if ":" not in node:
        return node
    # `n_123:1` → n_123
    # `inst/x:pin` → inst/x (instance, not net — best-effort)
    # `bus[3]:5` → bus[3]
    # We split on the LAST colon; the part before is the net/instance identifier
    head, _, _ = node.rpartition(":")
    return head


def extract_per_pair(spef_path: Path) -> list[dict]:
    """Streaming pass over SPEF: for each *D_NET <target>, accumulate
    (target, aggressor) → sum_cpl from *CAP coupling entries.
    """
    rows: list[dict] = []
    current_net: str | None = None
    in_cap = False
    pair_sums: dict[str, float] = {}  # aggressor_net -> running sum

    def _flush(target: str | None, sums: dict[str, float]) -> None:
        if target is None:
            return
        for aggr, c_val in sums.items():
            if c_val < 1e-4:
                continue
            if aggr == target:
                continue  # self-loop, skip
            rows.append({
                "target_net": target,
                "aggressor_net": aggr,
                "c_pair_fF": float(c_val),
            })

    with open(spef_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue

            if stripped.startswith("*D_NET"):
                _flush(current_net, pair_sums)
                pair_sums = {}
                tokens = stripped.split()
                current_net = tokens[1]
                in_cap = False
                continue
            if not current_net:
                continue
            if stripped.startswith("*CAP"):
                in_cap = True
                continue
            if stripped.startswith("*CONN") or stripped.startswith("*RES"):
                in_cap = False
                continue
            if stripped.startswith("*END"):
                _flush(current_net, pair_sums)
                pair_sums = {}
                current_net = None
                in_cap = False
                continue

            if in_cap and not stripped.startswith("*"):
                tokens = stripped.split()
                if len(tokens) == 4:
                    # Coupling: `<id> <node_a> <node_b> <c_val>`
                    try:
                        c_val = float(tokens[3])
                    except ValueError:
                        continue
                    aggr_node = tokens[2]
                    aggr_net = _normalize_node_to_net(aggr_node)
                    pair_sums[aggr_net] = pair_sums.get(aggr_net, 0.0) + c_val
    # Final flush if file doesn't end with *END
    _flush(current_net, pair_sums)
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    spefs = list(args.spef_dir.glob("intel22_*_starrc.spef"))
    if args.designs:
        spefs = [s for s in spefs
                 if any(d in s.stem for d in args.designs)]
    print(f">>> {len(spefs)} SPEF files to parse")

    summary = []
    for spef_path in sorted(spefs):
        # design_name = stem 'intel22_X_starrc' → 'intel22_X'
        design = spef_path.stem.replace("_starrc", "")
        out_path = args.out_dir / f"{design}.parquet"
        if out_path.exists():
            print(f"  {design}: SKIP (exists)")
            n_rows = len(pd.read_parquet(out_path))
            summary.append({"design": design, "n_pairs": n_rows, "status": "cached"})
            continue
        import time
        t0 = time.time()
        print(f">>> {design}: parsing {spef_path.name} ({spef_path.stat().st_size/1e9:.2f} GB) ...")
        rows = extract_per_pair(spef_path)
        elapsed = time.time() - t0
        df = pd.DataFrame(rows)
        df["design_name"] = design
        df = df[["design_name", "target_net", "aggressor_net", "c_pair_fF"]]
        df.to_parquet(out_path, index=False)
        print(f"    {design}: {len(df):,} pairs ({elapsed:.1f}s) → {out_path.name}")
        summary.append({
            "design": design,
            "n_pairs": int(len(df)),
            "elapsed_sec": float(elapsed),
            "status": "done",
        })

    pd.DataFrame(summary).to_csv(args.out_dir / "_extraction_summary.csv", index=False)
    print()
    print(f"✅ All done. Total pairs: {sum(s.get('n_pairs', 0) for s in summary):,}")
    print(f"  output: {args.out_dir}")


if __name__ == "__main__":
    main()
