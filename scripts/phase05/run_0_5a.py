#!/usr/bin/env python3
"""
Phase 0.5A: Cross-tile coupling fraction analysis.

For each net in a design, computes what fraction of its total coupling
capacitance comes from aggressors that are visible in at least one of
its tiles.

If coverage < 70%: cross-tile coupling loss is a major bottleneck.
If coverage > 90%: tiling captures most coupling; context may be fine.

Usage:
  python3 scripts/phase05/run_0_5a.py \
      --spef  golden_data/spef_data/intel22/intel22_gcd_f3_starrc.spef \
      --manifest /data/PEX_SSL/data/processed/intel22/dataset_manifest.csv \
      --tiles_dir /data/PEX_SSL/data/processed/intel22 \
      --design intel22_gcd_f3 \
      --out_dir analysis/phase05a
"""

import argparse
import gzip
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from spef_utils import parse_coupling_caps


# ---------------------------------------------------------------------------

def load_tile_aggressors(tile_path, target_net):
    """Return set of aggressor net names visible in this tile."""
    with gzip.open(tile_path, 'rb') as f:
        d = pickle.load(f)
    names = d.get('cuboid_net_names', [])
    return {n for n in names if n and n != target_net and n != 'UNKNOWN_PIN'}


def analyze(spef_path, manifest_path, tiles_dir, design, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load coupling caps from SPEF ---
    print("Parsing SPEF coupling caps ...")
    coupling = parse_coupling_caps(spef_path)
    print("  {} nets with coupling caps".format(len(coupling)))

    # --- 2. Load manifest for this design ---
    manifest = pd.read_csv(manifest_path)
    design_df = manifest[manifest['design_name'] == design].copy()
    print("  {} tiles for design '{}'".format(len(design_df), design))

    # net -> list of tile rel_paths
    net_tiles = defaultdict(list)
    for _, row in design_df.iterrows():
        net_tiles[row['net_name']].append(row['rel_path'])

    tiles_base = Path(tiles_dir)

    # --- 3. For each net: collect visible aggressors across all tiles ---
    print("Scanning tiles for visible aggressors ...")

    rows = []
    n_nets_processed = 0

    for net_name, tile_paths in net_tiles.items():
        if net_name not in coupling:
            continue  # no coupling caps in SPEF for this net

        total_cpl = sum(coupling[net_name].values())
        if total_cpl < 1e-9:
            continue

        # Collect all aggressors visible in ANY tile of this net
        visible_aggressors = set()
        for rel in tile_paths:
            full = tiles_base / rel
            if not full.exists():
                continue
            try:
                agg = load_tile_aggressors(full, net_name)
                visible_aggressors.update(agg)
            except Exception as e:
                print("  WARN: could not load {}: {}".format(rel, e))

        # Compute covered cap
        covered_cap = sum(
            cap for agg, cap in coupling[net_name].items()
            if agg in visible_aggressors
        )
        n_aggressors = len(coupling[net_name])
        n_visible    = sum(1 for agg in coupling[net_name] if agg in visible_aggressors)
        coverage_cap = covered_cap / total_cpl
        coverage_net = n_visible / n_aggressors if n_aggressors > 0 else 0.0

        rows.append({
            'net_name'     : net_name,
            'n_tiles'      : len(tile_paths),
            'n_aggressors' : n_aggressors,
            'n_visible'    : n_visible,
            'total_cpl_fF' : total_cpl,
            'covered_fF'   : covered_cap,
            'coverage_cap' : coverage_cap,   # fraction of cap covered
            'coverage_net' : coverage_net,   # fraction of aggressors visible
        })
        n_nets_processed += 1

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'coverage_per_net.csv', index=False)
    print("  Processed {} nets".format(n_nets_processed))

    # --- 4. Statistics ---
    cov_cap = df['coverage_cap'].values
    cov_net = df['coverage_net'].values

    print("\n" + "="*60)
    print("PHASE 0.5A RESULTS")
    print("="*60)
    print("\nCoverage by coupling CAP fraction:")
    print("  mean   : {:.3f}".format(float(cov_cap.mean())))
    print("  median : {:.3f}".format(float(np.median(cov_cap))))
    print("  p10    : {:.3f}".format(float(np.percentile(cov_cap, 10))))
    print("  p25    : {:.3f}".format(float(np.percentile(cov_cap, 25))))
    print("  min    : {:.3f}".format(float(cov_cap.min())))

    print("\nCoverage by aggressor NET count fraction:")
    print("  mean   : {:.3f}".format(float(cov_net.mean())))
    print("  median : {:.3f}".format(float(np.median(cov_net))))
    print("  p10    : {:.3f}".format(float(np.percentile(cov_net, 10))))
    print("  p25    : {:.3f}".format(float(np.percentile(cov_net, 25))))

    # Distribution buckets
    print("\nCapacity coverage distribution (per-net):")
    buckets = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.001)]
    labels  = ["<50%", "50-70%", "70-90%", ">90%"]
    for (lo, hi), lab in zip(buckets, labels):
        n = int(((cov_cap >= lo) & (cov_cap < hi)).sum())
        print("  {:8s}: {:4d} nets  ({:.1f}%)".format(
            lab, n, n / len(df) * 100))

    # Nets with worst coverage
    print("\nBottom 10 nets by cap coverage:")
    worst = df.nsmallest(10, 'coverage_cap')[
        ['net_name', 'n_tiles', 'n_aggressors', 'n_visible',
         'total_cpl_fF', 'covered_fF', 'coverage_cap']
    ]
    print(worst.to_string(index=False))

    # Global cap-weighted coverage
    total_all  = df['total_cpl_fF'].sum()
    covered_all = df['covered_fF'].sum()
    global_cov = covered_all / total_all if total_all > 0 else 0.0
    print("\nGlobal cap-weighted coverage: {:.3f}  ({:.2f} / {:.2f} fF)".format(
        global_cov, covered_all, total_all))

    # --- 5. Verdict ---
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    if global_cov < 0.70:
        print("  GO    - Cross-tile coupling loss is MAJOR (>{:.0f}% missed)".format(
            (1 - global_cov) * 100))
        print("          Tiling removal is strongly justified.")
    elif global_cov < 0.90:
        print("  GO    - Moderate cross-tile loss ({:.0f}% missed).".format(
            (1 - global_cov) * 100))
        print("          Larger context window or tiling removal advised.")
    else:
        print("  CAUTION - Most coupling is within tiles ({:.0f}% covered).".format(
            global_cov * 100))
        print("          Cross-tile coupling is not the dominant error source.")
    print("="*60)

    # --- 6. Optional plot ---
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(cov_cap, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
        axes[0].axvline(float(np.median(cov_cap)), color='red', ls='--',
                        label='median={:.2f}'.format(float(np.median(cov_cap))))
        axes[0].set_xlabel('Cap coverage fraction')
        axes[0].set_ylabel('# nets')
        axes[0].set_title('Fraction of coupling cap visible in tiles')
        axes[0].legend()

        axes[1].scatter(df['total_cpl_fF'], df['coverage_cap'],
                        alpha=0.5, s=15, c='steelblue')
        axes[1].axhline(0.9, color='green',  ls='--', lw=1, label='90%')
        axes[1].axhline(0.7, color='orange', ls='--', lw=1, label='70%')
        axes[1].set_xscale('log')
        axes[1].set_xlabel('Total coupling cap [fF] (log scale)')
        axes[1].set_ylabel('Cap coverage fraction')
        axes[1].set_title('Coverage vs net coupling magnitude')
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(str(out_dir / 'coverage_analysis.png'), dpi=120)
        plt.close(fig)
        print("  Saved: coverage_analysis.png")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Phase 0.5A: cross-tile coupling fraction")
    parser.add_argument('--spef',       required=True)
    parser.add_argument('--manifest',   required=True)
    parser.add_argument('--tiles_dir',  required=True)
    parser.add_argument('--design',     required=True)
    parser.add_argument('--out_dir',    default='analysis/phase05a')
    args = parser.parse_args()

    analyze(args.spef, args.manifest, args.tiles_dir, args.design, args.out_dir)


if __name__ == '__main__':
    main()
