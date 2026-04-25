#!/usr/bin/env python3
"""
Phase 0.5B: DEF-SPEF node alignment quality.

Checks whether SPEF node positions map cleanly onto tile abs_geometry
centroids (which represent DEF routing geometry). If the alignment is
poor, DEF topology nodes cannot be used as clean training targets.

Method:
  - Collect all SPEF node positions from *CONN section
  - Collect all abs_geometry centroids from tile pkl.gz files
  - For each SPEF node, find nearest abs_geometry centroid (KD-tree)
  - Report: mapping accuracy at ε=0.05, 0.1, 0.2 um
  - Report: fraction of SPEF node cap covered by well-mapped nodes

Usage:
  python3 scripts/phase05/run_0_5b.py \
      --spef  golden_data/spef_data/intel22/intel22_gcd_f3_starrc.spef \
      --manifest /data/PEX_SSL/data/processed/intel22/dataset_manifest.csv \
      --tiles_dir /data/PEX_SSL/data/processed/intel22 \
      --design intel22_gcd_f3 \
      --out_dir analysis/phase05b
"""

import argparse
import gzip
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from spef_utils import parse_node_positions, norm


# ---------------------------------------------------------------------------

def parse_node_ground_caps(spef_path):
    """
    Returns dict: node_key -> ground_cap_fF
    (only ground cap entries: 3-token *CAP lines)
    """
    gnd = defaultdict(float)
    current_net = None
    in_cap = False

    with open(spef_path, encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            if line.startswith('*D_NET'):
                tokens = line.split()
                current_net = norm(tokens[1]) if len(tokens) >= 2 else None
                in_cap = False
                continue
            if current_net is None:
                continue
            if line.startswith('*CAP'):
                in_cap = True; continue
            if line.startswith('*RES') or line.startswith('*END') or line.startswith('*CONN'):
                in_cap = False; continue
            if in_cap and line and line[0].isdigit():
                tokens = line.split()
                if len(tokens) == 3:
                    try:
                        gnd[norm(tokens[1])] += float(tokens[2])
                    except ValueError:
                        pass
    return dict(gnd)


def load_all_abs_centroids(manifest_path, tiles_dir, design):
    """
    Load abs_geometry centroids (x, y) from all tiles of a design.
    Returns np.array (M, 2) of unique 2D positions.
    """
    manifest = pd.read_csv(manifest_path)
    design_df = manifest[manifest['design_name'] == design]
    tiles_base = Path(tiles_dir)

    all_xy = set()
    n_tiles = 0
    for _, row in design_df.iterrows():
        full = tiles_base / row['rel_path']
        if not full.exists():
            continue
        try:
            with gzip.open(full, 'rb') as f:
                d = pickle.load(f)
            abs_geo = d.get('abs_geometries')
            if abs_geo is None or len(abs_geo) == 0:
                continue
            # centroid = (x, y) from abs_geometries columns [0, 1]
            for row_geo in abs_geo:
                all_xy.add((round(float(row_geo[0]), 4),
                             round(float(row_geo[1]), 4)))
            n_tiles += 1
        except Exception:
            pass

    print("  Loaded {} tiles, {} unique XY centroids".format(
        n_tiles, len(all_xy)))
    return np.array(sorted(all_xy), dtype=np.float64)


def analyze(spef_path, manifest_path, tiles_dir, design, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing SPEF node positions ...")
    node_pos = parse_node_positions(spef_path)
    print("  {} SPEF nodes with positions".format(len(node_pos)))

    print("Parsing SPEF node ground caps ...")
    node_cap = parse_node_ground_caps(spef_path)
    print("  {} nodes have ground cap".format(len(node_cap)))

    print("Loading tile abs_geometry centroids ...")
    centroids = load_all_abs_centroids(manifest_path, tiles_dir, design)

    if len(centroids) == 0:
        print("ERROR: no centroids loaded")
        return

    # Build KD-tree on 2D centroids
    tree = cKDTree(centroids)

    # For each SPEF node with known position, find nearest centroid
    epsilons = [0.05, 0.10, 0.20, 0.50]
    rows = []
    for node, (x, y) in node_pos.items():
        dist, _ = tree.query([x, y], k=1)
        cap = node_cap.get(node, 0.0)
        rows.append({'node': node, 'x': x, 'y': y,
                     'nearest_dist_um': dist, 'gnd_cap_fF': cap})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'node_alignment.csv', index=False)

    dists = df['nearest_dist_um'].values
    caps  = df['gnd_cap_fF'].values
    total_cap = caps.sum()

    print("\n" + "="*60)
    print("PHASE 0.5B RESULTS")
    print("="*60)
    print("\nSPEF node -> DEF geometry nearest-neighbor distance [um]:")
    print("  mean   : {:.4f}".format(float(dists.mean())))
    print("  median : {:.4f}".format(float(np.median(dists))))
    print("  p75    : {:.4f}".format(float(np.percentile(dists, 75))))
    print("  p90    : {:.4f}".format(float(np.percentile(dists, 90))))
    print("  p95    : {:.4f}".format(float(np.percentile(dists, 95))))
    print("  max    : {:.4f}".format(float(dists.max())))

    print("\nMapping quality at different thresholds:")
    print("  {:>8}  {:>12}  {:>12}  {:>14}".format(
        "epsilon", "nodes_mapped", "pct_nodes", "pct_cap_covered"))
    for eps in epsilons:
        mask = dists <= eps
        n_mapped = int(mask.sum())
        pct_n = n_mapped / len(dists) * 100
        pct_c = caps[mask].sum() / (total_cap + 1e-12) * 100
        print("  {:>8.3f}  {:>12,}  {:>11.1f}%  {:>13.1f}%".format(
            eps, n_mapped, pct_n, pct_c))

    # Verdict
    pct_nodes_01 = float((dists <= 0.10).mean()) * 100
    pct_cap_01   = float(caps[dists <= 0.10].sum() / (total_cap + 1e-12)) * 100

    print("\n" + "="*60)
    print("VERDICT  (threshold = 0.10 um)")
    print("="*60)
    if pct_cap_01 >= 95:
        print("  GO    - {:.1f}% of ground cap maps to DEF geometry within 0.10um".format(pct_cap_01))
        print("          DEF topology nodes are viable training targets.")
    elif pct_cap_01 >= 80:
        print("  WARN  - {:.1f}% of ground cap maps within 0.10um".format(pct_cap_01))
        print("          Acceptable but some label noise expected.")
    else:
        print("  CAUTION - Only {:.1f}% of ground cap maps within 0.10um".format(pct_cap_01))
        print("          Consider using SPEF node positions directly as targets.")
    print("="*60)

    # Optional plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].hist(np.clip(dists, 0, 0.5), bins=60,
                     color='steelblue', alpha=0.8, edgecolor='white')
        for eps, col in zip([0.05, 0.10, 0.20], ['green', 'orange', 'red']):
            axes[0].axvline(eps, color=col, ls='--', lw=1, label='ε={}'.format(eps))
        axes[0].set_xlabel('Distance to nearest DEF centroid [um]  (clipped at 0.5)')
        axes[0].set_ylabel('# SPEF nodes')
        axes[0].set_title('SPEF node -> DEF geometry alignment')
        axes[0].legend(fontsize=8)

        # Cap-weighted CDF
        sort_idx = np.argsort(dists)
        sorted_dist = dists[sort_idx]
        sorted_cap  = caps[sort_idx]
        cum_cap = np.cumsum(sorted_cap) / (total_cap + 1e-12)
        axes[1].plot(sorted_dist, cum_cap, color='steelblue', lw=1.5)
        axes[1].axvline(0.10, color='orange', ls='--', lw=1, label='ε=0.10')
        axes[1].axhline(0.95, color='green',  ls='--', lw=1, label='95%')
        axes[1].set_xlim(0, 0.5)
        axes[1].set_xlabel('Distance threshold [um]')
        axes[1].set_ylabel('Cumulative ground cap fraction')
        axes[1].set_title('Cap-weighted alignment CDF')
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(str(out_dir / 'alignment_analysis.png'), dpi=120)
        plt.close(fig)
        print("  Saved: alignment_analysis.png")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Phase 0.5B: DEF-SPEF node alignment")
    parser.add_argument('--spef',       required=True)
    parser.add_argument('--manifest',   required=True)
    parser.add_argument('--tiles_dir',  required=True)
    parser.add_argument('--design',     required=True)
    parser.add_argument('--out_dir',    default='analysis/phase05b')
    args = parser.parse_args()

    analyze(args.spef, args.manifest, args.tiles_dir, args.design, args.out_dir)


if __name__ == '__main__':
    main()
