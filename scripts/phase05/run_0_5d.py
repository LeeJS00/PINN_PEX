#!/usr/bin/env python3
"""
Phase 0.5D: Local context sufficiency analysis.

Two focused questions:
  A) Per-node coupling range breakdown:
     For each SPEF node, what fraction of its coupling cap comes from
     aggressors within 2um / 4um / 8um? This tells us whether the 4um
     tile window provides enough context, or whether a coarser voxel
     beyond 4um is necessary.

  B) Bad-net deep dive (req_msg[9], dpath_b_reg$out[9]):
     These had < 70% tile coverage in Phase 0.5A.  Show WHY: where are
     the missing coupling aggressors, and how far are they from the net?

Usage:
  python3 scripts/phase05/run_0_5d.py \
      --spef golden_data/spef_data/intel22/intel22_gcd_f3_starrc.spef \
      --manifest /data/PEX_SSL/data/processed/intel22/dataset_manifest.csv \
      --tiles_dir /data/PEX_SSL/data/processed/intel22 \
      --design intel22_gcd_f3 \
      --out_dir analysis/phase05d
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
from spef_utils import parse_coupling_caps, parse_node_positions, norm, net_of

WINDOW_HALF = 2.0   # tile half-width in um (4um window)
RADII = [2.0, 4.0, 6.0, 8.0, 12.0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def parse_coupling_by_node(spef_path):
    """
    Returns dict: target_node -> [(aggressor_node, cap_fF), ...]
    Only coupling (2-node) *CAP entries.
    """
    by_node = defaultdict(list)
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
                if len(tokens) >= 4:
                    try:
                        n1 = norm(tokens[1])
                        n2 = norm(tokens[2])
                        val = float(tokens[3])
                        if net_of(n1) != net_of(n2):
                            by_node[n1].append((n2, val))
                    except (ValueError, IndexError):
                        pass
    return dict(by_node)


def load_tile_info(manifest_path, tiles_dir, design):
    """
    Returns dict: net_name -> list of {origin, window, visible_aggressors}
    """
    manifest = pd.read_csv(manifest_path)
    design_df = manifest[manifest['design_name'] == design]
    tiles_base = Path(tiles_dir)

    info = defaultdict(list)
    for _, row in design_df.iterrows():
        p = tiles_base / row['rel_path']
        if not p.exists():
            continue
        try:
            with gzip.open(p, 'rb') as f:
                d = pickle.load(f)
            cx, cy = float(d['origin'][0]), float(d['origin'][1])
            names = {n for n in d['cuboid_net_names']
                     if n and n != row['net_name'] and n != 'UNKNOWN_PIN'}
            info[row['net_name']].append({
                'cx': cx, 'cy': cy,
                'xmin': cx - WINDOW_HALF, 'xmax': cx + WINDOW_HALF,
                'ymin': cy - WINDOW_HALF, 'ymax': cy + WINDOW_HALF,
                'visible': names,
            })
        except Exception:
            pass
    return dict(info)


# ---------------------------------------------------------------------------
# Analysis A: per-node coupling range breakdown
# ---------------------------------------------------------------------------

def analysis_a(node_couplings, node_pos, out_dir):
    """
    For each target node with coupling entries, compute cap fraction
    from aggressors within each radius band.
    """
    print("\n" + "="*60)
    print("ANALYSIS A: Per-node coupling range breakdown")
    print("="*60)

    rows = []
    n_no_pos = 0

    for tgt_node, pairs in node_couplings.items():
        p_tgt = node_pos.get(tgt_node) or node_pos.get(net_of(tgt_node))
        if p_tgt is None:
            n_no_pos += 1
            continue

        tx, ty = p_tgt
        total_cap = sum(v for _, v in pairs)
        if total_cap < 1e-9:
            continue

        cap_by_radius = {}
        for R in RADII:
            cap_in = sum(v for agg_node, v in pairs
                         if _dist(tx, ty, node_pos, agg_node) <= R)
            cap_by_radius[R] = cap_in

        rows.append({
            'node'      : tgt_node,
            'net'       : net_of(tgt_node),
            'total_cap' : total_cap,
            **{'frac_R{:.0f}'.format(R): cap_by_radius[R] / total_cap
               for R in RADII}
        })

    print("  Nodes analysed : {:,}".format(len(rows)))
    print("  No position    : {:,}".format(n_no_pos))

    if not rows:
        print("  ERROR: no data")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'node_range_breakdown.csv', index=False)

    print("\n  Fraction of coupling cap captured at radius R (distribution across nodes):")
    print("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "R(um)", "mean", "median", "p10", "p25"))
    for R in RADII:
        col = 'frac_R{:.0f}'.format(R)
        v = df[col].values
        print("  {:>6.1f}  {:>8.3f}  {:>8.3f}  {:>8.3f}  {:>8.3f}".format(
            R, float(v.mean()), float(np.median(v)),
            float(np.percentile(v, 10)), float(np.percentile(v, 25))))

    # Classify nodes by how much of their cap is within 4um
    frac4 = df['frac_R4'].values
    print("\n  Node classification by fraction-within-4um:")
    for lo, hi, lab in [(0.0, 0.5, '<50%'),
                         (0.5, 0.7, '50-70%'),
                         (0.7, 0.9, '70-90%'),
                         (0.9, 1.01, '>90%')]:
        n = int(((frac4 >= lo) & (frac4 < hi)).sum())
        pct_cap = float(df.loc[(frac4 >= lo) & (frac4 < hi), 'total_cap'].sum()
                        / df['total_cap'].sum() * 100)
        print("    {:8s}: {:4d} nodes  ({:.1f}% of total coupling cap)".format(
            lab, n, pct_cap))

    # Verdict
    global_frac4 = float((df['total_cap'].values *
                           df['frac_R4'].values).sum()) / float(df['total_cap'].sum())
    global_frac8 = float((df['total_cap'].values *
                           df['frac_R8'].values).sum()) / float(df['total_cap'].sum())

    print("\n  Cap-weighted fraction within 4um : {:.3f}".format(global_frac4))
    print("  Cap-weighted fraction within 8um : {:.3f}".format(global_frac8))
    print("  Marginal gain 4->8um             : {:.3f}".format(
        global_frac8 - global_frac4))

    print("\n  --> VERDICT:")
    if global_frac4 >= 0.90:
        print("  4um context captures {:.1f}% of coupling cap.".format(
            global_frac4 * 100))
        print("  Voxel beyond 4um gives only {:.1f}% marginal gain.".format(
            (global_frac8 - global_frac4) * 100))
        print("  Global context is a MINOR factor for coupling cap.")
        print("  -> Main problem is elsewhere (model capacity / training signal).")
    elif global_frac4 >= 0.75:
        print("  4um captures {:.1f}% but 8um gets {:.1f}%.".format(
            global_frac4 * 100, global_frac8 * 100))
        print("  A coarse voxel at 8um range would recover ~{:.1f}% more.".format(
            (global_frac8 - global_frac4) * 100))
        print("  -> Voxel context is MODERATELY useful.")
    else:
        print("  Only {:.1f}% within 4um — global context is CRITICAL.".format(
            global_frac4 * 100))
        print("  -> Voxel context beyond 4um is strongly justified.")

    # Optional plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # CDF of frac_R4 across nodes (cap-weighted)
        sorted_f = np.sort(frac4)
        w = df['total_cap'].values / df['total_cap'].sum()
        w_sorted = w[np.argsort(frac4)]
        cum_w = np.cumsum(w_sorted)
        axes[0].plot(sorted_f, cum_w * 100, color='steelblue', lw=2)
        axes[0].axvline(0.9, color='green', ls='--', lw=1, label='90%')
        axes[0].axvline(0.7, color='orange', ls='--', lw=1, label='70%')
        axes[0].set_xlabel('Fraction of node coupling within 4um')
        axes[0].set_ylabel('Cumulative cap fraction [%]')
        axes[0].set_title('Cap-weighted CDF: within-4um fraction per node')
        axes[0].legend()

        # Bar chart: marginal gain per radius step
        r_vals = RADII
        means = [float(df['frac_R{:.0f}'.format(R)].mean()) for R in r_vals]
        bars = [means[0]] + [means[i] - means[i-1] for i in range(1, len(means))]
        labels = ['0-{}um'.format(int(r_vals[0]))] + \
                 ['{}-{}um'.format(int(r_vals[i-1]), int(r_vals[i]))
                  for i in range(1, len(r_vals))]
        colors = ['steelblue', 'royalblue', 'orange', 'tomato', 'gray']
        axes[1].bar(range(len(bars)), [b * 100 for b in bars],
                    color=colors[:len(bars)], alpha=0.8, edgecolor='white')
        axes[1].set_xticks(range(len(bars)))
        axes[1].set_xticklabels(labels, rotation=20, ha='right')
        axes[1].set_ylabel('Mean coupling cap fraction [%]')
        axes[1].set_title('Marginal cap gain per radius band')

        fig.tight_layout()
        fig.savefig(str(out_dir / 'node_range_breakdown.png'), dpi=120)
        plt.close(fig)
        print("  Saved: node_range_breakdown.png")
    except ImportError:
        pass


def _dist(tx, ty, node_pos, agg_node):
    """Distance from (tx,ty) to agg_node's position. Returns inf if unknown."""
    p = node_pos.get(agg_node) or node_pos.get(net_of(agg_node))
    if p is None:
        return float('inf')
    return float(np.sqrt((p[0] - tx)**2 + (p[1] - ty)**2))


# ---------------------------------------------------------------------------
# Analysis B: bad-net deep dive
# ---------------------------------------------------------------------------

def analysis_b(bad_nets, coupling_caps, node_pos, tile_info, out_dir):
    """
    For each bad net: show spatial map of tile window, visible aggressors,
    and missing aggressors with their distances.
    """
    print("\n" + "="*60)
    print("ANALYSIS B: Bad-net deep dive")
    print("="*60)

    report_lines = []

    for net_name in bad_nets:
        print("\n  Net: {}".format(net_name))
        tiles = tile_info.get(net_name, [])
        cpl   = coupling_caps.get(net_name, {})
        total_cpl = sum(cpl.values())

        if not tiles:
            print("    No tiles found.")
            continue

        # Collect all visible aggressors across all tiles
        all_visible = set()
        for t in tiles:
            all_visible.update(t['visible'])

        # Missing aggressors
        missing = {agg: cap for agg, cap in cpl.items()
                   if agg not in all_visible}
        covered = total_cpl - sum(missing.values())
        print("    Tiles       : {}  windows: {}".format(
            len(tiles),
            ['{:.1f},{:.1f}'.format(t['cx'], t['cy']) for t in tiles]))
        print("    Coupling    : {:.4f} fF total, {:.1f}% covered".format(
            total_cpl, covered / total_cpl * 100 if total_cpl > 0 else 0))
        print("    Missing aggressors ({} of {} nets, {:.4f} fF):".format(
            len(missing), len(cpl), sum(missing.values())))

        # For each missing aggressor, find its position and distance to nearest tile
        print("    {:35s}  {:>10}  {:>12}  {:>10}".format(
            "aggressor", "cap(fF)", "pos(x,y)", "min_dist"))
        sorted_missing = sorted(missing.items(), key=lambda x: -x[1])
        for agg, cap in sorted_missing[:15]:
            pos = node_pos.get(agg)
            if pos is None:
                # try first node of this net
                for k in node_pos:
                    if net_of(k) == agg:
                        pos = node_pos[k]
                        break
            if pos is None:
                pos_str = "unknown"
                min_dist = float('nan')
            else:
                pos_str = "{:.2f},{:.2f}".format(pos[0], pos[1])
                min_dist = min(
                    np.sqrt((pos[0] - t['cx'])**2 + (pos[1] - t['cy'])**2)
                    for t in tiles)
            print("    {:35s}  {:>10.4f}  {:>12}  {:>10}".format(
                agg, cap, pos_str,
                "{:.2f}um".format(min_dist) if not np.isnan(min_dist) else "n/a"))

        # Key question: distance distribution of missing aggressors
        dists_missing = []
        caps_missing  = []
        for agg, cap in missing.items():
            pos = node_pos.get(agg)
            if pos is None:
                for k in node_pos:
                    if net_of(k) == agg:
                        pos = node_pos[k]
                        break
            if pos:
                min_dist = min(
                    np.sqrt((pos[0] - t['cx'])**2 + (pos[1] - t['cy'])**2)
                    for t in tiles)
                dists_missing.append(min_dist)
                caps_missing.append(cap)

        if dists_missing:
            d_arr = np.array(dists_missing)
            c_arr = np.array(caps_missing)
            print("\n    Distance of missing aggressors from nearest tile center:")
            print("    mean={:.2f}um  median={:.2f}um  max={:.2f}um".format(
                float(d_arr.mean()), float(np.median(d_arr)), float(d_arr.max())))
            cap_within = {R: float(c_arr[d_arr <= R].sum())
                          for R in [2, 4, 6, 8, 12]}
            total_missing = float(c_arr.sum())
            print("    Recoverable by expanding context:")
            for R, c in cap_within.items():
                print("      +{}um context -> recovers {:.4f} fF  ({:.1f}% of missing)".format(
                    R, c, c / total_missing * 100 if total_missing > 0 else 0))

        report_lines.append("Net: {}\n  covered={:.1f}%  missing_aggressors={}\n".format(
            net_name, covered/total_cpl*100 if total_cpl > 0 else 0, len(missing)))

    (out_dir / 'bad_nets_report.txt').write_text('\n'.join(report_lines))
    print("\n  Saved: bad_nets_report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0.5D: local context sufficiency")
    parser.add_argument('--spef',       required=True)
    parser.add_argument('--manifest',   required=True)
    parser.add_argument('--tiles_dir',  required=True)
    parser.add_argument('--design',     required=True)
    parser.add_argument('--bad_nets',   nargs='+',
                        default=["req_msg[9]", "dpath_b_reg$out[9]"])
    parser.add_argument('--out_dir',    default='analysis/phase05d')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Parsing SPEF ...")
    node_couplings = parse_coupling_by_node(args.spef)
    node_pos       = parse_node_positions(args.spef)
    coupling_caps  = parse_coupling_caps(args.spef)
    print("  {:,} nodes with coupling  |  {:,} node positions".format(
        len(node_couplings), len(node_pos)))

    print("Loading tile info ...")
    tile_info = load_tile_info(args.manifest, args.tiles_dir, args.design)
    print("  {:,} nets have tiles".format(len(tile_info)))

    analysis_a(node_couplings, node_pos, out_dir)
    analysis_b(args.bad_nets, coupling_caps, node_pos, tile_info, out_dir)

    print("\n  All outputs saved to: {}".format(out_dir))


if __name__ == '__main__':
    main()
