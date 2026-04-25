#!/usr/bin/env python3
"""
Phase 0.5C: Coupling distance distribution.

For every coupling capacitance entry in the SPEF, computes the Euclidean
distance between the two coupled nodes. Builds a cap-weighted CDF to
determine the search radius R that captures 90% of coupling cap.

This directly informs the local neighborhood radius R for the new
architecture's local node predictor.

Also: breaks down coverage by distance bucket to show how fast cap
accumulates as R increases.

Usage:
  python3 scripts/phase05/run_0_5c.py \
      --spef  golden_data/spef_data/intel22/intel22_gcd_f3_starrc.spef \
      --out_dir analysis/phase05c

  # Multiple designs for cross-design comparison:
  python3 scripts/phase05/run_0_5c.py \
      --spef  golden_data/spef_data/intel22/intel22_gcd_f3_starrc.spef \
              golden_data/spef_data/intel22/intel22_ibex_core_f3_starrc.spef \
      --out_dir analysis/phase05c_multi
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from spef_utils import parse_coupling_with_positions


# ---------------------------------------------------------------------------

def analyze(spef_paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for spef in spef_paths:
        p = Path(spef)
        if not p.exists():
            print("WARN: not found: {}".format(p))
            continue
        print("Parsing {} ...".format(p.name))
        entries = parse_coupling_with_positions(str(p))
        print("  {} coupling entries with positions".format(len(entries)))
        all_entries.extend(entries)

    if not all_entries:
        print("ERROR: no entries loaded")
        return

    # Compute Euclidean distances
    x1 = np.array([e[0] for e in all_entries])
    y1 = np.array([e[1] for e in all_entries])
    x2 = np.array([e[2] for e in all_entries])
    y2 = np.array([e[3] for e in all_entries])
    caps = np.array([e[4] for e in all_entries])

    dists = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    total_cap = caps.sum()

    # Save raw data
    rows = "\n".join("{:.4f},{:.6f}".format(d, c)
                     for d, c in zip(dists.tolist(), caps.tolist()))
    (out_dir / 'coupling_distances.csv').write_text(
        "distance_um,cap_fF\n" + rows)

    print("\n" + "="*60)
    print("PHASE 0.5C RESULTS")
    print("="*60)
    print("\nCoupling entry statistics:")
    print("  Total entries : {:,}".format(len(all_entries)))
    print("  Total cap     : {:.4f} fF".format(float(total_cap)))
    print("  Entries with dist=0: {:,}  ({:.1f}%)".format(
        int((dists < 0.001).sum()),
        float((dists < 0.001).mean()) * 100))

    print("\nDistance [um] distribution:")
    pcts = [10, 25, 50, 75, 90, 95, 99]
    for p in pcts:
        print("  p{:2d} : {:.4f} um".format(p, float(np.percentile(dists, p))))
    print("  max : {:.4f} um".format(float(dists.max())))

    # Cap-weighted percentiles
    sort_idx = np.argsort(dists)
    sd = dists[sort_idx]
    sc = caps[sort_idx]
    cum_cap = np.cumsum(sc) / (total_cap + 1e-12)

    print("\nCap-weighted distance percentiles:")
    for target_pct in [50, 70, 80, 90, 95, 99]:
        idx = int(np.searchsorted(cum_cap, target_pct / 100.0))
        idx = min(idx, len(sd) - 1)
        print("  {:.0f}% of cap within {:.4f} um".format(
            target_pct, float(sd[idx])))

    # Coverage at candidate radii
    radii = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0]
    print("\nRadius R -> cap coverage:")
    print("  {:>8}  {:>12}  {:>14}".format("R (um)", "n_entries", "cap_covered_%"))
    r90 = None
    r95 = None
    for R in radii:
        mask = dists <= R
        pct = float(caps[mask].sum() / (total_cap + 1e-12)) * 100
        print("  {:>8.1f}  {:>12,}  {:>13.1f}%".format(R, int(mask.sum()), pct))
        if r90 is None and pct >= 90.0:
            r90 = R
        if r95 is None and pct >= 95.0:
            r95 = R

    # Zero-distance entries analysis
    zero_mask = dists < 0.001
    if zero_mask.sum() > 0:
        print("\nZero-distance entries ({:.1f}% of cap):".format(
            float(caps[zero_mask].sum() / total_cap * 100)))
        print("  These are same-location nodes (via stacks, overlapping pins).")
        print("  Any finite radius R captures them automatically.")

    print("\n" + "="*60)
    print("VERDICT: Recommended local radius R")
    print("="*60)
    if r90 is not None:
        print("  R = {:.1f} um  captures 90% of coupling cap".format(r90))
    else:
        print("  R > {:.1f} um  needed for 90% coupling cap".format(radii[-1]))
    if r95 is not None:
        print("  R = {:.1f} um  captures 95% of coupling cap".format(r95))
    else:
        print("  R > {:.1f} um  needed for 95% coupling cap".format(radii[-1]))
    print("\n  Use this R for the local neighborhood in the new architecture.")
    print("="*60)

    # Optional plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram (distance, unweighted)
        clip_max = float(np.percentile(dists, 99))
        axes[0].hist(np.clip(dists, 0, clip_max), bins=80,
                     color='steelblue', alpha=0.8, edgecolor='white')
        axes[0].set_xlabel('Coupling distance [um]  (clipped at p99)')
        axes[0].set_ylabel('# entries')
        axes[0].set_title('Coupling distance distribution (count-weighted)')

        # Cap-weighted CDF
        plot_dists = np.linspace(0, min(clip_max, 20.0), 300)
        cov = np.array([float(caps[dists <= R].sum() / (total_cap + 1e-12))
                        for R in plot_dists])
        axes[1].plot(plot_dists, cov * 100, color='steelblue', lw=2)
        for R_line, col, lab in [(2.0, 'green', '2um'),
                                  (4.0, 'orange', '4um'),
                                  (8.0, 'red', '8um')]:
            axes[1].axvline(R_line, color=col, ls='--', lw=1, label=lab)
        axes[1].axhline(90, color='gray', ls=':', lw=1, label='90%')
        axes[1].set_xlabel('Radius R [um]')
        axes[1].set_ylabel('Cumulative coupling cap [%]')
        axes[1].set_title('Cap-weighted coverage CDF')
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(str(out_dir / 'coupling_distance_cdf.png'), dpi=120)
        plt.close(fig)
        print("\n  Saved: coupling_distance_cdf.png")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Phase 0.5C: coupling distance distribution")
    parser.add_argument('--spef', nargs='+', required=True)
    parser.add_argument('--out_dir', default='analysis/phase05c')
    args = parser.parse_args()

    analyze(args.spef, args.out_dir)


if __name__ == '__main__':
    main()
