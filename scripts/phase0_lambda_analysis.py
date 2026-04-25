#!/usr/bin/env python3
"""
Phase 0: lambda (charge density) feasibility validation.

Parses StarRC SPEF *RES/*CAP sections to compute per-segment charge density
lambda [fF/um] and validates whether it is a well-behaved, learnable quantity.

Key outputs:
  1. Conservation check  : sum(node_caps) vs *D_NET total cap
  2. Lambda distribution : grouped by (layer, width_bucket)
  3. CV analysis         : coefficient of variation per geometry group
  4. Via/pin cap stats   : lumped cap distribution (separate from wire lambda)
  5. Go/No-Go summary

Usage:
  python3 scripts/phase0_lambda_analysis.py \
      --spef golden_data/spef_data/intel22/intel22_gcd_f3_starrc.spef \
      --out_dir analysis/phase0_gcd

  # Run on multiple designs:
  python3 scripts/phase0_lambda_analysis.py \
      --spef golden_data/spef_data/intel22/intel22_gcd_f3_starrc.spef \
             golden_data/spef_data/intel22/intel22_ibex_core_f3_starrc.spef \
      --out_dir analysis/phase0_multi
"""

import re
import sys
import argparse
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


# ---------------------------------------------------------------------------
# SPEF parser
# ---------------------------------------------------------------------------

_RE_L    = re.compile(r'\$l=([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)')
_RE_W    = re.compile(r'\$w=([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)')
_RE_LVL  = re.compile(r'\$lvl=([0-9]+)')
_RE_DIR  = re.compile(r'\$dir=([01])')
_RE_VC   = re.compile(r'\$vc=([0-9]+)')
_RE_AREA = re.compile(r'\$a=([0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?)')


def _norm(name):
    return name.replace('\\', '').strip()


def parse_spef(spef_path, max_nets=None):
    """
    Parse one StarRC SPEF file.

    Returns dict: net_name -> {
        'total_cap'     : float,                  # from *D_NET header
        'wire_segs'     : list[dict],             # segments with $l > 0
        'lumped_nodes'  : list[dict],             # vias + pins (no $l or $l==0)
        'node_gnd_caps' : dict[node -> float],    # ground caps from *CAP
        'cpl_caps'      : list[dict],             # coupling entries from *CAP
    }
    """
    nets = {}
    current = None
    in_cap = in_res = False
    n_parsed = 0

    with open(spef_path, encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue

            if line.startswith('*D_NET'):
                tokens = line.split()
                if len(tokens) >= 3:
                    try:
                        name = _norm(tokens[1])
                        total = float(tokens[2])
                        current = name
                        nets[current] = {
                            'total_cap': total,
                            'wire_segs': [],
                            'lumped_nodes': [],
                            'node_gnd_caps': defaultdict(float),
                            'cpl_caps': [],
                        }
                        in_cap = in_res = False
                        n_parsed += 1
                        if max_nets and n_parsed >= max_nets:
                            break
                    except ValueError:
                        current = None
                continue

            if current is None:
                continue

            if line.startswith('*END'):
                current = None
                in_cap = in_res = False
                continue

            if line.startswith('*CAP'):
                in_cap = True
                in_res = False
                continue

            if line.startswith('*RES'):
                in_res = True
                in_cap = False
                continue

            if line.startswith('*'):
                in_cap = in_res = False
                continue

            nd = nets[current]

            # --- CAP section ---
            if in_cap and line[0].isdigit():
                tokens = line.split()
                if len(tokens) == 3:
                    # ground cap: IDX NODE VALUE
                    nd['node_gnd_caps'][_norm(tokens[1])] += float(tokens[2])
                elif len(tokens) >= 4:
                    # coupling: IDX NODE1 NODE2 VALUE
                    try:
                        nd['cpl_caps'].append({
                            'node1': _norm(tokens[1]),
                            'net2' : _norm(tokens[2]).split(':')[0],
                            'val'  : float(tokens[3]),
                        })
                    except ValueError:
                        pass

            # --- RES section ---
            if in_res and line[0].isdigit():
                tokens = line.split()
                if len(tokens) < 4:
                    continue
                try:
                    node1 = _norm(tokens[1])
                    node2 = _norm(tokens[2])
                    res   = float(tokens[3])
                except (ValueError, IndexError):
                    continue

                l_m   = _RE_L.search(line)
                w_m   = _RE_W.search(line)
                lvl_m = _RE_LVL.search(line)
                dir_m = _RE_DIR.search(line)
                vc_m  = _RE_VC.search(line)
                a_m   = _RE_AREA.search(line)

                length = float(l_m.group(1))   if l_m   else None
                width  = float(w_m.group(1))   if w_m   else None
                layer  = int(lvl_m.group(1))   if lvl_m else -1
                direc  = int(dir_m.group(1))   if dir_m else -1
                n_via  = int(vc_m.group(1))    if vc_m  else 0
                area   = float(a_m.group(1))   if a_m   else 0.0

                is_lumped = (
                    n_via > 0
                    or length is None
                    or length < 1e-4
                    or (width is not None and width > 1.0)
                )

                entry = {
                    'node1': node1, 'node2': node2,
                    'res': res, 'layer': layer,
                }
                if is_lumped:
                    entry.update({'n_via': n_via, 'area': area})
                    nd['lumped_nodes'].append(entry)
                else:
                    entry.update({
                        'length': length,
                        'width' : width if width is not None else 0.0,
                        'dir'   : direc,
                    })
                    nd['wire_segs'].append(entry)

    return nets


# ---------------------------------------------------------------------------
# Lambda computation
# ---------------------------------------------------------------------------

def compute_lambda(nd):
    """
    Distribute per-node ground caps to adjacent wire segments (length-weighted).

    Returns list of (segment_dict, lambda_gnd [fF/um]).
    Also returns unassigned_cap: total cap that could not be mapped to any segment.
    """
    segs = nd['wire_segs']
    node_gnd = nd['node_gnd_caps']

    if not segs:
        return [], sum(node_gnd.values())

    # node -> [(seg_idx, length)]
    node2segs = defaultdict(list)
    for i, s in enumerate(segs):
        node2segs[s['node1']].append((i, s['length']))
        node2segs[s['node2']].append((i, s['length']))

    seg_cap = np.zeros(len(segs))
    unassigned = 0.0

    for node, cap in node_gnd.items():
        connected = node2segs.get(node, [])
        if not connected:
            unassigned += cap
            continue
        total_len = sum(l for _, l in connected)
        if total_len < 1e-9:
            share = cap / len(connected)
            for idx, _ in connected:
                seg_cap[idx] += share
        else:
            for idx, l in connected:
                seg_cap[idx] += cap * (l / total_len)

    results = []
    for i, s in enumerate(segs):
        lam = seg_cap[i] / s['length'] if s['length'] > 1e-9 else 0.0
        results.append((s, lam))

    return results, unassigned


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def cv(values):
    """Coefficient of variation (std/mean). Returns 0 if mean==0."""
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float('nan')
    m = arr.mean()
    if abs(m) < 1e-12:
        return float('nan')
    return arr.std() / abs(m)


def bucket_width(w, step=0.01):
    """Round wire width to nearest step for grouping."""
    return round(w / step) * step


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(nets, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregation containers
    all_lambda    = []   # (layer, width_bucket, lambda, length, has_coupling, net)
    all_via_caps  = []   # lumped cap per lumped_node (sum of adjacent node gnd caps)
    conservation  = []   # (net, total_cap, recovered_cap, error_pct)

    n_nets_total     = len(nets)
    n_nets_no_segs   = 0
    n_segs_total     = 0
    n_lumped_total   = 0

    for net_name, nd in nets.items():
        total_cap = nd['total_cap']

        # nodes that have coupling (for flagging)
        coupled_nodes = set()
        for c in nd['cpl_caps']:
            coupled_nodes.add(c['node1'])

        lam_results, unassigned = compute_lambda(nd)
        n_segs_total   += len(lam_results)
        n_lumped_total += len(nd['lumped_nodes'])

        if not lam_results:
            n_nets_no_segs += 1

        # Conservation: sum of all node ground caps + sum of coupling caps
        sum_gnd = sum(nd['node_gnd_caps'].values())
        sum_cpl = sum(c['val'] for c in nd['cpl_caps'])
        recovered = sum_gnd + sum_cpl
        err_pct = abs(recovered - total_cap) / (total_cap + 1e-12) * 100.0
        conservation.append((net_name, total_cap, recovered, err_pct))

        # Lambda records
        for seg, lam in lam_results:
            has_cpl = (seg['node1'] in coupled_nodes or
                       seg['node2'] in coupled_nodes)
            all_lambda.append({
                'layer'    : seg['layer'],
                'width_bkt': bucket_width(seg['width']),
                'lambda'   : lam,
                'length'   : seg['length'],
                'has_cpl'  : int(has_cpl),
                'net'      : net_name,
            })

    # ------------------------------------------------------------------
    # 1. Conservation report
    # ------------------------------------------------------------------
    errs = np.array([r[3] for r in conservation])
    print("\n" + "="*60)
    print("1. CONSERVATION CHECK  (sum(node_caps) vs *D_NET total)")
    print("="*60)
    print("  Nets parsed       : {:,}".format(n_nets_total))
    print("  Nets with no segs : {:,}".format(n_nets_no_segs))
    print("  Wire segs total   : {:,}".format(n_segs_total))
    print("  Lumped nodes total: {:,}".format(n_lumped_total))
    print("  Error mean  : {:.4f} %".format(float(errs.mean())))
    print("  Error median: {:.4f} %".format(float(np.median(errs))))
    print("  Error p95   : {:.4f} %".format(float(np.percentile(errs, 95))))
    print("  Error max   : {:.4f} %".format(float(errs.max())))
    if errs.max() < 1.0:
        print("  --> PASS  (max error < 1%)")
    elif errs.max() < 5.0:
        print("  --> WARN  (max error < 5%)")
    else:
        print("  --> FAIL  (max error >= 5%, check SPEF coupling handling)")

    # Save conservation CSV
    cons_rows = "\n".join(
        "{},{:.6f},{:.6f},{:.4f}".format(n, t, r, e)
        for n, t, r, e in conservation
    )
    (out_dir / "conservation.csv").write_text(
        "net_name,total_cap_fF,recovered_cap_fF,error_pct\n" + cons_rows
    )
    print("  Saved: conservation.csv")

    # ------------------------------------------------------------------
    # 2. Lambda distribution by (layer, width)
    # ------------------------------------------------------------------
    lam_arr   = np.array([r['lambda']    for r in all_lambda])
    lay_arr   = np.array([r['layer']     for r in all_lambda])
    wid_arr   = np.array([r['width_bkt'] for r in all_lambda])
    len_arr   = np.array([r['length']    for r in all_lambda])
    cpl_arr   = np.array([r['has_cpl']   for r in all_lambda])

    print("\n" + "="*60)
    print("2. LAMBDA DISTRIBUTION  (all wire segments)")
    print("="*60)
    print("  Total segments : {:,}".format(len(lam_arr)))
    print("  Lambda [fF/um]:")
    print("    mean   : {:.4f}".format(float(lam_arr.mean())))
    print("    median : {:.4f}".format(float(np.median(lam_arr))))
    print("    std    : {:.4f}".format(float(lam_arr.std())))
    print("    p5     : {:.4f}".format(float(np.percentile(lam_arr, 5))))
    print("    p95    : {:.4f}".format(float(np.percentile(lam_arr, 95))))
    print("    min    : {:.4f}".format(float(lam_arr.min())))
    print("    max    : {:.4f}".format(float(lam_arr.max())))
    print("  Global CV      : {:.3f}".format(cv(lam_arr)))

    # ------------------------------------------------------------------
    # 3. CV analysis by (layer, width_bucket)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("3. CV ANALYSIS  by (layer, width_bucket)")
    print("="*60)
    print("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>6}".format(
        "layer", "width_um", "count", "mean_lam", "std_lam", "CV", "flag"))
    print("  " + "-"*58)

    group_stats = []
    layers = sorted(set(lay_arr.tolist()))
    for lyr in layers:
        mask_l = (lay_arr == lyr)
        widths = sorted(set(wid_arr[mask_l].tolist()))
        for wid in widths:
            mask = mask_l & (wid_arr == wid)
            vals = lam_arr[mask]
            if len(vals) < 3:
                continue
            m   = float(vals.mean())
            s   = float(vals.std())
            c   = s / abs(m) if abs(m) > 1e-12 else float('nan')
            flag = ""
            if not math.isnan(c):
                if c < 0.15:
                    flag = "EXCELLENT"
                elif c < 0.30:
                    flag = "GOOD"
                elif c < 0.50:
                    flag = "MODERATE"
                else:
                    flag = "HIGH"
            print("  {:>6}  {:>8.3f}  {:>8,}  {:>8.4f}  {:>8.4f}  {:>8.3f}  {:>6}".format(
                lyr, wid, int(mask.sum()), m, s, c, flag))
            group_stats.append((lyr, wid, int(mask.sum()), m, s, c))

    # Save group stats CSV
    gs_rows = "\n".join(
        "{},{:.4f},{},{:.6f},{:.6f},{:.4f}".format(lyr, wid, cnt, m, s, c)
        for lyr, wid, cnt, m, s, c in group_stats
    )
    (out_dir / "lambda_by_layer_width.csv").write_text(
        "layer,width_um,count,mean_lambda,std_lambda,cv\n" + gs_rows
    )
    print("  Saved: lambda_by_layer_width.csv")

    # ------------------------------------------------------------------
    # 4. Isolated vs coupled segment comparison
    # ------------------------------------------------------------------
    iso_mask = (cpl_arr == 0)
    cpl_mask = (cpl_arr == 1)
    print("\n" + "="*60)
    print("4. ISOLATED vs COUPLED SEGMENTS")
    print("="*60)
    if iso_mask.sum() > 0 and cpl_mask.sum() > 0:
        lam_iso = lam_arr[iso_mask]
        lam_cpl = lam_arr[cpl_mask]
        print("  Isolated  (no coupling on nodes): n={:,}  mean={:.4f}  cv={:.3f}".format(
            int(iso_mask.sum()), float(lam_iso.mean()), cv(lam_iso)))
        print("  Coupled   (has coupling on nodes): n={:,}  mean={:.4f}  cv={:.3f}".format(
            int(cpl_mask.sum()), float(lam_cpl.mean()), cv(lam_cpl)))
        ratio = float(lam_cpl.mean()) / float(lam_iso.mean()) if float(lam_iso.mean()) > 1e-9 else float('nan')
        print("  Coupled/Isolated mean ratio: {:.3f}  (>1 means coupling raises total cap)".format(ratio))
    else:
        print("  Not enough data for comparison.")

    # ------------------------------------------------------------------
    # 5. Lambda vs segment length
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("5. LAMBDA vs SEGMENT LENGTH")
    print("="*60)
    len_pcts = [10, 25, 50, 75, 90]
    thresholds = [float(np.percentile(len_arr, p)) for p in len_pcts]
    print("  Segment length percentiles (um): " +
          "  ".join("p{}={:.3f}".format(p, t) for p, t in zip(len_pcts, thresholds)))

    buckets = [(0, thresholds[1]), (thresholds[1], thresholds[3]), (thresholds[3], 1e9)]
    labels  = ["short (<p25)", "medium (p25-p75)", "long (>p75)"]
    for (lo, hi), lab in zip(buckets, labels):
        m = (len_arr >= lo) & (len_arr < hi)
        if m.sum() < 3:
            continue
        vals = lam_arr[m]
        print("  {:20s} n={:6,}  mean={:.4f}  cv={:.3f}".format(
            lab, int(m.sum()), float(vals.mean()), cv(vals)))

    # ------------------------------------------------------------------
    # 6. Via / lumped cap statistics (from all_lambda we can't derive,
    #    but we can check from net_data 'lumped_nodes' indirectly)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("6. UNASSIGNED (lumped) CAP per NET  (fraction going to vias/pins)")
    print("="*60)

    unassigned_fracs = []
    for net_name, nd in nets.items():
        total = nd['total_cap']
        if total < 1e-9:
            continue
        _, unassigned = compute_lambda(nd)
        sum_gnd = sum(nd['node_gnd_caps'].values())
        frac = unassigned / (sum_gnd + 1e-12) if sum_gnd > 1e-9 else 0.0
        unassigned_fracs.append(frac)

    ua = np.array(unassigned_fracs)
    print("  Fraction of gnd-cap unassigned (via/pin):")
    print("    mean  : {:.3f}".format(float(ua.mean())))
    print("    median: {:.3f}".format(float(np.median(ua))))
    print("    p95   : {:.3f}".format(float(np.percentile(ua, 95))))
    print("    max   : {:.3f}".format(float(ua.max())))
    if ua.mean() < 0.10:
        print("  --> Most cap is on wire segments (good for lambda approach)")
    elif ua.mean() < 0.30:
        print("  --> Moderate lumped component; need separate via/pin prediction")
    else:
        print("  --> Large lumped component; lambda approach needs hybrid design")

    # ------------------------------------------------------------------
    # 7. Save full lambda CSV for further analysis
    # ------------------------------------------------------------------
    lam_rows = "\n".join(
        "{},{:.4f},{:.6f},{:.4f},{}".format(
            r['layer'], r['width_bkt'], r['lambda'], r['length'], r['has_cpl'])
        for r in all_lambda
    )
    (out_dir / "all_lambda.csv").write_text(
        "layer,width_um,lambda_fF_per_um,length_um,has_coupling\n" + lam_rows
    )
    print("\n  Saved: all_lambda.csv  ({:,} rows)".format(len(all_lambda)))

    # ------------------------------------------------------------------
    # 8. Go / No-Go summary
    # ------------------------------------------------------------------
    cv_vals = [c for (_, _, _, _, _, c) in group_stats if not math.isnan(c)]
    pct_good = sum(1 for c in cv_vals if c < 0.30) / len(cv_vals) * 100.0 if cv_vals else 0.0
    cons_pass = float(errs.max()) < 5.0

    print("\n" + "="*60)
    print("GO / NO-GO SUMMARY")
    print("="*60)
    print("  Conservation max error : {:.2f}%  ({})".format(
        float(errs.max()), "PASS" if cons_pass else "FAIL"))
    print("  Groups with CV < 0.30  : {:.1f}%  (threshold: >= 60%)".format(pct_good))
    print("  Avg CV across groups   : {:.3f}".format(
        float(np.mean(cv_vals)) if cv_vals else float('nan')))
    print("  Unassigned cap mean    : {:.3f}".format(float(ua.mean())))

    if cons_pass and pct_good >= 60:
        verdict = "GO"
        detail  = "Lambda is sufficiently consistent. Proceed to Phase 1."
    elif cons_pass and pct_good >= 40:
        verdict = "GO (with caution)"
        detail  = ("Moderate variance. Use wire/via separation and ensure "
                   "local aggressor context in features.")
    else:
        verdict = "NO-GO"
        detail  = ("High variance or conservation failure. Consider "
                   "net-level supervision or via-separated training targets.")

    print("\n  VERDICT: {}".format(verdict))
    print("  Detail : {}".format(detail))
    print("="*60)

    # ------------------------------------------------------------------
    # Plots (optional)
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Plot 1: lambda histogram overall
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(lam_arr[lam_arr < np.percentile(lam_arr, 99)], bins=80, color='steelblue', alpha=0.8)
        ax.set_xlabel('lambda [fF/um]')
        ax.set_ylabel('count')
        ax.set_title('Lambda distribution (all wire segments, <p99)')
        fig.tight_layout()
        fig.savefig(str(out_dir / 'hist_lambda_overall.png'), dpi=100)
        plt.close(fig)

        # Plot 2: box plot of lambda by layer
        layers_with_data = sorted(set(lay_arr.tolist()))
        data_by_layer = [lam_arr[lay_arr == lyr] for lyr in layers_with_data]
        fig, ax = plt.subplots(figsize=(max(6, len(layers_with_data) * 0.8), 5))
        ax.boxplot(data_by_layer, labels=[str(l) for l in layers_with_data], sym='')
        ax.set_xlabel('Metal layer ($lvl)')
        ax.set_ylabel('lambda [fF/um]')
        ax.set_title('Lambda by metal layer')
        fig.tight_layout()
        fig.savefig(str(out_dir / 'boxplot_lambda_by_layer.png'), dpi=100)
        plt.close(fig)

        # Plot 3: CV by (layer, width)
        if group_stats:
            cv_labels = ["L{}W{:.3f}".format(l, w) for l, w, _, _, _, _ in group_stats]
            cv_values = [c if not math.isnan(c) else 0.0 for _, _, _, _, _, c in group_stats]
            fig, ax = plt.subplots(figsize=(max(8, len(cv_labels) * 0.3), 4))
            x = range(len(cv_labels))
            colors = ['green' if c < 0.15 else 'orange' if c < 0.30
                      else 'red' if c < 0.50 else 'darkred' for c in cv_values]
            ax.bar(x, cv_values, color=colors, alpha=0.8)
            ax.axhline(0.15, color='green',  ls='--', lw=1, label='CV=0.15')
            ax.axhline(0.30, color='orange', ls='--', lw=1, label='CV=0.30')
            ax.axhline(0.50, color='red',    ls='--', lw=1, label='CV=0.50')
            ax.set_xticks(list(x))
            ax.set_xticklabels(cv_labels, rotation=90, fontsize=7)
            ax.set_ylabel('CV (std/mean)')
            ax.set_title('Lambda CV by (layer, width)')
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(str(out_dir / 'bar_cv_by_layer_width.png'), dpi=100)
            plt.close(fig)

        print("\n  Saved plots: hist_lambda_overall.png, boxplot_lambda_by_layer.png, bar_cv_by_layer_width.png")

    except ImportError:
        print("\n  (matplotlib not available; skipping plots)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 0: lambda feasibility analysis")
    parser.add_argument('--spef', nargs='+', required=True,
                        help="One or more StarRC SPEF files")
    parser.add_argument('--out_dir', default='analysis/phase0',
                        help="Output directory for CSVs and plots")
    parser.add_argument('--max_nets', type=int, default=None,
                        help="Limit number of nets per file (for quick tests)")
    args = parser.parse_args()

    all_nets = {}
    for spef_path in args.spef:
        p = Path(spef_path)
        if not p.exists():
            print("WARN: not found: {}".format(p))
            continue
        print("Parsing {} ...".format(p.name))
        nets = parse_spef(p, max_nets=args.max_nets)
        print("  -> {} nets".format(len(nets)))
        # prefix net names with design to avoid collisions
        design = p.stem.replace('_starrc', '')
        for k, v in nets.items():
            all_nets["{}:{}".format(design, k)] = v

    if not all_nets:
        print("ERROR: no nets loaded")
        sys.exit(1)

    print("\nTotal nets loaded: {:,}".format(len(all_nets)))
    analyze(all_nets, args.out_dir)


if __name__ == '__main__':
    main()
