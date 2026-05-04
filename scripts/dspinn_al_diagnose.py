#!/usr/bin/env python3
"""
scripts/dspinn_al_diagnose.py

Run diagnostic checks on a DS-PINN AL run when the reporter flags MARGINAL or
FAILED. Outputs a JSON of evidence to feed back into Codex / next-version
proposal.

Diagnostic dimensions (all derived from log + CSV — no model reload):
  1. DS-PINN activation health trajectory (Z_macro magnitude, proj_out norm,
     FNO weight norm vs step). Flags: stagnation, explosion, regression.
  2. Loss component dynamics (does loss_aux_macro descend? does loss_cpl_direct
     plateau alongside loss_cpl_total?). Flags: aux head not training, CPL
     stuck at near-zero local minimum.
  3. Per-iteration validation evolution (best MAPE/CPL across iters).
     Flags: regression after iter N, plateau at v10b ceiling.
  4. AL data acquisition pattern (which designs/nets selected per iter).
     Flags: AL stuck on single design, low entropy nets selected.
  5. CTS/clk net forced injection counter (L4 logic). Flags: injection
     not firing or always firing.

Outputs:
  output_intel22/active_learning/<model_name>/diagnose_iter<N>.json

Usage:
  python3 scripts/dspinn_al_diagnose.py --model_name dspinn_v1_new
"""
import argparse
import json
import re
import sys
import statistics
from pathlib import Path

# Reuse parser from report module by importing it.
sys.path.insert(0, str(Path(__file__).parent))
from dspinn_al_report import (
    parse_log, V10B_BASELINE_MAPE, V10B_BASELINE_CPL,
    GATE1_CPL_STRONG, GATE1_CPL_PARTIAL, GATE1_CPL_MARGINAL,
)

RE_AL_ACQUIRED = re.compile(
    r"AL Acquired Top (\d+) Difficult Nets \(([0-9]+) Tiles\) from (\d+) Designs"
    r"\.\s+\(Avg Net Entropy: ([0-9.]+)\)"
)
RE_L4_INJECT = re.compile(r"\[L4\] Force-injected (\d+) CTS/clk nets")
RE_DESIGN_SPEF = re.compile(r"Generating Golden SPEF for (\S+)")
RE_ITER_TAG    = re.compile(r">>> \[Iteration (\d+)\]")


def parse_acquisition_events(log_path: Path) -> list[dict]:
    """Per-iteration AL acquisition records: design list, net count, entropy."""
    if not log_path.exists():
        return []
    text = log_path.read_text(errors='replace')
    events: list[dict] = []
    cur_iter = 0
    cur: dict | None = None
    for line in text.splitlines():
        m_it = RE_ITER_TAG.search(line)
        if m_it:
            if cur:
                events.append(cur)
            cur_iter = int(m_it.group(1))
            cur = {'iteration': cur_iter, 'designs': [], 'cts_inject': 0,
                   'nets': 0, 'tiles': 0, 'avg_entropy': None}
            continue
        if cur is None:
            continue
        m_acq = RE_AL_ACQUIRED.search(line)
        if m_acq:
            cur['nets']  = int(m_acq.group(1))
            cur['tiles'] = int(m_acq.group(2))
            cur['avg_entropy'] = float(m_acq.group(4))
            continue
        m_l4 = RE_L4_INJECT.search(line)
        if m_l4:
            cur['cts_inject'] = int(m_l4.group(1))
            continue
        m_spef = RE_DESIGN_SPEF.search(line)
        if m_spef:
            cur['designs'].append(m_spef.group(1))
    if cur:
        events.append(cur)
    return events


def diagnose_dspinn_health(records: list[dict]) -> dict:
    """Trace Z_macro / proj_out / FNO weight evolution. Detect failure modes."""
    probes = [r for r in records if 'zmacro_mean' in r]
    if not probes:
        return {'status': 'no_probes', 'message': 'No DS-PINN probe records found.'}

    flags = []

    # 1. Stagnation: proj_out |W| < 0.05 throughout
    proj_w_max = max(r['proj_w'] for r in probes)
    if proj_w_max < 0.05:
        flags.append("proj_out_stagnant: proj_out|W| never exceeded 0.05 — "
                    "Z_macro is not engaging downstream heads.")

    # 2. Explosion: Z_macro max > 100 at any probe
    zmacro_max = max(r['zmacro_max'] for r in probes)
    if zmacro_max > 100:
        flags.append(f"zmacro_explode: Z_macro|max| reached {zmacro_max:.2f} "
                    "— consider tighter cpl_macro_norm or smaller LR.")

    # 3. Regression: Z_macro mean drops below 10% of peak
    zm_peak = max(r['zmacro_mean'] for r in probes)
    zm_last = probes[-1]['zmacro_mean']
    if zm_last < 0.1 * zm_peak and len(probes) > 5:
        flags.append(f"zmacro_regress: Z_macro|mean| collapsed from peak "
                    f"{zm_peak:.3e} to {zm_last:.3e} — late training issue.")

    # 4. FNO weight stagnation
    fno_spec_first = probes[0]['fno_spec']
    fno_spec_last  = probes[-1]['fno_spec']
    if abs(fno_spec_last - fno_spec_first) < 0.05 and len(probes) > 10:
        flags.append(f"fno_stagnant: FNO[0] spec|W| moved <0.05 over training "
                    f"(start={fno_spec_first:.3f}, end={fno_spec_last:.3f}).")

    return {
        'status': 'ok' if not flags else 'concern',
        'flags': flags,
        'n_probes': len(probes),
        'proj_w_first':  probes[0]['proj_w'],
        'proj_w_last':   probes[-1]['proj_w'],
        'proj_w_max':    proj_w_max,
        'zmacro_mean_first': probes[0]['zmacro_mean'],
        'zmacro_mean_last':  probes[-1]['zmacro_mean'],
        'zmacro_max_overall': zmacro_max,
        'fno_spec_first': fno_spec_first,
        'fno_spec_last':  fno_spec_last,
    }


def diagnose_loss_dynamics(records: list[dict]) -> dict:
    """Analyze how each loss component evolves over training."""
    train_recs = [r for r in records if 'train_loss' in r]
    if len(train_recs) < 3:
        return {'status': 'insufficient_data'}

    flags = []
    # Take first quartile vs last quartile to compare early vs late training.
    q = max(1, len(train_recs) // 4)
    early = train_recs[:q]
    late  = train_recs[-q:]

    def avg(seq, key):
        vals = [r[key] for r in seq if key in r]
        return statistics.mean(vals) if vals else None

    components = ['loss_scale', 'loss_cpl_total', 'loss_distribution',
                  'loss_gnd', 'loss_cpl_direct', 'loss_aux_macro']
    evolution = {}
    for k in components:
        e = avg(early, k)
        l = avg(late, k)
        if e is None or l is None or e == 0:
            continue
        rel_change = (l - e) / e * 100
        evolution[k] = {'early': e, 'late': l, 'rel_change_pct': rel_change}

    # Flags
    if 'loss_aux_macro' in evolution:
        aux = evolution['loss_aux_macro']
        if aux['rel_change_pct'] > -10:
            flags.append(f"aux_not_descending: loss_aux_macro changed only "
                        f"{aux['rel_change_pct']:+.1f}% (early {aux['early']:.3f} "
                        f"→ late {aux['late']:.3f}) — Z_macro not predicting "
                        f"per-net total cap.")
    if 'loss_cpl_direct' in evolution and 'loss_cpl_total' in evolution:
        cpl_d = evolution['loss_cpl_direct']
        cpl_t = evolution['loss_cpl_total']
        if cpl_d['rel_change_pct'] > -5 and cpl_t['rel_change_pct'] < -50:
            flags.append("cpl_direct_stuck: loss_cpl_total descended "
                        f"{cpl_t['rel_change_pct']:.1f}% but loss_cpl_direct "
                        f"only {cpl_d['rel_change_pct']:.1f}% — CPL learning "
                        "the log-space mapping but not absolute magnitude.")

    return {
        'status': 'ok' if not flags else 'concern',
        'flags': flags,
        'evolution': evolution,
    }


def diagnose_iter_progression(records: list[dict]) -> dict:
    """Per-iter best validation. Detect regression / plateau."""
    iters_seen = sorted({r.get('iteration', 0) for r in records})
    per_iter = {}
    for it in iters_seen:
        in_iter = [r for r in records
                   if r.get('iteration') == it and 'net_mape' in r]
        if not in_iter:
            continue
        best = min(in_iter, key=lambda r: r['net_mape'])
        per_iter[it] = {
            'best_step': best.get('step'),
            'net_mape': best['net_mape'],
            'val_smape_tot': best.get('val_smape_tot'),
            'val_smape_gnd': best.get('val_smape_gnd'),
            'val_smape_cpl': best.get('val_smape_cpl'),
        }

    flags = []
    # Regression: any iter has worse MAPE than the previous best
    sorted_keys = sorted(per_iter.keys())
    best_so_far = float('inf')
    for it in sorted_keys:
        m = per_iter[it]['net_mape']
        if m > best_so_far * 1.10:
            flags.append(f"iter{it}_regression: MAPE {m:.2f}% is >10% worse "
                        f"than running best {best_so_far:.2f}%.")
        best_so_far = min(best_so_far, m)

    # CPL ceiling: if best CPL across all iters < 5% improvement vs v10b 320%
    best_cpl = min((p['val_smape_cpl'] for p in per_iter.values()
                    if p.get('val_smape_cpl')), default=float('inf'))
    if best_cpl > V10B_BASELINE_CPL * 0.95 and len(per_iter) >= 2:
        flags.append(f"cpl_ceiling: best CPL SMAPE across iters = {best_cpl:.1f}% "
                    f"(v10b ceiling {V10B_BASELINE_CPL}%) — DS-PINN not breaking "
                    "the CPL barrier.")

    return {
        'status': 'ok' if not flags else 'concern',
        'flags': flags,
        'per_iter': per_iter,
        'best_mape_seen': best_so_far if best_so_far != float('inf') else None,
        'best_cpl_seen':  best_cpl   if best_cpl   != float('inf') else None,
    }


def diagnose_data_acquisition(events: list[dict]) -> dict:
    """Which designs/nets is AL pulling? Stuck on one design? Low entropy?"""
    if not events:
        return {'status': 'no_acquisition_events'}

    flags = []
    all_designs = []
    for e in events:
        all_designs.extend(e.get('designs', []))
    from collections import Counter
    design_counter = Counter(all_designs)

    if len(events) >= 3:
        # If a single design dominates >70% of acquisitions, flag.
        total = sum(design_counter.values())
        if total > 0:
            top_design, top_count = design_counter.most_common(1)[0]
            ratio = top_count / total
            if ratio > 0.70:
                flags.append(f"design_concentration: '{top_design}' acquired "
                            f"{ratio:.0%} of times across {len(events)} iters.")

    entropies = [e['avg_entropy'] for e in events if e.get('avg_entropy')]
    if entropies and max(entropies) - min(entropies) < 0.01 and len(entropies) >= 3:
        flags.append(f"entropy_flat: avg net entropy varied <0.01 across iters "
                    f"({entropies[0]:.4f} ↔ {entropies[-1]:.4f}) — selector "
                    "not differentiating tiles.")

    cts_total = sum(e.get('cts_inject', 0) for e in events)
    if cts_total == 0 and len(events) >= 2:
        flags.append("no_cts_injection: L4 CTS/clk force-injection never fired "
                    "— may miss clock nets in training.")

    return {
        'status': 'ok' if not flags else 'concern',
        'flags': flags,
        'iters': len(events),
        'design_counter': dict(design_counter.most_common()),
        'entropies': entropies,
        'cts_total_injected': cts_total,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--output_root', default='/home/jslee/projects/PINNPEX/output_intel22')
    ap.add_argument('--iter', type=int, default=None)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    al_dir = output_root / 'active_learning' / args.model_name
    log_path = output_root / f'al_{args.model_name}.log'
    if not log_path.exists():
        candidates = list(output_root.glob(f'al_{args.model_name}*.log'))
        if candidates:
            log_path = candidates[0]
    if not log_path.exists():
        print(f"❌ Log file not found.")
        sys.exit(1)

    records = parse_log(log_path)
    events  = parse_acquisition_events(log_path)

    diagnostic = {
        'model_name': args.model_name,
        'log_path': str(log_path),
        'n_records': len(records),
        'iters_observed': sorted({r.get('iteration', 0) for r in records}),
        'dspinn_health': diagnose_dspinn_health(records),
        'loss_dynamics': diagnose_loss_dynamics(records),
        'iter_progression': diagnose_iter_progression(records),
        'data_acquisition': diagnose_data_acquisition(events),
    }

    target_iter = args.iter if args.iter is not None else (
        diagnostic['iters_observed'][-1] if diagnostic['iters_observed'] else 0)
    al_dir.mkdir(parents=True, exist_ok=True)
    out = al_dir / f'diagnose_iter{target_iter}.json'
    out.write_text(json.dumps(diagnostic, indent=2))

    # Console summary
    print(f"=== DS-PINN AL Diagnostic — {args.model_name} ===\n")
    for section_key, section in diagnostic.items():
        if not isinstance(section, dict) or 'flags' not in section:
            continue
        st = section.get('status', 'ok')
        flags = section.get('flags', [])
        icon = '🟢' if st == 'ok' else '🟠'
        print(f"{icon} {section_key}: {st}")
        for f in flags:
            print(f"    ⚠ {f}")
        print()

    print(f"Full diagnostic written to: {out}")


if __name__ == '__main__':
    main()
