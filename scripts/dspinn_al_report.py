#!/usr/bin/env python3
"""
scripts/dspinn_al_report.py

Parse a running or completed DS-PINN AL run and produce a structured report:
  - per-iter validation metrics (Total/GND/CPL SMAPE, net-level MAPE)
  - DS-PINN health (Z_macro magnitude, proj_out norm, FNO weight norm)
  - training loss components (loss_scale/cpl/gnd/aux)
  - verdict against roadmap thresholds (Gate 1, Gate 2)

Outputs:
  output_intel22/active_learning/<model_name>/report_iter<N>.md
  output_intel22/active_learning/<model_name>/metrics_iter<N>.json

Usage:
  python3 scripts/dspinn_al_report.py --model_name dspinn_v1_new
  python3 scripts/dspinn_al_report.py --model_name dspinn_v1_new --iter 1
"""
import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime

# Roadmap thresholds (from /home/jslee/.claude/plans/dspinn-roadmap.md).
V10B_BASELINE_MAPE  = 27.30
V10B_BASELINE_CPL   = 320.0      # CPL SMAPE % stuck at ≥320 across 80+ ckpts
GATE1_CPL_STRONG    = 150.0
GATE1_CPL_PARTIAL   = 250.0
GATE1_CPL_MARGINAL  = 310.0
GATE2_TARGET_MAPE   = 25.0
GATE2_STRETCH_MAPE  = 20.0


# Regex patterns for log parsing (matched in plain text — tqdm carriage returns ignored).
RE_STEP_LOSS = re.compile(
    r">>> \[FineTuner\] Step (\d+):\s+Train Loss = ([0-9.]+)"
    r"\s+loss_scale: ([0-9.]+)\s+loss_cpl_total: ([0-9.]+)"
    r"(?:\s+loss_cpl_vector: ([0-9.]+))?"          # v2 P1
    r"(?:\s+loss_cpl_ratio: ([0-9.]+))?"           # v3 β
    r"\s+loss_distribution: ([0-9.]+)\s+loss_gnd: ([0-9.]+)\s+\(w=([0-9.]+)\)"
    r"\s+loss_cpl_direct: ([0-9.]+)\s+\(w=([0-9.]+)\)"
    r"(?:\s+loss_aux_macro: ([0-9.]+)\s+\(w=([0-9.]+)\))?"
)
RE_VAL_SMAPE = re.compile(
    r"Validation SMAPE \[%\] -> Tot: ([0-9.]+) \| GND: ([0-9.]+) \| CPL: ([0-9.]+)"
)
RE_CUSTOM_LOSS = re.compile(
    r"Custom loss \[%\]\s+-> Tot: ([0-9.]+) \| GND: ([0-9.]+) \| CPL: ([0-9.]+)"
)
RE_REAL_SMAPE = re.compile(
    r"True SMAPE \[%\]\s+-> Tot: ([0-9.]+) \| GND: ([0-9.]+) \| CPL \(per-edge\): ([0-9.]+)"
)
RE_CPL_RATIO  = re.compile(r"CPL ratio \(med\)\s*:\s*([0-9.]+)%")
RE_NET_MAPE  = re.compile(r"Net-level MAPE\s+: ([0-9.]+)%")
RE_COMPOSITE = re.compile(r"Composite Score: ([0-9.]+)\s*(🌟 BEST!)?")
RE_Z_MACRO   = re.compile(
    r"\[DS-PINN\] Z_macro:\s+\|mean\|=([0-9.eE+-]+)\s+\|max\|=([0-9.eE+-]+)\s+std=([0-9.eE+-]+)"
)
RE_PROJ_FNO  = re.compile(
    r"\[DS-PINN\] proj_out:\s+\|W\|=([0-9.]+)\s+\|b\|=([0-9.]+)\s+"
    r"fno\[0\]:\s+spec\|W\|=([0-9.]+)\s+skip\|W\|=([0-9.]+)"
)
RE_ITER_TAG  = re.compile(r">>> \[Iteration (\d+)\]")
RE_ACQ       = re.compile(
    r"AL Acquired Top (\d+) Difficult Nets \(([0-9]+) Tiles\) from (\d+) Designs"
    r"\s+\(Avg Net Entropy: ([0-9.]+)\)"
)


def parse_log(log_path: Path) -> list[dict]:
    """Parse the AL log into a list of step records. Each record is a dict
    keyed by step number with all metrics observed at/around that step."""
    if not log_path.exists():
        return []
    text = log_path.read_text(errors='replace')

    # Split on "[Probe Alert - Step N]" boundaries to collate val/probe per step.
    # Strategy: walk lines top-down; track current iteration; on Step matches,
    # accumulate into the active record dict.
    records: list[dict] = []
    cur_iter = 0
    cur_rec: dict = {}

    def flush():
        if cur_rec:
            records.append(dict(cur_rec))

    for line in text.splitlines():
        m_iter = RE_ITER_TAG.search(line)
        if m_iter:
            cur_iter = int(m_iter.group(1))
            continue
        m_step = RE_STEP_LOSS.search(line)
        if m_step:
            flush()
            cur_rec = {
                'step': int(m_step.group(1)),
                'iteration': cur_iter,
                'train_loss': float(m_step.group(2)),
                'loss_scale': float(m_step.group(3)),
                'loss_cpl_total': float(m_step.group(4)),
                'loss_distribution': float(m_step.group(7)),
                'loss_gnd': float(m_step.group(8)),
                'w_gnd': float(m_step.group(9)),
                'loss_cpl_direct': float(m_step.group(10)),
                'w_cpl_direct': float(m_step.group(11)),
            }
            if m_step.group(5):
                cur_rec['loss_cpl_vector'] = float(m_step.group(5))
            if m_step.group(6):
                cur_rec['loss_cpl_ratio'] = float(m_step.group(6))
            if m_step.group(12):
                cur_rec['loss_aux_macro'] = float(m_step.group(12))
                cur_rec['w_aux_macro'] = float(m_step.group(13))
            continue
        m_val = RE_VAL_SMAPE.search(line)
        if m_val and cur_rec:
            cur_rec.update(
                val_smape_tot=float(m_val.group(1)),
                val_smape_gnd=float(m_val.group(2)),
                val_smape_cpl=float(m_val.group(3)),
            )
            continue
        m_loss = RE_CUSTOM_LOSS.search(line)
        if m_loss and cur_rec:
            cur_rec.update(
                custom_tot=float(m_loss.group(1)),
                custom_gnd=float(m_loss.group(2)),
                custom_cpl=float(m_loss.group(3)),
            )
            continue
        m_real = RE_REAL_SMAPE.search(line)
        if m_real and cur_rec:
            cur_rec.update(
                real_smape_tot=float(m_real.group(1)),
                real_smape_gnd=float(m_real.group(2)),
                real_smape_cpl=float(m_real.group(3)),
            )
            # Mirror into legacy keys so downstream verdict logic uses true SMAPE.
            cur_rec.update(
                val_smape_tot=float(m_real.group(1)),
                val_smape_gnd=float(m_real.group(2)),
                val_smape_cpl=float(m_real.group(3)),
            )
            continue
        m_ratio = RE_CPL_RATIO.search(line)
        if m_ratio and cur_rec:
            cur_rec['cpl_ratio_med'] = float(m_ratio.group(1))
            continue
        m_mape = RE_NET_MAPE.search(line)
        if m_mape and cur_rec:
            cur_rec['net_mape'] = float(m_mape.group(1))
            continue
        m_comp = RE_COMPOSITE.search(line)
        if m_comp and cur_rec:
            cur_rec['composite'] = float(m_comp.group(1))
            cur_rec['is_best']   = bool(m_comp.group(2))
            continue
        m_zm = RE_Z_MACRO.search(line)
        if m_zm and cur_rec:
            cur_rec['zmacro_mean'] = float(m_zm.group(1))
            cur_rec['zmacro_max']  = float(m_zm.group(2))
            cur_rec['zmacro_std']  = float(m_zm.group(3))
            continue
        m_pf = RE_PROJ_FNO.search(line)
        if m_pf and cur_rec:
            cur_rec['proj_w']    = float(m_pf.group(1))
            cur_rec['proj_b']    = float(m_pf.group(2))
            cur_rec['fno_spec']  = float(m_pf.group(3))
            cur_rec['fno_skip']  = float(m_pf.group(4))
            continue
    flush()
    return records


def find_iter_boundary(records: list[dict], target_iter: int) -> dict | None:
    """Return the last record at/before the end of the target_iter."""
    rec = None
    for r in records:
        if r.get('iteration') == target_iter and 'val_smape_cpl' in r:
            rec = r
    return rec


def best_at_iter(records: list[dict], target_iter: int) -> dict | None:
    """Best (lowest net_mape) record within the iteration."""
    candidates = [r for r in records
                  if r.get('iteration') == target_iter and 'net_mape' in r]
    if not candidates:
        return None
    return min(candidates, key=lambda r: r['net_mape'])


def verdict_for_iter1(rec: dict | None) -> tuple[str, str]:
    """Map iter1 CPL SMAPE to roadmap Gate-1 verdict."""
    if rec is None:
        return ("UNKNOWN", "Iter 1 has not produced validation metrics yet.")
    cpl = rec.get('val_smape_cpl', float('inf'))
    if cpl < GATE1_CPL_STRONG:
        return ("STRONG",
                f"CPL SMAPE {cpl:.1f}% < {GATE1_CPL_STRONG:.0f}% — "
                f"DS-PINN is delivering on the screening hypothesis.")
    if cpl < GATE1_CPL_PARTIAL:
        return ("PARTIAL",
                f"CPL SMAPE {cpl:.1f}% in 150-250% band — "
                f"partial improvement; continue to Iter 3 and compare.")
    if cpl < GATE1_CPL_MARGINAL:
        return ("MARGINAL",
                f"CPL SMAPE {cpl:.1f}% in 250-310% band — "
                f"marginal; consider unfreezing FNO earlier or boosting d_macro.")
    return ("FAILED",
            f"CPL SMAPE {cpl:.1f}% ≥ {GATE1_CPL_MARGINAL:.0f}% — "
            f"matches v10b ceiling; structural failure of MacroDensityFNO approach.")


def verdict_for_iter3(rec: dict | None) -> tuple[str, str]:
    """Map iter3 net MAPE to Gate-2 verdict."""
    if rec is None:
        return ("UNKNOWN", "Iter 3 has not completed validation.")
    mape = rec.get('net_mape', float('inf'))
    cpl  = rec.get('val_smape_cpl', float('inf'))
    if mape < GATE2_STRETCH_MAPE:
        return ("STRONG", f"Net MAPE {mape:.2f}% beats stretch goal {GATE2_STRETCH_MAPE}%.")
    if mape < GATE2_TARGET_MAPE:
        return ("STRONG", f"Net MAPE {mape:.2f}% beats target {GATE2_TARGET_MAPE}%.")
    if mape < V10B_BASELINE_MAPE:
        if cpl < V10B_BASELINE_CPL * 0.7:
            return ("PARTIAL",
                    f"Net MAPE {mape:.2f}% beats v10b {V10B_BASELINE_MAPE}% "
                    f"and CPL improved to {cpl:.1f}% — keep going.")
        return ("PARTIAL",
                f"Net MAPE {mape:.2f}% beats v10b {V10B_BASELINE_MAPE}% but "
                f"CPL still high ({cpl:.1f}%) — investigate.")
    return ("FAILED",
            f"Net MAPE {mape:.2f}% ≥ v10b {V10B_BASELINE_MAPE}% — "
            f"DS-PINN not improving over baseline, reassess approach.")


def render_markdown(model_name: str, target_iter: int, rec: dict | None,
                    records_in_iter: list[dict],
                    verdict: tuple[str, str]) -> str:
    out = []
    out.append(f"# DS-PINN AL Report — `{model_name}` Iter {target_iter}")
    out.append("")
    out.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    out.append("")

    if rec is None:
        out.append(f"## ⚠ Iter {target_iter} not yet completed")
        out.append("")
        out.append(f"No validation records found for iteration {target_iter}.")
        return "\n".join(out)

    # Headline verdict
    icons = {"STRONG": "🟢", "PARTIAL": "🟡", "MARGINAL": "🟠",
             "FAILED": "🔴", "UNKNOWN": "⚪"}
    out.append(f"## {icons.get(verdict[0], '⚪')} Verdict: **{verdict[0]}**")
    out.append("")
    out.append(verdict[1])
    out.append("")

    # Validation metrics
    out.append("## Validation metrics (best in iter)")
    out.append("")
    out.append("| Metric | Value | Reference |")
    out.append("|--------|------:|----------:|")
    out.append(f"| Net-level MAPE     | {rec.get('net_mape', float('nan')):.2f}% | "
               f"v10b: {V10B_BASELINE_MAPE}% |")
    out.append(f"| Total SMAPE        | {rec.get('val_smape_tot', float('nan')):.2f}% | — |")
    out.append(f"| GND SMAPE          | {rec.get('val_smape_gnd', float('nan')):.2f}% | — |")
    out.append(f"| CPL SMAPE          | {rec.get('val_smape_cpl', float('nan')):.2f}% | "
               f"v10b: {V10B_BASELINE_CPL}% |")
    out.append(f"| Composite Score    | {rec.get('composite', float('nan')):.2f} | — |")
    out.append(f"| Step               | {rec.get('step', '?')} | — |")
    out.append("")

    # DS-PINN health
    out.append("## DS-PINN activation health (latest probe in iter)")
    out.append("")
    last_probe = next((r for r in reversed(records_in_iter) if 'zmacro_mean' in r), None)
    if last_probe:
        out.append("| Indicator | Value | Diagnostic |")
        out.append("|-----------|------:|------------|")
        zm_mean = last_probe['zmacro_mean']
        out.append(f"| Z_macro \\|mean\\| | {zm_mean:.3e} | "
                   f"{'⚠ near zero' if zm_mean < 1e-3 else '✓ engaging'} |")
        out.append(f"| Z_macro \\|max\\|  | {last_probe['zmacro_max']:.3e} | — |")
        out.append(f"| Z_macro std       | {last_probe['zmacro_std']:.3e} | — |")
        pw = last_probe['proj_w']
        out.append(f"| proj_out \\|W\\|   | {pw:.3f} | "
                   f"{'⚠ frozen at 0' if pw < 1e-3 else '✓ trained'} |")
        out.append(f"| proj_out \\|b\\|   | {last_probe['proj_b']:.3f} | — |")
        out.append(f"| FNO[0] spec \\|W\\| | {last_probe['fno_spec']:.3f} | "
                   f"{'⚠ stagnant' if last_probe['fno_spec'] < 0.1 else '✓ trained'} |")
        out.append(f"| FNO[0] skip \\|W\\| | {last_probe['fno_skip']:.3f} | — |")
    else:
        out.append("_No DS-PINN probe records found yet._")
    out.append("")

    # Loss decomposition (last record in iter)
    out.append("## Loss decomposition (last step in iter)")
    out.append("")
    last_train = next((r for r in reversed(records_in_iter) if 'train_loss' in r), None)
    if last_train:
        out.append("| Component | Value | Weight |")
        out.append("|-----------|------:|------:|")
        out.append(f"| total      | {last_train.get('train_loss', float('nan')):.3f} | — |")
        out.append(f"| scale (log Y_total) | {last_train.get('loss_scale', float('nan')):.3f} | 3.0 |")
        out.append(f"| cpl_total  | {last_train.get('loss_cpl_total', float('nan')):.3f} | 1.0 |")
        out.append(f"| distribution | {last_train.get('loss_distribution', float('nan')):.3f} | 0.10 |")
        out.append(f"| gnd_direct | {last_train.get('loss_gnd', float('nan')):.3f} | "
                   f"{last_train.get('w_gnd', 0):.2f} |")
        out.append(f"| cpl_direct | {last_train.get('loss_cpl_direct', float('nan')):.3f} | "
                   f"{last_train.get('w_cpl_direct', 0):.2f} |")
        if 'loss_aux_macro' in last_train:
            out.append(f"| aux_macro  | {last_train.get('loss_aux_macro', float('nan')):.3f} | "
                       f"{last_train.get('w_aux_macro', 0):.2f} |")
    out.append("")

    # Trajectory snapshot
    out.append("## Validation trajectory in iter")
    out.append("")
    val_records = [r for r in records_in_iter if 'net_mape' in r]
    if val_records:
        out.append("| Step | Tot SMAPE | GND SMAPE | CPL SMAPE | Net MAPE | Best? |")
        out.append("|-----:|----------:|----------:|----------:|---------:|:-----:|")
        for r in val_records:
            mark = '🌟' if r.get('is_best') else ''
            out.append(f"| {r.get('step', '?')} | "
                       f"{r.get('val_smape_tot', float('nan')):.1f} | "
                       f"{r.get('val_smape_gnd', float('nan')):.1f} | "
                       f"{r.get('val_smape_cpl', float('nan')):.1f} | "
                       f"{r.get('net_mape', float('nan')):.2f}% | {mark} |")
    out.append("")

    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--iter', type=int, default=None,
                    help='Specific iteration to report on. Default: latest with val data.')
    ap.add_argument('--output_root', default='/home/jslee/projects/PINNPEX/output_intel22')
    args = ap.parse_args()

    output_root = Path(args.output_root)
    al_dir = output_root / 'active_learning' / args.model_name
    log_path = output_root / f'al_{args.model_name}.log'
    if not log_path.exists():
        # Fallback: any log file matching
        candidates = list(output_root.glob(f'al_{args.model_name}*.log'))
        if candidates:
            log_path = candidates[0]
    if not log_path.exists():
        print(f"❌ Log file not found: tried {output_root / f'al_{args.model_name}.log'}")
        sys.exit(1)

    records = parse_log(log_path)
    if not records:
        print(f"⚠ No step records parsed from {log_path}")
        sys.exit(1)

    # Determine iterations present
    iters_present = sorted({r.get('iteration', 0) for r in records})
    target_iter = args.iter if args.iter is not None else iters_present[-1]

    in_iter = [r for r in records if r.get('iteration') == target_iter]
    rec = best_at_iter(records, target_iter)

    # Compute verdict
    if target_iter == 1:
        verdict = verdict_for_iter1(rec)
    elif target_iter >= 3:
        verdict = verdict_for_iter3(rec)
    else:
        verdict = ("INFO", f"Iter {target_iter} is too early for a verdict; "
                           "Gate-1 evaluates at iter 1, Gate-2 at iter 3.")

    md = render_markdown(args.model_name, target_iter, rec, in_iter, verdict)
    al_dir.mkdir(parents=True, exist_ok=True)
    md_path   = al_dir / f"report_iter{target_iter}.md"
    json_path = al_dir / f"metrics_iter{target_iter}.json"
    md_path.write_text(md)
    payload = {
        'model_name': args.model_name,
        'iteration': target_iter,
        'iters_present': iters_present,
        'verdict': verdict[0],
        'verdict_msg': verdict[1],
        'best_record': rec,
        'last_train': next((r for r in reversed(in_iter) if 'train_loss' in r), None),
        'last_probe': next((r for r in reversed(in_iter) if 'zmacro_mean' in r), None),
        'val_trajectory': [r for r in in_iter if 'net_mape' in r],
        'baselines': {'v10b_mape': V10B_BASELINE_MAPE, 'v10b_cpl': V10B_BASELINE_CPL},
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(md)
    print(f"\nReport written to: {md_path}")
    print(f"Metrics JSON:      {json_path}")
    print(f"\nVerdict: {verdict[0]} — {verdict[1]}")


if __name__ == '__main__':
    main()
