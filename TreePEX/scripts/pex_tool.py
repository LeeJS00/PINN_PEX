"""pex_tool.py — TreePEX orchestrator: features → SPEF → golden compare.

Run end-to-end pipeline for one design.

  python pex_tool.py --design intel22_tv80s_f3
  python pex_tool.py --design intel22_nova_f3
  python pex_tool.py --all                 # both test designs
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
REPORTS = ROOT / "outputs" / "reports"
PY = "/tool/etc/python/install/3.11.9/bin/python3"

TEST_DESIGNS = ["intel22_tv80s_f3", "intel22_nova_f3"]


def run_one(design: str) -> dict:
    print(f"\n========== TreePEX ▶ {design} ==========\n", flush=True)
    timings = {"design": design}

    # Stage 1: inference
    t0 = time.time()
    rc = subprocess.run([PY, str(SCRIPTS / "02_inference.py"), "--design", design]).returncode
    timings["stage1_inference_s"] = round(time.time() - t0, 3)
    if rc != 0:
        timings["error"] = f"stage1 rc={rc}"; return timings

    # Stage 2: SPEF write
    t0 = time.time()
    rc = subprocess.run([PY, str(SCRIPTS / "03_write_spef.py"), "--design", design]).returncode
    timings["stage2_spef_write_s"] = round(time.time() - t0, 3)
    if rc != 0:
        timings["error"] = f"stage2 rc={rc}"; return timings

    # Stage 3: compare to golden
    t0 = time.time()
    rc = subprocess.run([PY, str(SCRIPTS / "04_compare_golden.py"), "--design", design]).returncode
    timings["stage3_compare_s"] = round(time.time() - t0, 3)
    if rc != 0:
        timings["error"] = f"stage3 rc={rc}"; return timings

    # Read report
    rep_json = REPORTS / f"{design}_report.json"
    if rep_json.exists():
        rep = json.loads(rep_json.read_text())
        timings.update({k: rep[k] for k in ("MAPE_tot_med", "MAPE_gnd_med",
                                              "MAPE_cpl_med", "R2_tot", "R2_gnd",
                                              "R2_cpl", "n_nets_compared",
                                              "spef_roundtrip_max_abs_err_fF")})
    timings["status"] = "OK"
    return timings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", type=str, default=None)
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    if args.all:
        designs = TEST_DESIGNS
    elif args.design:
        designs = [args.design]
    else:
        print("Provide --design <name> or --all"); return 1

    summaries = []
    for d in designs:
        s = run_one(d)
        summaries.append(s)

    print("\n========== SUMMARY ==========")
    for s in summaries:
        if "error" in s:
            print(f"  {s['design']}: ERROR — {s['error']}")
            continue
        wall = s["stage1_inference_s"] + s["stage2_spef_write_s"] + s["stage3_compare_s"]
        print(f"  {s['design']}: tot_med={s['MAPE_tot_med']:.3f}%  gnd={s['MAPE_gnd_med']:.2f}%  "
              f"cpl={s['MAPE_cpl_med']:.2f}%  R²={s['R2_tot']:.4f}  "
              f"n={s['n_nets_compared']:,}  total_wall={wall:.2f}s "
              f"(infer={s['stage1_inference_s']:.2f}s + spef={s['stage2_spef_write_s']:.2f}s + cmp={s['stage3_compare_s']:.2f}s)")
    out_json = REPORTS / "tool_summary.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"\n>>> wrote {out_json}")


if __name__ == "__main__":
    raise SystemExit(main())
