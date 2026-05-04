#!/usr/bin/env python3
"""
30_per_net_pin_caps.py — Aggregate Liberty pin caps to per-net features.

For each (design, signal_net), enumerate DEF instance-pin tuples, look up
cell-pin capacitance from `lib_pin_caps.parquet` (Liberty), and emit
per-net aggregates:
    pin_cap_total       Σ pin_cap (input + output, mostly input)
    pin_cap_input_total Σ pin_cap restricted to direction=input
    pin_cap_max         max pin cap on the net
    pin_cap_mean        mean pin cap (over matched pins)
    n_pins_lib_matched  count of pins with lib lookup hit
    n_input_pins_lib    count restricted to input
    n_output_pins_lib   count restricted to output

Output: per-design `pin_cap_<design>.parquet` + concatenated
`pin_caps_all_designs.parquet`. Join into v3 features for retrain.

Reuses DEF parsing logic from sister
`r_analytic_v3/scripts/build_features_v6_signal_obs.py`.
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

import configs.config as cfg


COMP_RE = re.compile(r"^-\s+(\S+)\s+(\S+)")
PIN_RE = re.compile(r"\(\s*(\S+)\s+(\S+)\s*\)")


def parse_def_components(def_path: Path) -> dict[str, str]:
    """Parse DEF COMPONENTS section → instance → cell_type."""
    out = {}
    in_comp = False
    with open(def_path, errors="ignore") as f:
        for line in f:
            s = line.rstrip()
            stripped = s.strip()
            if not in_comp:
                if s.startswith("COMPONENTS "):
                    in_comp = True
                continue
            if s.startswith("END COMPONENTS"):
                break
            m = COMP_RE.match(stripped)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def parse_signal_net_pin_lists(def_path: Path) -> dict[str, list[tuple[str, str]]]:
    """Parse DEF NETS → net_name → list of (instance, pin) tuples."""
    out = {}
    in_nets = False
    cur_net = None
    cur_pins = []
    seen_routed = False
    with open(def_path, errors="ignore") as f:
        for line in f:
            s = line.rstrip()
            if not in_nets:
                if s.startswith("NETS "):
                    in_nets = True
                continue
            if s.startswith("END NETS"):
                if cur_net is not None:
                    out[cur_net] = cur_pins
                break
            stripped = s.strip()
            if stripped.startswith("- "):
                if cur_net is not None:
                    out[cur_net] = cur_pins
                cur_net = stripped[2:].split()[0]
                cur_pins = []
                seen_routed = False
            elif cur_net is not None and not seen_routed:
                if any(stripped.startswith(p) for p in
                        ("+ ROUTED", "+ FREQUENCY", "+ SOURCE", "+ USE")):
                    seen_routed = True
                else:
                    cur_pins.extend(PIN_RE.findall(stripped))
    return out


def build_per_net_pin_caps(
    def_path: Path,
    comp_map: dict[str, str],
    net_pin_map: dict[str, list[tuple[str, str]]],
    pin_cap_lookup: dict[tuple[str, str], dict],
) -> pd.DataFrame:
    """For each net, sum pin caps over its (instance, pin) connections."""
    rows = []
    for net_name, pin_tuples in net_pin_map.items():
        caps_input = []
        caps_all = []
        n_input = 0
        n_output = 0
        n_matched = 0
        for inst, pin in pin_tuples:
            if inst.upper() == "PIN":
                continue
            cell_type = comp_map.get(inst)
            if cell_type is None:
                continue
            key = (cell_type, pin.lower())
            entry = pin_cap_lookup.get(key)
            if entry is None:
                continue
            n_matched += 1
            cap = entry.get("capacitance")
            direction = entry.get("direction")
            if cap is not None:
                caps_all.append(cap)
            if direction == "input":
                n_input += 1
                if cap is not None:
                    caps_input.append(cap)
            elif direction == "output":
                n_output += 1
        rows.append({
            "net_name": net_name,
            "pin_cap_total": float(sum(caps_all)) if caps_all else 0.0,
            "pin_cap_input_total": float(sum(caps_input)) if caps_input else 0.0,
            "pin_cap_max": float(max(caps_all)) if caps_all else 0.0,
            "pin_cap_mean": float(np.mean(caps_all)) if caps_all else 0.0,
            "n_pins_lib_matched": int(n_matched),
            "n_input_pins_lib": int(n_input),
            "n_output_pins_lib": int(n_output),
        })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lib-pin-caps", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/lib_pin_caps.parquet"),
    )
    p.add_argument(
        "--def-dir", type=Path,
        default=cfg.RAW_DIR,
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/pin_caps_per_net"),
    )
    p.add_argument(
        "--designs", nargs="*", default=None,
        help="design list (default: TRAIN+TEST DEFs from cfg)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> Loading Liberty pin caps: {args.lib_pin_caps}")
    lib_df = pd.read_parquet(args.lib_pin_caps)
    print(f"  {len(lib_df):,} pins, {lib_df.cell_type.nunique():,} cells")
    pin_cap_lookup = {
        (r.cell_type, r.pin_name.lower()): {
            "capacitance": r.capacitance, "direction": r.direction
        }
        for _, r in lib_df.iterrows()
    }
    print(f"  lookup table: {len(pin_cap_lookup):,} (cell, pin) keys")

    def_paths = []
    if args.designs:
        for d in args.designs:
            p = args.def_dir / f"{d}.def"
            if p.exists():
                def_paths.append(p)
    else:
        # All TRAIN + TEST defs
        for p in list(cfg.TRAIN_DEFS) + list(cfg.TEST_DEFS):
            if p.exists():
                def_paths.append(p)
    print(f">>> {len(def_paths)} DEFs to process")

    all_rows = []
    for def_path in def_paths:
        design = def_path.stem
        out_path = args.out_dir / f"{design}.parquet"
        if out_path.exists():
            print(f"  {design}: cached, loading existing")
            df = pd.read_parquet(out_path)
        else:
            print(f"  {design}: parsing DEF ...", flush=True)
            comp_map = parse_def_components(def_path)
            net_pin_map = parse_signal_net_pin_lists(def_path)
            df = build_per_net_pin_caps(def_path, comp_map, net_pin_map, pin_cap_lookup)
            df.to_parquet(out_path, index=False)
            print(f"    {design}: {len(df):,} nets, "
                  f"avg pin_cap_total={df.pin_cap_total.mean():.3f} fF, "
                  f"avg n_input={df.n_input_pins_lib.mean():.1f}")
        df["design_name"] = design
        all_rows.append(df)

    all_df = pd.concat(all_rows, ignore_index=True)
    all_path = args.out_dir / "pin_caps_all_designs.parquet"
    all_df.to_parquet(all_path, index=False)
    print()
    print(f"✅ {all_path}: {len(all_df):,} nets total")
    print(f"  designs: {all_df.design_name.nunique()}")
    print(f"  total pin cap distribution:")
    print(f"    median {all_df.pin_cap_total.median():.3f} fF  "
          f"mean {all_df.pin_cap_total.mean():.3f}  "
          f"P95 {all_df.pin_cap_total.quantile(0.95):.3f}  "
          f"max {all_df.pin_cap_total.max():.3f}")


if __name__ == "__main__":
    main()
