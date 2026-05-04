#!/usr/bin/env python3
"""
29_parse_liberty_pin_caps.py — Liberty .lib parser for cell pin capacitance.

Hint from user (2026-05-03): `tool/pdk/22nm/lib/*.lib` contains per-cell
per-pin Liberty data with intrinsic input pin capacitance — direct C_gnd
contribution that DEF/LEF features miss entirely.

Per-pin Liberty data:
    capacitance       : (fF) input pin gate+overlap+parasitic cap
    fall/rise_capacitance : (fF) more precise variant
    direction         : input | output | inout
    related_ground_pin: vssx (which power rail)

For each cell macro, extract all input pins + their capacitances. Output:
parquet `cell_pin_caps.parquet` with (cell_type, pin_name, direction,
capacitance, fall_capacitance, rise_capacitance).

Then per-net (in v3 features):
    pin_cap_total       = Σ capacitance over connected input pins
    pin_cap_max         = max
    pin_cap_mean        = mean
    n_input_pins_lib    = count of input pins (non-power)

These directly contribute to gnd cap (cell input capacitance loads ≡
C_gnd component reviewers care about).
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--lib", type=Path,
        default=Path("/home/jslee/projects/PINNPEX/tool/pdk/22nm/lib/b15_nn_p1222_2x1r2_tttt_0.750v_25c_nldm.lib"),
        help="Liberty file (default: tttt 0.75v 25C nominal)"
    )
    p.add_argument(
        "--out", type=Path,
        default=Path("/data/PINNPEX/data/processed_v3/intel22/lib_pin_caps.parquet"),
    )
    return p.parse_args()


CELL_RE = re.compile(r"^\s*cell\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*\{\s*$")
PIN_RE = re.compile(r"^\s*pin\s*\(\s*([A-Za-z0-9_\[\]\\:]+)\s*\)\s*\{\s*$")
CAP_RE = re.compile(r"^\s*capacitance\s*:\s*([0-9.eE+-]+)\s*;\s*$")
FALL_CAP_RE = re.compile(r"^\s*fall_capacitance\s*:\s*([0-9.eE+-]+)\s*;\s*$")
RISE_CAP_RE = re.compile(r"^\s*rise_capacitance\s*:\s*([0-9.eE+-]+)\s*;\s*$")
DIR_RE = re.compile(r"^\s*direction\s*:\s*([a-z_]+)\s*;\s*$")
GND_RE = re.compile(r"^\s*related_ground_pin\s*:\s*([A-Za-z0-9_]+)\s*;\s*$")


def parse_liberty(lib_path: Path) -> pd.DataFrame:
    """Streaming parser. Tracks brace depth for cell + pin scopes."""
    rows = []
    cur_cell = None
    cur_pin = None
    cur_pin_data: dict = {}
    depth = 0   # brace depth
    cell_open_depth = None   # depth at which the cell { opened
    pin_open_depth = None    # depth at which the pin { opened

    print(f">>> parsing {lib_path}")
    with open(lib_path, errors="ignore") as f:
        for ln_no, raw in enumerate(f, 1):
            line = raw.rstrip("\n")

            # First update brace depth from this line BEFORE matching scope changes
            # But we need to detect cell/pin opens before depth update.
            cell_m = CELL_RE.match(line)
            pin_m = PIN_RE.match(line)

            if cell_m and cur_pin is None:
                cur_cell = cell_m.group(1)
                cell_open_depth = depth   # before incrementing for this {
            elif pin_m and cur_cell is not None and cur_pin is None:
                cur_pin = pin_m.group(1)
                pin_open_depth = depth
                cur_pin_data = {
                    "cell_type": cur_cell,
                    "pin_name": cur_pin,
                    "direction": None,
                    "capacitance": None,
                    "fall_capacitance": None,
                    "rise_capacitance": None,
                    "related_ground_pin": None,
                }
            elif cur_pin is not None:
                # Inside a pin block: extract attributes
                m = DIR_RE.match(line)
                if m:
                    cur_pin_data["direction"] = m.group(1)
                m = CAP_RE.match(line)
                if m:
                    cur_pin_data["capacitance"] = float(m.group(1))
                m = FALL_CAP_RE.match(line)
                if m:
                    cur_pin_data["fall_capacitance"] = float(m.group(1))
                m = RISE_CAP_RE.match(line)
                if m:
                    cur_pin_data["rise_capacitance"] = float(m.group(1))
                m = GND_RE.match(line)
                if m:
                    cur_pin_data["related_ground_pin"] = m.group(1)

            # Update brace depth
            depth += line.count("{") - line.count("}")

            # Pin closed?
            if cur_pin is not None and depth <= pin_open_depth:
                rows.append(cur_pin_data)
                cur_pin = None
                pin_open_depth = None
                cur_pin_data = {}

            # Cell closed?
            if cur_cell is not None and cur_pin is None and depth <= cell_open_depth:
                cur_cell = None
                cell_open_depth = None

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    args = parse_args()
    df = parse_liberty(args.lib)
    print(f">>> parsed: {len(df):,} pin rows")
    print(f">>> unique cells: {df.cell_type.nunique():,}")
    print(f">>> direction distribution: {df.direction.value_counts().to_dict()}")
    print(f">>> capacitance present: {df.capacitance.notna().sum():,} ({df.capacitance.notna().mean()*100:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f">>> wrote {args.out}")

    print()
    print("=== input pins capacitance stats ===")
    inp = df[df.direction == "input"]
    print(f"  n_input_pins: {len(inp):,}")
    print(f"  capacitance: median {inp.capacitance.median():.4f} fF  "
          f"mean {inp.capacitance.mean():.4f}  "
          f"P5 {inp.capacitance.quantile(0.05):.4f}  "
          f"P95 {inp.capacitance.quantile(0.95):.4f}")
    print()
    print("Sample (first 5 input pins):")
    print(inp.head())


if __name__ == "__main__":
    main()
