"""Phase 5e — v4 features: aggregate cell-pin routing into signal-net features.

Discovery: DEF NETS routes only signal-level wires; the routing INSIDE each
standard cell (from cell pin shape to first via) lives in the cell LEF
and is reflected in golden RES but missing from our DEF-only features.

PINNPEX `DefStreamParser` already emits PIN/INST_PORT pseudo-nets that
carry these cell-internal segments — we just need to aggregate them into
the signal net they belong to.

Pipeline:
  1. Parse DEF NETS section → for each signal net: list of (inst, pin) tuples.
  2. Run PINNPEX parser to collect EVERY entity's segments, keyed by entity name.
  3. For each signal net, sum the segment contributions of its (inst, pin) entries
     onto its own nsq_M{i} / nvian_<NAME> / etc.

We also keep per-pin metal counts as separate features so the regression
can weight them differently from main-route metal.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/home/jslee/projects/PINNPEX")
sys.path.insert(0, str(PROJ))

from configs import config as cfg  # noqa: E402
from src.preprocessing.def_parser   import DefStreamParser   # noqa: E402
from src.preprocessing.layer_parser import LayerInfoParser   # noqa: E402
from src.preprocessing.lef_parser   import LefParser         # noqa: E402
from src.preprocessing.cell_parser  import CellLibParser     # noqa: E402

import importlib.util
_HERE = Path(__file__).resolve().parent
_V3 = _HERE.parent
_spec = importlib.util.spec_from_file_location(
    "compare_spef_pinpex",
    str(PROJ / "src" / "evaluation" / "compare_spef.py"),
)
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
parse_spef = _mod.parse_spef_with_coordinates

DEF_DIR  = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22")
SPEF_DIR = Path("/home/jslee/projects/PEX_SSL/golden_data/spef_data/intel22")

DESIGNS = [
    "intel22_aes_cipher_top_f3", "intel22_gcd_f3", "intel22_ibex_core_f3",
    "intel22_ldpc_decoder_802_3an_f3",
    "intel22_mc_top_f3", "intel22_spi_top_f3", "intel22_usbf_top_f3",
    "intel22_vga_enh_top_f3", "intel22_wb_conmax_top_f3", "intel22_nova_f3",
    "intel22_tv80s_f3",
]

PIN_RE = re.compile(r"\(\s+(\S+)\s+(\S+)\s+\)")
LAYER_M_RE = re.compile(r"^m(\d+)$", re.IGNORECASE)
VIA_NUM_RE = re.compile(r"^VIA(\d+)_", re.IGNORECASE)


def parse_signal_net_pin_lists(def_path):
    """Per signal net, list of (inst, pin) tuples from DEF NETS header."""
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


def segments_summary(segs):
    """Reduce a segments list (list of dict) to per-layer/per-via counts.

    Handles all segment types yielded by PINNPEX DefStreamParser:
      - WIRE: routed wire from (start, end) with width
      - RECT: small landing patch
      - VIA:  layer-transition via instance (named token)
      - INST_PORT / PIN: cell pin metal pad (rect on layer)
    """
    nsq = defaultdict(float)
    rsq = defaultdict(float)
    pin_nsq = defaultdict(float)   # pin-pad nsq (separate so regression can weight)
    via_name = defaultdict(int)
    n_seg = 0
    n_zero_l = 0
    n_pinpad = 0
    for s in segs:
        t = s["type"]
        if t == "WIRE":
            m = LAYER_M_RE.match(s["layer"].lower())
            if not m:
                continue
            lvl = int(m.group(1))
            p1, p2 = s["start"], s["end"]
            L = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
            W = max(float(s["width"]), 1e-6)
            if L == 0:
                n_zero_l += 1
                continue
            nsq[lvl] += L / W
            n_seg += 1
        elif t == "RECT":
            m = LAYER_M_RE.match(s["layer"].lower())
            if not m:
                continue
            lvl = int(m.group(1))
            x1,y1,x2,y2 = s["rect"]
            dx, dy = abs(x2-x1), abs(y2-y1)
            if dx <= 0 or dy <= 0:
                continue
            rsq[lvl] += max(dx, dy) / max(min(dx, dy), 1e-6)
        elif t == "VIA":
            via_name[s["name"]] += 1
        elif t in ("INST_PORT", "PIN"):
            # cell pin pad — rectangular metal shape on a layer
            m = LAYER_M_RE.match(s["layer"].lower())
            if not m:
                continue
            lvl = int(m.group(1))
            x1, y1, x2, y2 = s["pos"]
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            if dx <= 0 or dy <= 0:
                continue
            # treat as one square (= L/W with L=longer dim, W=shorter)
            pin_nsq[lvl] += max(dx, dy) / max(min(dx, dy), 1e-6)
            n_pinpad += 1
    return {"nsq": dict(nsq), "rsq": dict(rsq), "via_name": dict(via_name),
            "pin_nsq": dict(pin_nsq), "n_seg": n_seg, "n_zero_l": n_zero_l,
            "n_pinpad": n_pinpad}


def parser_resources():
    return (LayerInfoParser(cfg.LAYERS_INFO_PATH).parse(),
            LefParser(cfg.TECH_LEF_PATH).parse(),
            CellLibParser(cfg.CELL_LEF_PATH).parse())


def features_for_design(def_path, parser_res):
    layer_map, tech_lef, cell_lib = parser_res

    # Step A: DEF NETS pin lists per signal net
    print(f"    parsing pin lists ...", flush=True)
    sig_to_pins = parse_signal_net_pin_lists(def_path)
    print(f"    {len(sig_to_pins)} signal nets, "
          f"{sum(len(p) for p in sig_to_pins.values())} pin tuples")

    # Step B: PINNPEX parser → all entities (signal nets, pins, inst_ports)
    parser = DefStreamParser(str(def_path), layer_map, tech_lef, cell_lib)
    entity_segments = {}    # entity_name -> segments list
    for entity_name, _, segs in parser.parse():
        if not segs:
            continue
        entity_segments[entity_name] = segs

    # Step C: Build prefix index. PINNPEX naming convention (def_parser.py):
    #   INST_PORT segment: yielded as f"INST_PORT_{name}" where name itself is
    #     "INST_PORT_<inst>_<pin>" → final key = "INST_PORT_INST_PORT_<inst>_<pin>"
    #     Multiple cuboids per pin → multiple yields (one entity each).
    #   PIN segment: yielded as f"PIN_{name}" with name "PIN_<pin>_<idx>"
    #     → final key = "PIN_PIN_<pin>_<idx>".
    # We match by PREFIX since multiple entities share the same (inst,pin).
    inst_port_prefix_index = defaultdict(list)  # (inst,pin) -> [entity names]
    pin_prefix_index = defaultdict(list)         # pin -> [entity names]
    for ent in entity_segments:
        if ent.startswith("INST_PORT_INST_PORT_"):
            # parse <inst>_<pin> after the double prefix
            tail = ent[len("INST_PORT_INST_PORT_"):]
            inst_port_prefix_index[tail].append(ent)
        elif ent.startswith("PIN_PIN_"):
            tail = ent[len("PIN_PIN_"):]
            # strip trailing "_<index>" if any
            base = re.sub(r"_\d+$", "", tail)
            pin_prefix_index[base].append(ent)

    # Step D: build per-signal-net features
    rows = []
    for sig_net, pin_tuples in sig_to_pins.items():
        if sig_net not in entity_segments:
            continue
        # base features from signal-net's own segments
        base = segments_summary(entity_segments[sig_net])
        nsq_pin = defaultdict(float)
        rsq_pin = defaultdict(float)
        via_pin = defaultdict(int)
        n_pin_matched = 0
        for inst, pin in pin_tuples:
            matched_entities = []
            if inst.upper() == "PIN":
                # top-level chip pin
                matched_entities = pin_prefix_index.get(pin, [])
            else:
                # try exact key first
                key_exact = f"{inst}_{pin}"
                matched_entities = inst_port_prefix_index.get(key_exact, [])
            for ent_name in matched_entities:
                psum = segments_summary(entity_segments[ent_name])
                # cell-pin segments come back as pin_nsq (INST_PORT/PIN type)
                for k, v in psum["pin_nsq"].items():
                    nsq_pin[k] += v
                # in case the entity also has WIRE/RECT/VIA (unusual)
                for k, v in psum["nsq"].items():
                    nsq_pin[k] += v
                for k, v in psum["rsq"].items():
                    rsq_pin[k] += v
                for k, v in psum["via_name"].items():
                    via_pin[k] += v
            if matched_entities:
                n_pin_matched += 1

        row = {"net_name": sig_net,
               "n_segments": base["n_seg"],
               "n_zero_l_wire": base["n_zero_l"],
               "n_pins_total": len(pin_tuples),
               "n_pins_matched": n_pin_matched,
               "one": 1.0}
        for k, v in base["nsq"].items():
            row[f"nsq_M{k}"] = float(v)
        for k, v in base["rsq"].items():
            row[f"rsq_M{k}"] = float(v)
        for k, v in base["via_name"].items():
            row[f"nvian_{k}"] = int(v)
        # cell-pin (sub-net) contributions (separate columns)
        for k, v in nsq_pin.items():
            row[f"pin_nsq_M{k}"] = float(v)
        for k, v in rsq_pin.items():
            row[f"pin_rsq_M{k}"] = float(v)
        for k, v in via_pin.items():
            row[f"pin_nvian_{k}"] = int(v)
        rows.append(row)

    df = pd.DataFrame(rows).fillna(0.0)
    return df


def main():
    pres = parser_resources()
    cache = _V3 / "cache"
    for d in DESIGNS:
        out = cache / f"feat_v4_{d}.parquet"
        if out.exists():
            print(f"  [cache hit] {d}")
            continue
        print(f"  parsing {d} ...", flush=True)
        df = features_for_design(DEF_DIR / f"{d}.def", pres)
        try:
            g = parse_spef(SPEF_DIR / f"{d}_starrc.spef")
            gR = pd.DataFrame([{"net_name": n, "R_gold": float(info["total_res"])}
                                 for n, info in g.items()])
            df = df.merge(gR, on="net_name", how="left")
        except Exception as e:
            print(f"    [warn] golden missing for {d}: {e}")
        df["design"] = d
        df.to_parquet(out)
        print(f"    n_nets={len(df)}  cols={len(df.columns)}  "
              f"avg pins matched={df['n_pins_matched'].mean():.2f}/"
              f"{df['n_pins_total'].mean():.2f}")


if __name__ == "__main__":
    main()
