#!/usr/bin/env python3
"""
Online inference: DEF + LEF + layer_info → SPEF (no build_dataset.py, no disk tiles).

Usage:
    python3 scripts/run_inference.py \
        --def_path path/to/design.def \
        --model_ckpt output_intel22/active_learning/my_run/best_model.pth \
        --out_spef output.spef \
        [--gpu 0]

Input:  DEF + configs/config.py paths for LEF / layer_info
Output: SPEF with ML-predicted net-level total caps and analytical resistances

No SPEF file needed at inference time.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

import configs.config as cfg
from src.models.neural_field import DeepPEX_Model
from src.preprocessing.def_parser import DefStreamParser
from src.preprocessing.layer_parser import LayerInfoParser
from src.preprocessing.lef_parser import LefParser
from src.preprocessing.cell_parser import CellLibParser
from src.preprocessing.online_context import OnlineContextBuilder
from src.physics.materials import BEOLMaterialStack
from src.data.tensorizer import FeatureTensorizer
from src.utils.spef_writer import RCTopologyBuilder, NetCapWriter, SPEFWriter


# -------------------------------------------------------------------------
# Model loading

def load_model(ckpt_path, device):
    model = DeepPEX_Model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Strip _orig_mod. prefix if present (torch.compile artifacts)
    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    current = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in current and v.shape == current[k].shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model


# -------------------------------------------------------------------------
# Per-net inference: tile tensors → scatter_add → net-level totals

@torch.no_grad()
def predict_net(model, tiles, device, pad_to=cfg.NF_PAD_TO_CUBOIDS):
    """
    tiles: list of dicts from OnlineContextBuilder._make_tile()
    Returns: (C_gnd_total, C_cpl_dict) where C_cpl_dict = {aggressor_net_name: float}

    Note: C_cpl_dict is sparse — only nets that appear as aggressors in tiles.
    For now we return aggregated total coupling (no per-aggressor breakdown at
    inference without a full aggressor name registry). Per-aggressor coupling
    is deferred to a future phase once NameRegistry is wired online.
    """
    if not tiles:
        return 0.0, {}

    total_gnd = 0.0
    total_cpl = 0.0
    weight_sum = 0.0

    with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu', dtype=torch.bfloat16):
        for tile in tiles:
            tensor = torch.from_numpy(tile['tensor']).float()  # (N, 9)
            core_ratios = torch.from_numpy(tile['core_ratios']).float()  # (N,)
            is_target = torch.from_numpy(tile['is_target']).float()  # (N,)

            N = tensor.shape[0]
            if N == 0:
                continue

            # Pad to pad_to
            pad = pad_to - N
            if pad > 0:
                tensor = F.pad(tensor, (0, 0, 0, pad))
                core_ratios = F.pad(core_ratios, (0, pad))
                is_target = F.pad(is_target, (0, pad))
            elif pad < 0:
                tensor = tensor[:pad_to]
                core_ratios = core_ratios[:pad_to]
                is_target = is_target[:pad_to]

            mask = torch.zeros(pad_to, dtype=torch.bool)
            mask[:min(N, pad_to)] = True

            tensor = tensor.unsqueeze(0).to(device)       # (1, pad, 9)
            mask = mask.unsqueeze(0).to(device)            # (1, pad)
            core_ratios = core_ratios.to(device)
            is_target = is_target.to(device)

            preds = model(tensor, mask, compute_coupling=True)

            c_total = preds['c_total_phys'].float().squeeze(0)  # (pad,)
            c_gnd = preds['c_gnd_seg'].float().squeeze(0)       # (pad,)

            weighted_total = (c_total * is_target * core_ratios).sum().item()
            weighted_gnd = (c_gnd * is_target * core_ratios).sum().item()
            w = (is_target * core_ratios).sum().item()

            total_gnd += weighted_gnd
            total_cpl += max(0.0, weighted_total - weighted_gnd)
            weight_sum += w

    return max(0.0, total_gnd), {'__total_cpl__': max(0.0, total_cpl)}


# -------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Online PEX inference: DEF → SPEF (no disk tiles)")
    parser.add_argument('--def_path',    required=True, help="Routed DEF file")
    parser.add_argument('--model_ckpt',  required=True, help="Trained model checkpoint (.pth)")
    parser.add_argument('--out_spef',    required=True, help="Output SPEF path")
    parser.add_argument('--gpu',         type=int, default=cfg.GPU_ID, help="GPU index (-1 for CPU)")
    parser.add_argument('--pad_to',      type=int, default=cfg.NF_PAD_TO_CUBOIDS)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    t0 = time.time()

    # --- 1. Parse tech files ---
    print("Parsing tech files...")
    layer_info = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    tech_lef   = LefParser(cfg.TECH_LEF_PATH).parse()
    cell_lib   = CellLibParser(cfg.CELL_LEF_PATH).parse()
    mat_stack  = BEOLMaterialStack(layer_info)
    tensorizer = FeatureTensorizer(mat_stack)

    # --- 2. Parse DEF (two-pass via OnlineContextBuilder) ---
    print(f"Parsing DEF: {args.def_path}")
    def_parser = DefStreamParser(args.def_path, layer_info, tech_lef, cell_lib)

    ctx = OnlineContextBuilder(tensorizer, window_size=cfg.WINDOW_SIZE)
    ctx.build(def_parser)

    top_ports = [
        (name, 'I' if 'IN' in info.get('direction', '').upper()
               else 'O' if 'OUT' in info.get('direction', '').upper() else 'B')
        for name, info in getattr(def_parser, 'pins', {}).items()
        if info.get('type', 'PIN') == 'PIN'
    ]

    # --- 3. Load model ---
    print(f"Loading model: {args.model_ckpt}")
    model = load_model(args.model_ckpt, device)

    # --- 4. Inference + SPEF write ---
    out_path = Path(args.out_spef)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    design_name = Path(args.def_path).stem

    n_nets = 0
    t_inf = 0.0

    with open(out_path, 'w') as fh:
        writer = SPEFWriter(fh, design_name, top_ports)
        writer.write_header()

        for net_name, tiles in ctx.iter_net_tiles():
            # Rebuild DEF segments for this net from ctx._net_data
            nid = next(k for k, v in ctx._net_data.items() if v['name'] == net_name)
            segments = ctx._net_data[nid]['segments']

            t_s = time.time()
            C_gnd, C_cpl_dict = predict_net(model, tiles, device, args.pad_to)
            t_inf += time.time() - t_s

            # Build RC topology (deterministic from DEF)
            try:
                topology = RCTopologyBuilder(net_name, segments, top_ports, layer_info, tech_lef)
            except Exception as e:
                print(f"  WARN: topology build failed for {net_name}: {e}")
                continue

            # Distribute net-level caps to SPEF nodes
            net_writer = NetCapWriter(topology, C_gnd, C_cpl_dict)
            writer.stream_net_cap_writer(net_writer)
            n_nets += 1

    t_total = time.time() - t0
    print(f"\nDone: {n_nets} nets written to {out_path}")
    print(f"  Total wall time : {t_total:.1f}s")
    print(f"  Model inference : {t_inf:.1f}s")
    print(f"  Output SPEF     : {out_path}")


if __name__ == '__main__':
    main()
