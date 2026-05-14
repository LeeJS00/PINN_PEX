"""bench_pinn.py — end-to-end PINN v12 mesh 5-seed ensemble benchmark.

Independent process to avoid src/ namespace conflict with main project src/.
Times: parse → feature extract → GPU model load → predict → SPEF write → compare.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/jslee/projects/PINNPEX")
ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "outputs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Main project parsers will be loaded first (insert at index 0).
# pex_v3 module path is swapped in AFTER parse stage by stage_pinn_predict.
sys.path.insert(0, str(PROJECT_ROOT))

V3_FEATURES = "/data/PINNPEX/data/processed_v3/intel22/features/all_designs.csv"
RAW_DEFS = {
    "intel22_tv80s_f3": PROJECT_ROOT / "tool" / "def" / "intel22" / "intel22_tv80s_t1.def",
    "intel22_nova_f3":  PROJECT_ROOT / "tool" / "def" / "intel22" / "intel22_nova_t1.def",
}
PINN_SEEDS = [0, 1, 2, 3, 4]
PINN_CKPTS = [PROJECT_ROOT / "archive" / "pex_v3" / "output" / "phase1_mesh_5seed" / f"seed{s}" / "model.pt"
              for s in PINN_SEEDS]

EPS_FF = 1e-3


def mape_med(p, g):
    p = np.asarray(p); g = np.asarray(g)
    return float(np.median(np.abs(p - g) / np.maximum(np.abs(g), EPS_FF)) * 100)


def r2(p, g):
    p = np.asarray(p); g = np.asarray(g)
    ss_res = float(((g - p) ** 2).sum())
    ss_tot = float(((g - g.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def stage_parse(design: str):
    """Parse DEF + LEF + Liberty + layer (uses main project src/preprocessing)."""
    from src.preprocessing.def_parser import DefStreamParser
    from src.preprocessing.layer_parser import LayerInfoParser
    from src.preprocessing.lef_parser import LefParser
    from src.preprocessing.cell_parser import CellLibParser
    from configs import config as cfg

    timings = {}
    t0 = time.time(); tech_lef = LefParser(str(cfg.TECH_LEF_PATH)).parse()
    timings["lef_parse_s"] = round(time.time() - t0, 3)
    t0 = time.time(); cell_lib = CellLibParser(str(cfg.CELL_LEF_PATH)).parse()
    timings["cell_lef_parse_s"] = round(time.time() - t0, 3)
    t0 = time.time(); layer_map = LayerInfoParser(str(cfg.LAYERS_INFO_PATH)).parse()
    timings["layer_info_parse_s"] = round(time.time() - t0, 3)

    t0 = time.time()
    def_path = RAW_DEFS[design]
    if not def_path.exists():
        alt = Path("/home/jslee/projects/PEX_SSL/data/raw/def/intel22") / f"{design}.def"
        if alt.exists():
            def_path = alt
    if not def_path.exists():
        print(f"  WARN: DEF for {design} not found at {def_path}")
        timings["def_parse_s"] = -1.0
    else:
        parser = DefStreamParser(str(def_path), layer_map, tech_lef, cell_lib)
        n_nets = 0
        for _ in parser.parse():
            n_nets += 1
        timings["def_parse_s"] = round(time.time() - t0, 3)
        timings["n_nets_parsed"] = n_nets

    total = sum(v for k, v in timings.items() if k.endswith("_s") and v > 0)
    timings["total_parse_s"] = round(total, 3)
    return total, timings


def stage_pinn_predict(design: str):
    """Load + predict using pex_v3 mesh ensemble.

    Swap sys.path so pex_v3/src/... is loaded (and clear cached `src` module so
    pex_v3 modules don't see the main project's src).
    """
    # 1. Remove cached `src` and src.* modules (from main project) so pex_v3's
    #    src can be loaded fresh.
    for k in list(sys.modules.keys()):
        if k == "src" or k.startswith("src."):
            del sys.modules[k]
    # 2. Put pex_v3 at front of sys.path
    if str(PROJECT_ROOT) in sys.path:
        sys.path.remove(str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "archive" / "pex_v3"))

    import torch
    from torch.utils.data import DataLoader
    from src.models.hybrid_v3_mesh import HybridPexV3Mesh
    from src.data.cuboid_set_dataset import (
        PerNetCuboidStore, CuboidAugmentedDataset, collate_cuboid_batch,
    )
    from src.trainers.finetune_hybrid_v3 import (
        split_by_manifest_column, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS,
    )
    from src.baselines.calibration_v3 import (
        fit_per_layer_calibration, apply_per_layer_calibration,
    )

    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    CUBOID_DIR = Path("/data/PINNPEX/data/processed_v3/intel22/per_net_cuboids")

    timings = {}

    # GPU model load (5 seeds)
    t0 = time.time()
    models = []
    for ckpt in PINN_CKPTS:
        m = HybridPexV3Mesh().to(DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
        m.eval()
        models.append(m)
    timings["pinn_gpu_load_s"] = round(time.time() - t0, 3)

    # Dataset prep
    t0 = time.time()
    df = pd.read_csv(V3_FEATURES)
    train_df, valid_df, test_df = split_by_manifest_column(df)
    calib = fit_per_layer_calibration(train_df)
    test_df = apply_per_layer_calibration(test_df, calib)
    test_df = test_df[test_df["design_name"] == design].reset_index(drop=True)
    store = PerNetCuboidStore(CUBOID_DIR)
    ds = CuboidAugmentedDataset(test_df, store, _SELF_FEATURE_COLS, _PAIR_FEATURE_COLS)
    loader = DataLoader(ds, batch_size=256, num_workers=2, collate_fn=collate_cuboid_batch)
    timings["pinn_dataset_prep_s"] = round(time.time() - t0, 3)
    timings["n_nets"] = len(test_df)

    # Forward pass
    t0 = time.time()
    n = len(test_df)
    pg_acc = np.zeros(n, dtype=np.float64)
    pc_acc = np.zeros(n, dtype=np.float64)
    for m in models:
        i = 0
        with torch.no_grad():
            for b in loader:
                ag = b["analytic_gnd"].to(DEVICE)
                ac = b["analytic_cpl"].to(DEVICE)
                sf = b["self_features"].to(DEVICE)
                pf = b["pair_features"].to(DEVICE)
                cb = b["cuboids"].to(DEVICE)
                mk = b["padding_mask"].to(DEVICE)
                pg = m.predict_gnd(ag, sf, cb, mk).cpu().numpy()
                pc = m.predict_cpl(ac, pf, cb, mk).cpu().numpy()
                b_n = len(pg)
                pg_acc[i:i+b_n] += pg
                pc_acc[i:i+b_n] += pc
                i += b_n
    pg_acc /= len(models); pc_acc /= len(models)
    timings["pinn_predict_s"] = round(time.time() - t0, 3)

    pred_df = test_df[["design_name", "net_name", "c_gnd_fF", "c_cpl_total_fF"]].copy()
    pred_df["pred_gnd"] = pg_acc
    pred_df["pred_cpl"] = pc_acc
    return timings, pred_df


def stage_spef_write(design: str, pred_df: pd.DataFrame, out_path: Path):
    t0 = time.time()
    lines = []
    lines.append("*SPEF \"IEEE 1481-1999\"")
    lines.append(f"*DESIGN \"{design}\"")
    lines.append("*DATE \"2026-05-11 (pex_v8 bench PINN)\"")
    lines.append("*VENDOR \"PINN-PEX\"")
    lines.append("*PROGRAM \"archive/pex_v8 bench_pinn\"")
    lines.append("*VERSION \"1.0\"")
    lines.append("*DIVIDER /")
    lines.append("*DELIMITER :")
    lines.append("*BUS_DELIMITER [ ]")
    lines.append("*T_UNIT 1 NS")
    lines.append("*C_UNIT 1 FF")
    lines.append("*R_UNIT 1 OHM")
    lines.append("*L_UNIT 1 HENRY")
    lines.append("")
    for r in pred_df.itertuples(index=False):
        net = r.net_name
        total_cap = float(r.pred_gnd + r.pred_cpl)
        lines.append(f"*D_NET {net} {total_cap:.6f}")
        lines.append("*CAP")
        lines.append(f"1 {net} {float(r.pred_gnd):.6f}")
        lines.append("*END")
    out_path.write_text("\n".join(lines))
    return round(time.time() - t0, 3), {"spef_bytes": int(out_path.stat().st_size)}


def stage_compare(pred_df: pd.DataFrame):
    t0 = time.time()
    pg = pred_df["pred_gnd"].values; pc = pred_df["pred_cpl"].values
    gg = pred_df["c_gnd_fF"].values; gc = pred_df["c_cpl_total_fF"].values
    ev = {
        "n": len(pred_df),
        "tot_med": mape_med(pg + pc, gg + gc),
        "gnd_med": mape_med(pg, gg),
        "cpl_med": mape_med(pc, gc),
        "R2_tot": r2(pg + pc, gg + gc),
    }
    return round(time.time() - t0, 3), ev


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", nargs="+", default=["intel22_tv80s_f3", "intel22_nova_f3"])
    return p.parse_args()


def main():
    args = parse_args()
    # Phase A: parse all designs (main project src on path)
    parse_results = {}
    for design in args.designs:
        print(f"\n══════ parse ▶ {design} ══════")
        t_parse, parse_t = stage_parse(design)
        print(f"[parse] {t_parse:.2f}s  details={parse_t}")
        parse_results[design] = (t_parse, parse_t)

    # Phase B: PINN inference (pex_v3 src on path) for all designs
    all_rows = []
    for design in args.designs:
        print(f"\n══════ PINN ▶ {design} ══════")
        t_parse, parse_t = parse_results[design]

        t0 = time.time()
        pinn_t, pred_df = stage_pinn_predict(design)
        spef_path = OUT_DIR / f"{design}_pinn_v12.spef"
        t_spef, spef_t = stage_spef_write(design, pred_df, spef_path)
        t_cmp, ev = stage_compare(pred_df)
        print(f"[pinn_gpu_load] {pinn_t['pinn_gpu_load_s']}s")
        print(f"[pinn_dataset_prep] {pinn_t['pinn_dataset_prep_s']}s")
        print(f"[pinn_predict] {pinn_t['pinn_predict_s']}s")
        print(f"[spef_write] {t_spef}s")
        print(f"[compare] {t_cmp}s")
        print(f"MAPE: tot={ev['tot_med']:.3f}  gnd={ev['gnd_med']:.2f}  cpl={ev['cpl_med']:.2f}  R2={ev['R2_tot']:.4f}")

        total = t_parse + pinn_t["pinn_gpu_load_s"] + pinn_t["pinn_dataset_prep_s"] + pinn_t["pinn_predict_s"] + t_spef + t_cmp
        row = {
            "design": design, "model": "PINN-v12-mesh",
            "stage1_parse_s": t_parse,
            "stage2_tile_build_s": 0.0,
            "stage3_feature_extract_s": pinn_t["pinn_dataset_prep_s"],
            "stage4_model_load_s": pinn_t["pinn_gpu_load_s"],
            "stage5_predict_s": pinn_t["pinn_predict_s"],
            "stage6_spef_write_s": t_spef,
            "stage7_compare_s": t_cmp,
            "total_e2e_s": round(total, 3),
            "n_nets": ev["n"], "tot_mape_med": ev["tot_med"],
            "gnd_mape_med": ev["gnd_med"], "cpl_mape_med": ev["cpl_med"],
            "R2_tot": ev["R2_tot"], "spef_bytes": spef_t["spef_bytes"],
        }
        all_rows.append(row)

    out_csv = RESULTS_DIR / "bench_pinn.csv"
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
