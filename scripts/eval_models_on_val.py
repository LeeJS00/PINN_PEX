"""Post-hoc Net-MAPE evaluation on the predefined 1500-net val cache.

Two use cases:
  1. Recover Net-MAPE for runs whose stdout was not redirected
     (the user's m5_v3_baseline_seed{0..3} jobs).
  2. Sanity-check a freshly trained best_model.pth without re-launching
     the full AL pipeline.

Each --model entry is `<model_name>:<flags>` where flags is a comma-sep
list including `dspinn`, `gino`, `calib_path=PATH`. Empty flags = v2 recipe.

Example:
    python3 scripts/eval_models_on_val.py --gpu 0 \\
        --model m5_v3_baseline_seed0:dspinn,calib_path=none \\
        --model m5_v3_baseline_seed1:dspinn,calib_path=none \\
        --out output_intel22/active_learning/diag_phase_a/m5_eval.md
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import configs.config as cfg
from src.active_learning.oracle import FullChipPEXOracle
from src.data.replay_buffer import DesignLevelReplayBuffer
from src.models.neural_field import DeepPEX_Model
from src.preprocessing.layer_parser import LayerInfoParser  # noqa: F401  (parsed inside DeepPEX path)
from src.trainers.finetuner import NeuralFieldFinetuner


VAL_CACHE = REPO_ROOT / "output_intel22/active_learning/cache/predefined_valid_subset.csv"


def parse_model_arg(spec: str) -> dict:
    name, _, flag_str = spec.partition(":")
    flags = {}
    for f in (x.strip() for x in flag_str.split(",") if x.strip()):
        if "=" in f:
            k, v = f.split("=", 1)
            flags[k.strip()] = v.strip()
        else:
            flags[f.strip()] = True
    return {"name": name, "flags": flags}


def build_val_loader(device: str) -> torch.utils.data.DataLoader:
    if not VAL_CACHE.exists():
        raise FileNotFoundError(
            f"Val cache missing at {VAL_CACHE}. Run cache prewarm first."
        )
    val_df = pd.read_csv(VAL_CACHE)
    print(f"[val] loaded cache: {len(val_df)} tiles, "
          f"{val_df.groupby(['design_name', 'net_name']).ngroups} unique nets, "
          f"{val_df['design_name'].nunique()} designs")

    output_dir = Path(cfg.OUTPUT_DIR) / "active_learning"
    oracle = FullChipPEXOracle(output_dir)
    val_buffer = DesignLevelReplayBuffer(max_designs=20)

    def_map = {p.stem: p for p in cfg.TRAIN_DEFS + cfg.TEST_DEFS}
    for d_name in val_df["design_name"].unique():
        d_def_path = def_map.get(d_name)
        if d_def_path is None:
            print(f"[val] skipping design {d_name}: no DEF mapping")
            continue
        d_spef = oracle.generate_golden_spef(d_name, d_def_path)
        val_buffer.add_design(d_name, val_df[val_df["design_name"] == d_name], d_spef)

    return val_buffer.get_dataloader()


def evaluate_one(model_name: str, flags: dict, val_loader, device: str) -> dict:
    cfg._use_gino = "gino" in flags
    cfg._use_dspinn = "dspinn" in flags
    if "calib_path" in flags:
        v = flags["calib_path"]
        if v.lower() in {"none", "null", "off", ""}:
            cfg.CALIBRATION_INIT_PATH = None
        else:
            cfg.CALIBRATION_INIT_PATH = Path(v)
    print(f"[{model_name}] cfg: dspinn={cfg._use_dspinn} gino={cfg._use_gino} "
          f"calib_path={cfg.CALIBRATION_INIT_PATH}")

    model_dir = Path(cfg.OUTPUT_DIR) / "active_learning" / model_name
    ckpt = model_dir / "best_model.pth"
    if not ckpt.exists():
        print(f"[{model_name}] best_model.pth missing at {ckpt}; skipping.")
        return {"name": model_name, "net_mape": None, "note": "missing best_model.pth"}

    model = DeepPEX_Model(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    # Strip torch.compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    current = model.state_dict()
    filtered = {k: v for k, v in state.items()
                if k in current and v.shape == current[k].shape}
    dropped = [k for k in state if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    if dropped:
        print(f"[{model_name}] dropped {len(dropped)} mismatched keys (sample: {dropped[:3]})")

    finetuner = NeuralFieldFinetuner(model, device=device, lr=cfg.AL_LR)
    finetuner.model.eval()
    net_mape = finetuner.compute_net_mape(val_loader)
    print(f"[{model_name}] Net-level MAPE = {net_mape:.2f}%")

    return {"name": model_name, "net_mape": float(net_mape),
            "ckpt": str(ckpt), "flags": flags}


def write_report(rows: list[dict], out_path: Path, started_at: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# Post-hoc Net-MAPE on Val Cache ({started_at})")
    lines.append("")
    lines.append(f"Val cache: `{VAL_CACHE}`")
    lines.append("")
    lines.append("| Model | dspinn | gino | calib | Net MAPE | ckpt |")
    lines.append("|-------|:------:|:----:|:-----:|---------:|------|")
    for r in rows:
        flags = r.get("flags", {}) or {}
        lines.append("| {n} | {ds} | {g} | {c} | {m} | `{ck}` |".format(
            n=r["name"],
            ds="✓" if flags.get("dspinn") else "—",
            g="✓" if flags.get("gino") else "—",
            c=str(flags.get("calib_path", "default")),
            m=(f"{r['net_mape']:.2f}%" if r["net_mape"] is not None else "n/a"),
            ck=r.get("ckpt", "—"),
        ))
    lines.append("")
    valid = [r["net_mape"] for r in rows if r.get("net_mape") is not None]
    if valid:
        import statistics as _s
        m = _s.fmean(valid)
        s = _s.pstdev(valid) if len(valid) > 1 else 0.0
        lines.append(f"**Aggregate:** mean={m:.2f}% ± {s:.2f}% over {len(valid)} models")
        lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"[report] wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", default=[], required=True,
                   help='Repeatable. Format "<name>:<flags>", flags comma-sep '
                        '(dspinn,gino,calib_path=PATH).')
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", type=Path,
                   default=REPO_ROOT / "output_intel22/active_learning/diag_phase_a/post_hoc_eval.md")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    started_at = datetime.now().isoformat(timespec="seconds")

    val_loader = build_val_loader(device=device)

    rows = []
    for spec in args.model:
        m = parse_model_arg(spec)
        try:
            row = evaluate_one(m["name"], m["flags"], val_loader, device)
        except Exception as exc:
            print(f"[{m['name']}] FAILED: {exc!r}")
            row = {"name": m["name"], "net_mape": None,
                   "flags": m["flags"], "note": str(exc)}
        rows.append(row)

    write_report(rows, args.out, started_at)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
