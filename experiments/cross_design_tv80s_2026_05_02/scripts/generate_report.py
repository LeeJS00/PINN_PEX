"""Generate final markdown report combining all results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))

from configs import cfg
from src.data_loader import mape_per_net


def collect_test_csvs(roots) -> dict:
    """Returns {model_tag: dataframe with y_true, y_pred} keyed by parent_dir/seed."""
    out = {}
    for root in roots:
        rp = Path(root)
        if not rp.exists():
            continue
        for csv in sorted(rp.rglob("*__test.csv")):
            tag = f"{csv.parent.name}/{csv.stem.replace('__test','')}"
            try:
                df = pd.read_csv(csv)
                if {"y_true","y_pred"}.issubset(df.columns):
                    out[tag] = df
            except Exception:
                continue
    return out


def per_model_summary(test_csvs):
    rows = []
    for tag, df in test_csvs.items():
        ape = 100.0 * np.abs(df["y_pred"] - df["y_true"]) / np.maximum(df["y_true"], 1e-3)
        rows.append({
            "tag": tag,
            "n":  int(len(ape)),
            "mape_mean":   float(ape.mean()),
            "mape_median": float(ape.median()),
            "mape_p90":    float(np.percentile(ape, 90)),
            "mape_p99":    float(np.percentile(ape, 99)),
        })
    return pd.DataFrame(rows).sort_values("mape_mean")


def make_ensemble(test_csvs, val_csvs):
    """Aggregate and produce mean/median/blend ensembles."""
    if not test_csvs:
        return {}
    tags = sorted(test_csvs.keys())
    base = test_csvs[tags[0]][["design_name","net_name","y_true"]].copy() \
              if "design_name" in test_csvs[tags[0]].columns else test_csvs[tags[0]][["y_true"]].copy()
    for t in tags:
        base[t] = test_csvs[t]["y_pred"].values
    P = base[tags].to_numpy()
    yt = base["y_true"].to_numpy()
    out = {}
    out["ENS_mean"]   = pd.DataFrame({"y_true": yt, "y_pred": np.nanmean(P, axis=1)})
    out["ENS_median"] = pd.DataFrame({"y_true": yt, "y_pred": np.nanmedian(P, axis=1)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+",
                    default=[str(cfg.OUTPUT_DIR / "final_pipe"),
                             str(cfg.OUTPUT_DIR / "resmlp_v2"),
                             str(cfg.OUTPUT_DIR / "mlp_hand_v2")])
    ap.add_argument("--report", default=str(cfg.REPORTS_DIR / "FINAL_REPORT.md"))
    args = ap.parse_args()

    test_csvs = collect_test_csvs(args.roots)
    print(f"Found {len(test_csvs)} test CSVs")
    summary = per_model_summary(test_csvs)
    print(summary.to_string(index=False))

    # Ensembles
    ensembles = make_ensemble(test_csvs, None)
    ens_summary = []
    for k, df in ensembles.items():
        ape = 100.0 * np.abs(df["y_pred"] - df["y_true"]) / np.maximum(df["y_true"], 1e-3)
        ens_summary.append({"tag": k, "n": int(len(ape)),
                            "mape_mean": float(ape.mean()),
                            "mape_median": float(ape.median()),
                            "mape_p90": float(np.percentile(ape, 90)),
                            "mape_p99": float(np.percentile(ape, 99))})
    ens_df = pd.DataFrame(ens_summary).sort_values("mape_mean")
    print()
    print(ens_df.to_string(index=False))

    # Save
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(Path(args.report).parent / "per_model_summary.csv", index=False)
    ens_df.to_csv(Path(args.report).parent / "ensemble_summary.csv", index=False)

    # Markdown report
    lines = [
        "# Cross-design tv80s — Final Report",
        "",
        f"Generated from `{', '.join(args.roots)}`",
        "",
        f"**Total models evaluated:** {len(test_csvs)}",
        "",
        "## Setup",
        "- Train designs: 8 small intel22 chips (aes_cipher_top, gcd, ibex_core, ldpc_decoder_802_3an, mc_top, spi_top, usbf_top, vga_enh_top, wb_conmax_top — minus whichever served as val).",
        "- Validation: nova (or fallback) — used for early stopping.",
        "- Test: tv80s (full chip, all reachable nets).",
        "- Features: 114 hand-engineered (geometry, layer-aware, coupling, power shielding, analytic compact estimate). All SPEF-derived columns dropped to prevent label leakage.",
        "",
        "## Per-model MAPE (sorted by mean)",
        "",
        "| tag | n | mape_mean | mape_median | mape_p90 | mape_p99 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in summary.head(30).iterrows():
        lines.append(f"| `{r.tag}` | {r.n} | {r.mape_mean:.3f}% | {r.mape_median:.3f}% | {r.mape_p90:.2f}% | {r.mape_p99:.2f}% |")

    lines += ["", "## Ensembles", "", "| tag | n | mape_mean | mape_median | mape_p90 | mape_p99 |", "|---|---|---|---|---|---|"]
    for _, r in ens_df.iterrows():
        lines.append(f"| `{r.tag}` | {r.n} | {r.mape_mean:.3f}% | {r.mape_median:.3f}% | {r.mape_p90:.2f}% | {r.mape_p99:.2f}% |")

    lines += [
        "",
        "## Notes",
        "",
        "- MAPE is per-net, computed on tv80s only. Floor is 1e-3 fF (effectively all nets included).",
        "- ENS_mean = arithmetic mean of all model preds. ENS_median = median.",
    ]
    Path(args.report).write_text("\n".join(lines))
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
