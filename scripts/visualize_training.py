"""
PINN-PEX Training Dashboard
Generates an interactive HTML dashboard from training logs.

Usage:
    python3 scripts/visualize_training.py            # generate output_intel22/dashboard.html
    python3 scripts/visualize_training.py --serve    # also serve on http://localhost:8080
    python3 scripts/visualize_training.py --out /tmp/dashboard.html
"""
import re
import sys
import time
import argparse
import http.server
import threading
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
AL_DIR = PROJECT_ROOT / "output_intel22" / "active_learning"

RUNS = {
    "v6_fivedsgn": {
        "log": AL_DIR / "v6_fivedsgn_train.log",
        "color": "#1f77b4",
        "label": "v6 (8 iters, 10k steps, no physics fix)",
        "status": "✅ Complete",
    },
    "v8_fivedesign": {
        "log": Path("/tmp/v8_fivedesign.log"),
        "color": "#ff7f0e",
        "label": "v8 (fringe_scale + wide_density)",
        "status": "✅ Complete",
    },
    "v8b_railcpl": {
        "log": Path("/tmp/v8b_railcpl.log"),
        "color": "#2ca02c",
        "label": "v8b (fringe + rail_coupling)",
        "status": "✅ Complete",
    },
    "v9": {
        "log": AL_DIR.parent / "al_v9.log",
        "color": "#9467bd",
        "label": "v9 (VSS aggr + 10ch + Sakurai-Tamaru CPL + n_tiles)",
        "status": "🔄 Training",
    },
    "v10": {
        "log": AL_DIR.parent / "al_v10.log",
        "color": "#d62728",
        "label": "v10 (CPL warmup fix + direction-weighted ε_eff + K×3 LR + ibex)",
        "status": "🔄 Training",
    },
    "v10b": {
        "log": AL_DIR.parent / "al_v10b.log",
        "color": "#e377c2",
        "label": "v10b (v10 + lateral/broadside decomp + K×K layer-pair scale)",
        "status": "🔄 Training",
    },
}

EVAL_LOGS = {
    "v6_fivedsgn":  Path("/tmp/eval_v6_fivedsgn.log"),
    "v8b_railcpl":  Path("/tmp/eval_v8b_railcpl.log"),
    "v9":           AL_DIR.parent / "eval_v9.log",
}

# v7 OOD baseline (for comparison reference lines)
V7_REF = {
    "in_dist_mape": 26.63,
    "ood_mape_nova": 48.87,
    "ood_mape_tv80s": 50.51,
    "chip_gnd_nova": 0.290,
    "chip_gnd_tv80s": 0.232,
    "chip_cpl_nova": 0.708,
    "chip_cpl_tv80s": 0.545,
}

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
_P_ITER  = re.compile(r'\[Iteration (\d+)\]')
_P_STEP  = re.compile(
    r'Step (\d+): Train Loss = ([\d.]+).*?loss_gnd: ([\d.]+)(?:.*?loss_cpl_direct: ([\d.]+))?'
)
_P_SMAPE = re.compile(
    r'Validation SMAPE.*?Tot:\s*([\d.]+).*?GND:\s*([\d.]+).*?CPL:\s*([\d.]+)'
)
_P_MAPE  = re.compile(r'Net-level MAPE\s*:\s*([\d.]+)%')
_P_BEST  = re.compile(r'BEST!')
_P_BEST_CKPT = re.compile(r'Current best checkpoint net MAPE:\s*([\d.]+)%')


def parse_log(log_path: Path) -> pd.DataFrame:
    if not log_path or not log_path.exists():
        return pd.DataFrame()

    records = []
    current_iter = 0
    pending: dict = {}

    with open(log_path, errors='replace') as f:
        lines = f.readlines()

    for line in lines:
        m = _P_ITER.search(line)
        if m:
            current_iter = int(m.group(1))
            pending = {}
            continue

        m = _P_STEP.search(line)
        if m:
            pending = {
                'iter': current_iter,
                'step': int(m.group(1)),
                'train_loss': float(m.group(2)),
                'loss_gnd': float(m.group(3)),
                'loss_cpl_direct': float(m.group(4)) if m.group(4) else None,
                'tot_smape': None, 'gnd_smape': None, 'cpl_smape': None,
                'net_mape': None, 'is_best': False,
            }
            continue

        if not pending:
            continue

        m = _P_SMAPE.search(line)
        if m:
            pending['tot_smape'] = float(m.group(1))
            pending['gnd_smape'] = float(m.group(2))
            pending['cpl_smape'] = float(m.group(3))
            continue

        m = _P_MAPE.search(line)
        if m:
            pending['net_mape'] = float(m.group(1))
            continue

        if _P_BEST.search(line) and pending:
            pending['is_best'] = True

        # Commit when we see the Probe Alert (marks end of a step's output)
        if 'Probe Alert' in line and pending.get('net_mape') is not None:
            pending['label'] = f"I{pending['iter']}-{pending['step']}"
            records.append(dict(pending))
            pending = {}

    # Commit any trailing record (current step still running)
    if pending.get('net_mape') is not None:
        pending['label'] = f"I{pending['iter']}-{pending['step']}"
        records.append(dict(pending))

    return pd.DataFrame(records) if records else pd.DataFrame()


def parse_al_csv(run_key: str) -> pd.DataFrame:
    csvs = sorted((AL_DIR / run_key).glob("al_training_log_*.csv"))
    if not csvs:
        return pd.DataFrame()
    try:
        return pd.read_csv(csvs[-1])
    except Exception:
        return pd.DataFrame()


def parse_ood_log(log_path: Path) -> dict:
    """Extract chip-level OOD metrics printed by evaluator.py.

    Handles the actual evaluator output format:
      Design : intel22_nova_f3  ...
      Ground Cap    |  87564.892 |  49029.028 |  0.5599x
      Coupling Cap  | 122063.284 | 106579.598 |  0.8732x
      Total Cap     |    40.581  |   0.6235   |  2.77486
    """
    result = {}
    if not log_path or not log_path.exists():
        return result

    # Replace \r so tqdm progress bars don't interfere
    text = log_path.read_text(errors='replace').replace('\r', '\n')

    # Split into per-design blocks on the Design header line
    blocks = re.split(r'Design\s*:\s*(intel22_\w+)', text)
    # blocks = [pre, design1, block1, design2, block2, ...]
    for i in range(1, len(blocks) - 1, 2):
        design_full = blocks[i]   # e.g. intel22_nova_f3
        block       = blocks[i + 1]

        # Map design name to short key used in result dict
        if 'nova' in design_full:
            key = 'nova'
        elif 'tv80' in design_full:
            key = 'tv80s'
        else:
            continue

        # chip_gnd ratio:  "Ground Cap  | ... | ... | 0.5599x"
        m = re.search(r'Ground Cap\s*\|[^|]+\|[^|]+\|\s*([\d.]+)x?', block, re.IGNORECASE)
        if m:
            result[f'chip_gnd_{key}'] = float(m.group(1))

        # chip_cpl ratio:  "Coupling Cap | ... | ... | 0.8732x"
        m = re.search(r'Coupling Cap\s*\|[^|]+\|[^|]+\|\s*([\d.]+)x?', block, re.IGNORECASE)
        if m:
            result[f'chip_cpl_{key}'] = float(m.group(1))

        # Net-level total MAPE: "Total Cap  |  40.581 | ..."
        m = re.search(r'Total Cap\s*\|\s*([\d.]+)\s*\|', block, re.IGNORECASE)
        if m:
            result[f'mape_{key}'] = float(m.group(1))

    return result


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------
HOVER = "<b>%{customdata[0]}</b><br>MAPE: %{y:.2f}%<br>Iter %{customdata[1]}, Step %{customdata[2]}<extra></extra>"


def fig_mape(dfs: dict) -> go.Figure:
    fig = go.Figure()
    for key, df in dfs.items():
        if df.empty:
            continue
        cfg = RUNS[key]
        best_mape = df['net_mape'].min()
        best_row  = df.loc[df['net_mape'].idxmin()]

        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['net_mape'],
            mode='lines+markers',
            name=f"{cfg['label']}  (best={best_mape:.2f}%)",
            line=dict(color=cfg['color'], width=2),
            marker=dict(
                color=[cfg['color']] * len(df),
                size=[10 if r else 5 for r in df['is_best']],
                symbol=['star' if r else 'circle' for r in df['is_best']],
                line=dict(color='black', width=[1.5 if r else 0 for r in df['is_best']]),
            ),
            customdata=list(zip(df['label'], df['iter'], df['step'])),
            hovertemplate=HOVER,
        ))

        # Mark iteration boundaries
        prev_iter = -1
        for idx, row in df.iterrows():
            loc = df.index.get_loc(idx)
            if row['iter'] != prev_iter and prev_iter >= 0:
                fig.add_vline(x=loc - 0.5, line_dash='dot', line_color=cfg['color'],
                              line_width=1, opacity=0.4)
            prev_iter = row['iter']

    # v7 in-dist reference
    fig.add_hline(y=V7_REF['in_dist_mape'], line_dash='dash', line_color='red',
                  annotation_text=f"v7 in-dist {V7_REF['in_dist_mape']}%",
                  annotation_position='bottom right')

    fig.update_layout(
        title='Net-level MAPE — Training Progress  (★ = best_model.pth saved)',
        xaxis_title='Training Record (chronological)',
        yaxis_title='Net MAPE (%)',
        yaxis=dict(range=[20, 75]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=420, margin=dict(t=60, b=40),
    )
    return fig


def fig_gnd_smape(dfs: dict) -> go.Figure:
    fig = go.Figure()
    for key, df in dfs.items():
        if df.empty or 'gnd_smape' not in df or df['gnd_smape'].isna().all():
            continue
        cfg = RUNS[key]
        fig.add_trace(go.Scatter(
            x=list(range(len(df))),
            y=df['gnd_smape'],
            mode='lines+markers',
            name=cfg['label'],
            line=dict(color=cfg['color'], width=2),
            marker=dict(size=4),
            customdata=list(zip(df['label'], df['iter'], df['step'])),
            hovertemplate="<b>%{customdata[0]}</b><br>GND SMAPE: %{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        title='GND SMAPE — Fringe_scale Learning Effect',
        xaxis_title='Training Record', yaxis_title='GND SMAPE (%)',
        yaxis=dict(range=[25, 60]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=320, margin=dict(t=60, b=40),
    )
    return fig


def fig_loss(dfs: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Train Loss (log scale)', 'GND Loss + CPL Direct Loss'])
    for key, df in dfs.items():
        if df.empty:
            continue
        cfg = RUNS[key]
        x = list(range(len(df)))
        fig.add_trace(go.Scatter(x=x, y=df['train_loss'], name=cfg['label'],
                                 line=dict(color=cfg['color']), showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df['loss_gnd'], name=f"{key} gnd",
                                 line=dict(color=cfg['color'], dash='dot'), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=x, y=df['loss_cpl_direct'], name=f"{key} cpl",
                                 line=dict(color=cfg['color'], dash='dash'), showlegend=False), row=1, col=2)

    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_layout(
        title='Loss Components',
        height=320, margin=dict(t=60, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


def fig_al_progress(al_dfs: dict) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Labeled Tiles Added per Iteration', 'Pool Size Remaining'])
    for key, df in al_dfs.items():
        if df.empty:
            continue
        cfg = RUNS[key]
        fig.add_trace(go.Bar(x=df.get('iteration', range(len(df))),
                             y=df.get('labeled_size', []),
                             name=cfg['label'], marker_color=cfg['color']), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.get('iteration', range(len(df))),
                                 y=df.get('pool_size', []),
                                 name=cfg['label'], line=dict(color=cfg['color']),
                                 showlegend=False), row=1, col=2)
    fig.update_layout(
        title='Active Learning Progress',
        barmode='group', height=300, margin=dict(t=60, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Summary table HTML
# ---------------------------------------------------------------------------
def summary_html(dfs: dict, ood_results: dict) -> str:
    rows = []
    for key, df in dfs.items():
        cfg = RUNS[key]
        if df.empty:
            best_mape = cur_iter = cur_step = '—'
            gnd_best = '—'
        else:
            best_mape = f"{df['net_mape'].min():.2f}%"
            best_row  = df.loc[df['net_mape'].idxmin()]
            gnd_best  = f"{df['gnd_smape'].min():.2f}%" if 'gnd_smape' in df and not df['gnd_smape'].isna().all() else '—'
            last = df.iloc[-1]
            cur_iter  = str(int(last['iter']))
            cur_step  = str(int(last['step']))

        ood = ood_results.get(key, {})
        ood_nova  = f"{ood['mape_nova']:.2f}%"  if 'mape_nova'  in ood else '⏳'
        ood_tv80s = f"{ood['mape_tv80s']:.2f}%" if 'mape_tv80s' in ood else '⏳'
        gnd_ratio = f"{ood.get('chip_gnd_tv80s', '—'):.3f}×" if 'chip_gnd_tv80s' in ood else '⏳'
        cpl_ratio = f"{ood.get('chip_cpl_tv80s', '—'):.3f}×" if 'chip_cpl_tv80s' in ood else '⏳'

        rows.append(f"""
        <tr>
          <td><b>{key}</b><br><small style='color:#666'>{cfg['label']}</small></td>
          <td>{cfg['status']}</td>
          <td style='color:#d62728;font-weight:bold'>{best_mape}</td>
          <td>{gnd_best}</td>
          <td>I{cur_iter}/s{cur_step}</td>
          <td>{ood_nova}</td>
          <td>{ood_tv80s}</td>
          <td>{gnd_ratio}</td>
          <td>{cpl_ratio}</td>
        </tr>""")

    v7 = V7_REF
    rows.append(f"""
        <tr style='background:#fff3cd'>
          <td><b>v7_smalldsgn</b><br><small style='color:#666'>baseline (3 designs, CPL fix)</small></td>
          <td>✅ Complete</td>
          <td>{v7['in_dist_mape']}%</td>
          <td>—</td>
          <td>—</td>
          <td>{v7['ood_mape_nova']}%</td>
          <td>{v7['ood_mape_tv80s']}%</td>
          <td>{v7['chip_gnd_tv80s']}×</td>
          <td>{v7['chip_cpl_tv80s']}×</td>
        </tr>""")

    return f"""
    <table style='width:100%;border-collapse:collapse;font-size:14px'>
      <thead>
        <tr style='background:#2c3e50;color:white'>
          <th style='padding:8px;text-align:left'>Model</th>
          <th>Status</th>
          <th>Best In-Dist MAPE</th>
          <th>Best GND SMAPE</th>
          <th>Current Iter/Step</th>
          <th>OOD MAPE (nova)</th>
          <th>OOD MAPE (tv80s)</th>
          <th>GND Ratio (tv80s)</th>
          <th>CPL Ratio (tv80s)</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------
def build_html(out_path: Path, refresh_secs: int = 60) -> None:
    # Load data
    dfs = {k: parse_log(v['log']) for k, v in RUNS.items()}
    al_dfs = {k: parse_al_csv(k) for k in RUNS}
    ood_results = {k: parse_ood_log(p) for k, p in EVAL_LOGS.items()}

    # Best MAPEs for header cards
    cards_html = ""
    card_colors = {"v6_fivedsgn": "#1f77b4", "v8_fivedesign": "#ff7f0e", "v8b_railcpl": "#2ca02c", "v9": "#9467bd", "v10": "#d62728", "v10b": "#e377c2"}
    for key, df in dfs.items():
        best = f"{df['net_mape'].min():.2f}%" if not df.empty else "N/A"
        status = RUNS[key]['status']
        col = card_colors[key]
        cards_html += f"""
        <div style='flex:1;background:white;border-left:4px solid {col};
                    padding:16px;border-radius:4px;box-shadow:0 1px 4px #ccc'>
          <div style='color:#666;font-size:12px'>{key} {status}</div>
          <div style='font-size:28px;font-weight:bold;color:{col}'>{best}</div>
          <div style='color:#666;font-size:11px'>{RUNS[key]["label"]}</div>
        </div>"""

    # Charts
    charts_html = ""
    for fig_fn in [
        lambda: fig_mape(dfs),
        lambda: fig_gnd_smape(dfs),
        lambda: fig_loss(dfs),
        lambda: fig_al_progress(al_dfs),
    ]:
        try:
            fig = fig_fn()
            charts_html += pio.to_html(fig, full_html=False, include_plotlyjs=False)
        except Exception as e:
            charts_html += f"<p style='color:red'>Chart error: {e}</p>"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <meta http-equiv='refresh' content='{refresh_secs}'>
  <title>PINN-PEX Training Dashboard</title>
  <script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
           background: #f5f5f5; margin: 0; padding: 20px; }}
    .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 8px;
               margin-bottom: 20px; }}
    .cards {{ display: flex; gap: 16px; margin-bottom: 20px; }}
    .chart-box {{ background: white; border-radius: 8px; padding: 16px;
                  box-shadow: 0 1px 4px #ccc; margin-bottom: 16px; }}
    table td, table th {{ padding: 8px 12px; border-bottom: 1px solid #ddd; }}
    table tr:hover {{ background: #f9f9f9; }}
    .refresh-note {{ color: #999; font-size: 12px; }}
  </style>
</head>
<body>
  <div class='header'>
    <h1 style='margin:0'>⚡ PINN-PEX Training Dashboard</h1>
    <div style='margin-top:8px;opacity:0.8'>
      Last updated: {now} &nbsp;·&nbsp;
      <span class='refresh-note'>Auto-refresh every {refresh_secs}s</span>
    </div>
  </div>

  <div class='cards'>{cards_html}</div>

  <div class='chart-box'>
    <h3 style='margin-top:0'>📊 Model Comparison</h3>
    {summary_html(dfs, ood_results)}
    <p style='color:#666;font-size:12px;margin-top:8px'>
      ⏳ = OOD evaluation pending &nbsp;·&nbsp;
      v7 row = baseline reference &nbsp;·&nbsp;
      GND/CPL ratio = pred/golden chip-level capacitance (target ≥ 0.5×, ≥ 0.7×)
    </p>
  </div>

  <div class='chart-box'>{charts_html}</div>

  <p style='color:#999;font-size:12px;text-align:right'>
    Re-run <code>python3 scripts/visualize_training.py</code> to update manually
  </p>
</body>
</html>"""

    out_path.write_text(html, encoding='utf-8')
    print(f"✅ Dashboard written → {out_path}  ({refresh_secs}s auto-refresh)")


# ---------------------------------------------------------------------------
# HTTP server (optional --serve mode)
# ---------------------------------------------------------------------------
def serve(out_path: Path, port: int = 8080, refresh_secs: int = 30) -> None:
    directory = out_path.parent

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)
        def log_message(self, fmt, *args):
            pass  # silence request logs

    def rebuild_loop():
        while True:
            try:
                build_html(out_path, refresh_secs)
            except Exception as e:
                print(f"⚠️  Rebuild error: {e}")
            time.sleep(refresh_secs)

    t = threading.Thread(target=rebuild_loop, daemon=True)
    t.start()

    print(f"🌐 Serving on http://localhost:{port}/{out_path.name}")
    print(f"   SSH tunnel: ssh -L {port}:localhost:{port} <server>")
    print("   Ctrl-C to stop")
    with http.server.HTTPServer(('', port), Handler) as httpd:
        httpd.serve_forever()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PINN-PEX Training Dashboard')
    parser.add_argument('--out', default=str(AL_DIR.parent / 'dashboard.html'),
                        help='Output HTML path')
    parser.add_argument('--serve', action='store_true', help='Serve via HTTP on --port')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--refresh', type=int, default=60, help='Auto-refresh interval (seconds)')
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.serve:
        build_html(out_path, args.refresh)
        serve(out_path, args.port, args.refresh)
    else:
        build_html(out_path, args.refresh)
