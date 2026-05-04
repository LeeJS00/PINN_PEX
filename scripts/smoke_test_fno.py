#!/usr/bin/env python3
"""
scripts/smoke_test_fno.py

3가지 모델 변형 200-step smoke test.
  v10b_baseline  MLP only                    (현재 best 27.30%)
  gino_v1        Feature FNO enricher        (encoder→FNO→FluxRouter)
  dspinn_v1      Density FNO conditioning    (MacroDensityFNO→GND+CPL)

출력: compile latency / steady-state ms/step / peak GPU MiB /
      step-200 val MAPE / GND SMAPE / CPL SMAPE

Usage:
  python3 -u scripts/smoke_test_fno.py --gpu 1
  python3 -u scripts/smoke_test_fno.py --gpu 1 --cases v10b_baseline dspinn_v1
"""
import argparse, sys, gc, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd

import configs.config as cfg_module
from src.models.neural_field import DeepPEX_Model
from src.trainers.finetuner import NeuralFieldFinetuner
from src.active_learning.oracle import FullChipPEXOracle
from src.data.replay_buffer import DesignLevelReplayBuffer

# ── 케이스 정의 ───────────────────────────────────────────────────────────────
CASES = [
    dict(name='v10b_baseline', use_gino=False, use_dspinn=False,
         desc='MLP only — 현재 best 27.30%'),
    dict(name='gino_v1',       use_gino=True,  use_dspinn=False,
         desc='Feature FNO enricher (encoder→FNO→FluxRouter)'),
    dict(name='dspinn_v1',     use_gino=False, use_dspinn=True,
         desc='Density FNO conditioning (MacroDensityFNO → GND+CPL)'),
]

SMOKE_STEPS  = 200
WARMUP_STEPS = 1     # compile 은 첫 step에서 발생; 나머지 199를 steady-state로 측정
# ──────────────────────────────────────────────────────────────────────────────


def load_data(device: str, output_dir: Path):
    """predefined cache → (train_loader, val_loader). 케이스 간 공유."""
    oracle    = FullChipPEXOracle(output_dir)
    train_buf = DesignLevelReplayBuffer(max_designs=50)
    val_buf   = DesignLevelReplayBuffer(max_designs=10)
    def_map   = {p.stem: p for p in cfg_module.TRAIN_DEFS + cfg_module.TEST_DEFS}
    cache_dir = output_dir / "cache"

    for csv_path, buf, split in [
        (cache_dir / "predefined_train_subset.csv", train_buf, "train"),
        (cache_dir / "predefined_valid_subset.csv", val_buf,   "valid"),
    ]:
        assert csv_path.exists(), f"cache 없음: {csv_path}"
        df = pd.read_csv(csv_path)
        for d in df['design_name'].unique():
            dp = def_map.get(d)
            if dp:
                buf.add_design(d, df[df['design_name'] == d],
                               oracle.generate_golden_spef(d, dp))

    print(f"  Train tiles: {len(train_buf.all_data)} | Val tiles: {len(val_buf.all_data)}")
    return train_buf.get_dataloader(), val_buf.get_dataloader()


def build_model(case: dict, device: str) -> torch.nn.Module:
    cfg_module._use_gino   = case['use_gino']
    cfg_module._use_dspinn = case['use_dspinn']
    model = DeepPEX_Model(cfg_module).to(device)

    basis_dir = Path(cfg_module.OUTPUT_DIR) / "checkpoints" / cfg_module.RUN_NAME
    ckpts = sorted(basis_dir.glob("bem_ssl_ep*.pth"),
                   key=lambda x: int(x.stem.split('ep')[-1]))
    if ckpts:
        state = torch.load(ckpts[-1], map_location=device)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        cur      = model.state_dict()
        filtered = {k: v for k, v in state.items()
                    if k in cur and v.shape == cur[k].shape}
        model.load_state_dict(filtered, strict=False)
        model.freeze_ssl_layers()
    else:
        print("  ⚠️ SSL checkpoint 없음 — random init으로 진행")

    if hasattr(torch, 'compile'):
        model = torch.compile(model, dynamic=True)
    return model


def run_case(case: dict, train_loader, val_loader, device: str) -> dict:
    print(f"\n{'='*62}")
    print(f"  [{case['name']}]  {case['desc']}")
    print(f"{'='*62}")

    model     = build_model(case, device)
    finetuner = NeuralFieldFinetuner(model, device=device, lr=cfg_module.AL_LR)

    total_p    = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params  total={total_p/1e6:.2f}M  trainable={trainable_p/1e6:.2f}M")

    torch.cuda.reset_peak_memory_stats(device)

    # ── Warmup: step 1 (torch.compile 발생) ──────────────────────────────────
    t_compile_s = time.perf_counter()
    finetuner.train_steps(train_loader, val_loader=None,
                          max_steps=WARMUP_STEPS, save_dir=None,
                          report_every=999999)
    torch.cuda.synchronize(device)
    compile_s = time.perf_counter() - t_compile_s
    print(f"  compile+step1 latency: {compile_s:.1f}s")

    # ── Steady-state: 나머지 (SMOKE_STEPS - WARMUP_STEPS) steps ─────────────
    steady = SMOKE_STEPS - WARMUP_STEPS
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    finetuner.train_steps(train_loader, val_loader=None,
                          max_steps=steady, save_dir=None,
                          report_every=999999)
    torch.cuda.synchronize(device)
    elapsed_s  = time.perf_counter() - t0
    ms_per_step = elapsed_s / steady * 1000

    peak_mib    = torch.cuda.max_memory_allocated(device) / 1024**2
    eta_min     = ms_per_step * 10_000 / 1_000 / 60   # 10000-step iter 기준

    print(f"  Steady-state: {ms_per_step:.0f}ms/step  peak {peak_mib:.0f}MiB  ETA/iter≈{eta_min:.0f}min")

    # ── Validation ────────────────────────────────────────────────────────────
    print("  Running validation (step-200)...")
    val_score, s_tot, s_gnd, s_cpl = finetuner.evaluate(val_loader)
    net_mape = finetuner.compute_net_mape(val_loader)
    print(f"  Val MAPE={val_score*100:.1f}%  GND={s_gnd*100:.1f}%  CPL={s_cpl*100:.1f}%  NetMAPE={net_mape*100:.1f}%")

    result = dict(
        name         = case['name'],
        compile_s    = round(compile_s, 1),
        ms_per_step  = round(ms_per_step, 0),
        eta_min      = round(eta_min, 0),
        peak_mib     = round(peak_mib, 0),
        trainable_M  = round(trainable_p / 1e6, 2),
        val_mape     = round(val_score * 100, 1),
        gnd_smape    = round(s_gnd * 100, 1),
        cpl_smape    = round(s_cpl * 100, 1),
        net_mape     = round(net_mape * 100, 1),
    )

    del model, finetuner
    gc.collect()
    torch.cuda.empty_cache()
    return result


def print_summary(results: list):
    print(f"\n{'='*72}")
    print("  SMOKE TEST SUMMARY  (200 steps, steady-state = steps 2-200)")
    print(f"{'='*72}")
    print(f"  {'Model':<18} {'compile':>8} {'ms/stp':>7} {'ETA/iter':>9} {'PeakMiB':>8} "
          f"{'TrnM':>6} {'NetMAPE':>8} {'GND':>7} {'CPL':>8}")
    print(f"  {'-'*70}")
    for r in results:
        cpl_warn = ' ⚠' if r['cpl_smape'] > 150 else ''
        print(f"  {r['name']:<18} {r['compile_s']:>7.1f}s {r['ms_per_step']:>7.0f} "
              f"{r['eta_min']:>8.0f}m {r['peak_mib']:>8.0f} "
              f"{r['trainable_M']:>6.2f} "
              f"{r['net_mape']:>7.1f}% {r['gnd_smape']:>6.1f}% "
              f"{r['cpl_smape']:>6.1f}%{cpl_warn}")
    print(f"{'='*72}")
    print("  ETA/iter = projected runtime for one 10,000-step AL iteration")

    # 상대 비교
    if len(results) > 1:
        base = results[0]
        print(f"\n  Relative to [{base['name']}]:")
        for r in results[1:]:
            dt  = r['ms_per_step'] - base['ms_per_step']
            dmape = r['net_mape'] - base['net_mape']
            sign  = '+' if dmape >= 0 else ''
            print(f"    {r['name']}: speed {dt:+.0f}ms/step  NetMAPE {sign}{dmape:.1f}pp  "
                  f"CPL {r['cpl_smape']-base['cpl_smape']:+.1f}pp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',   type=int, default=1)
    parser.add_argument('--cases', nargs='+', default=None,
                        help='실행할 케이스 (default: 전체). 예: --cases gino_v1 dspinn_v1')
    args = parser.parse_args()

    device     = f"cuda:{args.gpu}"
    output_dir = Path(cfg_module.OUTPUT_DIR) / "active_learning"

    selected = CASES if not args.cases \
               else [c for c in CASES if c['name'] in args.cases]
    print(f"Smoke test on {device}: {[c['name'] for c in selected]}")
    print(f"Steps: {SMOKE_STEPS} (warmup={WARMUP_STEPS}, steady={SMOKE_STEPS-WARMUP_STEPS})\n")

    print("[Data Loading — 공유 데이터셋]")
    train_loader, val_loader = load_data(device, output_dir)

    results = []
    for case in selected:
        results.append(run_case(case, train_loader, val_loader, device))

    print_summary(results)


if __name__ == '__main__':
    main()
