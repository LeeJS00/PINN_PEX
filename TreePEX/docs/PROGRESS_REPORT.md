# TreePEX progress report — 2026-05-10 (compaction-ready snapshot)

## ★ Final deployable frontier (변동 없음)

**TreePEX 5-seed Tweedie XGBoost prediction-mean ensemble:**
- tv80s: tot_med **4.979 %** / gnd 18.02 / cpl 13.27 / R²(tot) 0.9940 / inference **0.171 s** for 3,169 nets
- nova:  tot_med **5.279 %** / gnd 17.40 / cpl 14.96 / R²(tot) 0.9911 / inference 0.185 s for 92,425 nets
- vs v12 PINN: −0.57 pp tv80s tot, **120× faster wall** (0.17 s vs 20.4 s)
- SPEF round-trip lossless (max abs err 5e-6 fF), IEEE 1481-1999 compatible

Tool: `TreePEX/scripts/pex_tool.py --all`. Models: `TreePEX/models/` (10 .json, ~120 MB).

## 4 % gap 진단 (정정 완료)

❌ 잘못된 진단 (모든 보고서에서 제거됨): "GDSII / substrate map 필요"
✅ 정확한 진단: **StarRC도 동일 DEF/LEF/Liberty/layer-stack 입력 사용** (NXTGRD pattern lookup + 3D field solver). 1.0 pp 격차는 input 부재가 아닌 representation expressivity 한계.

정정 완료 위치: RESULTS.md §0, TreePEX/REPORT.md, pex_v4/docs/H_FEATURES_RESULT.md, pex_v4/auto_4pct/reports/FINAL.md, pex_v5/reports/FINAL.md, TreePEX/presentation/make_ppt.py slide 14, memory project_TreePEX_deployable_demo.md.

## 실험 결과 종합 (4 % 도달 시도)

### 이전 라운드 (pex_v4 + pex_v5, 14 strategies)
- S4 Small + Tweedie 5-seed: 5.087 ± 0.049 (deployable best until TreePEX ensemble)
- TreePEX ensemble (5-seed predict-mean): **4.979** (current frontier)
- P2 oracle bound (per-bucket + oracle routing): 4.742 (NOT deployable)
- Big_combined / S6 / S9 / P1 quantile / P3 custom MAPE / P8 router: 모두 NEG

### 이번 라운드 (post-GDSII-correction)

| ID | 추가 정보 | 결과 | 판정 |
|---|---|---|---|
| **N4** | LightGBM (orthogonal tree class) | tv80s 5.108 ± 0.017 / nova 5.445 ± 0.010 | NEG — XGBoost와 동등 ceiling |
| **N1 Small_pair** | aggressor 자체 base 41 features (top-1/2/3) | tv80s 5.158 ± 0.055 / nova 5.503 ± 0.019 | NEG — top1 67 % missing → fallback noise |
| **N1 Big_pair** | 동일, deeper config | tv80s 5.230 ± 0.060 / nova 5.654 ± 0.041 | NEG — overfit 가속 |
| **N2 Small_membank** | k=20 NN over training nets, 7 prior features | tv80s 4.993 / nova 5.322 (5-seed ensemble) — flat vs frontier | **NEG** — tree model이 67-D 안에서 implicit하게 NN pattern을 잡음, explicit prior redundant |
| **N2 Big_membank** | 동일, deeper config | seed=42 DONE (24+24 min), 4 seeds remaining | **진행 중** |
| **N3** pair distribution (top-3 잘림 회복) | raw tile 재추출 (~13-21% per chunk) | **진행 중** | — |

## 활성 background 프로세스

```
N2 (PID 2418884): TreePEX/scripts/N2_memory_bank.py
  Status: 6/10 seed DONE (5× Small DONE = NEG; Big seed=42 DONE; 4 Big remaining)
  ETA: ~1.5-2 h for remaining Big seeds
  Log: TreePEX/scripts/runs_N2/run.log
  Output: TreePEX/scripts/runs_N2/{per_seed/, summary_*.json}
  Small eval script: TreePEX/scripts/compute_n2_small.py

N3 (kicked off in this session — restart): TreePEX/scripts/N3_pair_distribution.py
  Status: tile re-extraction starting
  ETA: ~3-4 h for raw tile pass (8 workers), then ~1 h training
  Log: TreePEX/scripts/runs_N3/extract.log
  Output: TreePEX/scripts/runs_N3/pair_distribution_features.csv (after extract)
```

## 다음 단계 결정 트리

1. **N2 결과 도착** (~1.5h)
   - tv80s ≤ 4.95 → 새 frontier 후보, 5-seed lock 검증
   - 5.0 < tv80s ≤ 5.1 → 약간 개선, 다른 trk 우선
   - tv80s > 5.1 → NEG, A path로 escalate
2. **N3 결과 도착** (~4h)
   - pair distribution features의 효과 측정
3. 둘 다 NEG → C (v12 cuboid encoder embedding stacking) 또는 더 큰 architectural change

## 시도 안 한 카테고리 (예비 path)

| 카테고리 | 메커니즘 | 인프라 | 기대 |
|---|---|---|---|
| C: v12 stacking | v12 cuboid encoder 출력을 XGBoost feature로 (stacking) | 3-5h (v12 inference 5-fold OOF) | -0.2 ~ -0.5 pp |
| D: 2.5D image + CNN | 각 net 주변 routing을 image 렌더링 → CNN encoder | multi-day | -0.3 ~ -0.8 pp |
| E: Per-pair LEARNED head with sum constraint | PyTorch infra, sum-supervised | 2-3 days | -0.3 ~ -0.7 pp |

## 중요 기록

- 모든 GDSII 핑계 표현 보고서 6곳에서 정정됨 (위 정정 위치 참조)
- pex_v3, pex_v4, pex_v5는 read-only (보존)
- TreePEX = current active workspace
- Models trained 2026-05-10: `TreePEX/models/tweedie_{gnd,cpl}_seed{42,0,1,2,3}.json`
