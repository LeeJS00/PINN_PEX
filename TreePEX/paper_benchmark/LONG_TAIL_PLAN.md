# Long-tail (Large-net) 보완 plan — TreePEX ASAP7

_Lock 2026-05-18. L11 specialist (commit ae0f7d8) 이후 long-tail gap 분석 + 다음 lever 후보._

## 🔄 Pivot 2026-05-18 (afternoon) — Refinement sprint replaces Tier A/B

User: "필수적인 요소들만 남기는 방식으로. model engineering 일종의?"

→ Long-tail accretion-by-lever 대신 ablation-driven pruning이 우선.
Tier A/B는 minimal canonical lock 후 재평가.

### Phase A result (2026-05-18 PM, 5-seed × 2 designs)
- **L5 calibration → 🗑 DROPPED**: mean MAPE Δ −0.03 / median Δ +0.02 / R² Δ 0 (3-gate ALL pass "no harm"). calibration.json → archive. See `~/.claude/.../memory/project_asap7_L12_L5_drop.md`.
- **Fanout XGB proxy → ✅ ESSENTIAL**: Ridge alternative +0.36 tv80s / +0.29 nova mean MAPE, nova R² −0.016.
- **L11 specialist → ✅ ESSENTIAL for nova R²**: tv80s improves without it but nova R² collapses 0.9694→0.9368 (Δ −0.033).

### Phase B retrains COMPLETE (5-seed each, 별도 출력 dir로 destructive overwrite 방지)

| ID | 변경 | 결과 (ΔMAPE vs canonical, tv80s/nova) | 결정 |
|---|---|---:|---|
| B1 | TREEPEX_L6_FANOUT_NOISE=0 (general + specialist) | +0.70 / +0.57 pp, R² −0.008/−0.013 | L6 σ=0.2 ESSENTIAL — KEEP |
| B2 | specialist depth=9→8 n_est=750→500 | −0.04 / −0.04 pp (improve), R² +0.0005 | **SIMPLIFY → d8 n500** (3× smaller weights) |
| B3 | V4 H3-off (V3-only 41-D, general + specialist) | +1.36 / +1.83 pp, R² −0.002/−0.014, D9 +2.29 pp | V4 H3 ESSENTIAL — KEEP |

### intel22 ablation (parallel, 2026-05-18 evening)

| Config | tv80s ΔMAPE | nova ΔMAPE | tv80s R² | nova R² | 결정 |
|---|---:|---:|---:|---:|---|
| A1 no_L5 | **−0.10 IMPROVE** | **−0.14 IMPROVE** | +0.0002 | +0.0001 | L5 DROP (intel22도) |
| A2 ridge proxy | +0.04 ns | +0.11 sig, D9 +1.08 | −0.0014 | −0.0023 | XGB proxy KEEP |

### Final lock 2026-05-18 evening (both PDKs)

`calibration.json` → archive (intel22 + ASAP7 둘 다). ASAP7 specialist d9 n750 → d8 n500 swap. 자세한 결과: `~/.claude/.../memory/project_refinement_sprint_v3_lock.md`.

Cold-inference final numbers (5-seed prediction-mean), warm/cold 분리:

**⚡ Warm path** (cached features, label-leak fanout — pex_tool / 02_inference path):

| PDK | Design | nets | MAPE_med / R² | Wall e2e |
|---|---|---:|---|---:|
| intel22 | tv80s_f3 | 3,169 | **4.95 % / 0.9936** | 11.27 s |
| intel22 | nova_f3 | 92,425 | **5.34 % / 0.9914** | 82.10 s |
| ASAP7 | tv80s_x1 | 3,328 | **6.72 % / 0.9854** | 9.68 s |

**❄️ Cold path** (DEF→features→inference, XGB proxy fanout — pex_cold path):

| PDK | Design | nets | MAPE_med / R² | Wall e2e |
|---|---|---:|---|---:|
| intel22 | tv80s_f3 | 3,280 | **4.954 % / 0.9933** | 68.31 s |
| intel22 | nova_f3 | 113,812 | **5.474 % / 0.9895** | 4767 s (80 m) |
| ASAP7 | tv80s_x1 | 3,328 | **7.001 % / 0.9827** | ~70 s |
| ASAP7 | nova_x1 | 125,499 | **7.925 % / 0.9699** | ~3249 s (54 m) |

Warm/cold Δ MAPE (Δ = fanout proxy OOS quality 효과):
- intel22 tv80s +0.00 pp (proxy near-perfect)
- intel22 nova  +0.14 pp (fanout proxy 12 % OOS MAPE)
- ASAP7   tv80s +0.28 pp (fanout proxy 18-20 % OOS MAPE)

---

## Feature pruning sprint (2026-05-19) — **F3+F4 모두 REJECT**

User pivot 후속: F1 XGB importances + F2 permutation importance로 28 dead features
식별 → F3a (28-drop) / F3b (41-drop) / F4 (V4 H3 top2+top3 drop) 5-seed retrain.

### Pruning variant 결과 (vs post-sprint 67-D canonical)

| Variant | Schema | ASAP7 tv80s ΔMAPE / R² | ASAP7 nova ΔMAPE / R² | intel22 tv80s | intel22 nova |
|---|---:|---:|---:|---:|---:|
| baseline (67-D) | 68-D | 9.157 / 0.9827 | 10.198 / 0.9699 | 6.207 / 0.9935 | 6.919 / 0.9695 |
| F3a Pruned-39 (28-drop) | 40-D | **+0.51 / −0.016** ⚠ | **+0.22 / −0.004** | +0.13 | **−0.16 improve** |
| F3b Pruned-26 (41-drop) | 27-D | **+0.55 / −0.017** ⚠ | **+0.38 / −0.005** | +0.14 | **−0.26 improve** |
| F4 V4-H3-top1-only | 56-D | +0.47 / R² ≈ | **+0.46 / R²≈** ⚠ | +0.17 | +0.29 |

**3-gate verdict**: F3a/F3b/F4 모두 REJECT (ASAP7 양쪽 tv80s ±0.20 / nova ±0.07 tol 위반).
**67-D canonical이 minimum viable**.

### Lesson

Permutation importance (F2, single-feature shuffle 다른 features fix)는
모델의 marginal contribution만 측정. **실제 retrain은 새 feature set으로
interactions 재학습 → permutation prediction과 부호조차 일치 안 함**.

특히:
- `top2_score` F2 marginal (0.83 pp), F4 retrain ASAP7 nova +0.46 pp
- 28-dead group drop: capacity bottleneck — 모델이 drop된 features 정보를
  남은 features로 흡수 못 함
- intel22 nova는 모든 variant에서 improve (PDK별 feature usage 매우 다름),
  but cross-PDK consistency가 paper claim일 시 invalid

상세: `~/.claude/.../memory/feedback_permutation_importance_pitfall.md`.
F3a/F3b/F4 weight 디렉토리 archive 처리 (rebuttal-시 재현용 보존).

**두 path 별도 표 의무** (user directive 2026-05-18): warm path 자체가 label-leak path
이므로 cold (proxy fanout) 와 절대 같은 column에 섞지 말 것.

### Acceptance gates (statistician protocol applied)
- Gate-1: paired Wilcoxon (per-net |error|) + Holm-Bonferroni
- Gate-2: 95% BCa CI excludes per-design tol (tv80s ±0.20 / nova ±0.07 pp)
- Gate-3: per-decile (D7/D8/D9) no regress > 0.5 pp
- Pre-registered table: `outputs/ablation/refine_v3_01/analysis_summary.csv`

---


## 현황 (2026-05-17 L11 lock 직후)

L11이 닫은 격차: ASAP7 nova R² **0.937 → 0.970** (+0.033). MAPE는 7.94→7.90 % 동등.
남은 문제:
1. **Decile-9 bimodal symmetric error** — FE_* prefix over-prediction / n_961xx prefix under-prediction.
   Monotonic L5 calibration으로 회복 불가능.
2. **tv80s 0.05 pp marginal regression** (6.96 → 7.00 %) — 6.8 % routed nets의 specialist preds가
   L5-calibrated canonical보다 조금 worse.
3. **Hard threshold discontinuity** — `total_wire_length_um > 15.35 μm` switching이 boundary net에서 jump.

## 0. Diagnostic gap (반나절, read-only)

| 단계 | 내용 | 산출물 |
|---|---|---|
| 0a | L11 residual을 **net-name prefix별 stratify** (FE_*, n_961xx, sram_*, cts_*, scratch_* …) | `diagnostics/L11_prefix_residual.csv` |
| 0b | Decile-9 residual vs 6 features Pearson (wire_length, n_cuboids, fanout, eps, n_aggressor, broadside_overlap) | scatter + ρ table |
| 0c | tv80s routed 225 net 중 L5-calib vs specialist diff | `diagnostics/tv80s_l11_regress_nets.csv` |

→ 결과로 Tier A/B 우선순위 데이터 기반 재배열.

## Tier A — Calibration polish (low risk, 1–2일, 도메인 리뷰어 필수)

| ID | Lever | 기대 Δ (nova / tv80s) | 비용 | 위험 | 결과 |
|---|---|---|---|---|---|
| **L11.b** | L5 isotonic 3-stage **refit on switched preds** | nova ≈, tv80s −0.05 pp 회복 | 30 min retrain | ★ low | ❌ **REJECTED 2026-05-18** (tv80s +0.022 pp / nova R² −0.0032; D9 bimodal not captured by monotone isotonic — see `project_asap7_L11b_neg`) |
| **L11.c** | `wire_length_um` 기반 **sigmoid soft routing** (12–18 μm smooth) | nova R² +0.002~0.005 | 1 h | ★ low | ⏸ deferred to rebuttal (cosmetic; not blocker) |
| **L11.d** | Threshold sweep 12 / 14 / 15.35 / 17 / 20 μm × 5-seed cold-eval | best trade-off curve | 3 h | ★ low | ⏸ deferred to rebuttal |

**Gate** (applied to L11.b): nova R² ≥ 0.97 유지 + tv80s MAPE ≤ 7.00 % — **FAIL**.

**Lesson** (2026-05-18): D9 nova holds 96.7 % SS, D9 tv80s holds 74.8 % SS.
Bimodal symmetric error (FE_* over-pred + n_961xx under-pred on nova; tv80s D9
signed +0.078 fF) cannot be resolved by **any** monotone post-hoc transform of
magnitude. Tier A capacity ≈ 0 pp on nova. **Skip remaining Tier A — direct to
Tier B L15 hierarchical** (or L13 10-seed for safety).

## Tier B — Model variants on large subset (medium, 3–5일)

| ID | Lever | 가설 | 기대 Δ | 위험 |
|---|---|---|---|---|
| **L13** | Specialist **5-seed → 10-seed** prediction-mean | variance σ²/10 | R² +0.003~0.005 | ★ low |
| **L14** | **Quantile loss** (q=0.5) specialist 대체/stack | Tweedie over-pred 보정 | nova MAPE −0.2 pp 가능 | ★ medium |
| **L15** | **Hierarchical 2-tier**: gold>3fF→{3–15 mid, >15 mega} | mega-net 분리 학습 | nova R² +0.005~0.01 | ★ medium |
| **L16** | **Cap-weighted MAPE loss** (`weight ∝ log(gold+1)`) | large-net loss 가중 | nova MAPE −0.1~0.3 pp | ★★ small-net regression 위험 |
| **L17** | **Stacked residual XGBoost on large subset만** | specialist 잔차 흡수 | nova MAPE −0.1 pp | ★★★ L8 catastrophic 재발 위험 |

**Gate**: nova R² ≥ 0.975 OR MAPE −0.1 pp, tv80s |Δ| ≤ 0.1 pp, intel22 |Δ| ≤ 0.05 pp.

## Tier C — Representation expansion (high cost, 1–2주, paper draft 이후 또는 rebuttal lever)

| ID | Lever | 가설 | 비용 | 회의 정도 |
|---|---|---|---|---|
| **L18** | **Net-name prefix embedding** (regex → 8-D) | bimodal error의 name cluster 흡수 | 1주 | ★★ PDK 일반성 깨질 위험 — cross-PDK 검증 필수 |
| **L19** | **Aggressor cap 768 → 2048** + intel22 동기화 | nova decile-9 hub-net saturation 추가 해소 | rebuild 4 h + retrain | ★ low (marginal expected) |
| **L20** | **Top-3 aggressor raw 3D bbox** (현 26-D scalar에 18-D raw geometry 추가) | bimodal 방향성을 aggressor 위치로 차별 | rebuild 1 day | ★★ medium |

## Skip 권장

- **L12 intel22 specialist** — intel22 nova R² 0.992 이미 포화, +0.001 expected.
- **GNN aggregation specialist** — TreePEX scalar+tree 철학 깨짐, paper narrative loss.
- **Per-net additional general ML stacking** — overfit + L8 패턴 재발 위험.

## Execution order (next 2 weeks)

1. **Day 1 AM**: Diagnostic 0a–0c → 결과로 Tier 순서 정렬
2. **Day 1 PM – Day 2**: L11.b + L11.c + L11.d (Tier A 일괄) → lock if pass gate
3. **Day 3**: L13 (10-seed specialist) — safety 좋음
4. **Day 4**: L14 (quantile) 또는 L15 (hierarchical) — diagnostic 결과로 택일
5. **Day 5–6**: 남은 Tier B 1개 추가 시도
6. **Day 7**: L19 (aggressor cap 2048) — 시간 남으면
7. **Day 8–14**: Paper draft 진입; L18/L20은 rebuttal 시 reserve

## Reviewer protocol

- 각 lever 적용 직전 **pex-domain-reviewer agent 필수 호출** —
  4.66 % feature ceiling / L8 catastrophic precedent / 5-seed P6 protocol 점검.
- Codex/Gemini deliberation은 L14/L16/L17 (loss 변경 또는 stacking) 에서만 추가.

## Acceptance gates 종합

| Tier | 통과 조건 (모두 만족) |
|---|---|
| A | nova R² ≥ 0.97, tv80s MAPE ≤ 7.00 %, intel22 |Δ| ≤ 0.05 pp |
| B | nova R² ≥ 0.975 OR nova MAPE −0.1 pp, tv80s |Δ| ≤ 0.1 pp, intel22 |Δ| ≤ 0.05 pp |
| C | nova MAPE −0.5 pp OR R² +0.01, otherwise abandon |
