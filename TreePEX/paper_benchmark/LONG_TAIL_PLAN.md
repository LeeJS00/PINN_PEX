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

Cold-inference final numbers (5-seed prediction-mean, mean MAPE):

| PDK | Design | nets | post-sprint MAPE / R² |
|---|---|---:|---|
| intel22 | tv80s_f3 | 3,280 | **6.207 / 0.9935** |
| intel22 | nova_f3 | 113,812 | **6.919 / 0.9895** |
| ASAP7 | tv80s_x1 | 3,328 | **9.157 / 0.9827** |
| ASAP7 | nova_x1 | 125,499 | **10.198 / 0.9699** |

(intel22 mean MAPE; CLAUDE.md의 4.98/5.28는 median MAPE — 보고 정의 다름.)

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
