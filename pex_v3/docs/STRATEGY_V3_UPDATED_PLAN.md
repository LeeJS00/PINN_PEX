# Strategy v3 — Updated Plan (2026-05-02)

_Supersedes the original 2026-05-01 plan in `memory/project_strategy_v3_paradigm_shift.md`_

## What this update addresses

After 25+ GPU-hours of measurement work, 4 specialist agents [ROLE PASS],
and 18+ substantive bugs caught (15 by agents on my code), the plan
needs to incorporate findings that materially change the paper claim
narrative and the implementation order.

---

## §1 — Lessons learned (paper-impacting)

### L1. **B1 4.66% headline benefits from gnd/cpl cancellation**

XGBoost on hand features achieves 4.66% median on **total_fF** but the
per-channel reality is:
- `gnd_fF`: median 20.6% (mean 27.8%, P95 84%)
- `cpl_fF`: median 12.4% (mean 16.7%, P95 45%)
- `total_fF`: median 4.66% (mean 5.95%, P95 16%) — **cancellation effect**

A2 audit + stratified report verified this is real, not a metric bug.
Cancellation is genuine partial-cancellation between gnd-overestimate
and cpl-underestimate.

**Implication**: Phase 1 paradigm contribution must be per-channel-honest.
Reporting only "total MAPE" is insufficient — reviewers will ask for
gnd/cpl breakdown and discover the cancellation.

### L2. **Phase 1 contribution narrative must shift to β strategy**

Original thesis: "Hybrid analytic+residual achieves <4% beating CNN-Cap/NAS-Cap."

Revised thesis (A2 + A5 consensus):

> **Phase 1 hybrid arch achieves gnd MAPE < 8% AND cpl MAPE < 8% on
> the same v3 valid split, beating XGBoost+features (gnd 20.6%, cpl
> 12.4%) by per-channel honest improvement, NOT by total cancellation.**

Secondary claim (γ): smaller in-dist→OOD gap than B1's +2.82pp.

This reframing is the project's strongest physics-grounded contribution.
The "per-channel honesty" story is what distinguishes a physics-based
method from a black-box regressor.

### L3. **B3 PINN's 32.89pp data-fix gain is paper-grade independently**

H1+H3 fixes alone (no architecture change) reduced legacy v10b
63.79±5.02% to 30.90±2.20% on the same protocol — Cohen's d=-8.5,
p<2e-5. This is publishable as a methodology paper.

**Decision needed**: Single combined paper vs split (1A methodology + 1B
paradigm). Recommendation: **lean toward combined**, with §"data fixes"
as Section 4 + §"hybrid arch" as Section 5, since reviewers want to see
both improvements stacked.

### L4. **Heteroscedastic pattern is in the GOOD direction**

```
Q4 (>5fF, 2453 nets):    total 3.42%  ← large nets accurate
Q3 (0.5-5fF, 4477):      total 4.45%
Q2 (0.05-0.5, 5662):     total 5.65%  ← small nets less accurate
```

Phase 1 should not assume legacy heteroscedastic pattern (Q1 over, Q3+
under) — current B1 has the OPPOSITE pattern (large nets accurate,
small nets harder). Loss design (Loss Rule 2 — heteroscedastic
weighting) needs revisit on v3 data, not legacy.

### L5. **OOD distribution shift is real, not overfitting**

B1 in-dist 4.66% → B1 OOD test 5.90% (+1.24pp on cancellation total;
likely much larger on per-channel). This is feature-distribution shift
between training designs (aes, ibex, ldpc, mc, spi, usbf, vga, wb_conmax,
gcd) and test designs (nova, tv80s).

XGBoost trees can't extrapolate beyond training feature ranges. **Phase 1
hybrid arch must demonstrate smaller OOD gap** to claim generalization
contribution (γ secondary).

### L6. **Cost gap matters — must justify with accuracy delta**

XGBoost ~3 KFLOPs/sample vs Phase 1 hybrid (per A5 estimate) ~9 MFLOPs/sample
= **3000× more expensive at inference**. Reviewer R6 (interim review)
will demand justification. The accuracy delta needs to be substantial
on per-channel — i.e., **gnd 20% → 5% AND cpl 12% → 5%**, not just
total 5% → 4%.

### L7. **Agent infrastructure earns its keep — but use Path A**

Custom `.claude/agents/*.md` files do NOT auto-register as
`subagent_type`. Use `subagent_type: "general-purpose"` + embed role-md
path in the prompt. 7 of 8 agents now [ROLE PASS] via this pattern;
caught 18 substantive bugs (15 from agents, 3 self-found during this
session).

### L8. **Multi-GPU parallel training works** — Decision O = B verified

5 seeds × 5 GPUs = 6h wall-clock vs 30h sequential. `06_run_pinn_multigpu.py`
template works. Two bugs caught + fixed: cfg.GPU_ID vs CUDA_VISIBLE_DEVICES,
and parser reading wrong stdout log (multi-process must each write to its
OWN log; never share).

### L9. **Determinism gates strict reproducibility**

`torch.compile` on legacy AL produces non-bit-reproducible same-seed
runs. Must set `TORCH_COMPILE_DISABLE=1` for paper runs. A7 audit #12.

### L10. **Synthetic Stage 2 Mode B is `[HYPOTHESIS]`-level**

A4 audit: drop from Phase 1 pretraining. Use Stage 1 (parallel plate)
+ Stage 2 Mode A (stacked dielectric series) only. Replace Mode B with
vector-fitted complex-image (Chow-Aksun) when Phase 1 hits its <4% gate.

### L11. **15 of 43 hand features were dead-zero before A3 fix**

`segments[0].get("layer_idx", 0)` defaulted to 0; legacy DEF segments
have `"layer"` (string), not `"layer_idx"`. ALL cuboids were assigned
layer 0 → 8 layer-histogram + 3 VSS-shielding + 3 density + 4 layer-stack
features all dead. Fix unblocked the B1 4.66% number; without fix, B1
would have been much worse AND the comparison would be misleading.

---

## §2 — Updated Phase plan

### Phase 0 — DONE ✅

H1 + H2 + H3 + M5 (scaffolded) shipped. v3 manifest 1.32M tiles.
99 unit tests passing. Replaced by Phase B + Phase 1 ongoing work.

### Phase B — partial DONE; reframed 🔄

| Sub-task | Status |
|---|---|
| B3 PINN 5-seed on v3 (legacy DeepPEX) | ✅ done — 30.90 ± 2.20pp |
| B1 XGBoost 5-seed on v3 (hand features) | ✅ done — 4.66 median (cancellation noted) |
| B1 vs B3 paired MWU | ✅ supported |
| Stratified report (per-channel × per-design × per-quartile) | ✅ done |
| **B4 Compact + GAM** (Sakurai linear + GBDT residual) | 🔄 next, ~3 days (A2 estimate) |
| **B2 ParaGraph** reproduction (no tuning, capped) | 🔄 next, 5-day capped (A2 estimate) |
| **Legacy v10b on v3 val** re-eval (closes A1 caveats #1+#2) | 🔄 next, ~1 day |
| OOD panel for B3 (test split = nova+tv80s) | 🔄 deferred until B4/B2 done |
| M5 SSL re-pretrain on clean v3 data | ⏳ deferred (11 GPU-h) |

### Phase 1 — IN PROGRESS 🚧 (Tier-0 foundation done)

| Sub-task | Status |
|---|---|
| `analytic_base_v3.py` differentiable parallel-plate + stacked | ✅ done (11/11 tests, gradcheck pass) |
| `residual_head_v3.py` bounded multiplier + curriculum | ✅ done (11/11 tests, day-1 zero-init) |
| `hybrid_v3.py` skeleton — combines analytic+residual, **per-channel heads** | 🚧 **this turn** |
| Synthetic-Stage-1+Mode-A pretrain harness | 🔄 next |
| K3 transfer canary script | 🔄 next |
| `mesh_v3.py` (A6 spec, 6 days, 24h MVP) | ⏳ deferred (Phase 1.5; current scalar-feature path enough) |
| Real BEOL finetune (`train_pattern_v3.py`) | ⏳ depends on transfer canary pass |
| Phase 1 5-seed eval | ⏳ depends on finetune |

### Phase 2 — pattern→full-net aggregation (paper #2 candidate)

Deferred until Phase 1 hits gate (gnd < 8%, cpl < 8%).

### Phase 3 — Cross-PDK (asap7)

Deferred until Phase 1 + Phase 2.

---

## §3 — Updated paper acceptance gates

### Phase 1 ship-readiness

Phase 1 model is "ready for paper" when ALL of:

- [ ] Hybrid arch builds + trains end-to-end (Tier 0: hybrid_v3.py done)
- [ ] Pretrain on Stage 1 + Stage 2 Mode A converges (residual stays at
      multiplier ≈ 1.0 for parallel-plate; small deviation for stacked)
- [ ] K3 transfer canary passes (≥50% loss drop in 1k steps over no-pretrain control)
- [ ] On v3 valid (5 seeds, last-step checkpoint, fixed manifest hash):
      - **gnd_fF median MAPE < 8%** (β primary)
      - **cpl_fF median MAPE < 8%** (β primary)
      - **total_fF median MAPE < 4%** (α anchor)
- [ ] On v3 test (cross-design OOD nova+tv80s):
      - In-dist→OOD gap < +2.82pp on total (γ secondary)
- [ ] Statistical: paired MWU vs B1 + B4 + B2 supported (p<0.05, |d|>0.5)
- [ ] Determinism: `TORCH_COMPILE_DISABLE=1`, manifest sha256 in
      provenance.json, dirty.patch dumped if dirty git
- [ ] Cost: inference FLOPs/sample < 100× XGBoost (currently estimated 3000×;
      may need pruning or quantization)

### Methodology paper (1A) acceptance

If we split papers, Phase 1A methodology paper requires:
- [ ] B1 + B3 + B4 + B2 baselines on identical eval set
- [ ] Per-channel breakdown (gnd / cpl / total) for each
- [ ] Stratified by quartile / design
- [ ] Cross-PDK section (deferred — this is what makes it 1A vs interim doc)

---

## §4 — Risk register (new/updated)

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Phase 1 hybrid can't beat XGBoost per-channel | medium | high | β-strategy is harder than α; if Phase 1 doesn't deliver, paper claim shrinks to 1A only |
| R2 | OOD gap larger than B1 (Phase 1 worse than tree boosting on transfer) | medium | high | XGBoost trees don't extrapolate; structural model SHOULD be better but no guarantee |
| R3 | Synthetic pretrain doesn't transfer (K3 canary fail) | medium | medium | Stage 1+Mode A only (drop hypothesis-Mode-B); fall back to direct real-data fine-tune |
| R4 | Inference cost gap unjustifiable | low-med | medium | Even 100× XGBoost is acceptable if accuracy delta is large; ~3000× may need optimization |
| R5 | Codex round 3 spec review unresponsive | observed | low | Already using A5+A6 audits as substitute; spec is mature enough to ship |
| R6 | M5 SSL not re-pretrained — encoder has H1-leak memory | observed | low | Encoder frozen; H1-leak signal is weak through frozen encoder; defer M5 unless results stagnate |
| R7 | Legacy v10b raw 5-seed not on v3 val | observed | low | A1 caveat #1+#2; Welch t-from-summary already gives p<2e-5 |

---

## §5 — Concrete next-action priorities (updated)

### This turn

1. ✅ Write this updated plan doc
2. 🚧 `hybrid_v3.py` — combines analytic + residual + **per-channel heads**
   (gnd_self_head + cpl_pair_head as separate MLPs, NOT a single total head)
3. 🚧 Tests for hybrid_v3 (zero-init day-1 = analytic, gradcheck through hybrid)
4. ⏳ Skeleton `pretrain_synthetic_v3.py` for Stage 1 + Stage 2 Mode A

### Next 1-2 days

5. B4 Compact + GAM body (3 days per A2; Sakurai features already live)
6. Legacy v10b on v3 val re-eval — closes A1 caveat #1+#2
7. dirty.patch dump in `manifest_hash.write_provenance` (A7 #11 fix)
8. `_deprecated_` prefix on Stage 2 Mode B with `raise NotImplementedError`

### Sprint after that

9. B2 ParaGraph 5-day capped reproduction
10. Pretrain on Stage 1 + Stage 2 Mode A → K3 canary
11. Real BEOL fine-tune
12. 5-seed Phase 1 eval

### Strategic

13. Decide single-paper vs 1A+1B at end of Phase B (next 2 weeks)
14. M5 SSL re-pretrain (deferred unless results stagnate)
15. mesh_v3.py (deferred to Phase 1.5)
