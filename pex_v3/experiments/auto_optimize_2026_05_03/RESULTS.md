# Auto-Optimize Sweep — Running Results

_Live tracking. Each variant entry updated as smoke / 5-seed / stack lock completes._

## Baseline (locked, 5-seed)

| Stratum | Median | std | best |
|---|---:|---:|---:|
| valid total | 8.272% | 0.342pp | 6.18% |
| test total | 8.272% | 0.342pp | — |
| test gnd | 20.49% | — | — |
| test cpl | 15.53% | — | — |
| Top-50 outliers | 269% | — | — |
| gnd Q4 | 42.6% | — | — |
| M3 layer gnd | 21.7% | — | — |

## Single-seed smokes (Phase 1)

| Variant | Lever | gnd | cpl | total | last_valid | Mode B | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| C1a Mode B-only | output post-process | +0.00 | — | +0.00 | — | 0% | FAIL (no-op) |
| **C1b full iso** | output post-process | **-1.20** | — | **-1.31** | — | **+38pp WORSE** | FAIL (collateral) |
| **ClampNorm** | norm clamp | +0.40 | -0.31 | -0.91 | **-2.03** | clean | NEAR-PASS (override) |
| **InputSubset** ⭐ | input mask | **-1.44** | -0.51 | **-1.05** | TBD | clean | **PASS** |

Δ vs baseline (negative = better).

## Phase 2 — 5-seed locks

### HybridPexV3MeshInputSubset — DONE (anti-overclaim regression)

| Stratum | Baseline median | InputSubset median | Δ |
|---|---:|---:|---:|
| last_test_total | 8.272% | **7.914%** | **-0.358pp** |
| last_test_gnd | 20.491% | 21.104% | **+0.613pp** ❌ |
| last_test_cpl | 15.528% | 15.755% | +0.227pp |
| best_valid_total | 6.258% | 6.145% | -0.113pp |

- Cohen's d = -0.217 (small), MWU p=0.69 (n.s. across-seed)
- Paired per-net Wilcoxon: n=95594, p=1.3e-224, median Δ=-0.296pp (significant per-net)
- Bootstrap 95% CI [7.266%, 9.249%] — wide, baseline overlap
- **Smoke seed 42 was lucky**; 5-seed median per-channel actually slightly worse

### HybridPexV3MeshClampNorm — DONE (5-seed REGRESSION, smoke was lucky)

| Stratum | Baseline | ClampNorm | Δ |
|---|---:|---:|---:|
| last_test_total | 8.272% | **8.960%** | **+0.688pp** ❌ |
| last_test_gnd | 20.491% | 20.874% | +0.383pp |
| last_test_cpl | 15.528% | 15.559% | +0.031pp |
| best_valid_total | 6.258% | 6.268% | +0.010pp |

- Cohen's d = +0.743 (medium WORSE), MWU p=0.151 (n.s. across-seed)
- Paired per-net Wilcoxon: median Δ +0.454pp WORSE per net
- Phase 2 instability NOT damped at 5-seed; smoke seed 42 was an outlier
- Last-step variance high (stdev 1.038pp vs baseline 0.383pp)

### HybridPexV3MeshInputSubsetClampNorm (stack) smoke — DONE

| | Combined smoke | IS smoke | CN smoke | Verdict |
|---|---:|---:|---:|---|
| test gnd | 19.25% | 19.05% | 20.89% | composition 작음 |
| test cpl | 15.13% | 15.02% | 15.22% | composition 작음 |
| test total | **6.98%** | 7.22% | 7.36% | both singles BEAT by 0.24-0.38pp |
| last_valid total | 6.95% | TBD | 6.66% | mid |
| Phase 2 max\|Δ\| | 3.31pp | — | ~1.5pp | partial regression |

Combined smoke beats both single smokes. Strict gate (≤6.80%) FAIL by 0.18pp.
Stack agent recommendation: SOFT NO-GO, but worth 5-seed lock to verify.

5-seed lock LAUNCHED on GPUs 0-4. ~20 min wall.

## Phase 2.5 — C1 isotonic refit on InputSubset val (CPU, instant)

| | Baseline | IS only 5-seed | **IS + C1 refit (per-seed)** |
|---|---:|---:|---:|
| total median | 8.272% | 7.914% | **7.192%** ⭐ |
| gnd median | 20.491% | 21.104% | **19.964%** |
| top50 median | 282.8% | 282.8% | 298.6% (+15.8pp collateral) |

**This is the best result so far.** -1.08pp total, -0.53pp per-channel gnd (REAL improvement, not cancellation). Top-50 collateral smaller than C1b on baseline (+38pp → +15.8pp).

PIDs: InputSubset 5-seed runner spawned 5 subprocesses (3368347-3368351), one per GPU.

## Phase 3 — Best stack composition (pending)

Candidate stacks to evaluate:
1. **Combined model** = InputSubset + ClampNorm (architecturally orthogonal, clean composition)
2. **Combined model + C1b post-correction** — only if C1b's val/test mismatch can be fixed by refitting on stack output
3. Single best: whichever 5-seed median is lowest

## Phase 4 — HERO.md (pending)

Will contain:
- Final 5-seed best-stack number with bootstrap 95% CI
- Cohen's d + paired Wilcoxon vs baseline
- Stratified MAPE (per-design / per-quartile / per-fanout / per-layer)
- Anti-overclaim disclosure
- Figures / tables paper-ready

## Levers KILLED (this sweep)

- C1a Mode B-only — no-op (val/test population mismatch makes 1D axis correction useless)
- C1b full distribution — bulk gain (-1.31pp total) at cost of Top-50 +14.5% — kill criterion FAIL

## Levers DEFERRED

- B1 per-pair Sakurai — Codex Round 2 verdict: defer until Phase 2 results decide
- A2 bounded-additive residual — deferred
- A3 cuboid→net hierarchical attention — deprioritized after A1 KILL

## Anti-overclaim discipline

- All 5-seed numbers go through `pex_v3/scripts/aggregate_ablation_summary.py` (Cohen's d + paired Wilcoxon + bootstrap CI on test_total median)
- Stratified MAPE via `pex_v3/scripts/stratify_eval.py` — must show per-stratum improvement, not just total
- HERO.md will report best-step + last-step + ±std (last-step preferred for paper)
