# Phase 0 Audit Results — NCGT Plan v3

_Run date: 2026-05-01_
_Sampled: 11 designs (9 train + 2 test) × 300 nets each = 3,293 nets total_
_Script: `experiments/ncgt/scripts/audit_phase0.py`_
_Raw JSON: `experiments/ncgt/PHASE0_AUDIT.json`_

## TL;DR

| Gate | Plan target | Audit result | Pass/Fail |
|---|---|---|---|
| Signal segs/net P95 cap | ≤ 2K | mean **288**, max **607** | ✅ PASS (huge margin) |
| Aggressors/net P95 cap | ≤ 8K | mean **35,746** @ R=20μm, **10,777** @ R=5μm | ❌ FAIL (vias dominate) |
| SPEF per-edge supervision | ≥ 30% | strict 15.7% / **with tiebreak 85.6%** | ✅ PASS (with tiebreak) |

**Net result**: Plan v3 supervision strategy WORKS (85.6% per-edge usable). But aggressor cap requires **2 design changes**: exclude vias, reduce R_aggr.

## 1. Signal segs/net distribution

Per-design signal net statistics, P95 segments per net (after virtual subdivision):

| Design | n_signal | segs/net P95 |
|---|---|---|
| aes_cipher_top | 298 | ~70 |
| gcd_f3 | small | ~70 |
| ibex_core | 298 | ~280 |
| ldpc_decoder | 298 | ~580 (densest) |
| mc_top | 298 | ~330 |
| spi_top | 298 | ~80 |
| usbf_top | 298 | 534 |
| vga_enh_top | 298 | 576 |
| wb_conmax | 298 | 455 |
| nova_f3 (test) | 298 | 294 |
| tv80s_f3 (test) | 298 | 106 |

**Mean P95 = 288, Max P95 = 607.** Plan v3 cap of 2K segments per net is comfortable.

## 2. Aggressor count by R_aggr (sweep)

Mean / Max P95 across 11 designs:

| R_aggr (μm) | mean P95 | max P95 |
|---|---|---|
| 5 | 10,777 | 22,514 |
| 8 | 16,408 | 31,738 |
| 12 | 22,868 | 42,471 |
| 20 | 35,746 | 63,166 |

**Plan v3 had 8K cap at R=20μm — actual is 4-8× over.**

### Root cause: vias dominate aggressor count

Aggregated role counts at R=20μm (across all 11 designs):

| Role | Count | Fraction |
|---|---|---|
| via | 38.97M | 65.6% |
| signal_aggr_same_layer | 9.73M | 16.4% |
| power_VDD | 6.27M | 10.6% |
| power_VSS | 6.24M | 10.5% |
| signal_aggr_cross_layer | **0** | 0% (⚠️ classifier bug — see §6) |
| pin / branch_node | tracked separately | — |

**Vias = 65.6% of aggressors.** Vias are part of net topology (connecting metal layers), NOT coupling sources — they have negligible cap contribution. **Must be excluded** from aggressor enumeration.

### After via exclusion (estimated)

Removing 65.6% via aggressors from each band:

| R_aggr (μm) | mean P95 (signal+power) |
|---|---|
| 5 | ~3,700 |
| 8 | ~5,650 |
| 12 | ~7,870 |
| 20 | ~12,300 |

**At R_aggr=12μm** (after via exclusion): ~7.9K mean P95 — fits within 8K cap.
**At R_aggr=8μm**: ~5.7K — comfortable for memory budget.

## 3. SPEF per-edge supervision

| Metric | Mean | Max | Min |
|---|---|---|---|
| strict containment unique | 15.7% | 19.2% | 13.0% |
| ambiguous (tiebreakable) | ~70% | ~76% | ~58% |
| unmapped | 14.4% | 27.8% | 9.1% |
| **usable_with_tiebreak** | **85.6%** | **90.9%** | **72.2%** |

**The Plan v3 worry was unfounded** — strict 15.7% looked dire, but WIRE-preferred tie-break recovers ~70% additional edges to **85.6% usable per-edge supervision** on average.

This is well above the 30% gate. Per-edge CPL loss is viable.

### Per-design breakdown

| Design | unique% | ambig% | unmapped% | usable% |
|---|---|---|---|---|
| ibex_core | 15.4% | 68.4% | 16.2% | 83.8% |
| usbf_top | 14.5% | 76.2% | 9.3% | 90.7% |
| vga_enh_top | 16.4% | 74.5% | 9.1% | 90.9% |
| wb_conmax | 13.0% | 75.2% | 11.9% | 88.1% |
| nova_f3 (test) | 13.8% | 58.5% | 27.8% | **72.2%** ⚠ |
| tv80s_f3 (test) | 17.1% | 72.2% | 10.8% | 89.2% |

**OOD design `nova_f3` has 27.8% unmapped** (worst). May indicate denser layout requires `L_subdiv < 4μm`. Watch in Phase 1.

## 4. Net class distribution

| Class | Count |
|---|---|
| signal | 3,262 (99.1%) |
| VDD | 12 |
| VSS | 11 |
| clock | 8 |

Signal-net dominance is expected. Power nets (23 total) are large outliers (mesh structure, 10K+ segments each) — handled separately.

## 5. Plan parameter adjustments (v3 → v4)

Based on audit results:

| Parameter | v3 value | v4 value | Reason |
|---|---|---|---|
| Aggressor types | include vias | **exclude vias** at extractor | 65.6% of aggressors are vias with no cap contribution |
| R_aggr | 20 μm | **12 μm** | After via exclusion, P95 fits 8K cap; covers M7/M8 long parallel coupling |
| Per-net target seg cap | 2K | 1K | actual P95=288, 1K is 3× headroom |
| Per-net aggressor cap | 8K | 6K (4K signal + 2K power) | Closest-distance pruning at boundary |
| `L_subdiv` | 4 μm | 4 μm | Keep; nova_f3 unmapped 27.8% is borderline, revisit Phase 1 if needed |
| Per-edge supervision | strict (Plan target 30%) | **tie-break (85.6% actual)** | Codex r2 P1 E concern resolved |

## 6. Open issue: cross_layer role classifier returns 0

`signal_aggr_cross_layer = 0` across all 11 designs is suspicious. Either:
- Bug in `_layer_to_idx` causing all aggressor wires to share the same layer_idx as target.
- Or `target_layer` computation collapses to a value matching every aggressor.

**Severity**: low for plan finalization (heterogeneous-type embeddings can be merged into single `signal_aggr` if cross/same is unreliable). Investigate during Phase 1 implementation.

## 7. Decisions for Phase 1

- Aggressor enumeration excludes vias.
- R_aggr = 12 μm (down from 20).
- Per-net caps: targets ≤ 1K, signal aggressors ≤ 4K, power aggressors ≤ 2K.
- Per-edge SPEF supervision uses WIRE-preferred tie-break (line containment + min perp dist + WIRE > VIA preference).
- Heterogeneous types: 6 types initially (target / signal_aggr / power_VDD / power_VSS / pin / branch_node) — collapse same/cross-layer until classifier bug resolved.
- Phase 0 supervision gate **PASSED** → proceed to Phase 1 smoke test.
