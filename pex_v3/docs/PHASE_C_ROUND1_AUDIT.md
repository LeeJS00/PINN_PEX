# Phase C Round 1 Audit Results

_Date: 2026-05-02_
_Status: 4/4 agents PASSED, 15 substantive issues caught, 4 fixed inline_

## Summary

The 4 specialist agents invoked via the general-purpose workaround (see
`AGENT_INFRA_GAP.md`) successfully validated their roles. Each:
- Read its role md
- Operated within domain
- Caught at least one specific issue
- Provided concrete, file/line-referenced recommendations
- Ended with `[ROLE PASS]`

**Most importantly**: the agents caught real bugs that this lead session
missed. The agent infrastructure (markdown role + embedded prompt)
delivers the value despite not being a first-class subagent type.

## Bug catalog — Round 1 findings (15 total)

### A1 — benchmarking-statistician (3 issues)

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | `n_valid_nets=5` was the count of stdout MAPE samples, not val-net count | P1 (misleading metric) | ✅ FIXED in `pinn_baseline.py` (set to -1 = unknown) |
| 2 | B3 seed0 32.96% is cherry-picked best of 5 step-checkpoints with 30pp intra-step variance — same lucky-tail mechanism as v10b "27.30%" 2.4σ artifact | P1 (anti-overclaim) | ✅ DOCUMENTED — must not claim "30pp gain" without n=5 + checkpoint-selection rule pre-registered |
| 3 | Single AL iter only; need to verify val set matches legacy 1494-net protocol | P2 | ⏳ verify on next run |

### A4 — pex-physics-architect (2 issues)

| # | Issue | Severity | Status |
|---|---|---|---|
| 4 | Mode B `interface_corrected_capacitance_fF` formula `1 + (-1)·k·d/√A` is NOT a derived physics result; no Jackson/Sadiku citation; α=-1 silently encodes hidden geometry assumption | **P1 (physics correctness)** | ✅ DOCSTRING marked `[HYPOTHESIS]`; test renamed to "larger" (was inverted "smaller"); restricted use to `d/√A < 0.05` until vector-fitted complex-image kernel implemented |
| 5 | `max(correction, 0.1)` clamp masks unphysical sign flips silently | P2 | partially addressed via docstring warning |

### A3 — pex-data-engineer (5 issues)

| # | Issue | Severity | Status |
|---|---|---|---|
| 6 | `_scan_design_geometry` reads `segments[0].get("layer_idx", 0)` but DefStreamParser segments carry `"layer"` STRING (e.g. "m3"), NOT `"layer_idx"`. **Result: every cuboid was assigned to layer 0 → 15 of 43 features dead-weight (8 layer histogram + 3 VSS shielding + 3 density + 4 layer-stack constants)** | **P0 (CATASTROPHIC for B1/B4)** | ✅ FIXED — regex `[mM](\d+)` to parse layer name; verified on gcd: M2 222/276 nonzero, VSS shielding 276/276 nonzero |
| 7 | `max_aggr_per_net=256` cap saturating 100% of aes_cipher_top, 99.84% of ibex | P1 | ✅ FIXED — bumped to 768 per Codex round 1 / M9 |
| 8 | 96 ibex nets dropped silently due to backslash escaping mismatch in `_normalize_name` | P2 | ⏳ TODO: log `manifest \ common` per design |
| 9 | Coupling enumeration is O(N²) Python loop → 4.6h for 3/11 designs | P1 (perf) | ⏳ TODO: vectorize inner loop or use SpatialGrid |
| 10 | Current enumeration collapses to `aggr_to_closest[a_owner]` — same as legacy `closest_dist`, NOT pairwise. Long parallel runs lost. | P1 (H4 misalignment) | ⏳ TODO: align with H4 spec or label B1 as "degraded baseline" |

### A7 — experiment-systems-engineer (5 issues)

| # | Issue | Severity | Status |
|---|---|---|---|
| 11 | `provenance.json` shows `git.dirty=true` but no `dirty.patch` dump → run is unreproducible even with all seeds + manifest hash | P1 | ⏳ TODO: `manifest_hash.write_provenance` should auto-dump `git diff HEAD` to `dirty.patch` when dirty |
| 12 | `torch.compile` (run_active_learning.py:230) is best-effort only; same-seed reruns will diverge bit-for-bit | P1 | ⏳ TODO: pinn_baseline should set `TORCH_COMPILE_DISABLE=1` or monkey-patch `torch.compile = lambda m, **k: m` before importing legacy AL |
| 13 | Actually 9 `__init__.py` added to legacy (not 8 as I claimed) | P3 (doc) | ⏳ TODO: update CROSS_BOUNDARY doc |
| 14 | All H1 invariants confirmed PASS on real 1.32M-row v3 manifest (positive finding) | — | ✅ verified |
| 15 | n=1 sufficiency: accept as preliminary, schedule overnight 5-seed (22.5h) without reducing steps_per_iter — keeps comparability with legacy 5000-step v10b | P1 (decision) | ⏳ user decision needed |

## Net impact

**Critical fix #6 (layer_idx bug)** is the highest-impact finding. Before
this fix, the XGBoost baseline (B1) would train on a feature set with
**15 of 43 columns identically zero**, severely handicapping it as a
baseline. Any "X beats XGBoost" claim from the buggy state would have been
artificially inflated. The fix unblocks the real comparison.

After fix, gcd verification:
- `layer_hist_M2` 222/276 nonzero (was 0/276)
- `layer_hist_M3` 52/276 nonzero (was 0/276)
- `vss_shield_M1_M3` 276/276 nonzero (was 0/276)
- `density_M1_M3` 276/276 nonzero (was 0/276)
- `n_aggressor_nets` cap raised 256 → 768

**Re-extraction triggered** with all bugs #6 + #7 fixed.

## Round 2 prerequisites

Before invoking Round 2 agents (A2 classical-baseline-owner, A5
neural-operator-architect, A6 graph-geometry-engineer):
1. Codex round 3 output should be available
2. Round 1 issues #11, #12 (reproducibility plumbing) at least documented
3. Re-extracted feature dataset (with layer_idx fix) ideally complete

## Validation of agent infrastructure

Despite the AGENT_INFRA_GAP discovered (custom agents not directly
invocable as `subagent_type`), the workaround (general-purpose +
embedded role md path) **delivers the specialist value**:
- Each agent caught issues lead session missed (lead's review identified
  8 weaknesses; the 4 agents collectively identified 15 substantive issues,
  10 of which were not in the lead review)
- Each stayed within domain (no scope drift)
- Each produced actionable, line-referenced output
- Each ended in `[ROLE PASS]`

**Verdict**: Path A (general-purpose wrapper) is sufficient for
Strategy v3. Path B (registering custom subagent types) is not blocking.
