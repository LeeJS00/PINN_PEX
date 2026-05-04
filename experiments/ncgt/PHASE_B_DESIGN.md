# Phase B — Edge Supervision Expansion

_Created: 2026-05-02_
_Status: Design — pre-staged while Phase A runs_
_Goal: 16% → 10-13% MAPE via denser supervision_

## 1. Problem statement

**Current supervision density**:
- 1 net-total CPL (`cpl_total` per net)
- 1 net-total GND (`gnd_total` per net)
- ~0.3-0.6% of enumerated edges have per-edge SPEF supervision (`is_supervised[e]`)

**Mismatch**:
- We enumerate ~5K-15K edges per net (E_local + E_mid + E_long)
- SPEF *CAP only has ~50-200 coupling entries per net (one per coupled aggressor net pair)
- Joint mapping success: per-edge `is_supervised` rate ≈ 0.3-0.6%
- Result: **96%+ of NN's edge predictions trained only via net-total constraint** (1 supervision point for 5K-15K edges)

**Consequence**:
- p95 = 37% (long-tail outliers dominate)
- Pearson r = 0.944 (rank good but magnitude off)
- Heteroscedasticity unfixed

## 2. Key insight — SPEF density structure

SPEF *CAP entries form a **per-aggressor-net coupling matrix**:

```
*D_NET target_net total_cap
*CAP
1  target:N1   target:N2     0.0010    ← intra-net cap (ignore)
2  target:N3   aggressor_A:M1  0.005   ← coupling to aggressor A
3  target:N3   aggressor_A:M5  0.003   ← coupling to aggressor A (different node)
4  target:N5   aggressor_B:M2  0.012   ← coupling to aggressor B
...
```

Per-aggressor-net total: `Σ c | (n1 ∈ target, n2 ∈ aggressor_A)`. This is **always available** (no node mapping needed):

```python
cpl_per_aggr_net[A] = Σ_{(n1,n2,c) ∈ *CAP, net(n1)=target, net(n2)=A} c
```

Already computed in `build_edge_supervision` (sample.cpl_per_aggr_net dict).

**This gives ~50-200 supervision points per net instead of 1.** Order of magnitude denser than net-total.

## 3. Design

### 3.1 Model output augmentation

Currently:
```
pred_cpl_per_edge: (E,)              # per-edge prediction
pred_cpl_total = pred_cpl_per_edge.sum()   # net-level aggregation
```

Add:
```
pred_cpl_per_aggr_net: (N_aggr_nets,)
                      = scatter_add(pred_cpl_per_edge, aggr_net_id_per_edge)
where aggr_net_id_per_edge[e] = sample['aggr_net_ids'][edge_index[1, e]]
```

`pred_cpl_per_aggr_net[a] = Σ_{e: aggr_net(e)=a} pred_cpl_per_edge[e]`

### 3.2 Loss expansion

```python
# Existing
loss_cpl_total = mape(pred_cpl_total, gt_cpl_total)

# New — Phase B
mask = gt_cpl_per_aggr_net > eps  # only supervise aggr nets with non-trivial coupling
loss_cpl_per_net = mape(pred_cpl_per_aggr_net[mask], gt_cpl_per_aggr_net[mask]).mean()

loss = ... + 1.0 * loss_cpl_per_net
```

Weight: starts at 1.0 (same as loss_cpl_total). After ablation: tune.

### 3.3 Sample data structure

`NCGTSample.cpl_per_aggr_net` is currently `Dict[int, float]`. Convert to dense tensor:

```python
# In to_torch():
n_aggr_nets = max(self.cpl_per_aggr_net.keys()) + 1  # max aggr_net_id
cpl_per_aggr_net_tensor = torch.zeros(n_aggr_nets)
for aggr_id, cpl in self.cpl_per_aggr_net.items():
    cpl_per_aggr_net_tensor[aggr_id] = cpl
out['gt_cpl_per_aggr_net'] = cpl_per_aggr_net_tensor
out['n_aggr_nets'] = torch.tensor(n_aggr_nets, dtype=torch.long)
```

### 3.4 Per-edge aggr_net_id lookup

`sample['aggr_net_ids']: (A,)` gives net id for each aggressor segment.
`edge_index[1, e]` gives aggressor segment idx.
Combine: `aggr_net_id_per_edge = sample['aggr_net_ids'][edge_index[1]]`  shape (E,)

Then `scatter_add(pred_cpl_per_edge, aggr_net_id_per_edge, dim_size=n_aggr_nets)`.

## 4. Expected effect

| Metric | Phase A (current) | Phase B (target) |
|---|---|---|
| Supervision density per net | 1 (net-total) | 50-200 (per-aggr-net) |
| Pearson r | 0.944 | 0.97+ |
| p95 MAPE | 37% | 25-30% |
| Mean MAPE | 16% | 10-13% |

Heuristic: each additional supervision point reduces variance approximately as 1/√n. Going from 1 → 50 = 7× variance reduction in residual error. If residual error variance dominates the 16% mean, this could close to 10-12%.

## 5. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `cpl_per_aggr_net` aggressor net ids inconsistent across samples | P1 | Use sample-local aggr_net_id (already done in build_sample) — verify |
| Some aggr_nets have only 1 SPEF entry → very noisy | P2 | Mask gt < threshold (e.g., 0.001 fF) |
| Memory: dense tensor of size max_aggr_net_id | P3 | Max n_aggr_nets seen per net is ~few hundred — small |
| Loss gradient explosion (50× more supervision points) | P2 | Start with weight=0.3, tune |
| Per-edge loss ≠ Per-net-aggr loss conflict | P2 | Drop per-edge loss when per-net-aggr active (redundant) |

## 6. Implementation plan

Edit list (estimated 50-80 LOC changes):

1. **`ncgt_dataset.py`**:
   - `NCGTSample.to_torch()`: add `gt_cpl_per_aggr_net` tensor + `n_aggr_nets`
   - Lines: ~10

2. **`ncgt_model.py`**:
   - `forward()`: add `pred_cpl_per_aggr_net` via scatter_add
   - Lines: ~15

3. **`train_ncgt.py`**:
   - `compute_losses()`: add `loss_cpl_per_net`
   - `hand_tuned_combined()`: add weight for new loss
   - `evaluate()`: track per-aggr-net MAPE in metrics
   - Lines: ~30

4. **5-seed wrapper**:
   - Add C4 config: vanilla + per-aggr-net supervision
   - Lines: ~5

## 7. Validation protocol

After implementation:
1. Smoke test: 1-net forward — verify `pred_cpl_per_aggr_net` shape matches `gt_cpl_per_aggr_net`
2. Single-design 5-seed: compare to Phase A C1 (16.22% baseline)
3. Multi-design 5-seed: compare to Phase A C2
4. If improvement <2pp: rollback (Plan v4 §5 gate criterion)
5. If improvement ≥2pp: proceed to Phase C (physics base accuracy)

## 8. Activation order

1. Phase A complete → record C1/C2/C3 baseline
2. Implement Phase B changes (~1 hour)
3. Smoke test (~5 min)
4. Phase B 5-seed runs (~1.5-2 hours, 2 configs × 5 seeds)
5. Compare: Phase B mean vs Phase A mean (Mann-Whitney U test)
6. Decision: keep / tune / rollback

Then Phase C (physics base accuracy improvement: etch-stop + ILD series capacitance).

## 9. Open questions

- Should we drop `is_supervised` per-edge loss when per-aggr-net active? Likely YES (redundant signal).
- How to handle aggressor nets without SPEF entry (`cpl_per_aggr_net[a] = 0`)? They're truly uncoupled — supervise as zero.
- What's the right weight for `loss_cpl_per_net` vs `loss_cpl_total`? Start equal (1.0), tune from ablation.

## 10. Code changes — pre-staged (apply after Phase A)

See:
- `/home/jslee/projects/PINNPEX/experiments/ncgt/PHASE_B_PATCH.md` (to be written) for actual diffs
