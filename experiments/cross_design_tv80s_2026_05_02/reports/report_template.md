# Cross-design tv80s test — autonomous 10h run, 2026-05-02

**Goal**: per-net `total_cap_fF` MAPE < 4% on tv80s, training on 9 small intel22 designs.

**Setup**
- Workspace: `experiments/cross_design_tv80s_2026_05_02/` (isolated; does not modify `pex_v3/` or other experiment trees)
- Train designs (9): aes_cipher_top, gcd, ibex_core, ldpc_decoder_802_3an, mc_top, spi_top, usbf_top, vga_enh_top, wb_conmax_top
- Validation: nova
- Test: tv80s (full chip)
- Source: cuboid pkls at `/data/PINNPEX/data/processed_v3/intel22/<design>/` + golden SPEF (StarRC)

**Pipeline**
1. Per-net feature extraction from cuboid pkls + SPEF targets → parquet cache
   - v1: 60 features, basic geometry + coupling
   - v2: 114 features with proper layer mapping, per-layer wirelength/area, top-k aggressor area, 1/d weighted coupling, refined compact_total
2. GBDT 5-seed: LightGBM + XGBoost + CatBoost on `log(total_cap_fF)`
3. DeepSet/MLP: 3-stream cuboid encoder (target/aggressor/power) + hand-feature MLP → `log(total_cap)`
4. Stack: per-row blending tuned on val
5. Bootstrap CI on per-net MAPE

**Results to fill**

| Model | Seeds | Val MAPE mean | Test MAPE mean (tv80s) | Test MAPE median | P90 | P99 |
|-------|-------|---------------|------------------------|------------------|-----|-----|
| LightGBM v1 |   |   |   |   |   |   |
| XGBoost  v1 |   |   |   |   |   |   |
| CatBoost v1 |   |   |   |   |   |   |
| LightGBM v2 |   |   |   |   |   |   |
| XGBoost  v2 |   |   |   |   |   |   |
| CatBoost v2 |   |   |   |   |   |   |
| DeepSet     |   |   |   |   |   |   |
| Blend mean  |   |   |   |   |   |   |
| Blend median|   |   |   |   |   |   |
| Blend val-tuned |   |   |   |   |   |   |

**Key features by importance** (top-15)

(filled at end)

**Lessons learned**

(filled at end)
