"""PINNPEX EDA-style PEX pipeline.

Stages:
  1. extract_geometry — parse DEF + LEF + layers.info → per-net cuboid pkls
  2. build_features  — cuboid pkls → 145-dim hand features + 3-stream cuboid arrays
  3. predict_caps    — features → per-net total_cap_fF (super-ensemble inference)
  4. decompose_caps  — total_cap → c_gnd + per-pair coupling distribution
  5. compute_R       — cuboids + layer stack → analytic per-net resistance
  6. write_spef      — assemble all → IEEE 1481-1999 SPEF

CLI: scripts/predict_spef_e2e.py
"""
