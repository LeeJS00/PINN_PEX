"""c_gnd per-cuboid allocator variants — owned by pex-gnd-allocator-owner.

Each module exports a single function with this contract:

    allocate_gnd(
        segments: list[SegmentRecord],   # net's WIRE segments
        c_gnd_total: float,              # per-net total in fF
        layer_info: dict,                # parsed layer.info
    ) -> dict[node_id, float]            # per-topology-node gnd cap in fF

Variants in this package:
- baseline.py             : v3 length × width × ε × 0.22 placeholder (current frontier)
- analytic_sakurai.py     : Sakurai-Tamaru top + bottom plate + fringe (TBD)
- nnls_calibrated.py      : NNLS-fit per-layer multiplier on Sakurai (TBD)
"""
