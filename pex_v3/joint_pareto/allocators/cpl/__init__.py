"""c_cpl per-aggressor allocator variants — owned by pex-cpl-allocator-owner.

Each module exports a single function with this contract:

    allocate_cpl(
        target_segments: list[SegmentRecord],
        global_index: list[SegmentRecord],
        kdtree: cKDTree,
        target_net_name: str,
        c_cpl_total: float,
        layer_info: dict,
        max_dist_um: float = 5.0,
        top_k: int = 20,
    ) -> dict[aggr_net_name, float]      # per-aggressor cap in fF

Variants in this package:
- baseline.py            : v3 midpoint-distance² geometric (current frontier)
- overlap_3d.py          : 3D overlap area × ε / d_inter (TBD)
- layer_aware.py         : separate lateral / vertical / via-cap pathways (TBD)
"""
