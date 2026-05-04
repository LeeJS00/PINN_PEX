"""
test_spef_parser.py — verify the one-pass SPEF stream parser.

The new pex_v3 SPEF parser gives all-nets in O(1) file scan vs legacy O(N²).
We verify it on a synthetic SPEF fragment to confirm it returns the same
shape as expected.
"""
from __future__ import annotations
from pathlib import Path

import pytest

from src.baselines.feature_dataset import (
    stream_spef_nets,
    parse_spef_to_dict,
    _normalize_name,
    _extract_net_name,
)


SAMPLE_SPEF = """\
*SPEF "IEEE 1481-1998"
*DESIGN "test_design"
*DATE "2026-05-01"

*D_NET net_a 1.234
*CONN
*I net_a:1 I
*CAP
1 net_a:2 0.500
2 net_a:3 0.300
3 net_a:4 net_b:5 0.200
4 net_a:6 net_c:7 0.100
*RES
1 net_a:1 net_a:2 5.0 // res annotation
2 net_a:2 net_a:3 3.0
*END

*D_NET net_b 0.700
*CAP
5 net_b:1 0.700
*END
"""


def test_normalize_name_strips_backslash():
    assert _normalize_name(r"net\[5\]") == "net[5]"


def test_extract_net_name_drops_node_id():
    assert _extract_net_name("net_a:42") == "net_a"


def test_stream_spef_nets_yields_two_records(tmp_path):
    spef_path = tmp_path / "test.spef"
    spef_path.write_text(SAMPLE_SPEF)
    records = list(stream_spef_nets(spef_path))
    assert len(records) == 2
    names = [r["net_name"] for r in records]
    assert names == ["net_a", "net_b"]


def test_stream_spef_nets_total_cap_correct(tmp_path):
    spef_path = tmp_path / "test.spef"
    spef_path.write_text(SAMPLE_SPEF)
    recs = list(stream_spef_nets(spef_path))
    assert recs[0]["total_cap_fF"] == 1.234
    assert recs[1]["total_cap_fF"] == 0.700


def test_stream_spef_nets_ground_cap(tmp_path):
    spef_path = tmp_path / "test.spef"
    spef_path.write_text(SAMPLE_SPEF)
    recs = list(stream_spef_nets(spef_path))
    # net_a ground cap = 0.500 + 0.300 = 0.800
    assert recs[0]["ground_cap_fF"] == pytest.approx(0.800)
    # net_b ground cap = 0.700
    assert recs[1]["ground_cap_fF"] == pytest.approx(0.700)


def test_stream_spef_nets_coupling_caps(tmp_path):
    spef_path = tmp_path / "test.spef"
    spef_path.write_text(SAMPLE_SPEF)
    recs = list(stream_spef_nets(spef_path))
    # net_a couples to net_b (0.200) and net_c (0.100); total = 0.300
    assert recs[0]["c_cpl_total_fF"] == pytest.approx(0.300)
    assert recs[0]["coupled_caps"]["net_b"] == pytest.approx(0.200)
    assert recs[0]["coupled_caps"]["net_c"] == pytest.approx(0.100)


def test_stream_spef_nets_resistance_with_annotation(tmp_path):
    spef_path = tmp_path / "test.spef"
    spef_path.write_text(SAMPLE_SPEF)
    recs = list(stream_spef_nets(spef_path))
    # net_a: 5.0 (annotated) + 3.0 (simple) = 8.0
    assert recs[0]["total_res_ohm"] == pytest.approx(8.0)


def test_parse_spef_to_dict_keys_are_net_names(tmp_path):
    spef_path = tmp_path / "test.spef"
    spef_path.write_text(SAMPLE_SPEF)
    d = parse_spef_to_dict(spef_path)
    assert set(d.keys()) == {"net_a", "net_b"}
    assert d["net_a"]["total_cap_fF"] == 1.234


def test_stream_spef_nets_handles_missing_end(tmp_path):
    """Tolerate SPEFs that don't have a trailing *END (some StarRC variants)."""
    spef_path = tmp_path / "test.spef"
    # Drop the final *END
    truncated = SAMPLE_SPEF.rsplit("*END", 1)[0]
    spef_path.write_text(truncated)
    recs = list(stream_spef_nets(spef_path))
    # We still get net_a (closed by next *D_NET) and net_b (closed by EOF)
    assert len(recs) == 2
