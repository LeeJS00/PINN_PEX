"""
Diag: SPEF unit header verification for calibration extractor.

Hard gate for the data-driven calibration init pipeline (docs/distillation_log.md
§3.1 Step 0B): all TRAIN_SPEFS + TEST_SPEFS must declare *C_UNIT 1.0 FF so we
can treat parsed cap values as fF directly without scaling.

Exit codes:
    0 = all SPEFs verified OK (C_UNIT == "1.0 FF")
    1 = at least one SPEF deviates (must scale at extraction time, or abort)
    2 = at least one SPEF missing or unreadable
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import configs.config as cfg


C_UNIT_RE = re.compile(r"^\*C_UNIT\s+([0-9eE+\-.]+)\s+(\S+)")


def parse_units(spef_path: Path) -> tuple[float, str] | None:
    """Return (factor, unit) from *C_UNIT line, or None if not found in first 50 lines."""
    with open(spef_path, "r") as f:
        for i, line in enumerate(f):
            if i > 50:
                break
            m = C_UNIT_RE.match(line.strip())
            if m:
                return float(m.group(1)), m.group(2)
    return None


def main() -> int:
    paths: list[Path] = list(cfg.TRAIN_SPEFS) + list(cfg.TEST_SPEFS)
    print(f">>> Checking C_UNIT on {len(paths)} SPEFs (TRAIN + TEST)")
    bad = 0
    missing = 0
    for p in paths:
        if not p.exists():
            print(f"  [MISSING] {p}")
            missing += 1
            continue
        result = parse_units(p)
        if result is None:
            print(f"  [NO_HEADER] {p.name}")
            bad += 1
            continue
        factor, unit = result
        ok = (abs(factor - 1.0) < 1e-9) and (unit.upper() == "FF")
        flag = "OK" if ok else "MISMATCH"
        print(f"  [{flag}] {p.name}: *C_UNIT {factor} {unit}")
        if not ok:
            bad += 1

    print()
    if missing > 0:
        print(f"FAIL: {missing} SPEF(s) missing on disk")
        return 2
    if bad > 0:
        print(f"FAIL: {bad}/{len(paths)} SPEF(s) deviate from 1.0 FF")
        return 1
    print(f"PASS: all {len(paths)} SPEFs use C_UNIT 1.0 FF")
    return 0


if __name__ == "__main__":
    sys.exit(main())
