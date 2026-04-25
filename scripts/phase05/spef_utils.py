"""
Shared SPEF parsing utilities for Phase 0.5 experiments.
"""
import re
from collections import defaultdict

_RE_FLOAT = r'[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?'


def norm(name):
    return name.replace('\\', '').strip()


def net_of(node_str):
    """'clock:35' -> 'clock',  'inst:pin' -> 'inst:pin' (no colon means port/pin)."""
    parts = node_str.split(':')
    return norm(parts[0])


def parse_coupling_caps(spef_path):
    """
    Returns dict: target_net -> {aggressor_net: total_coupling_fF}

    Counts only *CAP entries with two nodes (coupling caps).
    Each entry is added to BOTH nets' views (symmetric).
    """
    coupling = defaultdict(lambda: defaultdict(float))
    current_net = None
    in_cap = False

    with open(spef_path, encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue

            if line.startswith('*D_NET'):
                tokens = line.split()
                current_net = norm(tokens[1]) if len(tokens) >= 2 else None
                in_cap = False
                continue

            if current_net is None:
                continue

            if line.startswith('*CAP'):
                in_cap = True
                continue
            if line.startswith('*RES') or line.startswith('*END') or line.startswith('*CONN'):
                in_cap = False
                continue

            if in_cap and line and line[0].isdigit():
                tokens = line.split()
                if len(tokens) >= 4:
                    try:
                        n1_net = net_of(tokens[1])
                        n2_net = net_of(tokens[2])
                        val = float(tokens[3])
                        if n1_net != n2_net:
                            coupling[n1_net][n2_net] += val
                            coupling[n2_net][n1_net] += val
                    except (ValueError, IndexError):
                        pass

    return dict(coupling)


def parse_node_positions(spef_path):
    """
    Returns dict: 'net:id' -> (x_um, y_um)

    Parses *CONN section:
      *P port_name I/O *C x y        -> node key = port_name
      *I inst:pin  I/O *C x y        -> node key = inst:pin
      *N net:id        *C x y        -> node key = net:id
    """
    positions = {}
    current_net = None
    in_conn = False

    re_c = re.compile(r'\*C\s+(' + _RE_FLOAT + r')\s+(' + _RE_FLOAT + r')')

    with open(spef_path, encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue

            if line.startswith('*D_NET'):
                tokens = line.split()
                current_net = norm(tokens[1]) if len(tokens) >= 2 else None
                in_conn = False
                continue

            if current_net is None:
                continue

            if line.startswith('*CONN'):
                in_conn = True
                continue
            if line.startswith('*CAP') or line.startswith('*RES') or line.startswith('*END'):
                in_conn = False
                continue

            if in_conn:
                m_c = re_c.search(line)
                if not m_c:
                    continue
                x, y = float(m_c.group(1)), float(m_c.group(2))

                if line.startswith('*P'):
                    # *P port_name I/O *C x y
                    tokens = line.split()
                    if len(tokens) >= 2:
                        positions[norm(tokens[1])] = (x, y)

                elif line.startswith('*I'):
                    # *I inst:pin I/O *C x y
                    tokens = line.split()
                    if len(tokens) >= 2:
                        positions[norm(tokens[1])] = (x, y)

                elif line.startswith('*N'):
                    # *N net:id *C x y
                    tokens = line.split()
                    if len(tokens) >= 2:
                        positions[norm(tokens[1])] = (x, y)

    return positions


def parse_coupling_with_positions(spef_path):
    """
    Returns list of (x1, y1, x2, y2, cap_fF) for every coupling *CAP entry
    where both nodes have known positions.

    Used for Phase 0.5C distance analysis.
    """
    node_pos = parse_node_positions(spef_path)

    entries = []
    current_net = None
    in_cap = False

    with open(spef_path, encoding='utf-8', errors='ignore') as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('//'):
                continue

            if line.startswith('*D_NET'):
                tokens = line.split()
                current_net = norm(tokens[1]) if len(tokens) >= 2 else None
                in_cap = False
                continue

            if current_net is None:
                continue

            if line.startswith('*CAP'):
                in_cap = True
                continue
            if line.startswith('*RES') or line.startswith('*END') or line.startswith('*CONN'):
                in_cap = False
                continue

            if in_cap and line and line[0].isdigit():
                tokens = line.split()
                if len(tokens) >= 4:
                    try:
                        n1 = norm(tokens[1])
                        n2 = norm(tokens[2])
                        val = float(tokens[3])
                        if net_of(n1) == net_of(n2):
                            continue  # ground cap, skip

                        # resolve position: try full node name, then net name
                        p1 = node_pos.get(n1) or node_pos.get(net_of(n1))
                        p2 = node_pos.get(n2) or node_pos.get(net_of(n2))
                        if p1 and p2:
                            entries.append((p1[0], p1[1], p2[0], p2[1], val))
                    except (ValueError, IndexError):
                        pass

    return entries
