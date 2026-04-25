# DeepPEX_v2/src/utils/spef_parser.py
import re

def normalize_name(name: str) -> str:
    """
    이름 비교를 위해 정규화합니다.
    1. 역슬래시(\) 제거 (Escape 문자 무시)
    2. 공백 제거
    """
    if not name: return ""
    return name.replace('\\', '').strip()
    
def extract_net_name(node_str: str) -> str:
    """
    SPEF 노드 이름(예: 'req_msg[8]:3')에서 콜론(:) 앞의 net_name만 추출 후 정규화
    """
    base = node_str.split(':')[0]
    return normalize_name(base)


def parse_spef_values(spef_path: str, target_net_name: str):
    """
    SPEF에서 Total Cap과 Total Res를 추출합니다.
    Returns: {'cap': float, 'res': float}
    """
    total_cap = 0.0
    total_res = 0.0
    ground_cap = 0.0
    coupled_caps = {}

    # Name Escaping
    candidates = {target_net_name}
    escaped = target_net_name.replace('[', '\\[').replace(']', '\\]').replace('/', '\\/')
    candidates.add(escaped)
    escaped = escaped.replace('\\', '') # SPEF 내부에서는 \ 자체가 escape 문자
    candidates.add(escaped)

    # Patterns
    p_d_net = re.compile(r'\*D_NET\s+(\S+)\s+([0-9\.\+eE\-]+)')
    # *RES 섹션: 1 node1 node2 value
    p_res = re.compile(r'\d+\s+\S+\s+\S+\s+([0-9\.\+eE\-]+)+\s+\/.\s+\$')

    in_res_section = False
    in_cap_section = False
    is_target_net = False

    try:
        with open(spef_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                if line.startswith('*D_NET'):
                    tokens = line.split()
                    if len(tokens) < 3:
                        continue
                    
                    match = p_d_net.match(line)
                    if match:
                        net = match.group(1)
                        if normalize_name(net) in candidates:
                            total_cap = float(match.group(2))
                            is_target_net = True
                            continue
                            # D_NET을 찾았으면 RES 섹션 탐색 준비 (보통 D_NET 뒤에 나옴)
                            
                # 2. Resistance (*RES)
                # SPEF 구조상 *D_NET 뒤에 *RES가 나오고, 그 안에 여러 넷의 저항이 나열됨.
                # 하지만 *D_NET이 요약본이고 *RES가 상세본이라 구조가 복잡함.
                # Mini-DEF StarRC 결과는 보통 넷이 1~2개뿐이므로 *RES 섹션 전체를 합산해도 무방함(VSS 제외).
                
                if is_target_net:
                    # 커플링 캡 탐색
                    if line.startswith('*CAP'):
                        in_cap_section = True
                        in_res_section = False
                        continue

                    if line.startswith('*RES'):
                        in_res_section = True
                        in_cap_section = False
                        continue
                    
                    if line.startswith('*END'):             
                        break

                    if in_cap_section:
                        tokens = line.split()
                        # [Case 1] Ground Cap (Self-Capacitance)
                        # Format: ID NODE VALUE (e.g., "27 n_1402:2 0.0270041")
                        if len(tokens) == 3:
                            try:
                                ground_cap += float(tokens[2])
                            except ValueError:
                                pass

                        # [Case 2] Coupling Cap (Cross-Capacitance)
                        # Format: ID NODE1 NODE2 VALUE (e.g., "1 n_1402:10 clk:73 0.0042")
                        elif len(tokens) >= 4:
                            n1 = extract_net_name(tokens[1])
                            n2 = extract_net_name(tokens[2])
                            
                            try:
                                val = float(tokens[3])
                                # Target Net이 아닌 쪽이 Aggressor
                                aggr = n2 if n1 in candidates else n1
                                coupled_caps[aggr] = coupled_caps.get(aggr, 0.0) + val
                            except ValueError:
                                pass

                    if in_res_section:
                        # *RES 섹션 내에서 타겟 넷의 저항만 발라내기는 까다로움 (노드 이름 매핑 필요).
                        # 하지만 Mini-DEF는 Target Net 하나만 Routing 되어 있고, 
                        # Aggressor는 VSS(Ground) 처리되거나 저항 추출 대상이 아님.
                        # 따라서 *RES 섹션의 모든 저항 값을 더하면 Target Net의 총 저항 근사치가 됨.
                        match = p_res.match(line)
                        if match:
                            total_res += float(match.group(1))
                            # print(float(match.group(1)))
                        else:
                            print('Unexpected RES line format:', line.strip())

        return {'cap': total_cap, 'res': total_res, 'ground_cap': ground_cap, 'coupled_caps': coupled_caps}
             
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    # Test Logic
    spef_path = "/home/jslee/projects/PEX_SSL/golden_data/spef_data/gcd_post_route.spef" # 생성하신 테스트 파일 경로
    target_net_name = "n_1402"
    result = parse_spef_values(spef_path, target_net_name)
    if result:
        print(f"Total Cap : {result['cap']} fF")
        print(f"Ground Cap: {result['ground_cap']} fF")
        print(f"Total Res : {result['res']} Ohm")
        print(f"Couplings :")
        for aggr, val in result['coupled_caps'].items():
            print(f"  - {aggr}: {val} fF")
    else:
        print("Failed to parse SPEF values.")