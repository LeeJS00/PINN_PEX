# scripts/parse_full_spef.py
import sys
import os
import re
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm

# Project Root Setup
sys.path.append(str(Path(__file__).parent.parent))
from configs import config

def normalize_name(name: str) -> str:
    """
    SPEF의 Escaped Name을 일반 이름으로 변환합니다.
    예: 'ctrl_state\$out\[0\]' -> 'ctrl_state$out[0]'
    """
    if not name: return ""
    return name.replace('\\', '').strip()

def parse_entire_spef(spef_path, csv_out_path):
    print(f"Processing SPEF: {spef_path.name}")
    
    data_rows = []
    
    # Regex Patterns
    # *D_NET netName capVal
    re_d_net = re.compile(r'^\*D_NET\s+(\S+)\s+([0-9\.\+eE\-]+)')
    # 1 node1 node2 resVal
    re_res_line = re.compile(r'^\d+\s+\S+\s+\S+\s+([0-9\.\+eE\-]+)')
    
    current_net = None
    current_cap = 0.0
    current_res = 0.0
    in_res_section = False
    
    def save_current_net():
        nonlocal current_net, current_cap, current_res
        if current_net:
            # 정규화된 이름으로 저장
            norm_name = normalize_name(current_net)
            data_rows.append({
                'net_name': norm_name,
                'full_cap': current_cap,
                'full_res': current_res
            })
    
    try:
        with open(spef_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # 1. New Net Start
                if line.startswith('*D_NET'):
                    # 이전 넷 저장
                    save_current_net()
                    
                    match = re_d_net.match(line)
                    if match:
                        current_net = match.group(1)
                        current_cap = float(match.group(2))
                        current_res = 0.0
                        in_res_section = False
                    else:
                        current_net = None
                    continue
                
                if current_net:
                    if line.startswith('*RES'):
                        in_res_section = True
                        continue
                    elif line.startswith('*END'):
                        save_current_net()
                        current_net = None
                        in_res_section = False
                        continue
                    elif line.startswith('*CONN') or line.startswith('*CAP'):
                        in_res_section = False
                        continue
                    
                    # 2. Resistance Summation
                    if in_res_section:
                        # 숫자 시작 라인만 파싱
                        if line[0].isdigit():
                            match = re_res_line.match(line)
                            if match:
                                current_res += float(match.group(1))

        # Last net save
        save_current_net()

    except Exception as e:
        print(f"❌ Error parsing {spef_path}: {e}")
        return

    # Save to CSV
    if data_rows:
        df = pd.DataFrame(data_rows)
        # def_name 컬럼 추가 (나중에 merge할 때 key로 사용)
        df['def_name'] = spef_path.stem + ".def" 
        
        csv_out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_out_path, index=False)
        print(f"✅ Saved {len(df)} nets to {csv_out_path}")
    else:
        print(f"⚠️ No nets found in {spef_path}")

def main():
    parser = argparse.ArgumentParser(description="Parse Full SPEF files to CSV for quick verification.")
    # 기본 경로는 config나 사용자 환경에 맞게 조정
    parser.add_argument('--spef_dir', type=str, default="data/spef", help="Directory containing .spef files")
    parser.add_argument('--out_dir', type=str, default="data/spef_parsed", help="Directory to save .csv files")
    args = parser.parse_args()

    in_path = Path(args.spef_dir)
    out_path = Path(args.out_dir)
    
    if not in_path.exists():
        print(f"❌ SPEF directory not found: {in_path}")
        # Fallback to search in current dir or spef_data
        if Path("spef_data").exists():
            print("Found 'spef_data', using it instead.")
            in_path = Path("spef_data")
        else:
            return

    spef_files = list(in_path.glob("*.spef"))
    if not spef_files:
        print(f"❌ No .spef files found in {in_path}")
        return

    print(f"Found {len(spef_files)} SPEF files. Converting to CSV...")
    
    for spef_file in tqdm(spef_files):
        csv_name = spef_file.stem + "_full_rc.csv"
        parse_entire_spef(spef_file, out_path / csv_name)

if __name__ == "__main__":
    main()