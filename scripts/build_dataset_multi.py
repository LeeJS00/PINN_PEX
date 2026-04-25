# scripts/build_dataset_multi.py
import sys
import pandas as pd
import subprocess
from pathlib import Path
from tqdm import tqdm
import shutil

sys.path.append(str(Path(__file__).parent.parent))
import configs.config as cfg

def main():
    # Root Output Dir
    root_out_dir = Path(cfg.PROCESSED_DIR)
    root_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = root_out_dir / "dataset_manifest.csv"
    # [NEW] PT Root Dir 설정
    pt_root_dir = Path(cfg.PROCESSED_DIR).parent / "intel22_pt"
    pt_root_dir.mkdir(parents=True, exist_ok=True)
    pt_manifest_path = pt_root_dir / "dataset_manifest.csv"
    global_net_map_path = pt_root_dir / "global_net_mapping.csv"
    # 1. Load Existing Manifest & Processed Designs
    processed_designs = set()
    if manifest_path.exists():
        try:
            manifest_df = pd.read_csv(manifest_path)
            processed_designs = set(manifest_df['design_name'].unique())
            print(f">>> Loaded existing manifest. Found {len(processed_designs)} already processed designs.")
        except Exception as e:
            print(f"⚠️ Failed to read manifest: {e}. Starting fresh.")
            manifest_df = pd.DataFrame()
    else:
        manifest_df = pd.DataFrame()

    # 2. Collect All Designs (Train + Test)
    train_stems = {p.stem for p in cfg.TRAIN_DEFS}
    test_stems = {p.stem for p in cfg.TEST_DEFS}
    all_defs = list(set(cfg.TRAIN_DEFS + cfg.TEST_DEFS))
    
    print(f">>> [Multi-Build] Total {len(all_defs)} designs in config.")
    
    for def_path in all_defs:
        if not def_path.exists():
            print(f"⚠️ Warning: DEF file not found: {def_path}")
            continue
            
        design_name = def_path.stem
        # [NEW] Skip Logic: 이미 매니페스트에 등록된 디자인인지 확인
        if design_name in processed_designs:
            print(f"\n>>> ⏭️ Skipping Design: {design_name} (Already in manifest)")
            continue
            
        design_out_dir = root_out_dir / design_name
        design_out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n>>> Processing Design: {design_name}")
        print(f"    Output Dir: {design_out_dir}")
        print(f"    PT Out Dir: {pt_root_dir / design_name}")
        # Call build_dataset.py with specific output directory
        # (This reuses the robust logic from the single-file script)
        cmd = [
            "python3", "scripts/build_dataset.py",
            "--def_path", str(def_path),
            "--out_dir", str(design_out_dir),
            "--pt_out_dir", str(pt_root_dir / design_name),
            "--num_workers", "64"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            
            # Collect generated map for this design
            # build_dataset.py generates {dir_name}_map.csv inside the parent of out_dir?
            # Or usually inside out_dir parent. Let's check build_dataset.py logic.
            # It saves to: out_path.parent / f"{out_path.name}_map.csv"
            # Here out_path is design_out_dir. So map is at root_out_dir / {design_name}_map.csv
            
            local_map_path = root_out_dir / f"{design_name}_map.csv"
            
            if local_map_path.exists():
                df = pd.read_csv(local_map_path)
                # Add Relative Path Column
                # e.g., "gcd/gcd__net1.pkl"
                df['rel_path'] = df['sample_filename'].apply(lambda x: f"{design_name}/{x}")
                df['design_name'] = design_name
                
                if design_name in test_stems:
                    df['split'] = 'test'
                else:
                    # Train/Valid Split (Random stratified within the design)
                    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
                    valid_count = int(len(df) * cfg.VALID_RATIO)
                    df['split'] = 'train'
                    if valid_count > 0:
                        df.loc[:valid_count-1, 'split'] = 'valid'

                # 매니페스트 누적 및 즉시 저장
                manifest_df = pd.concat([manifest_df, df], ignore_index=True)
                manifest_df.to_csv(manifest_path, index=False)
                pt_df = manifest_df.copy()
                pt_df['sample_filename'] = pt_df['sample_filename'].str.replace('.pkl.gz', '.pt', regex=False)
                pt_df['rel_path'] = pt_df['rel_path'].str.replace('.pkl.gz', '.pt', regex=False)
                pt_df.to_csv(pt_manifest_path, index=False)
                local_net_map = root_out_dir / f"{design_name}_net_mapping.csv"
                if local_net_map.exists():
                    nm_df = pd.read_csv(local_net_map)
                    nm_df['design_name'] = design_name
                    if global_net_map_path.exists():
                        global_nm_df = pd.read_csv(global_net_map_path)
                        global_nm_df = pd.concat([global_nm_df, nm_df], ignore_index=True)
                    else:
                        global_nm_df = nm_df
                    global_nm_df.to_csv(global_net_map_path, index=False)
                    shutil.copy(local_net_map, pt_root_dir / local_net_map.name)
                    
                processed_designs.add(design_name)
                # Cleanup local map file (merge into global manifest)
                local_map_path.unlink()
                print(f"    ✅ Successfully appended {len(df)} samples to manifest.")
            else:
                print(f"⚠️ Warning: No map generated for {design_name}")

        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {design_name}: {e}")

    # 4. Summary
    print(f"\n=== Build Complete ===")
    if not manifest_df.empty:
        print(f"Total Samples in Manifest: {len(manifest_df)}")
        print("\n--- Split Summary ---")
        print(manifest_df['split'].value_counts())
        print("\n--- Design Summary ---")
        print(manifest_df.groupby(['split', 'design_name']).size())
    else:
        print("No data generated.")

if __name__ == "__main__":
    main()