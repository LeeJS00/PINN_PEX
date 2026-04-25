# src/utils/experiment_utils.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import configs.config as cfg

def get_cross_n_split(train_design_names, test_design_names=None, valid_ratio=0.1, seed=42):
    """
    Manifest를 로드하여 지정된 Design Group에 따라 Train/Valid/Test DataFrame을 반환합니다.
    
    Args:
        train_design_names (list): 학습에 사용할 디자인 이름들 (e.g. ['gcd_post_route', 'aes'])
        test_design_names (list): 테스트에 사용할 디자인 이름들 (Unseen)
        valid_ratio (float): Train Design 내에서 Validation으로 떼어낼 비율 (Segment 기준)
        
    Returns:
        train_df, valid_df, test_df
    """
    manifest_path = Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Run build_dataset_multi.py first.")
        
    df = pd.read_csv(manifest_path)
    
    # 1. Filter Test Set (Unseen Designs)
    test_df = pd.DataFrame()
    if test_design_names:
        test_df = df[df['design_name'].isin(test_design_names)].reset_index(drop=True)
        
    # 2. Filter Train Pool (Seen Designs)
    train_pool = df[df['design_name'].isin(train_design_names)].reset_index(drop=True)
    
    if train_pool.empty:
        raise ValueError(f"No samples found for train designs: {train_design_names}")
        
    # 3. Train/Valid Split (Segment-wise random split)
    # 같은 디자인 내에서 Net Segment 단위로 섞습니다.
    train_df, valid_df = train_test_split(train_pool, test_size=valid_ratio, random_state=seed)
    
    print(f"=== Cross-N Split Statistics ===")
    print(f"Train Designs : {train_design_names}")
    print(f"Test Designs  : {test_design_names}")
    print(f"--------------------------------")
    print(f"Train Samples : {len(train_df)}")
    print(f"Valid Samples : {len(valid_df)} ({valid_ratio*100}%)")
    print(f"Test Samples  : {len(test_df)}")
    print("================================")
    
    return train_df, valid_df, test_df