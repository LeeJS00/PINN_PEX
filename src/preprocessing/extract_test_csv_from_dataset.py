import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent.parent))
import configs.config as cfg

manifest_path = Path("/home/jslee/projects/PEX_SSL/data/processed/intel22__0403/dataset_manifest.csv")
manifest_df = pd.read_csv(manifest_path)
test_df = manifest_df[manifest_df['split'] == 'test']
test_df.to_csv("/home/jslee/projects/PEX_SSL/output_intel22__0403/active_learning/cache/test_manifest.csv", index=False)