"""
Quick evaluation on small in-distribution designs.
Avoids the hour-long TEST_DEFS run by using 'valid' split tiles
from gcd_f3 / spi_top_f3 which have golden SPEFs available.
"""
import sys, argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import torch
import pandas as pd
import configs.config as cfg
from src.models.neural_field import DeepPEX_Model
from src.evaluation.evaluator import run_direct_eval

DEFAULT_DESIGNS = ['intel22_gcd_f3', 'intel22_spi_top_f3', 'intel22_aes_cipher_top_f3']


def main():
    parser = argparse.ArgumentParser(description='Quick in-distribution MAPE eval')
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--designs', nargs='+', default=DEFAULT_DESIGNS,
                        help='Design names to evaluate')
    parser.add_argument('--split', default='valid',
                        help='Manifest split to use (valid / train / all)')
    args = parser.parse_args()

    DEVICE = f"cuda:{args.gpu}"
    DATA_DIR = Path(cfg.PROCESSED_DIR)
    MODEL_DIR = Path(cfg.OUTPUT_DIR) / "active_learning" / args.model_name

    model = DeepPEX_Model(cfg).to(DEVICE)
    ckpt = MODEL_DIR / "best_model.pth"
    if not ckpt.exists():
        print(f"No checkpoint: {ckpt}")
        sys.exit(1)

    print(f">>> Loading checkpoint: {ckpt}")
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    manifest = pd.read_csv(DATA_DIR / "dataset_manifest.csv", low_memory=False)
    if args.split == 'all':
        eval_df = manifest
    else:
        eval_df = manifest[manifest['split'] == args.split]
    eval_df = eval_df[eval_df['design_name'].isin(args.designs)].reset_index(drop=True)

    if eval_df.empty:
        print(f"No tiles for designs={args.designs} split={args.split}")
        sys.exit(1)

    print(f"Designs: {eval_df['design_name'].unique().tolist()}")
    print(f"Total tiles: {len(eval_df):,}")

    run_direct_eval(model, eval_df, DATA_DIR, MODEL_DIR, DEVICE)


if __name__ == '__main__':
    main()
