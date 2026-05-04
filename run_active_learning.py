# run_active_learning.py
import torch
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import gc
from pathlib import Path
import argparse
import os
import random
import numpy as np


def apply_global_seed(seed: int) -> None:
    """Seed every RNG that affects training reproducibility.

    torch.compile + cudnn benchmarking remain non-deterministic by design;
    the goal here is to give each seed a distinct, reproducible *trajectory*,
    not bit-exact reproduction.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

sys.path.append(str(Path(__file__).parent))

from src.models.neural_field import DeepPEX_Model
from src.models.baselines import GNN_Cap
from src.active_learning.selector import PhysicsSelector
from src.active_learning.oracle import FullChipPEXOracle
from src.data.replay_buffer import DesignLevelReplayBuffer
from src.preprocessing.layer_parser import LayerInfoParser
from src.physics.materials import BEOLMaterialStack
from src.trainers.finetuner import NeuralFieldFinetuner
import configs.config as cfg
from src.utils.profiler import RuntimeProfiler

def load_or_create_predefined_cache(pool_df, train_buffer, val_buffer, oracle, def_map, cache_dir, num_train=2000, num_val=1500):
    """
    [Fast Engineering Mode]
    모델의 구조, 피처, 그래디언트를 빠르게 테스트하기 위해 고정된 크기의 Train/Valid 셋을 캐싱합니다.

    Returns the set of (design_name, net_name) pairs reserved for validation,
    so callers can anti-join them out of the AL pool.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache_path = cache_dir / "predefined_train_subset.csv"
    val_cache_path = cache_dir / "predefined_valid_subset.csv"

    # 1. 캐시 히트 (Cache Hit): 1초 만에 로딩
    if train_cache_path.exists() and val_cache_path.exists():
        print(f"\n>>> ⚡ [FAST MODE] Loading Predefined Datasets from {cache_dir.name}...")
        train_df = pd.read_csv(train_cache_path)
        val_df = pd.read_csv(val_cache_path)

        # Detect & repair legacy caches contaminated by the pre-fix sampler.
        # Older caches sometimes shared (design,net) pairs across both files,
        # which leaked validation nets into the train buffer.
        train_pairs = set(zip(train_df['design_name'], train_df['net_name']))
        val_pairs   = set(zip(val_df['design_name'],   val_df['net_name']))
        leaked_pairs = train_pairs & val_pairs
        if leaked_pairs:
            print(f"  ⚠️ [Cache Audit] {len(leaked_pairs)} (design,net) pairs found in BOTH caches "
                  f"— removing from TRAIN cache to preserve validation purity.")
            mask = ~pd.MultiIndex.from_arrays(
                [train_df['design_name'], train_df['net_name']]
            ).isin(leaked_pairs)
            train_df = train_df[mask].reset_index(drop=True)

        for d_name in val_df['design_name'].unique():
            d_def_path = def_map.get(d_name)
            if d_def_path:
                d_spef = oracle.generate_golden_spef(d_name, d_def_path)
                val_buffer.add_design(d_name, val_df[val_df['design_name'] == d_name], d_spef)

        for d_name in train_df['design_name'].unique():
            d_def_path = def_map.get(d_name)
            if d_def_path:
                d_spef = oracle.generate_golden_spef(d_name, d_def_path)
                train_buffer.add_design(d_name, train_df[train_df['design_name'] == d_name], d_spef)

        print(f"  - Train Tiles: {len(train_buffer.all_data)} | Valid Tiles: {len(val_buffer.all_data)}")
        return val_pairs

    # 2. 캐시 미스 (Cache Miss): 최초 1회 생성
    # Order: build VALIDATION cache first, then TRAIN cache with validation
    # nets excluded. The manifest split is tile-level so 12.3% of nets have
    # tiles in both train and valid splits; sampling them independently can
    # land the same (design,net) pair in both caches. Anti-joining keeps val
    # purity at the cost of slightly fewer train candidates.
    print("\n>>> 🐢 [CACHE MISS] Creating Predefined Datasets for the first time...")

    candidates_val = pool_df[pool_df['split'] == 'valid'].copy()
    candidates_val = candidates_val[~candidates_val['design_name'].str.contains('mpeg')]
    val_unique = candidates_val[['design_name', 'net_name']].drop_duplicates()
    val_sample = val_unique.sample(n=min(num_val, len(val_unique)), random_state=42)
    val_full = pd.merge(pool_df, val_sample, on=['design_name', 'net_name'])

    print(">>> Building Validation Cache...")
    for d_name in val_full['design_name'].unique():
        d_def_path = def_map.get(d_name)
        if d_def_path:
            d_spef = oracle.generate_golden_spef(d_name, d_def_path)
            val_buffer.add_design(d_name, val_full[val_full['design_name'] == d_name], d_spef)
    if not val_buffer.all_data.empty:
        val_buffer.all_data.to_csv(val_cache_path, index=False)
        print(f"  ✅ Saved valid cache: {len(val_buffer.all_data)} tiles.")

    val_pairs = set(zip(val_sample['design_name'], val_sample['net_name']))

    candidates_train = pool_df[pool_df['split'] == 'train'].copy()
    candidates_train = candidates_train[~candidates_train['design_name'].str.contains('mpeg')]
    train_unique = candidates_train[['design_name', 'net_name']].drop_duplicates()
    train_unique = train_unique[
        ~pd.MultiIndex.from_arrays(
            [train_unique['design_name'], train_unique['net_name']]
        ).isin(val_pairs)
    ]
    train_sample = train_unique.sample(n=min(num_train, len(train_unique)), random_state=42)
    train_full = pd.merge(pool_df, train_sample, on=['design_name', 'net_name'])

    print(">>> Building Train Cache...")
    for d_name in train_full['design_name'].unique():
        d_def_path = def_map.get(d_name)
        if d_def_path:
            d_spef = oracle.generate_golden_spef(d_name, d_def_path)
            train_buffer.add_design(d_name, train_full[train_full['design_name'] == d_name], d_spef)
    if not train_buffer.all_data.empty:
        train_buffer.all_data.to_csv(train_cache_path, index=False)
        print(f"  ✅ Saved train cache: {len(train_buffer.all_data)} tiles.")

    return val_pairs

def prepare_net_centric_validation(pool_df, val_buffer, oracle, def_map, num_val_nets=100):
    """
    타일 단위로 쪼개진 Validation 셋의 모순을 해결하기 위해,
    선택된 넷(Net)을 구성하는 '모든 타일'을 원본 데이터에서 끌어모아 완전한 넷을 구축합니다.
    """
    # 1. Validation 후보 넷 추출 ('mpeg' 제외)
    valid_candidates = pool_df[pool_df['split'] == 'valid'].copy()
    valid_candidates = valid_candidates[~valid_candidates['design_name'].str.contains('mpeg')]
    
    # 2. 유니크한 (design_name, net_name) 조합 추출 후 N개 임의 샘플링
    unique_nets = valid_candidates[['design_name', 'net_name']].drop_duplicates()
    if len(unique_nets) > num_val_nets:
        sampled_nets = unique_nets.sample(n=num_val_nets, random_state=42)
    else:
        sampled_nets = unique_nets
        
    # 3. 선택된 넷을 구성하는 "모든" 타일을 원본 pool_df에서 싹쓸이 (Train/Test 파편화 복구)
    val_full_tiles = pd.merge(pool_df, sampled_nets, on=['design_name', 'net_name'])
    print(f"\n>>> 🛡️ Building Validation Set: {len(sampled_nets)} Fully-Assembled Nets ({len(val_full_tiles)} tiles)...")
    
    # 4. 디자인별로 Oracle 추출 및 Buffer 주입
    for v_design in val_full_tiles['design_name'].unique():
        v_def_path = def_map.get(v_design)
        if v_def_path:
            v_spef = oracle.generate_golden_spef(v_design, v_def_path)
            design_val_tiles = val_full_tiles[val_full_tiles['design_name'] == v_design]
            val_buffer.add_design(v_design, design_val_tiles, v_spef)
            
    # 5. 데이터 누수(Leakage) 방지: 학습 풀에서 Validation에 쓰인 타일 영구 삭제
    pool_df = pool_df[~pool_df['sample_filename'].isin(val_full_tiles['sample_filename'])]
    pool_df = pool_df[pool_df['split'] == 'train'].reset_index(drop=True)
    
    val_loader = val_buffer.get_dataloader() if not val_buffer.all_data.empty else None
    return pool_df, val_loader


def main(args):
    apply_global_seed(args.seed)
    print(f">>> Global seed set to {args.seed}")

    MAX_BUDGET_RATIO = cfg.AL_MAX_BUDGET_RATIO
    MIN_ENTROPY_THRESHOLD = cfg.AL_MIN_ENTROPY_THRESHOLD
    TRAIN_STEPS_PER_ITER = (
        args.steps_per_iter if getattr(args, 'steps_per_iter', None) is not None
        else cfg.AL_TRAIN_STEPS_PER_ITER
    )
    AL_SAMPLING_METHOD = cfg.AL_SAMPLING_METHOD
    USE_FAST_ENGINEERING_MODE = True
    GPU_ID = args.gpu if args.gpu is not None else cfg.GPU_ID
    DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"

    DATA_DIR = Path(cfg.PROCESSED_DIR)
    basis_ckpt_dir = Path(cfg.OUTPUT_DIR) / "checkpoints" / cfg.RUN_NAME
    OUTPUT_DIR = Path(cfg.OUTPUT_DIR) / "active_learning"
    MODEL_DIR = OUTPUT_DIR / f"{args.model_name}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Init
    layer_map = LayerInfoParser(cfg.LAYERS_INFO_PATH).parse()
    mat_stack = BEOLMaterialStack(layer_map)
    
    if args.model_type == "DeepPEX":
        model = DeepPEX_Model(cfg).to(DEVICE)
        # Load SSL Weights
        ssl_ckpts = sorted(list(basis_ckpt_dir.glob("bem_ssl_ep*.pth")), key=lambda x: int(x.stem.split('ep')[-1]))
        if len(ssl_ckpts) == 0:
            print(f"⚠️ Warning: SSL Basis not found. Training from scratch (Unstable).")
        else:
            latest_ckpt = ssl_ckpts[-1]
            print(f">>> Loading Checkpoint: {latest_ckpt.name}")
            state = torch.load(latest_ckpt, map_location=DEVICE)
            
            # Checkpoint Keys may vary. Handling module prefix if DDP was used.
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            
            # [CRITICAL FIX] Filter out mismatched tensors (e.g. modified cpl_mlp dimensions)
            current_state = model.state_dict()
            filtered_state = {}
            for k, v in state.items():
                if k in current_state and v.shape == current_state[k].shape:
                    filtered_state[k] = v
                else:
                    print(f"⚠️ Skipping weight '{k}' due to shape mismatch or missing key.")
                
            missing, unexpected = model.load_state_dict(filtered_state, strict=False)
            model.freeze_ssl_layers() # SSL 지식 보존 및 Finetune 파라미터 격리
            print(f"✅ Loaded & Frozen Basis Model. (Missing Keys: {len(missing)} - Normal for new Physics layers)")
    
    elif args.model_type == "GNNCap":
        model = GNN_Cap(cfg).to(DEVICE)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, dynamic=True)
    
    # [NEW] 도입된 Full-Chip 파이프라인 모듈
    selector = PhysicsSelector(model, mat_stack, device=DEVICE)
    oracle = FullChipPEXOracle(OUTPUT_DIR)
    train_buffer = DesignLevelReplayBuffer(max_designs=50)
    val_buffer = DesignLevelReplayBuffer(max_designs=10)
    finetuner = NeuralFieldFinetuner(model, device=DEVICE, lr=cfg.AL_LR)
    
    # 매핑용 DEF 딕셔너리 생성
    def_map = {p.stem: p for p in cfg.TRAIN_DEFS + cfg.TEST_DEFS}
    
    # 2. Pool Load
    map_path = Path(cfg.PROCESSED_DIR) / "dataset_manifest.csv"
    if not map_path.exists():
        raise FileNotFoundError(f"Manifest not found: {map_path}")
        
    pool_df = pd.read_csv(map_path)
    
    if USE_FAST_ENGINEERING_MODE:
        print("\n>>> 🚀 [FAST ENGINEERING MODE] Using Predefined Datasets...")
        val_pairs = load_or_create_predefined_cache(pool_df, train_buffer, val_buffer, oracle, def_map, cache_dir=OUTPUT_DIR / "cache")
        val_loader = val_buffer.get_dataloader()
        pool_df = pool_df[pool_df['split'] == 'train'].reset_index(drop=True)
        # Anti-join: a net with mixed train/valid tiles can leave its train
        # tiles in pool_df after the split filter, letting AL re-label and
        # train on a net that's already in val_buffer. Drop them.
        if val_pairs:
            pre = len(pool_df)
            pool_df = pool_df[
                ~pd.MultiIndex.from_arrays(
                    [pool_df['design_name'], pool_df['net_name']]
                ).isin(val_pairs)
            ].reset_index(drop=True)
            dropped = pre - len(pool_df)
            if dropped > 0:
                print(f"  🛡️ [Leak Guard] Removed {dropped} train-pool tiles "
                      f"belonging to validation nets ({len(val_pairs)} pairs).")
    elif 'split' in pool_df.columns:
        pool_df, val_loader = prepare_net_centric_validation(pool_df, val_buffer, oracle, def_map, num_val_nets=100)
    else:
        val_loader = None
    
    available_designs = pool_df['design_name'].unique()

    if AL_SAMPLING_METHOD == "Sorted":
        al_target_designs = sorted(available_designs)[:3]
        print(f"\n🎯 [Contribution] Restricting AL Pool to 3 Designs: {al_target_designs}")
    elif AL_SAMPLING_METHOD == "SSL":
        TARGET_DESIGN_COUNT = 3
        print(f"\n>>> 🔍 Macro-Scanning {len(available_designs)} designs for uncertainty...")
        design_entropy = {}
        
        for d_name in available_designs:
            d_pool = pool_df[pool_df['design_name'] == d_name]
            # 각 디자인에서 무작위로 100개 타일만 스캔하여 대략적인 불확실성 측정
            scan_sample = d_pool.sample(min(100, len(d_pool)))
            _, _, avg_ent = selector.evaluate_pool(scan_sample, DATA_DIR, batch_size=cfg.AL_BATCH_SIZE)
            design_entropy[d_name] = avg_ent
            
        # 엔트로피(불확실성)가 가장 높은 순으로 정렬하여 Top-K 추출
        sorted_designs = sorted(design_entropy.items(), key=lambda x: x[1], reverse=True)
        # If mpeg is in top-N, skip!
        for i, (name, ent) in enumerate(sorted_designs):
            if 'mpeg' in name:
                sorted_designs.pop(i)
                break

        al_target_designs = [x[0] for x in sorted_designs[:TARGET_DESIGN_COUNT]]
        
        print("\n🎯 [Autonomous AL] Model selected the top-3 most difficult designs:")
        for i, (name, ent) in enumerate(sorted_designs):
            if i < TARGET_DESIGN_COUNT:
                print(f"  [{i+1}] {name} (Entropy: {ent:.4f})  <-- Selected")
            else:
                print(f"  [{i+1}] {name} (Entropy: {ent:.4f})")

    elif AL_SAMPLING_METHOD == 'Predefined':
        al_target_designs = cfg.AL_PREDEFINED_DESIGNS
        print(f"\n🎯 [Predefined] Restricting AL Pool to {al_target_designs}")

    pool_df = pool_df[pool_df['design_name'].isin(al_target_designs)].reset_index(drop=True)
    
    total_samples = len(pool_df)
    session_budget_path = MODEL_DIR / "al_session_budget.csv"
    if session_budget_path.exists():
        session_budget_path.unlink() # 새 실험 시 초기화

    total_nets = len(pool_df[['design_name', 'net_name']].drop_duplicates())
    TOTAL_NET_BUDGET = int(total_nets * MAX_BUDGET_RATIO) # 예: 전체 넷의 5%만 학습
    NETS_PER_ITER = max(5, TOTAL_NET_BUDGET // getattr(cfg, 'AL_FINE_ITERS', 10))
    
    print(f"=== Active Learning Plan (Full-Chip Assembly Paradigm) ===")
    print(f"Unlabeled Pool : {total_nets} Nets ({total_samples} tiles)")
    print(f"Hard Budget Cap: {TOTAL_NET_BUDGET} Nets ({MAX_BUDGET_RATIO*100}%)")
    print(f"Target Entropy : < {MIN_ENTROPY_THRESHOLD}")
    print("==========================================================")
    
    log_csv_path = MODEL_DIR / f"al_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_data = []

    # 3. AL Loop
    current_labeled_count = 0
    iteration = 0
    al_profiler = RuntimeProfiler(MODEL_DIR / "al_macro_runtime.csv")

    max_iters = getattr(args, 'max_iters', None)

    while True:
        print(f"\n>>> [Iteration {iteration}] Starting AL Cycle...")
        if max_iters is not None and iteration >= max_iters:
            print(f"\n>>> 🛑 [STOP] --max_iters cap reached ({iteration}/{max_iters}).")
            break
        if len(pool_df) == 0: break
        if current_labeled_count + NETS_PER_ITER > TOTAL_NET_BUDGET:
            print(f"\n>>> 🛑 [STOP] Hard Net Budget Cap Exhausted ({current_labeled_count}/{TOTAL_NET_BUDGET}).")
            break
            
        # (1) 불확실성이 높은 '타일'들 선별
        # 1. Pool에서 무작위로 타일을 뽑아 평가 (속도를 위해 전체의 10%만 샘플링)
        MAX_POOL_EVAL = 5000
        if len(pool_df) > MAX_POOL_EVAL:
            scan_pool = pool_df.sample(n=MAX_POOL_EVAL, random_state=iteration)
        else:
            scan_pool = pool_df.sample(frac=1.0, random_state=iteration) # 작을 때는 전부 스캔
        
        # 2. Selector를 수정하지 않고, 반환된 점수(Entropy)를 Dataframe에 병합
        # tile_entropies_df, max_ent, avg_ent = selector.evaluate_pool(scan_pool, DATA_DIR, batch_size=cfg.AL_BATCH_SIZE)
        fast_batch_size = cfg.AL_BATCH_SIZE * 4 
        tile_entropies_df, max_ent, avg_ent = selector.evaluate_pool(scan_pool, DATA_DIR, batch_size=fast_batch_size)
        if avg_ent < MIN_ENTROPY_THRESHOLD:
            print(f"\n>>> 🎯 [STOP] Target Accuracy Achieved! (Avg Entropy {avg_ent:.4f} < {MIN_ENTROPY_THRESHOLD}).")
            break

        # -----------------------------------------------------------------
        # [CRITICAL SHIFT] Strict Net-Level Expansion
        # AL이 선택한 타일이 속한 '넷(Net)'을 식별하고, 그 '넷'의 모든 타일만 가져옴.
        # 같은 디자인이라도 선택받지 못한 다른 쉬운 넷들은 절대 가져오지 않음!
        # -----------------------------------------------------------------
        # selected_df = pool_df[pool_df['sample_filename'].isin(candidates)]
        # unique_nets = selected_df[['design_name', 'net_name']].drop_duplicates()
        
        # # 선택된 어려운 넷들에 속한 타일만 풀칩 조립을 위해 싹쓸이
        # nets_to_add_df = pd.merge(pool_df, unique_nets, on=['design_name', 'net_name'])
        # 3. Net 단위로 GroupBy 하여 Max Entropy 산출
        net_entropy = tile_entropies_df.groupby(['design_name', 'net_name'])['entropy'].max().reset_index()
        
        # 4. 엔트로피가 가장 높은 상위 N개의 넷(Net) 선정
        top_nets = net_entropy.sort_values(by='entropy', ascending=False).head(NETS_PER_ITER)

        # [L4] CTS/버퍼 넷 강제 주입: entropy 기반 선택만으론 이 넷들이 빠질 수 있음
        cts_pattern = r'(?i)(^CTS_|^clk|_clk|^FE_PHN|^FE_OFN.*clk|^buf_clk|^clock)'
        selected_names = set(top_nets['net_name'])
        if not any(top_nets['net_name'].str.contains(cts_pattern, regex=True)):
            cts_candidates = pool_df[
                pool_df['net_name'].str.contains(cts_pattern, regex=True) &
                ~pool_df['net_name'].isin(selected_names)
            ][['design_name', 'net_name']].drop_duplicates()
            if len(cts_candidates) > 0:
                cts_inject = cts_candidates.sample(min(3, len(cts_candidates)), random_state=42)
                cts_rows = net_entropy[net_entropy['net_name'].isin(cts_inject['net_name'])]
                top_nets = pd.concat([top_nets, cts_rows], ignore_index=True).drop_duplicates('net_name')
                print(f"  [L4] Force-injected {len(cts_inject)} CTS/clk nets into AL batch.")

        unique_nets = top_nets[['design_name', 'net_name']]
        
        # 5. 선정된 Net들에 속한 풀칩 내 '모든 타일' 확보
        nets_to_add_df = pd.merge(pool_df, unique_nets, on=['design_name', 'net_name'])
        designs_in_batch = nets_to_add_df['design_name'].unique()
        # print(f"\n>>> 🏷️ AL Acquired {len(unique_nets)} Difficult Nets ({len(nets_to_add_df)} Tiles) from {len(designs_in_batch)} Designs.")
        print(f"\n>>> 🏷️ AL Acquired Top {len(unique_nets)} Difficult Nets ({len(nets_to_add_df)} Tiles) from {len(designs_in_batch)} Designs. (Avg Net Entropy: {top_nets['entropy'].mean():.4f})")
        
        # (2) Oracle 캐시 로드 및 버퍼 주입
        al_profiler.start("Oracle_SPEF_Gen")
        for design_name in designs_in_batch:
            def_path = def_map.get(design_name)
            if not def_path: continue
            print(f"\n>>> 🏷️ Generating Golden SPEF for {design_name}...")
            spef_path = oracle.generate_golden_spef(design_name, def_path)
            design_tiles_df = nets_to_add_df[nets_to_add_df['design_name'] == design_name].copy()
            train_buffer.add_design(design_name, design_tiles_df, spef_path)
        al_profiler.stop("Oracle_SPEF_Gen")

        # (3) 누적 관리 및 Pool에서 삭제
        labeled_tiles_this_iter = train_buffer.all_data[
            train_buffer.all_data['sample_filename'].isin(nets_to_add_df['sample_filename'])
        ].copy()
        
        # (3) 정답이 기입된 데이터를 누적 관리 및 CSV 저장
        labeled_tiles_this_iter['al_iteration'] = iteration + 1
        write_header = not session_budget_path.exists()
        labeled_tiles_this_iter.to_csv(session_budget_path, mode='a', header=write_header, index=False)
        # current_labeled_count += len(nets_to_add_df)
        current_labeled_count += len(unique_nets)
        
        # 이번에 학습한 넷들은 Pool에서 영구 제외
        pool_df = pool_df[~pool_df['sample_filename'].isin(nets_to_add_df['sample_filename'])]
            
        # (4) Finetuning 가동
        al_profiler.start("Model_Finetuning")
        train_loader = train_buffer.get_dataloader()
        trained_nets = len(train_buffer.all_data['net_name'].unique())
        print(f">>> [FineTuner] Training for {TRAIN_STEPS_PER_ITER} steps on {len(train_buffer.all_data)} tiles ({trained_nets} Core Nets)...")

        try:
            loss = finetuner.train_steps(train_loader, val_loader=val_loader, max_steps=TRAIN_STEPS_PER_ITER, save_dir=MODEL_DIR, al_iter=iteration)
        except BaseException as e:
            import traceback
            crash_path = MODEL_DIR / f"crash_iter{iteration+1}_{datetime.now().strftime('%H%M%S')}.txt"
            with open(crash_path, 'w') as f:
                f.write(traceback.format_exc())
                try:
                    f.write(torch.cuda.memory_summary())
                except Exception:
                    pass
            torch.save(model.state_dict(), MODEL_DIR / f"emergency_iter{iteration+1}.pth")
            print(f"[CRASH] Iteration {iteration+1} training failed. Crash log: {crash_path}")
            raise
        al_profiler.stop("Model_Finetuning")
        
        # 로깅 및 모델 저장
        log_entry = {
            'iteration': iteration + 1,
            'train_loss': loss,
            'pool_size': len(pool_df),
            'labeled_size': current_labeled_count,
            'avg_entropy': avg_ent
        }
        log_data.append(log_entry)
        pd.DataFrame(log_data).to_csv(log_csv_path, index=False)

        torch.save(model.state_dict(), MODEL_DIR / f"model_iter_{iteration+1}.pth")
        del train_loader
        gc.collect()
        torch.cuda.empty_cache()
        al_profiler.save_and_reset("AL_Cycle", f"Iter_{iteration}")
        iteration += 1

    print("\n>>> Active Learning Complete.")
    print(f"Total Labeled Tiles Used: {current_labeled_count} / TOTAL_NET_BUDGET")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Active Learning for PEX Net Labeling")
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--model_type', default="DeepPEX", choices=['DeepPEX', 'GNNCap'])
    parser.add_argument('--seed', type=int, default=0,
                        help='Global RNG seed applied to torch / numpy / random / cuda. '
                             'Used by the 5-seed measurement protocol.')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='Cap the AL loop at this many iterations (e.g. 1 for the '
                             '5-seed 5000-step measurement). None = use config budget.')
    parser.add_argument('--steps_per_iter', type=int, default=None,
                        help='Override cfg.AL_TRAIN_STEPS_PER_ITER for this run.')
    args = parser.parse_args()
    main(args)
