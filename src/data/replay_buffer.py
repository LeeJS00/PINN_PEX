# src/data/replay_buffer.py
import torch
import pandas as pd
from collections import defaultdict
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Sampler

from src.data.datasets import NeuralFieldFinetuneDataset, robust_collate
import configs.config as cfg

from src.evaluation.compare_spef import parse_spef_with_coordinates


class NetGroupedSampler(Sampler):
    """동일한 넷(Net)에 속한 타일들을 무조건 같은 배치(Batch)에 넣도록 그룹핑합니다."""
    def __init__(self, df, batch_nets=1, max_tiles_per_batch=128, net_weights=None):
        self.df = df
        self.batch_nets = batch_nets
        self.max_tiles = max_tiles_per_batch

        # 넷 단위로 인덱스 그룹핑
        self.net_to_indices = defaultdict(list)
        for idx, net_name in enumerate(df['net_name']):
            self.net_to_indices[net_name].append(idx)

        # [L2 MegaNet fix] 드랍 대신 서브샘플링 — CTS/버퍼 넷도 훈련에 참여
        self.net_names = list(self.net_to_indices.keys())
        mega_count = sum(1 for inds in self.net_to_indices.values() if len(inds) > self.max_tiles)
        if mega_count > 0:
            print(f"  [Sampler] {mega_count} Mega-Nets (>{self.max_tiles} tiles) → subsampled per iter (labels pre-scaled).")

        # Cap-weighted sampling: sqrt(cap) so large-cap nets are sampled more often.
        # net_weights is a dict {net_name: weight}. Falls back to uniform if None.
        if net_weights is not None:
            raw = np.array([float(net_weights.get(n, 1.0)) for n in self.net_names], dtype=np.float64)
            raw = np.clip(raw, a_min=1e-6, a_max=None)
            self.sampling_probs = raw / raw.sum()
        else:
            self.sampling_probs = None

    def __iter__(self):
        if self.sampling_probs is not None:
            # Weighted sampling without replacement (Gumbel-top-k trick)
            gumbel = -np.log(-np.log(np.random.uniform(0, 1, len(self.net_names)) + 1e-20) + 1e-20)
            order = np.argsort(-(np.log(self.sampling_probs + 1e-20) + gumbel))
            shuffled_nets = [self.net_names[i] for i in order]
        else:
            shuffled_nets = self.net_names.copy()
            np.random.shuffle(shuffled_nets)

        batch = []
        nets_in_batch = 0
        for net in shuffled_nets:
            indices = self.net_to_indices[net]
            if len(indices) > self.max_tiles:
                # 매 에폭마다 다른 타일을 무작위 샘플 → 전체 커버리지 확보
                sampled = np.random.choice(indices, self.max_tiles, replace=False).tolist()
            else:
                sampled = indices
            batch.extend(sampled)
            nets_in_batch += 1

            if nets_in_batch >= self.batch_nets:
                yield batch
                batch = []
                nets_in_batch = 0

        if batch: yield batch

    def __len__(self):
        return len(self.net_names) // self.batch_nets


class DesignLevelReplayBuffer:
    def __init__(self, max_designs=10):
        self.max_designs = max_designs
        self.all_data = pd.DataFrame()
        self.data_dir = Path(cfg.PROCESSED_DIR)
        self.golden_cache = {} # {design_name: {net_name: {'total': x, 'gnd': y}}}

    def _parse_golden_spef(self, design_name, spef_path):
        """Golden SPEF에서 Net별 풀칩 Total/GND 캡 파싱"""
        net_stats = defaultdict(lambda: {'total': 0.0, 'gnd': 0.0, 'cpl': defaultdict(float)})
        nets = parse_spef_with_coordinates(spef_path)
        
        for net_name, data in nets.items():
            net_stats[net_name]['total'] = data['total_cap']
            net_stats[net_name]['gnd'] = sum(data['gnd_caps'].values())
            for node, aggrs in data['cpl_caps'].items():
                for aggr_net, cap in aggrs.items():
                    net_stats[net_name]['cpl'][aggr_net] += cap
        self.golden_cache[design_name] = net_stats

    def add_design(self, design_name, tiles_df, golden_spef_path):
        """
        Active Learning에서 선택된 Design 1개의 모든 타일을 버퍼에 추가하고, 
        Golden 풀칩 캡을 Global Label로 주입합니다.
        """
        if design_name not in self.golden_cache:
            self._parse_golden_spef(design_name, golden_spef_path)
            
        g_dict = self.golden_cache[design_name]
        tiles_df = tiles_df.copy()
        # [CRITICAL SHIFT] 타일의 오염된 로컬 캡을 버리고, 풀칩 캡(Global Label)으로 덮어쓰기!
        tiles_df['capacitance'] = tiles_df['net_name'].map(lambda n: g_dict.get(n, {}).get('total', 0.0))
        tiles_df['ground_cap'] = tiles_df['net_name'].map(lambda n: g_dict.get(n, {}).get('gnd', 0.0))
        tiles_df['coupled_caps'] = tiles_df['net_name'].map(lambda n: dict(g_dict.get(n, {}).get('cpl', {})))
        # 정답이 0이거나 SPEF에 없는 넷(Dummy 등) 필터링
        tiles_df = tiles_df[tiles_df['capacitance'] > 0.0].copy()
        
        if self.all_data.empty:
            self.all_data = tiles_df
        else:
            self.all_data = pd.concat([tiles_df, self.all_data])
            
    def get_dataloader(self):
        if self.all_data.empty: raise ValueError("No data in buffer!")

        # Valid 라벨(0.0보다 큰 캡)만 학습
        train_df = self.all_data[self.all_data['capacitance'] > 0.0].reset_index(drop=True)
        print(f"  [Buffer] Training on Valid Budget: {len(train_df)} tiles")

        # [L2 MegaNet fix] scatter_add는 K/N 타일의 기여만 합산하므로,
        # MegaNet(N > 128)의 타겟 라벨을 coverage=K/N 비율로 pre-scale한다.
        # 이렇게 하면 scaled_target ≈ scatter_add(K tiles) — 학습 신호가 편향되지 않음.
        max_tiles = 128
        net_counts = train_df.groupby('net_name').size()
        coverage = (net_counts.clip(upper=max_tiles) / net_counts).rename('_cov')
        train_df = train_df.join(coverage, on='net_name')
        mega_nets = (train_df['_cov'] < 1.0)
        if mega_nets.any():
            train_df.loc[mega_nets, 'capacitance'] *= train_df.loc[mega_nets, '_cov']
            train_df.loc[mega_nets, 'ground_cap']  *= train_df.loc[mega_nets, '_cov']
            train_df.loc[mega_nets, 'coupled_caps'] = train_df.loc[mega_nets].apply(
                lambda r: {k: v * r['_cov'] for k, v in r['coupled_caps'].items()}, axis=1
            )
        train_df = train_df.drop(columns='_cov')

        dataset = NeuralFieldFinetuneDataset(self.data_dir, train_df, pad_size=1024)

        # Cap-weighted net sampling: sqrt(cap) so large-cap nets are more likely seen
        # each epoch. Combined with batch_nets > 1, this reduces gradient variance and
        # improves learning for large-cap nets (20-100fF bucket was under-predicted).
        batch_nets = getattr(cfg, 'AL_BATCH_NETS', 1)
        batch_max_tiles = getattr(cfg, 'AL_MAX_TILES_PER_BATCH', max_tiles)
        net_cap_max = train_df.groupby('net_name')['capacitance'].max()
        net_weights = (net_cap_max.clip(lower=0.01) ** 0.5).to_dict()
        sampler = NetGroupedSampler(train_df, batch_nets=batch_nets,
                                     max_tiles_per_batch=batch_max_tiles,
                                     net_weights=net_weights)

        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=robust_collate,
                            num_workers=cfg.AL_NUM_WORKERS, prefetch_factor=2,
                            persistent_workers=False, pin_memory=True)
        return loader
