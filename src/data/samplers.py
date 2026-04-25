# src/data/samplers.py (새로 생성하거나 finetuner.py 상단에 추가)
import pandas as pd
from torch.utils.data import Sampler
from collections import defaultdict
import random

class NetGroupedSampler(Sampler):
    """
    풀칩 조립의 모순을 해결하기 위해, 같은 Net에 속한 타일들을 하나의 배치로 묶어주는 샘플러
    """
    def __init__(self, dataset_df, batch_size=64):
        self.dataset_df = dataset_df
        self.batch_size = batch_size
        self.net_to_indices = defaultdict(list)
        
        # 넷 단위로 타일 인덱스 그룹화
        for idx, net_name in enumerate(self.dataset_df['net_name']):
            self.net_to_indices[net_name].append(idx)
            
        self.net_names = list(self.net_to_indices.keys())

    def __iter__(self):
        random.shuffle(self.net_names)
        batch = []
        for net_name in self.net_names:
            indices = self.net_to_indices[net_name]
            
            # 한 넷의 타일들을 배치에 추가
            for idx in indices:
                batch.append(idx)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        
        # 남은 찌꺼기 타일들 반환
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return (len(self.dataset_df) + self.batch_size - 1) // self.batch_size