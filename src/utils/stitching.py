# src/utils/stitching.py
import pandas as pd
import numpy as np
from configs import config

class NetStitcher:
    def __init__(self, window_size=config.WINDOW_SIZE[0], overlap=config.TILING_OVERLAP):
        self.W = window_size
        self.O = overlap

    def stitch_net(self, tiles_df):
        """
        하나의 Net을 구성하는 Tile들의 예측값을 받아 가중 합산(Stitching)합니다.
        tiles_df: DataFrame containing ['capacitance', 'resistance', 'tile_idx', 'total_windows']
        """
        # 타일 인덱스로 정렬
        tiles = tiles_df.sort_values('tile_idx')
        num_tiles = len(tiles)
        
        if num_tiles == 0:
            return 0.0, 0.0
        
        if num_tiles == 1:
            # 타일이 1개면 가중치 1.0 (그대로 사용)
            return tiles['capacitance'].iloc[0], tiles['resistance'].iloc[0]

        # 가중치 계산 (Trapezoidal Weighting)
        # Start/End: (W - O/2) / W
        # Middle:    (W - O) / W
        
        # 1. 기본적으로 Middle 가중치 적용
        weight_middle = (self.W - self.O) / self.W
        weights = np.full(num_tiles, weight_middle)
        
        # 2. 양쪽 끝 타일 가중치 보정
        weight_edge = (self.W - (self.O / 2)) / self.W
        weights[0] = weight_edge
        weights[-1] = weight_edge
        
        # 3. Capacitance 합산 (Parallel Sum)
        stitched_cap = np.sum(tiles['capacitance'].values * weights)
        
        # 4. Resistance 합산 (Series Sum)
        # 저항은 겹치는 구간을 0.5배 해서 더하거나, 유효 길이만큼만 더해야 함.
        # Cap과 동일한 비율 논리를 적용하면 근사적으로 맞음.
        stitched_res = np.sum(tiles['resistance'].values * weights)
        
        return stitched_cap, stitched_res

    def stitch_dataframe(self, df):
        """
        전체 데이터프레임을 받아서 Net별로 Stitching을 수행하고 결과를 반환합니다.
        """
        results = []
        
        # Group by unique Net (DEF + NetName)
        grouped = df.groupby(['def_name', 'net_name'])
        
        for (def_name, net_name), group in grouped:
            cap, res = self.stitch_net(group)
            
            # 메타데이터는 첫 번째 타일 것 사용
            first_row = group.iloc[0]
            
            results.append({
                'def_name': def_name,
                'net_name': net_name,
                'stitched_cap': cap,
                'stitched_res': res,
                'num_tiles': len(group)
            })
            
        return pd.DataFrame(results)