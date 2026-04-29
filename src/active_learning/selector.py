# src/active_learning/selector.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from src.data.datasets import NeuralFieldSSLDataset, robust_collate

class PhysicsSelector:
    def __init__(self, model, material_stack=None, device='cuda'):
        self.model = model
        self.device = device

    def evaluate_pool(self, pool_df, data_dir, batch_size=8):
        """
        [NEW] Hierarchical Net-Centric AL을 위한 평가 함수.
        입력된 데이터프레임(타일들)의 엔트로피를 모두 측정하여 반환합니다.
        
        Returns:
            evaluated_df: 'entropy' 컬럼이 추가된 원본 DataFrame
            max_ent (float): 전체 타일 중 최대 엔트로피
            avg_ent (float): 전체 타일의 평균 엔트로피
        """
        print(f">>> Scanning {len(pool_df)} tiles using Flux Entropy...")
        
        # DataFrame 복사 및 인덱스 초기화 (결과 매핑 시 순서 보장을 위해)
        eval_df = pool_df.reset_index(drop=True).copy()
        
        dataset = NeuralFieldSSLDataset(data_dir, eval_df)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=robust_collate, num_workers=16, pin_memory=True)
        
        scores = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Scoring Flux Uncertainty"):
                if batch is None: continue
                cuboids, mask, meta_dict = batch
                cuboids, mask = cuboids.to(self.device), mask.to(self.device)
                
                # Coupling 연산 활성화하여 확률 분포(P) 획득
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    n_tiles = meta_dict.get('n_tiles', None)
                    endpoint_prox = meta_dict.get('endpoint_prox', None)
                    if isinstance(n_tiles, torch.Tensor): n_tiles = n_tiles.to(self.device)
                    if isinstance(endpoint_prox, torch.Tensor): endpoint_prox = endpoint_prox.to(self.device)
                    preds = self.model(cuboids, mask, compute_coupling=True, n_tiles=n_tiles, endpoint_prox=endpoint_prox)
                
                is_target = preds['is_target'] # (B, N)
                p_gnd = preds['p_gnd']         # (B, N)
                
                # -------------------------------------------------------------
                # [우아한 테크닉] KCL Flux Entropy (불확실성) 계산
                # H = - P_gnd * log(P_gnd) - sum(P_cpl * log(P_cpl))
                # -------------------------------------------------------------
                B_size, N_size = p_gnd.shape
                ent_cpl_nodes = torch.zeros(B_size, N_size, device=self.device)
                
                sparse_cpl = preds['sparse_cpl']
                if len(sparse_cpl['b_idx']) > 0:
                    b_e = sparse_cpl['b_idx']
                    src_e = sparse_cpl['src_idx']
                    p_cpl_e = sparse_cpl['p_cpl'] # (E,)
                    
                    # p_cpl_e가 0이 되는 것을 방지하기 위한 clamp
                    ent_cpl_e = -p_cpl_e * torch.log(torch.clamp(p_cpl_e, min=1e-9))
                    # Scatter Add로 타겟(Source) 노드 단위로 엔트로피 취합
                    ent_cpl_nodes.index_put_((b_e, src_e), ent_cpl_e, accumulate=True)
                
                ent_gnd = -p_gnd * torch.log(torch.clamp(p_gnd, min=1e-9))
                total_entropy = ent_gnd + ent_cpl_nodes # (B, N)
                
                # 타겟 큐보이드들의 엔트로피 평균값 산출 (배경, 어그레서 제외)
                tile_entropy = (total_entropy * is_target.float()).sum(dim=1) / torch.clamp(is_target.float().sum(dim=1), min=1.0)
                
                scores.extend(tile_entropy.cpu().tolist())
                
        # -------------------------------------------------------------
        # 결과 취합 (혹여 로딩에 실패한 타일이 있을 수 있으므로 길이를 맞춰줍니다)
        # -------------------------------------------------------------
        eval_df = eval_df.iloc[:len(scores)].copy()
        eval_df['entropy'] = scores
        
        avg_score = eval_df['entropy'].mean()
        max_score = eval_df['entropy'].max()
         
        return eval_df, max_score, avg_score