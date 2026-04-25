# src/utils/naming.py
import re
import pandas as pd
from pathlib import Path

def sanitize_name(name: str) -> str:
    """
    파일 시스템 및 쉘에서 안전하게 사용할 수 있도록 Net 이름을 변환합니다.
    Rules:
    - '/' (Hierarchy) -> '_'
    - '[', ']' (Bus)  -> '_'
    - '\', '$' (Escape/Var) -> '_'
    - 중복된 '_'는 하나로 축소
    """
    if not isinstance(name, str): return str(name)
    
    # 1. Replace unsafe chars with underscore
    safe_name = re.sub(r'[\/\[\]\\\$]', '_', name)
    
    # 2. Collapse multiple underscores (e.g. __ -> _)
    safe_name = re.sub(r'_+', '_', safe_name)
    
    # 3. Strip leading/trailing underscores
    safe_name = safe_name.strip('_')
    
    return safe_name

class NameRegistry:
    """
    원본 이름과 변환된 이름(Safe Name) 간의 매핑을 관리하고 저장합니다.
    """
    def __init__(self):
        self.original_to_safe = {}
        self.safe_to_original = {}

    def register(self, original_name: str) -> str:
        """이름을 등록하고 Safe Name을 반환합니다. 충돌 시 처리 로직 포함."""
        if original_name in self.original_to_safe:
            return self.original_to_safe[original_name]
        
        base_safe = sanitize_name(original_name)
        safe_name = base_safe
        
        # Collision Handling (드물지만, a/b와 a_b가 동시에 존재할 경우)
        counter = 1
        while safe_name in self.safe_to_original:
            # 충돌 발생 시 원본이 같은지 확인
            if self.safe_to_original[safe_name] == original_name:
                break
            # 다른 원본이 같은 safe_name을 점유 중이면 renaming
            safe_name = f"{base_safe}_v{counter}"
            counter += 1
            
        self.original_to_safe[original_name] = safe_name
        self.safe_to_original[safe_name] = original_name
        return safe_name

    def get_original(self, safe_name: str):
        return self.safe_to_original.get(safe_name, None)

    def save_csv(self, path: Path):
        """매핑 테이블을 CSV로 저장"""
        data = [
            {'original_name': orig, 'safe_name': safe} 
            for orig, safe in self.original_to_safe.items()
        ]
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
        
    def load_csv(self, path: Path):
        """CSV에서 매핑 테이블 로드"""
        if not path.exists(): return
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            orig = row['original_name']
            safe = row['safe_name']
            self.original_to_safe[orig] = safe
            self.safe_to_original[safe] = orig