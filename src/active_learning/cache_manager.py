# src/active_learning/cache_manager.py
import sqlite3
import json
import hashlib
import gzip
from pathlib import Path
import pandas as pd

class PEXCacheManager:
    def __init__(self, db_path="master_starrc_history.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        if not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
        """SQLite 데이터베이스 초기화 및 인덱스 생성 (WAL 모드로 동시성 확보)"""
        with sqlite3.connect(str(self.db_path)) as conn:
            # WAL (Write-Ahead Logging) 모드 활성화: 멀티프로세스 동시 접근 허용
            conn.execute('PRAGMA journal_mode=WAL;')
            conn.execute('PRAGMA synchronous=NORMAL;')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS pex_history (
                    hash_key TEXT PRIMARY KEY,
                    sample_filename TEXT,
                    net_name TEXT,
                    capacitance REAL,
                    resistance REAL,
                    coupled_caps TEXT,
                    status TEXT,
                    runtime REAL,
                    timestamp TEXT
                )
            ''')
            # 빠른 검색을 위한 파일명 인덱스 추가
            conn.execute('CREATE INDEX IF NOT EXISTS idx_filename ON pex_history(sample_filename);')

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """파일의 실제 바이너리 내용을 기반으로 SHA-256 물리적 지문(Fingerprint) 생성"""
        if not filepath.exists(): return "UNKNOWN"
        hasher = hashlib.sha256()
        with open(filepath, 'rb') as f:
            # 파일이 크지 않으므로 한 번에 읽어서 해싱 (필요시 chunking)
            hasher.update(f.read())
        return hasher.hexdigest()

    def get_cached_results(self, filenames: list, data_dir: Path) -> pd.DataFrame:
        """
        요청된 파일 리스트에 대해 캐시 히트(Hit)된 결과만 DataFrame으로 반환합니다.
        물리적 해시(Fingerprint)가 일치하는 경우에만 유효한 캐시로 인정합니다.
        """
        if not filenames: return pd.DataFrame()
        
        # 1. 파일명 기반으로 1차 DB 고속 조회
        placeholders = ','.join(['?'] * len(filenames))
        # [CRITICAL FIX] SQLite의 "Too many SQL variables" (Limit 999) 에러 방지용 청킹(Chunking)
        chunk_size = 900
        dfs =[]
        
        with sqlite3.connect(str(self.db_path)) as conn:
            for i in range(0, len(filenames), chunk_size):
                chunk = filenames[i:i + chunk_size]
                placeholders = ','.join(['?'] * len(chunk))
                query = f"SELECT * FROM pex_history WHERE sample_filename IN ({placeholders}) AND status='OK'"
                
                df_chunk = pd.read_sql_query(query, conn, params=chunk)
                if not df_chunk.empty:
                    dfs.append(df_chunk)
            
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if df.empty: return pd.DataFrame()

        # 2. 2차 검증: 물리적 해시(Fingerprint) 대조
        valid_rows = []
        for _, row in df.iterrows():
            fname = row['sample_filename']
            design_key = fname.split('__')[0]
            full_path = data_dir / design_key / fname
            
            current_hash = self.compute_file_hash(full_path)
            
            # 해시가 일치할 때만 캐시 히트로 인정 (파일이 변경되었으면 Miss 처리)
            if current_hash == row['hash_key']:
                # JSON 문자열을 dict로 복원
                row_dict = row.to_dict()
                row_dict['coupled_caps'] = json.loads(row_dict['coupled_caps']) if row_dict['coupled_caps'] else {}
                row_dict['source'] = 'MASTER_CACHE'
                valid_rows.append(row_dict)

        return pd.DataFrame(valid_rows)

    def update_cache(self, results_list: list, data_dir: Path):
        """새로 StarRC를 돌린 결과(List of dict)를 DB에 일괄 삽입(Upsert)합니다."""
        if not results_list: return
        
        records = []
        for res in results_list:
            fname = res['sample_filename']
            design_key = fname.split('__')[0]
            full_path = data_dir / design_key / fname
            
            h_key = self.compute_file_hash(full_path)
            cpl_caps_json = json.dumps(res.get('coupled_caps', {}))
            
            records.append((
                h_key, fname, res.get('net_name'), res.get('capacitance'), 
                res.get('resistance'), cpl_caps_json, res.get('status'), 
                res.get('runtime', 0.0), res.get('timestamp')
            ))
            
        with sqlite3.connect(self.db_path) as conn:
            # UPSERT: 해시 키가 같으면 최신 런타임/상태로 덮어쓰기
            conn.executemany('''
                INSERT INTO pex_history (hash_key, sample_filename, net_name, capacitance, resistance, coupled_caps, status, runtime, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hash_key) DO UPDATE SET
                    capacitance=excluded.capacitance,
                    resistance=excluded.resistance,
                    coupled_caps=excluded.coupled_caps,
                    status=excluded.status,
                    runtime=excluded.runtime,
                    timestamp=excluded.timestamp
            ''', records)