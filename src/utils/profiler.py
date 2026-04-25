# ==========================================================
# FILE: src/utils/profiler.py
# ==========================================================
import time
import torch
import pandas as pd
from collections import defaultdict
from pathlib import Path

class RuntimeProfiler:
    def __init__(self, log_path="runtime_analysis.csv"):
        self.log_path = Path(log_path)
        self.starts = {}
        self.accumulated = defaultdict(float)
        self.counts = defaultdict(int)

    def start(self, key):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        self.starts[key] = time.time()

    def stop(self, key):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        if key in self.starts:
            elapsed = time.time() - self.starts[key]
            self.accumulated[key] += elapsed
            self.counts[key] += 1
            del self.starts[key]

    def save_and_reset(self, phase, step_or_iter):
        row = {'Phase': phase, 'Step_Iter': step_or_iter}
        for k, v in self.accumulated.items():
            row[f"{k}_Total(s)"] = round(v, 4)
            row[f"{k}_Avg(s)"] = round(v / max(1, self.counts[k]), 4)
        
        df = pd.DataFrame([row])
        header = not self.log_path.exists()
        df.to_csv(self.log_path, mode='a', header=header, index=False)
        
        self.accumulated.clear()
        self.counts.clear()