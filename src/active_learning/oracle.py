# src/active_learning/oracle.py (New Architecture)
import subprocess
from pathlib import Path
import configs.config as cfg

class FullChipPEXOracle:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_golden_spef(self, design_name, def_path):
        """
        타일링이나 맹글링 없이 원본 Full-Chip DEF를 StarRC에 넣어 100% 무결한 Golden SPEF를 추출합니다.
        """
        for spef_path in cfg.TRAIN_SPEFS:
            if design_name in spef_path.stem:
                golden_spef_path = spef_path
                return golden_spef_path
            
        print(f">>> [Oracle] Firing up StarRC for Full-Chip Design: {design_name}...")
        
        # TCL 템플릿 로드 및 경로 치환 (기존 run_starrc_template.tcl 활용)
        tcl_template_path = Path(cfg.PEX_TEMPLATE_PATH)
        with open(tcl_template_path, 'r') as f:
            tcl_content = f.read()
            
        tcl_content = tcl_content.replace('__DESIGN_NAME__', design_name)
        tcl_content = tcl_content.replace('__DEF_FILE_PATH__', str(def_path))
        tcl_content = tcl_content.replace('__SPEF_FILE_PATH__', str(golden_spef_path))
        
        run_dir = self.output_dir / f"run_{design_name}"
        run_dir.mkdir(exist_ok=True)
        
        tcl_file = run_dir / "starrc.cmd"
        with open(tcl_file, 'w') as f:
            f.write(tcl_content)
            
        # StarRC 실행
        log_file = run_dir / "starrc.log"
        cmd = f"StarRC {tcl_file} > {log_file} 2>&1"
        subprocess.run(cmd, shell=True, check=True)
        
        print(f"✅ [Oracle] Successfully generated Golden SPEF: {golden_spef_path.name}")
        return golden_spef_path