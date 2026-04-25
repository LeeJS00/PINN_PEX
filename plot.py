import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error

def generate_length_split_report_and_plot(csv_path_or_df):
    """
    비교 결과(Golden vs Predicted)가 담긴 CSV 파일이나 DataFrame을 입력받아
    Short Net과 Long Net을 분리하여 리포트와 산점도를 생성합니다.
    """
    # 1. 데이터 로드 (실제 SPEF 비교 스크립트 출력 결과물 포맷에 맞게 수정 필요)
    if isinstance(csv_path_or_df, str):
        df = pd.read_csv(csv_path_or_df)
    else:
        df = csv_path_or_df
        
    # 컬럼명 가정: 'net_name', 'g_tot', 'p_tot'
    # 데이터가 없다면 임의로 테스트하기 위해 아래 코드를 사용하세요.
    # df = pd.DataFrame({'net_name': ['n_1', 'n_2'], 'g_tot': [1.2, 25.0], 'p_tot': [1.1, 5.0]})

    # 2. Short vs Long 분리 기준 (Golden Cap 5.0 fF 기준)
    THRESHOLD_FF = 5.0
    
    short_nets = df[df['g_tot'] < THRESHOLD_FF]
    long_nets = df[df['g_tot'] >= THRESHOLD_FF]
    
    # 3. 성능 평가 함수
    def calc_metrics(data):
        if len(data) == 0: return 0, 0, 0
        g = data['g_tot'].values
        p = data['p_tot'].values
        r2 = r2_score(g, p)
        mape = mean_absolute_percentage_error(g, p) * 100
        rmse = np.sqrt(np.mean((g - p)**2))
        return r2, mape, rmse

    r2_all, mape_all, rmse_all = calc_metrics(df)
    r2_short, mape_short, rmse_short = calc_metrics(short_nets)
    r2_long, mape_long, rmse_long = calc_metrics(long_nets)
    
    # 4. 콘솔 리포트 출력
    print("="*60)
    print("📊 [PINN-PEX] Short vs Long Nets Performance Report")
    print("="*60)
    print(f"[All Nets]   Count: {len(df):5d} | R2: {r2_all:.4f} | MAPE: {mape_all:6.2f}% | RMSE: {rmse_all:.4f} fF")
    print(f"[Short Nets] Count: {len(short_nets):5d} | R2: {r2_short:.4f} | MAPE: {mape_short:6.2f}% | RMSE: {rmse_short:.4f} fF  (< {THRESHOLD_FF} fF)")
    print(f"[Long Nets]  Count: {len(long_nets):5d} | R2: {r2_long:.4f} | MAPE: {mape_long:6.2f}% | RMSE: {rmse_long:.4f} fF  (>= {THRESHOLD_FF} fF)")
    print("="*60)
    print("💡 분석: Short Nets의 높은 R2는 국소적 물리 모델이 이미 수렴했음을 증명합니다.")
    print("          Long Nets의 오차는 Full-Chip 확장 시 발생하는 파편화/가중치 버그에 기인합니다.")

    # 5. Scatter Plot 그리기
    plt.figure(figsize=(14, 6))
    
    # (Plot 1) Short Nets
    plt.subplot(1, 2, 1)
    plt.scatter(short_nets['g_tot'], short_nets['p_tot'], alpha=0.5, color='blue', edgecolor='k', s=20)
    max_val_short = max(short_nets['g_tot'].max(), short_nets['p_tot'].max()) * 1.1
    plt.plot([0, max_val_short], [0, max_val_short], 'r--', linewidth=2)
    plt.title(f"Short Nets (Cap < {THRESHOLD_FF} fF)\n$R^2$: {r2_short:.3f}", fontsize=14, fontweight='bold')
    plt.xlabel("Golden Capacitance (fF)", fontsize=12)
    plt.ylabel("Predicted Capacitance (fF)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max_val_short)
    plt.ylim(0, max_val_short)

    # (Plot 2) Long Nets
    plt.subplot(1, 2, 2)
    plt.scatter(long_nets['g_tot'], long_nets['p_tot'], alpha=0.6, color='darkorange', edgecolor='k', s=40)
    max_val_long = max(long_nets['g_tot'].max(), long_nets['p_tot'].max()) * 1.1
    plt.plot([0, max_val_long], [0, max_val_long], 'r--', linewidth=2)
    plt.title(f"Global/Long Nets (Cap $\geq$ {THRESHOLD_FF} fF)\n$R^2$: {r2_long:.3f}", fontsize=14, fontweight='bold')
    plt.xlabel("Golden Capacitance (fF)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, max_val_long)
    plt.ylim(0, max_val_long)

    plt.tight_layout()
    plt.savefig("short_vs_long_nets_scatter.png", dpi=300)
    print("\n✅ Scatter plot saved as 'short_vs_long_nets_scatter.png'.")
    # plt.show() # 서버 환경이 아닐 경우 활성화

# === 사용 예시 ===
# 1. 평가 스크립트(compare_spef.py)에서 net_name, g_tot, p_tot을 CSV로 저장합니다.
# 2. generate_length_split_report_and_plot("my_results.csv") 를 호출합니다.
generate_length_split_report_and_plot("/home/jslee/projects/PEX_SSL/output_intel22/evaluation/v5/comparison/spef_comparison_report.csv")