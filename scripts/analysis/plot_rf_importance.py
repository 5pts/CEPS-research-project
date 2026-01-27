"""生成随机森林特征重要性条形图"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "results" / "phase3" / "ordinal_rf_feature_importance.csv"
OUTPUT_FILE = WORKSPACE / "figures" / "report_phase3" / "rf_feature_importance.png"

def main():
    df = pd.read_csv(DATA_FILE)

    # 中文标签映射
    label_map = {
        'cog_score': '认知能力',
        'ses_pca': '家庭SES',
        'linking_idx': '师生关系',
        'bonding_idx': '同伴关系',
        'hukou_type': '户籍类型'
    }

    df['label'] = df['feature'].map(label_map)
    df = df.sort_values('importance', ascending=True)

    # 颜色：社会资本用蓝色，其他用灰色
    colors = ['#4472C4' if f in ['linking_idx', 'bonding_idx'] else '#7F7F7F'
              for f in df['feature']]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df['label'], df['importance'], color=colors, edgecolor='white')

    # 添加数值标签
    for bar, val in zip(bars, df['importance']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=11)

    ax.set_xlabel('重要性得分 (Gini Importance)', fontsize=12)
    ax.set_title('随机森林变量重要性排名', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.5)

    # 添加图例说明
    ax.text(0.95, 0.05, '蓝色 = 学校社会资本\n灰色 = 控制变量',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
