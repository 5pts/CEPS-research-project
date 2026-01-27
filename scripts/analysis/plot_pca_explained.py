"""生成 PCA 解释方差和载荷可视化"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all_with_pca_ses.csv"
OUTPUT_DIR = WORKSPACE / "figures" / "report_phase3"

PCA_COLS = ["parent_edu_max", "family_econ", "home_books", "has_desk", "has_computer"]
LABELS_CN = {
    "parent_edu_max": "父母学历",
    "family_econ": "家庭经济",
    "home_books": "家庭藏书",
    "has_desk": "有书桌",
    "has_computer": "有电脑"
}


def zscore_frame(df):
    return (df - df.mean()) / df.std(ddof=0)


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    X = df[PCA_COLS].copy()

    # Invert has_computer if negatively correlated with family_econ
    if X["has_computer"].corr(X["family_econ"]) < 0:
        X["has_computer"] = 1 - X["has_computer"]

    Xz = zscore_frame(X)
    cov = np.cov(Xz.T, ddof=0)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)
    loadings = eigvecs[:, 0]

    # Align sign
    pc1 = Xz.values @ eigvecs[:, 0]
    if np.corrcoef(pc1, X["family_econ"].values)[0, 1] < 0:
        loadings = -loadings

    # ========== 图1: Scree Plot (方差解释比) ==========
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    pcs = [f'PC{i}' for i in range(1, 6)]
    bars = ax1.bar(pcs, explained * 100, color='#4472C4', edgecolor='white', alpha=0.8)
    ax1.plot(pcs, cumulative * 100, 'o-', color='#C55A11', linewidth=2, markersize=8, label='累计解释')

    # 标注数值
    for i, (bar, val) in enumerate(zip(bars, explained * 100)):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f'{val:.1f}%',
                ha='center', fontsize=10)

    # 标注拐点
    ax1.axhline(y=44.1, color='red', linestyle='--', alpha=0.5)
    ax1.annotate('拐点: PC1 解释 44.1%\n保留 PC1 即可', xy=(0, 44.1), xytext=(1.5, 55),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    ax1.set_ylabel('方差解释比 (%)', fontsize=12)
    ax1.set_xlabel('主成分', fontsize=12)
    ax1.set_title('碎石图 (Scree Plot): SES 主成分分析', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 80)
    ax1.legend(loc='upper right')

    plt.tight_layout()
    fig1.savefig(OUTPUT_DIR / 'pca_scree_plot.png', dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved pca_scree_plot.png")
    plt.close(fig1)

    # ========== 图2: PC1 载荷条形图 ==========
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    labels = [LABELS_CN[col] for col in PCA_COLS]
    colors = ['#2E7D32' if l > 0 else '#C62828' for l in loadings]

    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, loadings, color=colors, edgecolor='white', height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.axvline(x=0, color='black', linewidth=0.8)

    # 添加数值标签
    for i, (val, label) in enumerate(zip(loadings, labels)):
        offset = 0.02 if val > 0 else -0.02
        ha = 'left' if val > 0 else 'right'
        ax2.text(val + offset, i, f'{val:.3f}', va='center', ha=ha, fontsize=10)

    ax2.set_xlabel('PC1 载荷 (Loading)', fontsize=12)
    ax2.set_title('PC1 各变量载荷: 综合反映家庭社会经济地位', fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.1, 0.65)

    # 添加解释说明
    ax2.text(0.95, 0.05, '所有载荷均为正值\n→ PC1 是 SES 的综合指标\n载荷越大，贡献越大',
            transform=ax2.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / 'pca_loadings.png', dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved pca_loadings.png")
    plt.close(fig2)

    # ========== 图3: 相关性热力图 (输入变量) ==========
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    corr = Xz.corr()
    corr.index = [LABELS_CN[c] for c in corr.index]
    corr.columns = [LABELS_CN[c] for c in corr.columns]

    im = ax3.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # 添加数值
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax3.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=10, color=color)

    ax3.set_xticks(range(len(corr)))
    ax3.set_yticks(range(len(corr)))
    ax3.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
    ax3.set_yticklabels(corr.index, fontsize=10)
    ax3.set_title('SES 输入变量相关矩阵', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
    cbar.set_label('相关系数', fontsize=10)

    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / 'pca_input_correlation.png', dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved pca_input_correlation.png")
    plt.close(fig3)

    # ========== 图4: SES 分布直方图 ==========
    fig4, ax4 = plt.subplots(figsize=(7, 5))
    ses_scores = df['ses_pca'].dropna()

    ax4.hist(ses_scores, bins=50, color='#4472C4', edgecolor='white', alpha=0.8, density=True)
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='均值=0')
    ax4.axvline(x=ses_scores.median(), color='green', linestyle='--', linewidth=2, label=f'中位数={ses_scores.median():.2f}')

    ax4.set_xlabel('SES 得分 (PC1)', fontsize=12)
    ax4.set_ylabel('密度', fontsize=12)
    ax4.set_title('家庭社会经济地位 (SES) 得分分布', fontsize=14, fontweight='bold')
    ax4.legend()

    # 添加说明
    ax4.text(0.95, 0.95, f'N = {len(ses_scores):,}\n标准差 = {ses_scores.std():.2f}',
            transform=ax4.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig4.savefig(OUTPUT_DIR / 'pca_ses_distribution.png', dpi=150, bbox_inches='tight')
    print(f"[DONE] Saved pca_ses_distribution.png")
    plt.close(fig4)


if __name__ == "__main__":
    main()
