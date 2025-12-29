import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set plotting style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all.csv"
OUTPUT_DIR = WORKSPACE / "rescued_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_distributions(df):
    print("Plotting Distributions...")
    
    # 1. Expectation Distribution
    plt.figure(figsize=(8, 6), dpi=300)
    ax = sns.countplot(x='expect_college', data=df, palette='viridis')
    plt.title("学生教育期望分布\n(N={}, Data Source: CEPS Wave 2)".format(len(df)), fontsize=14, pad=20)
    plt.xlabel("期望层级 (0=非本科, 1=本科及以上)", fontsize=12)
    plt.ylabel("学生人数 (Count)", fontsize=12)
    plt.xticks([0, 1], ['无本科意愿', '期望上本科'])
    
    # Add labels
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
                    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rescued_expect_dist.png")
    plt.close()
    
    # 2. Social Capital Distributions
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    sns.histplot(df['bonding_idx'].dropna(), kde=True, ax=ax[0], color='#4c72b0', line_kws={'linewidth': 2})
    ax[0].set_title("Bonding SC (师生关系) 分布直方图", fontsize=13)
    ax[0].set_xlabel("标准化得分 (Z-Score)", fontsize=11)
    ax[0].set_ylabel("频数", fontsize=11)
    
    sns.histplot(df['bridging_idx'].dropna(), kde=True, ax=ax[1], color='#dd8452', line_kws={'linewidth': 2})
    ax[1].set_title("Bridging SC (同伴氛围) 分布直方图", fontsize=13)
    ax[1].set_xlabel("标准化得分 (Z-Score)", fontsize=11)
    ax[1].set_ylabel("频数", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rescued_sc_dist.png")
    plt.close()

def plot_interactions(df):
    print("Plotting Interactions...")
    
    # Binning Bonding SC for visualization
    # Use labels 1-5 for better readability
    df['bonding_bin'] = pd.qcut(df['bonding_idx'], q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    
    # Hukou Interaction
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Convert hukou to string labels for legend
    df['Hukou_Label'] = df['hukou_type'].map({0: '城市 (Urban)', 1: '农村 (Rural)'})
    
    sns.lineplot(data=df, x='bonding_bin', y='expect_college', hue='Hukou_Label', 
                 style='Hukou_Label', markers=True, dashes=False,
                 palette={'城市 (Urban)': '#1f77b4', '农村 (Rural)': '#d62728'}, linewidth=2.5, markersize=8)
                 
    plt.title("图2：师生关系对不同户籍学生升学信心的调节效应\n(Interaction Effect of Bonding SC by Hukou Status)", fontsize=14, pad=20)
    plt.xlabel("师生关系五等分 (Bonding SC Quintiles)\n1=最差, 5=最好", fontsize=12)
    plt.ylabel("升学信心概率 (Probability of College Expectation)", fontsize=12)
    plt.ylim(0, 1.0) # Ensure Y axis is 0-1 probability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='户籍类型', fontsize=10, title_fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rescued_interaction_hukou.png")
    plt.close()

def main():
    if not DATA_FILE.exists():
        print("Rescued data not found.")
        return
        
    df = pd.read_csv(DATA_FILE)
    plot_distributions(df)
    plot_interactions(df)
    print(f"Figures saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
