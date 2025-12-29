import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pygam import LogisticGAM, s, l
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all.csv"
FIGURES_DIR = WORKSPACE / "figures" / "ml_insights"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prep_data():
    print("[INFO] Loading data...")
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE, low_memory=False)
    else:
        raise FileNotFoundError(f"Cleaned data not found at {DATA_FILE}")
    
    # Map variables from rescued schema to ML schema
    # Rescued Schema: bonding_idx, bridging_idx, ses_self, hukou_type (0=Urban, 1=Rural), expect_college (0/1)
    
    df["bonding_sc_idx"] = df["bonding_idx"]
    df["bridging_sc_idx"] = df["bridging_idx"]
    df["ses_idx"] = df["ses_self"]
    df["hukou_num"] = df["hukou_type"]
    df["expect_univ_bin"] = df["expect_college"]

    # 准备建模数据
    model_df = df[["bonding_sc_idx", "bridging_sc_idx", "ses_idx", "hukou_num", "expect_univ_bin"]].copy()
    
    # 去除缺失值
    len_before = len(model_df)
    model_df = model_df.dropna()
    print(f"[INFO] Data loaded. Rows: {len_before} -> {len(model_df)} (after dropna)")
    
    return model_df

def run_gam_analysis(df):
    print("[INFO] Running GAM analysis...")
    X = df[["bonding_sc_idx", "bridging_sc_idx", "ses_idx", "hukou_num"]]
    y = df["expect_univ_bin"]
    
    # LogisticGAM: s() for spline terms (non-linear), l() for linear terms
    # Hukou is binary, so use linear term or factor term. SES we assume linear or spline.
    # Model: expect ~ s(bonding) + s(bridging) + s(ses) + l(hukou)
    gam = LogisticGAM(s(0) + s(1) + s(2) + l(3))
    
    try:
        gam.fit(X, y)
    except Exception as e:
        print(f"[ERROR] GAM fitting failed: {e}")
        return

    print("[INFO] GAM fitted. Generating partial dependence plots...")
    
    # 1. Bonding SC Partial Dependence
    plt.figure(figsize=(8, 5))
    XX = gam.generate_X_grid(term=0)
    plt.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX))
    plt.plot(XX[:, 0], gam.partial_dependence(term=0, X=XX, width=.95)[1], c='r', ls='--')
    plt.title("GAM: 师生关系 (Bonding SC) 对升学信心的非线性效应")
    plt.xlabel("Bonding SC Index (Z-score)")
    plt.ylabel("Partial Dependence (Log Odds)")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / "GAM_Bonding_Partial_Dependence.png")
    plt.close()

    # 2. Bridging SC Partial Dependence
    plt.figure(figsize=(8, 5))
    XX = gam.generate_X_grid(term=1)
    plt.plot(XX[:, 1], gam.partial_dependence(term=1, X=XX))
    plt.plot(XX[:, 1], gam.partial_dependence(term=1, X=XX, width=.95)[1], c='r', ls='--')
    plt.title("GAM: 同伴氛围 (Bridging SC) 对升学信心的非线性效应")
    plt.xlabel("Bridging SC Index (Z-score)")
    plt.ylabel("Partial Dependence (Log Odds)")
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / "GAM_Bridging_Partial_Dependence.png")
    plt.close()
    
    print(f"[DONE] GAM plots saved to {FIGURES_DIR}")

def run_decision_tree_analysis(df):
    print("[INFO] Running Decision Tree analysis for thresholds...")
    X = df[["bonding_sc_idx", "bridging_sc_idx", "ses_idx", "hukou_num"]]
    y = df["expect_univ_bin"]
    
    # 使用浅层树 (max_depth=3) 来寻找最具解释力的阈值
    dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
    dt.fit(X, y)
    
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=X.columns, class_names=["Low Expect", "High Expect"], filled=True, fontsize=10)
    plt.title("决策树：识别社会资本的关键阈值 (Thresholds)")
    plt.savefig(FIGURES_DIR / "Decision_Tree_Thresholds.png")
    plt.close()
    print(f"[DONE] Decision tree plot saved to {FIGURES_DIR}")

def run_interaction_visualization(df):
    print("[INFO] Visualizing Interactions (Bonding * SES)...")
    # 将 SES 分为高低两组
    median_ses = df["ses_idx"].median()
    low_ses = df[df["ses_idx"] < median_ses]
    high_ses = df[df["ses_idx"] >= median_ses]
    
    plt.figure(figsize=(8, 5))
    
    # 使用简单的多项式拟合或 LOWESS 来展示趋势
    # 这里用 seaborn 的 regplot 风格手动实现 (poly order=2 to allow curve)
    
    for name, sub_df in [("Low SES", low_ses), ("High SES", high_ses)]:
        if len(sub_df) < 10: continue
        # Binning for clearer scatter
        sub_df = sub_df.sort_values("bonding_sc_idx")
        # Rolling mean or Polynomial fit
        z = np.polyfit(sub_df["bonding_sc_idx"], sub_df["expect_univ_bin"], 2)
        p = np.poly1d(z)
        x_range = np.linspace(sub_df["bonding_sc_idx"].min(), sub_df["bonding_sc_idx"].max(), 100)
        plt.plot(x_range, p(x_range), label=f"{name} (Poly Fit)")
        
    plt.title("交互效应：不同 SES 背景下师生关系对升学信心的影响")
    plt.xlabel("Bonding SC Index (Z-score)")
    plt.ylabel("Prob. of High Expectation")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(FIGURES_DIR / "Interaction_Bonding_SES.png")
    plt.close()
    print(f"[DONE] Interaction plot saved to {FIGURES_DIR}")

def main():
    try:
        df = load_and_prep_data()
        if df.empty:
            print("[ERROR] No data available after preprocessing.")
            return
            
        run_gam_analysis(df)
        run_decision_tree_analysis(df)
        run_interaction_visualization(df)
        
        print("\n[SUCCESS] ML Analysis complete.")
        print(f"Results are stored in: {FIGURES_DIR}")
        print("See figures_readme.md for interpretation.")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
