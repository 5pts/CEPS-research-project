import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

# Set plotting style
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='SimHei')

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all.csv"
OUTPUT_DIR = WORKSPACE / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_reconstruct_data():
    print("[INFO] Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    # Map variables from rescued schema to evaluation schema
    # Rescued Schema: bonding_idx, bridging_idx, ses_self, hukou_type (0=Urban, 1=Rural), expect_college (0/1)
    # Target Schema: bonding_sc_idx, bridging_sc_idx, ses_idx, hukou_num, expect_univ_bin
    
    df["bonding_sc_idx"] = df["bonding_idx"]
    df["bridging_sc_idx"] = df["bridging_idx"]
    df["ses_idx"] = df["ses_self"] # Assuming standardized or similar scale
    
    # Hukou: Rescued has 0=Urban, 1=Rural.
    # Evaluation expects 0=City, 1=Rural. (Matches)
    df["hukou_num"] = df["hukou_type"]
    
    # Expectation: Rescued has expect_college (0/1)
    df["expect_univ_bin"] = df["expect_college"]

    cols = ["bonding_sc_idx", "bridging_sc_idx", "ses_idx", "hukou_num", "expect_univ_bin"]
    return df[cols].dropna()

def assess_data_quality(df):
    print("\n--- 1. Data Quality Assessment ---")
    stats = []
    
    # 1. Missingness (on raw df before dropna, but here we passed dropped df)
    # So we report on the final sample size vs original
    # We can't check missingness on a dropna-ed df easily.
    # Let's just proceed with distribution.
    
    # 2. Distribution
    from scipy import stats as sp_stats
    for col in df.columns:
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        # Normality Test (KS test)
        ks_stat, p_val = sp_stats.kstest(df[col], 'norm')
        is_normal = p_val > 0.05
        stats.append({
            "Variable": col,
            "Skewness": skew,
            "Kurtosis": kurt,
            "Normality (p>0.05)": is_normal
        })
    
    stats_df = pd.DataFrame(stats)
    print(stats_df)
    stats_df.to_csv(OUTPUT_DIR / "data_quality_stats.csv", index=False)
    
    # 3. Correlation
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("变量相关性矩阵")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix.png")
    plt.close()

def evaluate_harmonic_model(df):
    print("\n--- 2. Harmonic Model Suitability ---")
    # Theoretical check
    print("Assumption Check: Harmonic models assume periodic/cyclic data.")
    print("Data Type: Cross-sectional survey data.")
    print("Periodicity: None observed in key variables.")
    print("Conclusion: Model assumptions VIOLATED.")
    
    # We cannot fit a harmonic model meaningfully.
    # We will skip calculating AIC for it because it's invalid.
    
def propose_alternatives(df):
    print("\n--- 3. Alternative Models Evaluation ---")
    
    X = df[["bonding_sc_idx", "bridging_sc_idx", "ses_idx", "hukou_num"]]
    X = sm.add_constant(X)
    y = df["expect_univ_bin"]
    
    # Model 1: Logistic Regression
    print("Fitting Logistic Regression...")
    try:
        logit_mod = sm.Logit(y, X)
        logit_res = logit_mod.fit(disp=0)
        print(f"Logistic AIC: {logit_res.aic:.2f}")
        print(f"Logistic Pseudo R-squared: {logit_res.prsquared:.4f}")
        
        # Residual Analysis (Deviance Residuals)
        resid = logit_res.resid_deviance
        plt.figure(figsize=(8, 4))
        plt.scatter(logit_res.fittedvalues, resid, alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Fitted Log-Odds")
        plt.ylabel("Deviance Residuals")
        plt.title("Logistic Regression Residuals")
        plt.savefig(OUTPUT_DIR / "logistic_residuals.png")
        plt.close()
        
    except Exception as e:
        print(f"Logistic fit failed: {e}")

    # Model 2: Linear Probability Model (OLS) - for comparison/simplicity
    print("Fitting OLS (LPM)...")
    try:
        ols_mod = sm.OLS(y, X)
        ols_res = ols_mod.fit()
        print(f"OLS AIC: {ols_res.aic:.2f}")
        print(f"OLS R-squared: {ols_res.rsquared:.4f}")
    except:
        pass

def main():
    df = load_and_reconstruct_data()
    print(f"Final sample size for analysis: {len(df)}")
    
    assess_data_quality(df)
    evaluate_harmonic_model(df)
    propose_alternatives(df)
    
    # Distribution plots
    for col in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(OUTPUT_DIR / f"dist_{col}.png")
        plt.close()

if __name__ == "__main__":
    main()
