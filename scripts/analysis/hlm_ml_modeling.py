#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2 Analysis: Pure GEE Modeling & ML Exploration
Author: Senior Statistical Engineer (Role-Play)
Date: 2025-12-30
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# --- Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all.csv"
OUT_DIR = WORKSPACE / "results" / "phase2"
FIG_DIR = WORKSPACE / "figures" / "final"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and preprocess data for modeling."""
    print("[INFO] Loading data...")
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
    
    df = pd.read_csv(DATA_FILE)
    
    # Rename columns to match analysis plan
    # expect_college -> expect_univ_bin
    # ses_self -> ses_idx
    # hukou_type -> hukou_num
    rename_map = {
        "expect_college": "expect_univ_bin",
        "ses_self": "ses_idx",
        "hukou_type": "hukou_num"
    }
    df = df.rename(columns=rename_map)
    
    # Ensure numeric types
    cols = ["expect_univ_bin", "bonding_idx", "bridging_idx", "ses_idx", "hukou_num", "cog_score", "schids"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # Drop missing for modeling
    len_orig = len(df)
    df_clean = df[cols].dropna()
    print(f"[INFO] Data loaded. N={len_orig} -> N={len(df_clean)} (Analysis Set)")
    
    return df_clean

def run_gee_models(df):
    """Run Pure GEE models (Binomial Family, Exchangeable Correlation)."""
    results = {}
    
    print("\n[STEP 1] Pure GEE Modeling (Statsmodels)")
    
    # Define common GEE parameters
    # Family: Binomial (for binary outcome)
    # Covariance Structure: Exchangeable (standard for clustered data like schools)
    fam = sm.families.Binomial()
    cov = sm.cov_struct.Exchangeable()
    
    # 1. Main Effect Model
    print("  > Running Main Effects Model (GEE Binomial)...")
    formula_main = "expect_univ_bin ~ bonding_idx + bridging_idx + ses_idx + hukou_num + cog_score"
    
    try:
        # Note: smf.gee is the formula API for GEE
        md_main = smf.gee(formula_main, "schids", df, cov_struct=cov, family=fam)
        mdf_main = md_main.fit()
        print(mdf_main.summary().tables[1])
        results['main_model'] = mdf_main
    except Exception as e:
        print(f"    [Error] GEE Main Model failed: {e}")
        raise e

    # 2. Interaction Model (Bonding * SES)
    print("  > Running Interaction Model (Bonding * SES)...")
    formula_int_ses = "expect_univ_bin ~ bonding_idx * ses_idx + bridging_idx + hukou_num + cog_score"
    
    try:
        md_int_ses = smf.gee(formula_int_ses, "schids", df, cov_struct=cov, family=fam)
        mdf_int_ses = md_int_ses.fit()
        results['int_ses_model'] = mdf_int_ses
        
        # Check if interaction term exists and print p-value
        if 'bonding_idx:ses_idx' in mdf_int_ses.pvalues:
            pval = mdf_int_ses.pvalues['bonding_idx:ses_idx']
            print(f"    Interaction Term (bonding_idx:ses_idx) p-value: {pval:.4f}")
        else:
            print("    [Warning] Interaction term 'bonding_idx:ses_idx' not found in model results.")
            
    except Exception as e:
        print(f"    [Error] GEE Interaction Model failed: {e}")
        raise e

    return results

def run_ml_exploration(df):
    """Run Random Forest for Feature Importance."""
    print("\n[STEP 2] ML Exploration (Random Forest)")
    
    features = ["bonding_idx", "bridging_idx", "ses_idx", "hukou_num", "cog_score"]
    X = df[features]
    y = df["expect_univ_bin"]
    
    # Standardize not strictly needed for RF, but good practice
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_scaled, y)
    
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print("  > Feature Importances:")
    print(importances)
    
    return importances, rf

def plot_interaction(model, df, out_path):
    """Plot Interaction Effects (Bonding * SES) using GEE prediction."""
    print("  > Plotting Interaction Effects...")
    
    # Create prediction grid
    # We need to construct a DataFrame that matches the model's exog design
    # but GEE predict expects a DataFrame if formula was used.
    
    b_range = np.linspace(df['bonding_idx'].min(), df['bonding_idx'].max(), 100)
    
    # High SES vs Low SES (e.g., 75th vs 25th percentile)
    ses_high = df['ses_idx'].quantile(0.75)
    ses_low = df['ses_idx'].quantile(0.25)
    
    # Fix other controls at mean/mode
    # Note: schids is needed for GEE internal structure but for prediction of mean response 
    # we can assign a dummy or existing group. Actually predict(exog) usually works without groups 
    # if we are just predicting marginal means. 
    # However, statsmodels GEE predict might require the group column to be present if the formula uses it?
    # No, formula API usually handles it. Let's create a dummy DataFrame.
    
    mean_bridging = df['bridging_idx'].mean()
    mode_hukou = df['hukou_num'].mode()[0]
    mean_cog = df['cog_score'].mean()
    
    # Construct DataFrame for High SES
    df_high = pd.DataFrame({
        'bonding_idx': b_range,
        'ses_idx': ses_high,
        'bridging_idx': mean_bridging,
        'hukou_num': mode_hukou,
        'cog_score': mean_cog
    })
    
    # Construct DataFrame for Low SES
    df_low = pd.DataFrame({
        'bonding_idx': b_range,
        'ses_idx': ses_low,
        'bridging_idx': mean_bridging,
        'hukou_num': mode_hukou,
        'cog_score': mean_cog
    })
    
    # Predict
    # GEE predict returns the fitted values (probability for Binomial)
    preds_high = model.predict(df_high)
    preds_low = model.predict(df_low)
        
    plt.figure(figsize=(8, 6))
    plt.plot(b_range, preds_high, label=f"高 SES (75%)", color='#2ecc71', linewidth=2.5)
    plt.plot(b_range, preds_low, label=f"低 SES (25%)", color='#e74c3c', linewidth=2.5)
    
    plt.title("补偿效应验证：师生关系对不同家庭背景学生的影响差异 (GEE Model)", fontsize=14, pad=20)
    plt.xlabel("师生关系 (Bonding SC) 标准分", fontsize=12)
    plt.ylabel("期望上本科的概率 (Predicted Prob)", fontsize=12)
    plt.legend(fontsize=11)
    sns.despine()
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_forest(model, out_path):
    """Plot Forest Plot of Coefficients."""
    print("  > Plotting Forest Plot...")
    
    # Extract coefficients and CI
    coefs = model.params.drop("Intercept", errors='ignore')
    conf = model.conf_int().drop("Intercept", errors='ignore')
    conf.columns = ['lower', 'upper']
    
    # Errors for errorbar
    errors = [coefs - conf['lower'], conf['upper'] - coefs]
    
    # Colors: Highlight Significant ones
    pvalues = model.pvalues.drop("Intercept", errors='ignore')
    colors = ['#2c3e50' if p < 0.05 else '#95a5a6' for p in pvalues]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(coefs, range(len(coefs)), xerr=errors, fmt='o', color='black', ecolor='gray', capsize=5, markersize=0)
    plt.scatter(coefs, range(len(coefs)), c=colors, s=100, zorder=10)
    
    plt.yticks(range(len(coefs)), coefs.index, fontsize=11)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    plt.title("回归系数森林图 (Log Odds - GEE)", fontsize=14)
    plt.xlabel("Coefficient Estimate", fontsize=12)
    
    sns.despine(left=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        df = load_data()
        
        # Step 2: Modeling (Pure GEE)
        gee_results = run_gee_models(df)
        ml_importances, rf_model = run_ml_exploration(df)
        
        # Step 3: Visualization
        plot_interaction(gee_results['int_ses_model'], df, FIG_DIR / "final_interaction_plot.png")
        plot_forest(gee_results['main_model'], FIG_DIR / "final_forest_plot.png")
        
        # Save Text Report
        report_path = OUT_DIR / "model_summary_hlm.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Pure GEE Model Summary (Binomial / Exchangeable) ===\n")
            f.write(gee_results['main_model'].summary().as_text())
            f.write("\n\n=== Interaction Model Summary ===\n")
            f.write(gee_results['int_ses_model'].summary().as_text())
            f.write("\n\n=== Random Forest Feature Importance ===\n")
            f.write(ml_importances.to_string())
            
        print(f"\n[SUCCESS] Analysis Complete. Results saved to {OUT_DIR}")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
