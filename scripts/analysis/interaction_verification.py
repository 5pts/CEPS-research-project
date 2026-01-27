import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.sandwich_covariance import cov_cluster
from pathlib import Path

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all_with_pca_ses.csv"
OUTPUT_FILE = WORKSPACE / "results" / "phase3" / "interaction_verification_report.txt"

def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * 0
    return (series - series.mean()) / std

def load_and_prep():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    # Select columns
    cols = ["expect_edu_raw", "bonding_idx", "linking_idx", "ses_pca", "hukou_type", "cog_score", "clsids"]
    df = df[cols].dropna()
    
    # Drop "10=无所谓"
    df = df[df["expect_edu_raw"] != 10].copy()
    df["expect_edu_raw"] = df["expect_edu_raw"].astype(int)
    
    # Standardize continuous variables
    for col in ["bonding_idx", "linking_idx", "ses_pca", "cog_score"]:
        df[f"{col}_z"] = zscore(df[col])
        
    return df

def run_model(df, formula_name, predictors):
    y = df["expect_edu_raw"]
    
    # Start with base columns that exist in df
    base_cols = [p for p in predictors if p in df.columns]
    X = df[base_cols].copy()
    
    # Add interaction terms manually
    if "linking_x_ses" in predictors:
        X["linking_x_ses"] = df["linking_idx_z"] * df["ses_pca_z"]
    if "bonding_x_ses" in predictors:
        X["bonding_x_ses"] = df["bonding_idx_z"] * df["ses_pca_z"]
    if "linking_x_hukou" in predictors:
        X["linking_x_hukou"] = df["linking_idx_z"] * df["hukou_type"]
        
    # Ensure X has all requested predictors in correct order
    X = X[predictors]
    
    model = OrderedModel(y, X, distr="logit")
    res = model.fit(method="bfgs", disp=False)
    return res

def cluster_robust_stats(result, df, predictors):
    cov = cov_cluster(result, df["clsids"])
    se = np.sqrt(np.diag(cov))
    params = result.params
    
    # OrderedModel params include coefficients followed by cutpoints
    # We only care about coefficients for the table usually, but let's print all
    
    table = pd.DataFrame({
        "coef": params,
        "robust_se": se,
        "z": params / se,
        "p": 2 * (1 - stats.norm.cdf(np.abs(params / se)))
    })
    return table

import scipy.stats as stats

def main():
    df = load_and_prep()
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Interaction Verification for CEPS Hypotheses ===\n\n")
        
        # 1. Main Effects (Re-check)
        f.write("--- Model 1: Main Effects (H1) ---\n")
        predictors_1 = ["bonding_idx_z", "linking_idx_z", "ses_pca_z", "hukou_type", "cog_score_z"]
        res_1 = run_model(df, "Main Effects", predictors_1)
        f.write(str(res_1.summary()))
        f.write("\n\n")
        
        # 2. Interaction Model (H2)
        # H2a: Teacher Linking * SES
        # H2b: Peer Bonding * SES
        # H2c: Teacher Linking * Hukou
        
        f.write("--- Model 2: Interactions (H2) ---\n")
        predictors_2 = predictors_1 + ["linking_x_ses", "bonding_x_ses", "linking_x_hukou"]
        
        # We need to compute these columns in run_model, passing the list of names
        res_2 = run_model(df, "Interactions", predictors_2)
        
        f.write(str(res_2.summary()))
        f.write("\n\n")
        
        # Robust SE for Model 2
        f.write("--- Model 2: Robust SE (Clustered by class) ---\n")
        # Need to reconstruct the X matrix with interactions for correct shape if I were to pass it, 
        # but here we just pass the result and the group variable.
        # Wait, cov_cluster needs the result object to have the correct input data? 
        # The result object stores the model and data.
        
        table_2 = cluster_robust_stats(res_2, df, predictors_2)
        f.write(table_2.to_string())
        f.write("\n\n")
        
        f.write("--- Interpretation Guide ---\n")
        f.write("H2a: Linking benefits Low-SES > High-SES. Expect 'linking_x_ses' < 0.\n")
        f.write("H2b: Bonding benefits Low-SES > High-SES. Expect 'bonding_x_ses' < 0.\n")
        f.write("H2c: Linking benefits Rural > Urban. Expect 'linking_x_hukou' > 0 (assuming Rural=1).\n")
        
    print(f"Report generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
