import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from scipy import stats

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all_with_pca_ses.csv"

def zscore(series):
    return (series - series.mean()) / series.std()

def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    # Prep data
    cols = ["expect_edu_raw", "bonding_idx", "linking_idx", "ses_pca", "hukou_type", "cog_score"]
    df = df[cols].dropna()
    df = df[df["expect_edu_raw"] != 10].copy() # Drop 10
    
    # Z-scores
    for c in ["bonding_idx", "linking_idx", "ses_pca", "cog_score"]:
        df[f"{c}_z"] = zscore(df[c])
        
    # --- Check Raw Data Slopes ---
    # Define Low vs High SES
    # Low: < -0.5 SD, High: > 0.5 SD (Just to get distinct groups)
    low_ses = df[df["ses_pca_z"] < -0.5]
    high_ses = df[df["ses_pca_z"] > 0.5]
    
    print(f"N Low SES: {len(low_ses)}")
    print(f"N High SES: {len(high_ses)}")
    
    # Simple linear regression (Expectation ~ Bonding) for each group to see "Raw Slope"
    # Note: Expectation is ordinal, but linear slope gives a rough idea of the visual trend
    res_low = stats.linregress(low_ses["bonding_idx_z"], low_ses["expect_edu_raw"])
    res_high = stats.linregress(high_ses["bonding_idx_z"], high_ses["expect_edu_raw"])
    
    print("\n--- Raw Data Linear Slopes (Bonding -> Expectation) ---")
    print(f"Low SES Slope:  {res_low.slope:.4f}")
    print(f"High SES Slope: {res_high.slope:.4f}")
    
    if res_low.slope > res_high.slope:
        print(">> Raw Data shows Low SES has STEEPER slope (Compensatory visual).")
    else:
        print(">> Raw Data shows High SES has STEEPER slope (Matthew visual).")
        
    # --- Check Model Implied Slopes ---
    # Coefficient from interaction model (PCA SES): Main Bonding = 0.1464, Interaction = 0.0452
    # Slope = 0.1464 + 0.0452 * SES_Z
    
    mean_low_ses_z = low_ses["ses_pca_z"].mean()
    mean_high_ses_z = high_ses["ses_pca_z"].mean()
    
    model_slope_low = 0.1464 + 0.0452 * mean_low_ses_z
    model_slope_high = 0.1464 + 0.0452 * mean_high_ses_z
    
    print("\n--- Model Implied Slopes (Latent Scale) ---")
    print(f"Mean Low SES Z: {mean_low_ses_z:.2f} -> Model Slope: {model_slope_low:.4f}")
    print(f"Mean High SES Z: {mean_high_ses_z:.2f} -> Model Slope: {model_slope_high:.4f}")
    
    if model_slope_low > model_slope_high:
        print(">> Model implies Low SES has STEEPER slope.")
    else:
        print(">> Model implies High SES has STEEPER slope (Consistent with +coef).")

if __name__ == "__main__":
    main()
