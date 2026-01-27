import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pathlib import Path

DATA_FILE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总\rescued_data\merged_rescued_all_with_pca_ses.csv")

def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    
    # Prep
    df = df[df["expect_edu_raw"] != 10].dropna(subset=["expect_edu_raw", "bonding_idx", "linking_idx", "ses_pca", "hukou_type", "cog_score"])
    df["expect_edu_raw"] = df["expect_edu_raw"].astype(int)
    
    # Standardize
    for c in ["bonding_idx", "linking_idx", "ses_pca", "cog_score"]:
        df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std()
        
    # Interactions
    df["linking_x_ses"] = df["linking_idx_z"] * df["ses_pca_z"]
    df["bonding_x_ses"] = df["bonding_idx_z"] * df["ses_pca_z"]
    df["linking_x_hukou"] = df["linking_idx_z"] * df["hukou_type"]

    X = df[["bonding_idx_z", "linking_idx_z", "ses_pca_z", "hukou_type", "cog_score_z", 
            "linking_x_ses", "bonding_x_ses", "linking_x_hukou"]]
    y = df["expect_edu_raw"]
    
    mod = OrderedModel(y, X, distr="logit")
    res = mod.fit(method="bfgs", disp=False)
    
    print(res.summary())

if __name__ == "__main__":
    main()
