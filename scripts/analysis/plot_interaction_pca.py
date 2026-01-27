import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.miscmodels.ordinal_model import OrderedModel


WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all_with_pca_ses.csv"
OUT_FILE = WORKSPACE / "figures" / "report_phase3" / "interaction_plot_pca.png"


def zscore(series):
    return (series - series.mean()) / series.std()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    df = pd.read_csv(DATA_FILE, low_memory=False)
    cols = [
        "expect_edu_raw",
        "bonding_idx",
        "linking_idx",
        "ses_pca",
        "hukou_type",
        "cog_score",
    ]
    df = df[cols].dropna()
    df = df[df["expect_edu_raw"].between(1, 9)]

    for c in ["bonding_idx", "linking_idx", "ses_pca", "cog_score"]:
        df[f"{c}_z"] = zscore(df[c])

    df["linking_x_ses"] = df["linking_idx_z"] * df["ses_pca_z"]
    df["bonding_x_ses"] = df["bonding_idx_z"] * df["ses_pca_z"]
    df["linking_x_hukou"] = df["linking_idx_z"] * df["hukou_type"]

    y = df["expect_edu_raw"].astype(int)
    X = df[
        [
            "bonding_idx_z",
            "linking_idx_z",
            "ses_pca_z",
            "hukou_type",
            "cog_score_z",
            "linking_x_ses",
            "bonding_x_ses",
            "linking_x_hukou",
        ]
    ]

    res = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    params = res.params
    cut_6_7 = params["6/7"]

    # Plot bonding vs P(expectation >= 7)
    bonding_grid = np.linspace(-2.5, 2.5, 60)
    ses_low = df["ses_pca_z"].quantile(0.25)
    ses_high = df["ses_pca_z"].quantile(0.75)

    def p_ge_7(bonding_z, ses_z):
        linking_z = 0.0
        hukou = 0.0
        cog_z = 0.0
        linking_x_ses = linking_z * ses_z
        bonding_x_ses = bonding_z * ses_z
        linking_x_hukou = linking_z * hukou

        xb = (
            params["bonding_idx_z"] * bonding_z
            + params["linking_idx_z"] * linking_z
            + params["ses_pca_z"] * ses_z
            + params["hukou_type"] * hukou
            + params["cog_score_z"] * cog_z
            + params["linking_x_ses"] * linking_x_ses
            + params["bonding_x_ses"] * bonding_x_ses
            + params["linking_x_hukou"] * linking_x_hukou
        )
        p_le_6 = sigmoid(cut_6_7 - xb)
        return 1 - p_le_6

    prob_low = [p_ge_7(b, ses_low) for b in bonding_grid]
    prob_high = [p_ge_7(b, ses_high) for b in bonding_grid]

    plt.figure(figsize=(7, 5))
    plt.plot(bonding_grid, prob_high, color="#2ca02c", label="High SES (75%)")
    plt.plot(bonding_grid, prob_low, color="#d62728", label="Low SES (25%)")
    plt.xlabel("Peer Bonding (z-score)")
    plt.ylabel("P(Expectation ≥ Bachelor)")
    plt.title("Bonding × SES (Ordered Logit, PCA SES)")
    plt.legend()
    plt.tight_layout()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FILE, dpi=300)
    print(f"[DONE] Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
