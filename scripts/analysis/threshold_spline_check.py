import math
from pathlib import Path

import numpy as np
import pandas as pd
from patsy import dmatrix
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats


WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all_with_pca_ses.csv"
REPORT_FILE = WORKSPACE / "results" / "phase3" / "threshold_spline_report.txt"
FIG_DIR = WORKSPACE / "figures" / "report_phase3"


def zscore(series):
    return (series - series.mean()) / series.std()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def fit_ordered_logit(y, X):
    model = OrderedModel(y, X, distr="logit")
    return model.fit(method="bfgs", disp=False)


def spline_basis(x, df=4, degree=3, prefix="spl"):
    basis = dmatrix(
        f"0 + bs(x, df={df}, degree={degree}, include_intercept=False)",
        {"x": x},
        return_type="dataframe",
    )
    basis.columns = [f"{prefix}_{i}" for i in range(basis.shape[1])]
    return basis


def lr_test(llf_full, llf_base, df_full, df_base):
    lr_stat = 2 * (llf_full - llf_base)
    df = max(df_full - df_base, 1)
    p_val = stats.chi2.sf(lr_stat, df)
    return lr_stat, df, p_val


def plot_threshold(res, base_row, var_grid, spline_df, prefix, title, out_path):
    params = res.params
    cut_6_7 = params["6/7"]

    basis = spline_basis(var_grid, df=spline_df, degree=3, prefix=prefix)
    X = pd.DataFrame(np.repeat(base_row.values, len(var_grid), axis=0), columns=base_row.columns)
    X[basis.columns] = basis.values

    xb = X @ params[X.columns]
    p_le_6 = sigmoid(cut_6_7 - xb)
    p_ge_7 = 1 - p_le_6

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.plot(var_grid, p_ge_7, color="#2c7fb8")
    plt.xlabel("z-score")
    plt.ylabel("P(Expectation ≥ Bachelor)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)


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

    y = df["expect_edu_raw"].astype(int)

    results = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for var in ["bonding_idx_z", "linking_idx_z"]:
        spline_df = 4
        base_cols = ["ses_pca_z", "hukou_type", "cog_score_z"]
        linear_cols = [var] + base_cols

        X_linear = df[linear_cols]
        res_linear = fit_ordered_logit(y, X_linear)

        basis = spline_basis(df[var], df=spline_df, degree=3, prefix=var)
        X_spline = pd.concat([basis, df[base_cols]], axis=1)
        res_spline = fit_ordered_logit(y, X_spline)

        lr_stat, df_lr, p_val = lr_test(res_spline.llf, res_linear.llf, len(res_spline.params), len(res_linear.params))

        results.append((var, res_linear.llf, res_spline.llf, lr_stat, df_lr, p_val))

        # Plot spline effect holding controls at mean
        base_row = pd.DataFrame(
            {
                "ses_pca_z": [df["ses_pca_z"].mean()],
                "hukou_type": [0.0],
                "cog_score_z": [df["cog_score_z"].mean()],
            }
        )
        grid = np.linspace(-2.5, 2.5, 80)
        plot_threshold(
            res_spline,
            base_row,
            grid,
            spline_df,
            var,
            f"{var.replace('_z','').replace('_',' ').title()} Spline Effect",
            FIG_DIR / f"threshold_{var.replace('_z','')}_spline.png",
        )

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        f.write("Threshold (Spline) Check for Ordered Logit\n")
        f.write("=========================================\n")
        f.write("Spline df=4, degree=3. Compare linear vs spline via LR test.\n\n")
        for var, llf_lin, llf_spl, lr_stat, df_lr, p_val in results:
            f.write(f"{var}:\n")
            f.write(f"  LLF linear: {llf_lin:.3f}\n")
            f.write(f"  LLF spline: {llf_spl:.3f}\n")
            f.write(f"  LR stat: {lr_stat:.3f}, df={df_lr}, approx p~{p_val:.4f}\n\n")

    print(f"[DONE] Saved {REPORT_FILE}")


if __name__ == "__main__":
    main()
