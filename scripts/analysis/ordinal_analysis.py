import math
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel


WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_FILE = WORKSPACE / "rescued_data" / "merged_rescued_all_with_pca_ses.csv"
OUTPUT_DIR = WORKSPACE / "results" / "phase3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def zscore(series):
    std = series.std()
    if std == 0 or pd.isna(std):
        return series * 0
    return (series - series.mean()) / std


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing data file: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    required = [
        "expect_edu_raw",
        "bonding_idx",
        "linking_idx",
        "ses_pca",
        "hukou_type",
        "cog_score",
        "clsids",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def prep_model_df(df, drop_code_10=True):
    model_df = df[
        [
            "expect_edu_raw",
            "bonding_idx",
            "linking_idx",
            "ses_pca",
            "hukou_type",
            "cog_score",
            "clsids",
        ]
    ].copy()
    model_df = model_df.dropna()
    if drop_code_10:
        model_df = model_df[model_df["expect_edu_raw"] != 10]
    model_df["expect_edu_raw"] = model_df["expect_edu_raw"].astype(int)
    for col in ["bonding_idx", "linking_idx", "ses_pca", "cog_score"]:
        model_df[f"{col}_z"] = zscore(model_df[col])
    return model_df


def fit_ordered_logit(model_df):
    y = model_df["expect_edu_raw"]
    X = model_df[
        ["bonding_idx_z", "linking_idx_z", "ses_pca_z", "hukou_type", "cog_score_z"]
    ]
    model = OrderedModel(y, X, distr="logit")
    res = model.fit(method="bfgs", disp=False)
    return res


def fit_mnlogit(model_df):
    y = model_df["expect_edu_raw"]
    X = model_df[
        ["bonding_idx_z", "linking_idx_z", "ses_pca_z", "hukou_type", "cog_score_z"]
    ]
    X = sm.add_constant(X, has_constant="add")
    mn = sm.MNLogit(y, X)
    mn_res = mn.fit(method="newton", disp=False)
    return mn_res


def lr_test_ordered_vs_mnlogit(ord_res, mn_res):
    llf_ord = ord_res.llf
    llf_mn = mn_res.llf
    k_ord = len(ord_res.params)
    k_mn = mn_res.params.size
    lr_stat = 2 * (llf_mn - llf_ord)
    df = max(k_mn - k_ord, 1)
    p_val = stats.chi2.sf(lr_stat, df)
    return lr_stat, df, p_val


def mnlogit_sign_check(mn_res, predictors):
    params = mn_res.params
    summary = {}
    for col in predictors:
        if col not in params.index:
            continue
        signs = np.sign(params.loc[col])
        pos_share = (signs > 0).mean()
        neg_share = (signs < 0).mean()
        summary[col] = (pos_share, neg_share)
    return summary


def ttest_group_10_vs_1to9(df):
    df = df.dropna(subset=["expect_edu_raw"])
    group_10 = df[df["expect_edu_raw"] == 10]
    group_1to9 = df[(df["expect_edu_raw"] >= 1) & (df["expect_edu_raw"] <= 9)]
    results = {}
    for col in ["ses_pca", "cog_score", "bonding_idx", "linking_idx"]:
        a = group_10[col].dropna()
        b = group_1to9[col].dropna()
        if len(a) < 2 or len(b) < 2:
            results[col] = ("insufficient", np.nan, np.nan)
            continue
        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        results[col] = ("ok", t_stat, p_val)
    return results, len(group_10), len(group_1to9)


def chi2_hukou_group_10_vs_1to9(df):
    df = df.dropna(subset=["expect_edu_raw", "hukou_type"])
    group_10 = df[df["expect_edu_raw"] == 10]
    group_1to9 = df[(df["expect_edu_raw"] >= 1) & (df["expect_edu_raw"] <= 9)]
    if group_10.empty or group_1to9.empty:
        return None
    table = pd.crosstab(
        ["10"] * len(group_10) + ["1-9"] * len(group_1to9),
        pd.concat([group_10["hukou_type"], group_1to9["hukou_type"]], ignore_index=True),
    )
    chi2, p_val, dof, exp = stats.chi2_contingency(table)
    return chi2, p_val, dof


def cluster_robust_table(result, groups):
    cov = cov_cluster(result, groups)
    params = result.params
    se = np.sqrt(np.diag(cov))
    z = params / se
    p = 2 * stats.norm.sf(np.abs(z))
    ci_low = params - 1.96 * se
    ci_high = params + 1.96 * se
    table = pd.DataFrame(
        {
            "coef": params,
            "robust_se": se,
            "z": z,
            "p": p,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    return table


def main():
    df = load_data()
    report_path = OUTPUT_DIR / "ordinal_model_report.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write("=== Ordinal Logit Analysis (Education Expectation 1-9) ===\n")
        f.write(f"Data source: {DATA_FILE}\n")

        counts = df["expect_edu_raw"].value_counts(dropna=False).sort_index()
        f.write("\n--- Raw Distribution (expect_edu_raw) ---\n")
        f.write(counts.to_string())
        f.write("\n\n")

        ttest_results, n10, n19 = ttest_group_10_vs_1to9(df)
        f.write("--- Group 10 ('无所谓') vs 1-9: t-tests ---\n")
        f.write(f"n(10)={n10}, n(1-9)={n19}\n")
        for col, (status, t_stat, p_val) in ttest_results.items():
            if status == "ok":
                f.write(f"{col}: t={t_stat:.3f}, p={p_val:.4f}\n")
            else:
                f.write(f"{col}: insufficient data\n")
        chi2_out = chi2_hukou_group_10_vs_1to9(df)
        if chi2_out:
            chi2, p_val, dof = chi2_out
            f.write(f"hukou_type: chi2={chi2:.3f}, dof={dof}, p={p_val:.4f}\n")
        f.write("\n")

        f.write("--- Main Model: Ordered Logit (drop 10) ---\n")
        model_df = prep_model_df(df, drop_code_10=True)
        f.write(f"Rows: {len(model_df)}\n")
        f.write(f"Classes (clsids): {model_df['clsids'].nunique()}\n\n")

        ord_res = fit_ordered_logit(model_df)
        f.write("Ordered logit summary (MLE):\n")
        f.write(str(ord_res.summary()))
        f.write("\n\n")
        try:
            robust_table = cluster_robust_table(ord_res, model_df["clsids"])
            f.write("Ordered logit (cluster-robust SE by clsids):\n")
            f.write(robust_table.to_string(float_format=lambda x: f"{x:.4f}"))
            f.write("\n\n")
        except Exception as exc:
            f.write(f"Cluster-robust SE failed: {exc}\n\n")

        f.write("--- Proportional Odds Check (LR test vs Multinomial Logit) ---\n")
        try:
            mn_res = fit_mnlogit(model_df)
            lr_stat, df_lr, p_val = lr_test_ordered_vs_mnlogit(ord_res, mn_res)
            f.write(f"LR stat={lr_stat:.3f}, df={df_lr}, p={p_val:.4f}\n")
            f.write("\nMultinomial logit summary (MLE):\n")
            f.write(str(mn_res.summary()))
            f.write("\n\nSign check (share of positive/negative coef by outcome):\n")
            sign_summary = mnlogit_sign_check(
                mn_res,
                ["bonding_idx_z", "linking_idx_z", "ses_pca_z", "hukou_type", "cog_score_z"],
            )
            for col, (pos_share, neg_share) in sign_summary.items():
                f.write(f"{col}: pos={pos_share:.2f}, neg={neg_share:.2f}\n")
        except Exception as e:
            f.write(f"MNLogit failed: {e}\n")

    print(f"[DONE] Report saved to: {report_path}")


if __name__ == "__main__":
    main()
