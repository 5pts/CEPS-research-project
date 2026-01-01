#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Forensic Statistical Diagnosis for CEPS rescued data.
Outputs figures and a text report with red/yellow/green flags.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
DATA_CSV = WORKSPACE / "rescued_data" / "merged_rescued_all.csv"
OUT_DIR = WORKSPACE / "rescued_reports"
FIG_DIR = WORKSPACE / "rescued_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def traffic_light(value: float, low: float, high: float) -> str:
    """
    low/high are thresholds for red/yellow; green otherwise.
    """
    if value < low:
        return "红灯"
    if value < high:
        return "黄灯"
    return "绿灯"


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 5:
        return np.nan
    return df.iloc[:, 0].corr(df.iloc[:, 1])


def load_teacher_parent_proxies() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load teacher/parent data only for metadata usage.
    We will derive parent_expect from parent DTA if present.
    Teacher DTA does not include direct student-level teacher-student relation,
    so we rely on rescued teacher_praise/teacher_talk as proxy.
    """
    teacher_path = WORKSPACE / "任课教师数据" / "cepsw2teacherCN.dta"
    parent_path = WORKSPACE / "家长数据" / "cepsw2parentCN.dta"
    teacher_df = None
    parent_df = None
    try:
        import pyreadstat
    except Exception:
        return None, None

    if teacher_path.exists():
        teacher_df, _ = pyreadstat.read_dta(str(teacher_path))
    if parent_path.exists():
        parent_df, _ = pyreadstat.read_dta(str(parent_path))
        # Keep only ids + parent expectation if possible
        cols = [c for c in ["ids", "be13", "w2be13"] if c in parent_df.columns]
        parent_df = parent_df[cols].copy() if cols else None
        if parent_df is not None:
            if "be13" in parent_df.columns:
                parent_df = parent_df.rename(columns={"be13": "parent_expect"})
            if "w2be13" in parent_df.columns:
                parent_df = parent_df.rename(columns={"w2be13": "parent_expect"})
    return teacher_df, parent_df


def main() -> int:
    if not DATA_CSV.exists():
        print(f"[ERR] Missing data: {DATA_CSV}")
        return 1

    df = pd.read_csv(DATA_CSV)

    # Harmonize expected names
    # expect_univ_bin -> expect_college (rescued data)
    if "expect_univ_bin" not in df.columns and "expect_college" in df.columns:
        df["expect_univ_bin"] = df["expect_college"]
    # ses_idx -> ses_self (rescued data)
    if "ses_idx" not in df.columns and "ses_self" in df.columns:
        df["ses_idx"] = df["ses_self"]
    # hukou_num -> hukou_type (rescued data)
    if "hukou_num" not in df.columns and "hukou_type" in df.columns:
        df["hukou_num"] = df["hukou_type"]

    # Load parent expectation if available
    _, parent_df = load_teacher_parent_proxies()
    if parent_df is not None and "ids" in df.columns and "ids" in parent_df.columns:
        df = df.merge(parent_df, on="ids", how="left")

    # Build a teacher proxy from student reports (teacher_praise + teacher_talk)
    # This is still student-reported, but used as a distinct component proxy.
    if "teacher_praise" in df.columns and "teacher_talk" in df.columns:
        tp = df["teacher_praise"]
        tt = df["teacher_talk"]
        df["teacher_proxy"] = zscore(tp) + zscore(tt)
    else:
        df["teacher_proxy"] = np.nan

    report_lines = []

    # 1) Common Method Bias
    report_lines.append("1) 同源方差诊断 (Common Method Bias)")
    corr_teacher = safe_corr(df["bonding_idx"], df["teacher_proxy"])
    corr_parent = safe_corr(df["bonding_idx"], df["parent_expect"]) if "parent_expect" in df.columns else np.nan
    tl_teacher = traffic_light(corr_teacher, 0.2, 0.35) if not math.isnan(corr_teacher) else "黄灯"
    tl_parent = traffic_light(corr_parent, 0.2, 0.35) if not math.isnan(corr_parent) else "黄灯"
    report_lines.append(f"- bonding_idx vs teacher_proxy corr: {corr_teacher:.3f} [{tl_teacher}]")
    if "parent_expect" in df.columns:
        report_lines.append(f"- bonding_idx vs parent_expect corr: {corr_parent:.3f} [{tl_parent}]")
    else:
        report_lines.append("- parent_expect missing in merged data; only teacher_proxy used.")

    # Model: expect_univ_bin ~ teacher_proxy
    model_df = df[["expect_univ_bin", "teacher_proxy"]].dropna()
    if len(model_df) > 10:
        X = sm.add_constant(model_df["teacher_proxy"])
        y = model_df["expect_univ_bin"]
        logit_tp = sm.Logit(y, X).fit(disp=False)
        report_lines.append("Teacher-proxy Logit:")
        report_lines.append(logit_tp.summary2().as_text())
    else:
        report_lines.append("Teacher-proxy Logit: insufficient data.")

    # 2) SES vs Hukou logical conflict
    report_lines.append("\n2) SES vs 户籍逻辑检验")
    corr_ses_hukou = safe_corr(df["ses_idx"], df["hukou_num"])
    tl_ses_hukou = traffic_light(abs(corr_ses_hukou), 0.2, 0.35) if not math.isnan(corr_ses_hukou) else "黄灯"
    report_lines.append(f"- corr(ses_idx, hukou_num): {corr_ses_hukou:.3f} [{tl_ses_hukou}]")

    # Model A: Logit ~ bonding * ses
    df_a = df[["expect_univ_bin", "bonding_idx", "ses_idx"]].dropna()
    df_b = df[["expect_univ_bin", "bonding_idx", "hukou_num"]].dropna()

    def fit_logit(frame: pd.DataFrame, inter_var: str):
        frame = frame.copy()
        frame["interaction"] = frame["bonding_idx"] * frame[inter_var]
        X = sm.add_constant(frame[["bonding_idx", inter_var, "interaction"]])
        y = frame["expect_univ_bin"]
        return sm.Logit(y, X).fit(disp=False)

    if len(df_a) > 10:
        logit_a = fit_logit(df_a, "ses_idx")
        report_lines.append("Model A (bonding * ses_idx) coefficients:")
        report_lines.append(logit_a.summary2().tables[1].to_string())
    else:
        logit_a = None
        report_lines.append("Model A: insufficient data.")

    if len(df_b) > 10:
        logit_b = fit_logit(df_b, "hukou_num")
        report_lines.append("Model B (bonding * hukou_num) coefficients:")
        report_lines.append(logit_b.summary2().tables[1].to_string())
    else:
        logit_b = None
        report_lines.append("Model B: insufficient data.")

    # Plot interaction effects
    def plot_interaction(model, frame, inter_var, out_path):
        if model is None:
            return
        x_vals = np.linspace(frame["bonding_idx"].quantile(0.05), frame["bonding_idx"].quantile(0.95), 50)
        inter_levels = frame[inter_var].quantile([0.2, 0.5, 0.8]).values
        plt.figure(figsize=(6, 4))
        for lvl in inter_levels:
            X = pd.DataFrame({
                "const": 1.0,
                "bonding_idx": x_vals,
                inter_var: lvl,
                "interaction": x_vals * lvl,
            })
            preds = model.predict(X)
            plt.plot(x_vals, preds, label=f"{inter_var}={lvl:.2f}")
        plt.title(f"Interaction: bonding_idx x {inter_var}")
        plt.xlabel("bonding_idx")
        plt.ylabel("P(expect_univ_bin=1)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    if logit_a is not None:
        plot_interaction(logit_a, df_a, "ses_idx", FIG_DIR / "interaction_bonding_ses.png")
    if logit_b is not None:
        plot_interaction(logit_b, df_b, "hukou_num", FIG_DIR / "interaction_bonding_hukou.png")

    # 3) MNAR check
    report_lines.append("\n3) 非随机缺失诊断 (MNAR)")
    df["missing_teacher_data"] = df["teacher_praise"].isna() | df["teacher_talk"].isna()
    mnar_df = df[["missing_teacher_data", "ses_idx", "hukou_num"]].dropna()
    if len(mnar_df) > 10:
        X = sm.add_constant(mnar_df[["ses_idx", "hukou_num"]])
        y = mnar_df["missing_teacher_data"].astype(int)
        mnar_logit = sm.Logit(y, X).fit(disp=False)
        report_lines.append("Missingness Logit (missing_teacher_data ~ ses_idx + hukou_num):")
        report_lines.append(mnar_logit.summary2().tables[1].to_string())
    else:
        report_lines.append("Missingness Logit: insufficient data.")

    miss_grp = df.loc[df["missing_teacher_data"], "ses_idx"].dropna()
    ok_grp = df.loc[~df["missing_teacher_data"], "ses_idx"].dropna()
    if len(miss_grp) > 3 and len(ok_grp) > 3:
        tstat, pval = stats.ttest_ind(miss_grp, ok_grp, equal_var=False)
        diff = miss_grp.mean() - ok_grp.mean()
        tl_mnar = "红灯" if pval < 0.05 and abs(diff) > 0 else "黄灯" if pval < 0.1 else "绿灯"
        report_lines.append(f"SES mean diff (missing - non-missing): {diff:.3f}, p={pval:.4f} [{tl_mnar}]")
    else:
        report_lines.append("SES mean diff: insufficient data.")

    # 4) Distribution + OLS residuals
    report_lines.append("\n4) 变量分布与模型假设压力测试")
    bond = df["bonding_idx"].dropna()
    skew = stats.skew(bond) if len(bond) > 5 else np.nan
    kurt = stats.kurtosis(bond, fisher=True) if len(bond) > 5 else np.nan
    report_lines.append(f"bonding_idx skewness: {skew:.3f}")
    report_lines.append(f"bonding_idx kurtosis (Fisher): {kurt:.3f}")

    # Histogram
    plt.figure(figsize=(6, 4))
    sns.histplot(bond, kde=True, color="#3b6ea5")
    plt.title("bonding_idx distribution")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bonding_hist.png", dpi=150)
    plt.close()

    # QQ plot
    plt.figure(figsize=(5, 5))
    sm.qqplot(bond, line="45", fit=True)
    plt.title("bonding_idx QQ plot")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bonding_qq.png", dpi=150)
    plt.close()

    # OLS residuals for expect_univ_bin
    ols_df = df[["expect_univ_bin", "bonding_idx", "ses_idx", "hukou_num"]].dropna()
    if len(ols_df) > 10:
        X = sm.add_constant(ols_df[["bonding_idx", "ses_idx", "hukou_num"]])
        y = ols_df["expect_univ_bin"]
        ols = sm.OLS(y, X).fit()
        resid = ols.resid
        plt.figure(figsize=(6, 4))
        sns.histplot(resid, kde=True, color="#2a9d8f")
        plt.title("OLS residuals (expect_univ_bin)")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "ols_resid_hist.png", dpi=150)
        plt.close()
        report_lines.append("OLS summary:")
        report_lines.append(ols.summary().as_text())
    else:
        report_lines.append("OLS: insufficient data.")

    report_path = OUT_DIR / "forensic_diagnosis_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"[OK] Report saved: {report_path}")
    print(f"[OK] Figures saved in: {FIG_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
