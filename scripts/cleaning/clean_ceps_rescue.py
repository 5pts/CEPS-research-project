#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CEPS 数据清洗与整合脚本 (Rescued Version 2.1)
- 目的：最大化保留样本量，修复因变量选择不当导致的样本流失
- 策略：
  1. 多源数据（家长、教师、学校）交叉填补
  2. 教师数据按班级聚合（均值/众数）避免结构性偏差
  3. 使用全员必答题替代跳答题

变量口径（统一）：
- bonding_idx：同辈关系（横向）= 同伴/班级氛围条目合成
- linking_idx：师生关系（纵向）= 教师表扬 + 与教师交流
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pyreadstat

# Paths
WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
RAW_DIR = WORKSPACE
OUTPUT_DIR = WORKSPACE / "rescued_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = WORKSPACE / "rescued_reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# File Paths
FILES = {
    "student": RAW_DIR / "学生数据" / "cepsw2studentCN.dta",
    "parent": RAW_DIR / "家长数据" / "cepsw2parentCN.dta",
    "teacher": RAW_DIR / "任课教师数据" / "cepsw2teacherCN.dta",
    "principal": RAW_DIR / "校领导学校数据" / "cepsw2principalCN.dta",
}


def load_data(name, path):
    if not path.exists():
        print(f"[WARN] File not found: {path}")
        return None
    print(f"[INFO] Loading {name} from {path}...")
    df, _meta = pyreadstat.read_dta(str(path))
    return df


def aggregate_teacher_data(df):
    """
    聚合教师数据到班级层级 (Class Level)
    - 连续变量：取均值
    - 分类变量：取众数
    """
    print("--- Aggregating Teacher Data ---")
    if "w2clsids" in df.columns:
        df = df.rename(columns={"w2clsids": "clsids"})

    if "clsids" not in df.columns:
        print("[ERR] clsids not found in teacher data.")
        return None

    agg_dict = {}
    if "hr01" in df.columns:
        agg_dict["hr01"] = lambda x: x.mode()[0] if not x.mode().empty else np.nan
    if "hr02" in df.columns:
        agg_dict["hr02"] = "mean"

    if not agg_dict:
        return df.drop_duplicates(subset=["clsids"])  # Fallback

    grouped = df.groupby("clsids").agg(agg_dict).reset_index()
    print(f"Aggregated Teacher Data: {len(df)} rows -> {len(grouped)} classes")
    return grouped


def zscore(s):
    return (s - s.mean()) / s.std()


def main():
    # 1. Load Raw Data
    stu_df = load_data("Student", FILES["student"])
    par_df = load_data("Parent", FILES["parent"])
    tea_df = load_data("Teacher", FILES["teacher"])
    sch_df = load_data("School", FILES["principal"])

    if stu_df is None:
        print("Critical Error: Student data missing.")
        return

    # 2. Pre-process Auxiliary Data
    tea_clean = aggregate_teacher_data(tea_df) if tea_df is not None else None

    par_clean = None
    if par_df is not None:
        # Use parent economic condition items when available
        cols = ["ids", "w2be23", "w2be25"]
        cols = [c for c in cols if c in par_df.columns]
        par_clean = par_df[cols].copy()

    sch_clean = None
    if sch_df is not None:
        cols = ["schids", "pla01", "pla04"]
        cols = [c for c in cols if c in sch_df.columns]
        sch_clean = sch_df[cols].copy()
        sch_clean = sch_clean.rename(columns={"pla01": "school_loc", "pla04": "school_type"})

    # 3. Merge Raw Data (Student Centric)
    print("--- Merging Raw Datasets ---")
    merged = stu_df.copy()

    if par_clean is not None:
        merged = pd.merge(merged, par_clean, on="ids", how="left")

    if tea_clean is not None:
        if "w2clsids" in merged.columns:
            if "clsids" in merged.columns:
                merged = merged.drop(columns=["w2clsids"])
            else:
                merged = merged.rename(columns={"w2clsids": "clsids"})
        if "clsids" in merged.columns and "clsids" in tea_clean.columns:
            merged = pd.merge(merged, tea_clean, on="clsids", how="left")

    if sch_clean is not None:
        if "w2schids" in merged.columns:
            if "schids" in merged.columns:
                merged = merged.drop(columns=["w2schids"])
            else:
                merged = merged.rename(columns={"w2schids": "schids"})
        if "schids" in merged.columns and "schids" in sch_clean.columns:
            merged = pd.merge(merged, sch_clean, on="schids", how="left")

    print(f"Merged Raw Shape: {merged.shape}")

    # 4. Feature Engineering & Imputation (Rescue Logic)
    print("--- Applying Rescue Logic ---")

    # 4.1 Target: Education Expectation (w2b18)
    if "w2b18" in merged.columns:
        s_exp = pd.to_numeric(merged["w2b18"], errors="coerce")
        merged["expect_edu_raw"] = s_exp
        merged["expect_college"] = (s_exp >= 7).astype(float)
        merged.loc[s_exp.isna(), "expect_college"] = np.nan
    else:
        merged["expect_college"] = np.nan

    # 4.2 SES (w2a09) -> parent w2be23/w2be25 -> Class Mean
    merged["ses_combined"] = merged.get("w2a09")
    if "w2be23" in merged.columns:
        merged["ses_combined"] = merged["ses_combined"].fillna(merged["w2be23"])
    if "w2be25" in merged.columns:
        merged["ses_combined"] = merged["ses_combined"].fillna(merged["w2be25"])
    if "clsids" in merged.columns:
        class_ses_mean = merged.groupby("clsids")["ses_combined"].transform("mean")
        merged["ses_combined"] = merged["ses_combined"].fillna(class_ses_mean)
    merged["ses_self"] = merged["ses_combined"]

    # 4.3 Hukou (w2a18) -> school_loc -> Mode
    hukou_map = {1: 1, 2: 0, 3: 0, 4: np.nan}
    merged["hukou_type"] = merged["w2a18"].map(hukou_map)
    if "school_loc" in merged.columns:
        school_hukou_proxy = merged["school_loc"].map({1: 0, 2: 0, 3: 0, 4: 1})
        merged["hukou_type"] = merged["hukou_type"].fillna(school_hukou_proxy)
    if "clsids" in merged.columns:
        def get_mode(x):
            m = x.mode()
            return m[0] if not m.empty else np.nan
        class_hukou_mode = merged.groupby("clsids")["hukou_type"].transform(get_mode)
        merged["hukou_type"] = merged["hukou_type"].fillna(class_hukou_mode)
    global_mode = merged["hukou_type"].mode()[0]
    merged["hukou_type"] = merged["hukou_type"].fillna(global_mode)

    # 4.4 Linking SC (Teacher Praise + Talk) -> linking_idx
    praise_cols = ["w2b0507", "w2b0508", "w2b0509"]
    praise_cols = [c for c in praise_cols if c in merged.columns]
    if praise_cols:
        merged["teacher_praise"] = merged[praise_cols].mean(axis=1)
    else:
        merged["teacher_praise"] = np.nan

    if "w2c09" in merged.columns:
        merged["teacher_talk"] = merged["w2c09"].map({1: 1, 2: 0, 3: 0})
    else:
        merged["teacher_talk"] = np.nan

    tp = merged["teacher_praise"].fillna(merged["teacher_praise"].mean())
    tt = merged["teacher_talk"].fillna(merged["teacher_talk"].mean())
    linking_raw = zscore(tp) + zscore(tt)
    merged["linking_idx"] = zscore(linking_raw)

    # 4.5 Bonding SC (Peer/Class Climate) -> bonding_idx
    peer_cols = ["w2b0605", "w2b0606", "w2b0607"]
    peer_cols = [c for c in peer_cols if c in merged.columns]
    if peer_cols:
        peer_mean = merged[peer_cols].mean(axis=1)
        merged["bonding_idx"] = zscore(peer_mean.fillna(peer_mean.mean()))
    else:
        merged["bonding_idx"] = np.nan

    # 4.6 Cognition
    if "w2cogscore" in merged.columns:
        merged["cog_score"] = merged["w2cogscore"]
        merged.loc[merged["cog_score"] == 0, "cog_score"] = np.nan
    else:
        merged["cog_score"] = np.nan

    # 5. Final Selection & Saving
    final_cols = [
        "ids",
        "clsids",
        "schids",
        "expect_college",
        "expect_edu_raw",
        "bonding_idx",
        "linking_idx",
        "teacher_praise",
        "teacher_talk",
        "ses_self",
        "hukou_type",
        "cog_score",
    ]

    if "hr01" in merged.columns:
        final_cols.append("hr01")
    if "hr02" in merged.columns:
        final_cols.append("hr02")

    final_cols = [c for c in final_cols if c in merged.columns]
    final_df = merged[final_cols].copy()

    len_before = len(final_df)
    final_df = final_df.dropna(subset=["expect_college"])
    print(f"Rows dropped due to missing Target: {len_before - len(final_df)}")

    out_path = OUTPUT_DIR / "merged_rescued_all.csv"
    final_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Saved rescued data to {out_path}")

    missing_counts = final_df.isna().sum()
    with open(REPORT_DIR / "merged_data_quality_v2.txt", "w", encoding="utf-8") as f:
        f.write("Merged Data Quality Report (Rescue V2.1 - OFFICIAL)\n")
        f.write("=================================================\n")
        f.write(f"Total Rows: {len(final_df)}\n")
        f.write("Linkage Keys Preserved: ids, clsids, schids\n")
        f.write("Strategies Applied:\n")
        f.write(" - Teacher Data: Aggregated by Class (Mode/Mean)\n")
        f.write(" - SES: Imputed with Parent Econ (w2be23/w2be25) & Class Mean\n")
        f.write(" - Hukou: Imputed with School Location\n\n")
        f.write("Missing Values After Rescue:\n")
        f.write(missing_counts.to_string())


if __name__ == "__main__":
    main()
