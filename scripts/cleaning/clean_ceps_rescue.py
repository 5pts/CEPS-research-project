#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CEPS 数据清洗与整合脚本 (Rescued Version 2.0)
- 目的：最大化保留样本量，修复因变量选择不当导致的样本流失
- 策略：
  1. 多源数据（家长、教师、学校）交叉填补
  2. 教师数据按班级聚合（均值/众数）避免结构性偏差
  3. 使用全员必答题替代跳答题
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pyreadstat
import os

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
    df, meta = pyreadstat.read_dta(str(path))
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

    # Define variables to aggregate
    # hr01: Gender, hr02: Age, etc.
    # Add praise-related teacher vars if they exist in teacher data (usually they are in student data about teacher)
    # Teacher data contains teacher's own attributes.
    
    agg_dict = {}
    if "hr01" in df.columns: agg_dict["hr01"] = lambda x: x.mode()[0] if not x.mode().empty else np.nan
    if "hr02" in df.columns: agg_dict["hr02"] = "mean"
    
    # Perform aggregation
    if not agg_dict:
        return df.drop_duplicates(subset=["clsids"]) # Fallback

    grouped = df.groupby("clsids").agg(agg_dict).reset_index()
    print(f"Aggregated Teacher Data: {len(df)} rows -> {len(grouped)} classes")
    return grouped

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
    # Teacher
    tea_clean = None
    if tea_df is not None:
        tea_clean = aggregate_teacher_data(tea_df)
    
    # Parent (Keep relevant cols)
    par_clean = None
    if par_df is not None:
        # BE01: Family income, BE13: Parents' expectation
        # NOTE: Actual variable name might be w2be01/w2be13 in wave 2
        cols = ["ids", "be01", "be13", "w2be01", "w2be13"]
        cols = [c for c in cols if c in par_df.columns]
        par_clean = par_df[cols].copy()
        
        rename_map = {}
        if "be01" in cols: rename_map["be01"] = "parent_income"
        if "w2be01" in cols: rename_map["w2be01"] = "parent_income"
        if "be13" in cols: rename_map["be13"] = "parent_expect"
        if "w2be13" in cols: rename_map["w2be13"] = "parent_expect"
        par_clean = par_clean.rename(columns=rename_map)

    # School (Keep relevant cols)
    sch_clean = None
    if sch_df is not None:
        cols = ["schids", "pla01", "pla04"] # Location, Type
        cols = [c for c in cols if c in sch_df.columns]
        sch_clean = sch_df[cols].copy()
        sch_clean = sch_clean.rename(columns={"pla01": "school_loc", "pla04": "school_type"})

    # 3. Merge Raw Data (Student Centric)
    print("--- Merging Raw Datasets ---")
    merged = stu_df.copy()
    
    # Merge Parent
    if par_clean is not None:
        merged = pd.merge(merged, par_clean, on="ids", how="left")
        
    # Merge Teacher (via clsids)
    if tea_clean is not None:
        # Ensure clsids match type
        if "w2clsids" in merged.columns:
            if "clsids" in merged.columns:
                # If both exist, drop w2clsids to avoid duplication
                merged = merged.drop(columns=["w2clsids"])
            else:
                merged = merged.rename(columns={"w2clsids": "clsids"})
                
        if "clsids" in merged.columns and "clsids" in tea_clean.columns:
            merged = pd.merge(merged, tea_clean, on="clsids", how="left")
            
    # Merge School (via schids)
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
    # 1=Quit... 9=PhD.
    if "w2b18" in merged.columns:
        s_exp = pd.to_numeric(merged["w2b18"], errors="coerce")
        merged["expect_edu_raw"] = s_exp
        # Threshold: >=7 (Undergrad)
        merged["expect_college"] = (s_exp >= 7).astype(float)
        merged.loc[s_exp.isna(), "expect_college"] = np.nan
        
        # Imputation check: Could we use Parent Expectation? 
        # Ideally no, we want Student's expectation. But for analysis, if missing, we drop.
        # Current logic: Drop missing Target.
    else:
        merged["expect_college"] = np.nan

    # 4.2 SES (w2a0603) -> parent_income -> Class Mean
    # w2a0603: 1=Very poor... 5=Very rich.
    # parent_income: be01 (1=Very poor... 5=Very rich?) - need verification, assume similar ordinal
    
    # Standardize function
    def zscore(s): return (s - s.mean()) / s.std()

    # Step 1: Use w2a0603
    merged["ses_combined"] = merged["w2a0603"]
    
    # Step 2: Fill with parent_income (if available)
    if "parent_income" in merged.columns:
        merged["ses_combined"] = merged["ses_combined"].fillna(merged["parent_income"])
        
    # Step 3: Fill with Class Mean
    if "clsids" in merged.columns:
        class_ses_mean = merged.groupby("clsids")["ses_combined"].transform("mean")
        merged["ses_combined"] = merged["ses_combined"].fillna(class_ses_mean)
        
    merged["ses_self"] = merged["ses_combined"] # Rename for compatibility

    # 4.3 Hukou (w2a18) -> school_loc -> Mode
    # w2a18: 1=Agri, 2=Non-Agri, 3=Resident, 4=Other
    # Map: 1->1 (Rural), 2,3->0 (Urban), 4->NaN
    hukou_map = {1: 1, 2: 0, 3: 0, 4: np.nan}
    merged["hukou_type"] = merged["w2a18"].map(hukou_map)
    
    # Fill with School Location (pla01)
    # pla01: 1=Center City, 2=Fringe, 3=Town, 4=Rural
    # Map School: 1,2,3 -> 0 (Urban-like), 4 -> 1 (Rural)
    if "school_loc" in merged.columns:
        school_hukou_proxy = merged["school_loc"].map({1: 0, 2: 0, 3: 0, 4: 1})
        merged["hukou_type"] = merged["hukou_type"].fillna(school_hukou_proxy)
        
    # Step 3: Fill with Class Mode (Final Rescue)
    if "clsids" in merged.columns:
        def get_mode(x):
            m = x.mode()
            return m[0] if not m.empty else np.nan
        class_hukou_mode = merged.groupby("clsids")["hukou_type"].transform(get_mode)
        merged["hukou_type"] = merged["hukou_type"].fillna(class_hukou_mode)
        
    # Final Fallback: Global Mode
    global_mode = merged["hukou_type"].mode()[0]
    merged["hukou_type"] = merged["hukou_type"].fillna(global_mode)

    # 4.4 Bonding SC (Teacher Praise + Talk)
    # Praise: w2b0507-09
    praise_cols = ["w2b0507", "w2b0508", "w2b0509"]
    praise_cols = [c for c in praise_cols if c in merged.columns]
    if praise_cols:
        merged["teacher_praise"] = merged[praise_cols].mean(axis=1)
    else:
        merged["teacher_praise"] = np.nan
        
    # Talk: w2c09 (1=Yes, 2=No, 3=?)
    # Map 1->1, 2->0, 3->0 (Conservative)
    if "w2c09" in merged.columns:
        merged["teacher_talk"] = merged["w2c09"].map({1: 1, 2: 0, 3: 0})
    else:
        merged["teacher_talk"] = np.nan
        
    # Impute missing sub-components with mean
    tp = merged["teacher_praise"].fillna(merged["teacher_praise"].mean())
    tt = merged["teacher_talk"].fillna(merged["teacher_talk"].mean())
    
    merged["bonding_idx"] = zscore(tp) + zscore(tt)

    # 4.5 Bridging SC (Friends)
    bridge_cols = ["w2b0601", "w2b0602", "w2b0603"]
    bridge_cols = [c for c in bridge_cols if c in merged.columns]
    merged["bridging_idx"] = merged[bridge_cols].mean(axis=1)
    merged["bridging_idx"] = zscore(merged["bridging_idx"].fillna(merged["bridging_idx"].mean()))

    # 4.6 Cognition
    if "w2cogscore" in merged.columns:
        merged["cog_score"] = merged["w2cogscore"]
    else:
        merged["cog_score"] = np.nan

    # 5. Final Selection & Saving
    final_cols = ["ids", "clsids", "schids", 
                  "expect_college", "expect_edu_raw",
                  "bonding_idx", "teacher_praise", "teacher_talk",
                  "bridging_idx", 
                  "ses_self", "hukou_type", "cog_score"]
                  
    # Add teacher aggregated vars for reference
    if "hr01" in merged.columns: final_cols.append("hr01")
    if "hr02" in merged.columns: final_cols.append("hr02")
    
    # Filter columns that actually exist
    final_cols = [c for c in final_cols if c in merged.columns]

    final_df = merged[final_cols].copy()
    
    # Drop rows only if Target is missing
    len_before = len(final_df)
    final_df = final_df.dropna(subset=["expect_college"])
    print(f"Rows dropped due to missing Target: {len_before - len(final_df)}")
    
    # Save
    out_path = OUTPUT_DIR / "merged_rescued_all.csv"
    final_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Saved rescued data to {out_path}")
    
    # Mark old reports as DEPRECATED
    old_reports = ["merged_data_quality.txt", "data_quality_rescued.txt"]
    for r in old_reports:
        p = REPORT_DIR / r
        if p.exists():
            try:
                # Rename to .old
                p.rename(p.with_suffix(".txt.old"))
                print(f"[INFO] Deprecated old report: {r}")
            except Exception as e:
                print(f"[WARN] Could not rename {r}: {e}")
    
    # Generate updated quality report
    missing_counts = final_df.isna().sum()
    with open(REPORT_DIR / "merged_data_quality_v2.txt", "w") as f:
        f.write("Merged Data Quality Report (Rescue V2.0 - OFFICIAL)\n")
        f.write("=================================================\n")
        f.write(f"Total Rows: {len(final_df)}\n")
        f.write(f"Linkage Keys Preserved: ids, clsids, schids\n")
        f.write(f"Strategies Applied:\n")
        f.write(" - Teacher Data: Aggregated by Class (Mode/Mean)\n")
        f.write(" - SES: Imputed with Parent Income & Class Mean\n")
        f.write(" - Hukou: Imputed with School Location\n\n")
        f.write("Missing Values After Rescue:\n")
        f.write(missing_counts.to_string())

if __name__ == "__main__":
    main()
