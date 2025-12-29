
import pandas as pd
import numpy as np
from pathlib import Path
import pyreadstat
import sys

# Paths
WORKSPACE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总")
RAW_DIR = WORKSPACE
OUTPUT_FILE = WORKSPACE / "data_quality_report.txt"

FILES = {
    "student": RAW_DIR / "学生数据" / "cepsw2studentCN.dta",
    "parent": RAW_DIR / "家长数据" / "cepsw2parentCN.dta",
    "teacher": RAW_DIR / "任课教师数据" / "cepsw2teacherCN.dta",
    "principal": RAW_DIR / "校领导学校数据" / "cepsw2principalCN.dta",
}

def analyze():
    report = []
    
    def log(msg):
        print(msg)
        report.append(msg)

    log("=== CEPS Data Quality & Attrition Analysis (Aligned with Rescue V2.0) ===\n")

    # 1. Load Student Data (Base Population)
    log("--- 1. Student Data Analysis ---")
    if not FILES["student"].exists():
        log("Error: Student file not found.")
        return
    
    # 统一口径1：学生主键使用 ids
    df_stu, meta = pyreadstat.read_dta(str(FILES["student"]))
    n_original = len(df_stu)
    log(f"Original Student Sample Size (ids): {n_original}")

    # 统一口径3：关键变量集合 (含 Rescue 涉及的变量)
    key_vars = {
        "w2b18": "Educational Expectation (Target)",
        "w2a0603": "SES (Student Self-reported)",
        "w2a18": "Hukou (Student Self-reported)",
        "w2c09": "Talk to Teacher (Skip=Valid)",
        "w2b0507": "Teacher Praise (Math)",
        "w2b0508": "Teacher Praise (Chinese)",
        "w2b0509": "Teacher Praise (English)",
        "w2b0601": "Friend Study Hard",
        "w2b0602": "Friend Expect College",
        "w2b0603": "Friend Good Grades"
    }

    log("\n[Raw Missing Rates for Key Student Variables]")
    high_missing_vars = []
    
    for var, desc in key_vars.items():
        if var in df_stu.columns:
            # 统一口径4：区分真缺失与无效编码
            # w2c09: 1=Yes, 2=No, 3=Skip(Valid No) -> Only NaN is missing
            if var == "w2c09":
                 # Check if 3 exists
                 n_skips = (df_stu[var] == 3).sum()
                 n_miss = df_stu[var].isna().sum()
                 log(f"  {var} ({desc}): {n_miss} NaN, {n_skips} Skips (Valid=No)")
            else:
                n_miss = df_stu[var].isna().sum()
                log(f"  {var} ({desc}): {n_miss} missing")
                
            pct_miss = (n_miss / n_original) * 100
            if pct_miss > 20:
                high_missing_vars.append((var, desc, pct_miss))
        else:
            log(f"  {var} ({desc}): NOT FOUND in dataset")

    # 2. Parent Data Analysis (SES Rescue Source)
    log("\n--- 2. Parent Data Analysis (SES Rescue Source) ---")
    if FILES["parent"].exists():
        df_par, _ = pyreadstat.read_dta(str(FILES["parent"]))
        
        # Check if be01/w2be01 exists in parent data
        inc_var = "be01"
        if "w2be01" in df_par.columns: inc_var = "w2be01"
        elif "be01" in df_par.columns: inc_var = "be01"
        else: inc_var = None

        if inc_var is None:
            log("Error: 'be01' or 'w2be01' (Family Income) not found in Parent Data.")
        else:
            # Merge on ids
            merged_par = pd.merge(df_stu[["ids"]], df_par, on="ids", how="left")
            
            # Check be01 (Family Income) availability for SES rescue
            # Logic: If w2a0603 is missing, do we have be01?
            if "w2a0603" in df_stu.columns:
                stu_ses_missing = df_stu["w2a0603"].isna()
                rescue_potential = merged_par.loc[stu_ses_missing, inc_var].notna().sum()
                
                log(f"Student SES Missing: {stu_ses_missing.sum()}")
                log(f"Recoverable via Parent Income ({inc_var}): {rescue_potential}")
                log(f"Remaining Missing SES after Parent Rescue: {stu_ses_missing.sum() - rescue_potential}")
            else:
                 log("Error: 'w2a0603' not found in Student Data")
    
    # 3. Teacher Data Analysis (Rescue Source for Bonding)
    log("\n--- 3. Teacher Data Analysis (Aligned Aggregation) ---")
    if FILES["teacher"].exists():
        df_tea, _ = pyreadstat.read_dta(str(FILES["teacher"]))
        
        # 统一口径5：教师数据合并口径
        # Clean Rescue Script Logic:
        # 1. Rename w2clsids -> clsids
        # 2. Group by clsids -> Aggregation (Mode/Mean)
        
        if "w2clsids" in df_tea.columns:
            df_tea = df_tea.rename(columns={"w2clsids": "clsids"})
        
        # Simulate Aggregation (Simplified for QA check)
        # We just need to know if the CLASS has *any* teacher data
        unique_classes_with_data = df_tea["clsids"].unique()
        
        # Check Student Linkage
        stu_cls_var = "w2clsids"
        if "w2clsids" in df_stu.columns: stu_cls_var = "w2clsids"
        elif "clsids" in df_stu.columns: stu_cls_var = "clsids"
        else: stu_cls_var = None

        if stu_cls_var:
             # In student data, it might be w2clsids
             stu_classes = df_stu[stu_cls_var]
             # How many students are in classes that have teacher data?
             students_with_teacher_data = stu_classes.isin(unique_classes_with_data).sum()
             
             log(f"Students with Linked Teacher Data (via {stu_cls_var} -> clsids): {students_with_teacher_data}")
             log(f"Linkage Failure: {n_original - students_with_teacher_data} ({(n_original - students_with_teacher_data)/n_original*100:.2f}%)")
        else:
             log("Error: clsids/w2clsids not found in Student Data")

    # 4. School Data Analysis (Rescue Source for Hukou)
    log("\n--- 4. School Data Analysis (Hukou Rescue Source) ---")
    if FILES["principal"].exists():
         df_sch, _ = pyreadstat.read_dta(str(FILES["principal"]))
         # Check Linkage
         stu_sch_var = "w2schids"
         if "w2schids" in df_stu.columns: stu_sch_var = "w2schids"
         elif "schids" in df_stu.columns: stu_sch_var = "schids"
         else: stu_sch_var = None

         if stu_sch_var: # Clean script renames w2schids -> schids
             stu_sch_ids = df_stu[stu_sch_var]
             # Check match
             # Ensure school data has schids
             sch_key_var = "schids"
             if "w2schids" in df_sch.columns and "schids" not in df_sch.columns:
                 df_sch = df_sch.rename(columns={"w2schids": "schids"})
             
             if "schids" in df_sch.columns:
                 matched = stu_sch_ids.isin(df_sch["schids"]).sum()
                 log(f"Students with Linked School Data (via {stu_sch_var} -> schids): {matched}")
                 
                 # Check Hukou Rescue Potential
                 # If w2a18 is missing, do we have school data?
                 if "w2a18" in df_stu.columns:
                     stu_hukou_missing = df_stu["w2a18"].isna()
                     # Assuming all linked schools have location data (pla01)
                     # We can approximate by just checking linkage for missing rows
                     missing_rows_sch_ids = df_stu.loc[stu_hukou_missing, stu_sch_var]
                     recoverable = missing_rows_sch_ids.isin(df_sch["schids"]).sum()
                     log(f"Student Hukou Missing: {stu_hukou_missing.sum()}")
                     log(f"Recoverable via School Location: {recoverable}")
                 else:
                     log("Error: w2a18 not found in Student Data")
             else:
                 log("Error: schids not found in School Data")
         else:
             log("Error: schids/w2schids not found in Student Data")

    # 5. Final Retention Simulation (Based on Rescue V2.0 Logic)
    log("\n--- 5. Retention Simulation (Rescue V2.0) ---")
    # Condition: w2b18 must exist. All others imputed.
    if "w2b18" in df_stu.columns:
        n_retained = df_stu["w2b18"].notna().sum()
        log(f"Final Retained Sample (Target Available): {n_retained}")
        log(f"Final Attrition Rate: {100 - (n_retained/n_original)*100:.2f}%")
    
    # Save Report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"\nReport saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze()
