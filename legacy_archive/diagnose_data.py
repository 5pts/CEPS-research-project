import pandas as pd
import pyreadstat
import numpy as np

file_path = r"c:\Users\13926\Desktop\CEPS数据汇总\学生数据\cepsw2studentCN.dta"

print("--- DIAGNOSTIC START ---")

try:
    # Read first 1000 rows to be faster but representative
    df, meta = pyreadstat.read_dta(file_path, row_limit=1000)
    
    print(f"Columns loaded: {df.shape[1]}")
    
    # Check w2b18
    if "w2b18" in df.columns:
        print("\n[w2b18] Raw Values (Head):")
        print(df["w2b18"].head(20))
        
        # Print Labels if available
        if "w2b18" in meta.variable_value_labels:
            print("\n[w2b18] Labels:")
            labels = meta.variable_value_labels["w2b18"]
            for k, v in labels.items():
                print(f"  {k}: {v}")
        else:
            print("\n[w2b18] No Labels Found in Metadata")

        print("\n[w2b18] Value Counts:")
        print(df["w2b18"].value_counts(dropna=False))
        
        # Test conversion
        s_num = pd.to_numeric(df["w2b18"], errors='coerce')
        print("\n[w2b18] After to_numeric:")
        print(s_num.head())
        print("NaN count:", s_num.isna().sum())
        
        # Test Threshold
        print("\n[Threshold Check]")
        print(">= 6 count:", (s_num >= 6).sum())
        print(">= 7 count:", (s_num >= 7).sum())
        
    else:
        print("[ERROR] w2b18 NOT FOUND")

    # Check Teacher Columns
    teacher_path = r"c:\Users\13926\Desktop\CEPS数据汇总\任课教师数据\cepsw2teacherCN.dta"
    if "teacher" in file_path or True: # Force check
        try:
             df_tea, _ = pyreadstat.read_dta(teacher_path, row_limit=10)
             print("\n[Teacher Data] Columns:")
             print([c for c in df_tea.columns if "cls" in c.lower() or "id" in c.lower()])
        except Exception as e:
            print(f"Teacher load error: {e}")
        
    # Check Interaction / Correlation
    if "w2b18" in df.columns and "w2b0507" in df.columns:
        # Re-create variables
        s_exp = pd.to_numeric(df["w2b18"], errors='coerce')
        # Try different cutoffs
        y_6 = (s_exp >= 6).astype(int)
        y_7 = (s_exp >= 7).astype(int)
        y_8 = (s_exp >= 8).astype(int)
        
        # Bonding proxy (just one item for quick check)
        x = pd.to_numeric(df["w2b0507"], errors='coerce') # Math teacher praise
        
        print("\n[Correlations]")
        print(f"Bonding vs Expect>=6: {x.corr(y_6):.3f}")
        print(f"Bonding vs Expect>=7: {x.corr(y_7):.3f}")
        print(f"Bonding vs Expect>=8: {x.corr(y_8):.3f}")
        
        print("\n[Means by Cutoff]")
        print(f"Mean Expect>=6: {y_6.mean():.3f}")
        print(f"Mean Expect>=7: {y_7.mean():.3f}")
        print(f"Mean Expect>=8: {y_8.mean():.3f}")

except Exception as e:
    print(f"Error: {e}")

print("--- DIAGNOSTIC END ---")
