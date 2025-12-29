
import pandas as pd
import pyreadstat
from pathlib import Path

FILE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总\学生数据\cepsw2studentCN.dta")

def inspect_values():
    df, meta = pyreadstat.read_dta(str(FILE))
    
    print("--- w2a0603 (SES Self) Value Counts ---")
    if "w2a0603" in df.columns:
        print(df["w2a0603"].value_counts(dropna=False).sort_index())
    
    print("\n--- w2a18 (Hukou) Value Counts ---")
    if "w2a18" in df.columns:
        print(df["w2a18"].value_counts(dropna=False).sort_index())

    print("\n--- w2c09 (Teacher Talk) Value Counts ---")
    if "w2c09" in df.columns:
        print(df["w2c09"].value_counts(dropna=False).sort_index())

if __name__ == "__main__":
    inspect_values()
