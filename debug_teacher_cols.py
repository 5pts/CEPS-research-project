import pandas as pd
import pyreadstat
from pathlib import Path

FILE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总\任课教师数据\cepsw2teacherCN.dta")

def check():
    if not FILE.exists():
        print("File not found")
        return
        
    df, meta = pyreadstat.read_dta(str(FILE), row_limit=10)
    print("Columns in Teacher Data:")
    cols = list(df.columns)
    print(cols)
    
    # Check for HR01/HR02 variants
    hr_cols = [c for c in cols if "hr" in c.lower()]
    print(f"HR Cols: {hr_cols}")

if __name__ == "__main__":
    check()
