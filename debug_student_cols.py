import pandas as pd
import pyreadstat
from pathlib import Path

FILE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总\学生数据\cepsw2studentCN.dta")

def check():
    if not FILE.exists():
        print("File not found")
        return
        
    df, meta = pyreadstat.read_dta(str(FILE), row_limit=10)
    print("Columns in Student Data:")
    cols = list(df.columns)
    id_cols = [c for c in cols if "id" in c.lower() or "sch" in c.lower()]
    print(id_cols)
    
    if "w2schids" in cols: print("Found: w2schids")
    else: print("Missing: w2schids")
    
    if "schids" in cols: print("Found: schids")
    else: print("Missing: schids")

if __name__ == "__main__":
    check()
