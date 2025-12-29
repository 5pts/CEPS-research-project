
import pandas as pd
import pyreadstat
from pathlib import Path

FILES = {
    "parent": Path(r"c:\Users\13926\Desktop\CEPS数据汇总\家长数据\cepsw2parentCN.dta"),
    "student": Path(r"c:\Users\13926\Desktop\CEPS数据汇总\学生数据\cepsw2studentCN.dta"),
}

def inspect_cols():
    print("--- Inspecting Parent Data Columns ---")
    if FILES["parent"].exists():
        df, meta = pyreadstat.read_dta(str(FILES["parent"])) # Read full to be safe, or just meta if supported
        cols = [c for c in df.columns if "be01" in c.lower() or "income" in c.lower()]
        print(f"Candidates for Income: {cols}")
        # Check exact match
        print(f"Has 'be01'? {'be01' in df.columns}")
        print(f"Has 'BE01'? {'BE01' in df.columns}")
        
    print("\n--- Inspecting Student Data ID Columns ---")
    if FILES["student"].exists():
        df, meta = pyreadstat.read_dta(str(FILES["student"]))
        id_cols = [c for c in df.columns if "id" in c.lower()]
        print(f"ID Columns: {id_cols}")

if __name__ == "__main__":
    inspect_cols()
