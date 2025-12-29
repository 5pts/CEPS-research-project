
import pandas as pd
import pyreadstat
from pathlib import Path

FILE = Path(r"c:\Users\13926\Desktop\CEPS数据汇总\学生数据\cepsw2studentCN.dta")

def check_ids():
    df, meta = pyreadstat.read_dta(str(FILE), metadata_only=True)
    print("Columns:", meta.column_names)
    print("Has schids?", "schids" in meta.column_names)
    print("Has w2schids?", "w2schids" in meta.column_names)

if __name__ == "__main__":
    check_ids()
