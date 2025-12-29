import pyreadstat
import pandas as pd

dta_file = r"c:\Users\13926\Desktop\CEPS数据汇总\学生数据\cepsw2studentCN.dta"

try:
    df, meta = pyreadstat.read_dta(dta_file)
    print("Variable Label for w2c11:", meta.column_names_to_labels.get("w2c11"))
    print("Value Labels for w2c11:", meta.variable_value_labels.get("w2c11"))
    print("Unique values in data:", df["w2c11"].unique())
except Exception as e:
    print(e)
