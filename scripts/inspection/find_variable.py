import pyreadstat
import pandas as pd

dta_file = r"c:\Users\13926\Desktop\CEPS数据汇总\学生数据\cepsw2studentCN.dta"

try:
    # df, meta = pyreadstat.read_dta(dta_file, metadata_only=True)
    # metadata_only is not supported in some versions?
    # Let's read small rows
    df, meta = pyreadstat.read_dta(dta_file, row_limit=10)
    
    # Try to decode labels
    print("Scanning variables...")
    for col, label in meta.column_names_to_labels.items():
        try:
            # Attempt to fix encoding
            # pyreadstat often reads as utf-8 or latin-1 but original is gb18030
            # If it looks like garbage, it might be latin-1 interpretation of gb bytes
            decoded_label = label
            try:
                decoded_label = label.encode('latin1').decode('gb18030')
            except:
                pass
            
            # Search keywords
            if any(k in decoded_label for k in ["期望", "学历", "读", "上学", "大学"]):
                print(f"MATCH: {col} - {decoded_label}")
                # Print unique values
                print(f"   Values: {df[col].unique()}")
        except Exception as e:
            pass
            
    # Check w2b05 series labels
    print("\nChecking w2b05 labels:")
    for i in range(1, 10):
        c = f"w2b05{i:02d}"
        if c in meta.column_names:
            label = meta.column_names_to_labels.get(c, "")
            try:
                label = label.encode('latin1').decode('gb18030')
            except:
                pass
            print(f"{c}: {label}")


except Exception as e:
    print(e)
