import sys, traceback
from pathlib import Path
try:
    import PyPDF2
except Exception:
    traceback.print_exc(); sys.exit(1)
sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
path = Path(r"Paper_LaTeX/report_summary.pdf")
reader = PyPDF2.PdfReader(path.open('rb'))
print('pages', len(reader.pages))
for i,p in enumerate(reader.pages):
    text = p.extract_text() or ''
    print('\n--- Page', i+1, '---')
    print(text[:1600].replace('\n',' '))
