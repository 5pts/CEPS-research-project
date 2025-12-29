
# Data Archive Manifest (legacy_archive)
**Date:** 2025-12-28
**Purpose:** Archive deprecated data processing scripts and reports to prevent misuse.
**Official Version:** Rescue V2.0 (See `rescued_data/merged_rescued_all.csv`)

## Archived Files & Folders

| File/Folder | Original Purpose | Reason for Deprecation |
| :--- | :--- | :--- |
| `cleaned/` | Old cleaning output (V1) | High attrition rate (~50%), missing variables. Replaced by `rescued_data/`. |
| `cleaned_reports/` | Old quality reports | Based on V1 data. Replaced by `rescued_reports/`. |
| `data_quality_report_v1.txt` | Initial quality analysis | Mismatched variable definitions. Replaced by `analyze_data_quality.py` output. |
| `merged_data_quality.txt.old` | Old merge report | Missing SES rescue logic. Replaced by `merged_data_quality_v2.txt`. |
| `data_quality_rescued.txt.old` | Intermediate rescue report | Inconsistent linkage keys. Replaced by `merged_data_quality_v2.txt`. |
| `clean_ceps.py` | Original cleaning script | Strict drop logic caused high attrition. Replaced by `clean_ceps_rescue.py`. |
| `diagnose_data.py` | Initial diagnosis script | Outdated variable mapping. |

## Usage Policy
- **DO NOT USE** files in this directory for current analysis.
- **REFERENCE ONLY**: Use these files only if you need to reproduce historical errors or compare improvements.
