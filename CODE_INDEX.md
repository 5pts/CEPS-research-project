# Code Index

This project organizes scripts by task. Paths are relative to the project root.

## scripts/cleaning
- `scripts/cleaning/clean_ceps_rescue.py`: Rescue V2.0 cleaning, merging, and imputation pipeline.

## scripts/quality
- `scripts/quality/analyze_data_quality.py`: Data quality, linkage, and retention checks aligned with Rescue V2.0.

## scripts/analysis
- `scripts/analysis/evaluate_suitability.py`: Model suitability and diagnostics (harmonic vs alternatives).
- `scripts/analysis/ml_analysis.py`: GAM and decision-tree exploration of nonlinear effects.
- `scripts/analysis/visualize_rescued.py`: Visualizations based on rescued dataset.

## scripts/reporting
- `scripts/reporting/create_topic_doc.py`: Generate topic definition document.
- `scripts/reporting/generate_final_report.py`: Build final report output.
- `scripts/reporting/generate_progress_report.py`: Build progress report output.

## scripts/inspection
- `scripts/inspection/check_ids.py`: Check presence of key ID fields in student data.
- `scripts/inspection/find_variable.py`: Locate variables in datasets.
- `scripts/inspection/inspect_dta.py`: Quick inspection of .dta files.
- `scripts/inspection/inspect_meta.py`: Inspect metadata (labels, value mappings).
- `scripts/inspection/inspect_vars.py`: Inspect variable sets and distributions.

## Other key folders
- `cleaned/`, `cleaned_reports/`: Outputs from baseline cleaning.
- `rescued_data/`, `rescued_reports/`: Outputs from Rescue V2.0.
- `evaluation/`: Model evaluation outputs and figures.
