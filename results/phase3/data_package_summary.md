# Data Package Summary (for Advisor Email)

## 1) Cleaned Dataset
- File: `rescued_data/merged_rescued_all_with_pca_ses.csv`
- Format: CSV (UTF-8)
- Rows: 9,827
- Columns: 18
- Key identifiers: `ids` (student), `clsids` (class)

### Variables
- `expect_edu_raw` (w2b18): educational expectation (1-10). Note: 10 = "无所谓".
- `expect_college`: binary expectation (1 if expect >=本科, else 0).
- `bonding_idx`: peer bonding index (constructed; horizontal ties).
  - Items: w2b0605, w2b0606, w2b0607 (peer/class climate).
- `linking_idx`: teacher-student linking index (constructed; vertical ties).
  - Items: teacher_praise = mean(w2b0507, w2b0508, w2b0509); teacher_talk = w2c09 (1→1, 2/3→0).
  - Scaling: zscore(zscore(teacher_praise) + zscore(teacher_talk)).
- `ses_pca`: SES composite (PCA index from parent education, family economic status, home books, and household assets).
- `ses_pca_group`: SES group (0=lower, 1=higher) based on PCA split.
- `ses_self`: legacy SES composite (w2a09, w2be23/w2be25, class-mean rescue).
- `hukou_type`: hukou (0 urban, 1 rural).
- `cog_score`: cognitive score.
- `teacher_praise`, `teacher_talk`: components for teacher-student linking.

### Missingness (post-cleaning)
- `cog_score` has missing values after treating 0 as non-participation.
- Minor remaining missingness: `teacher_praise` (~0.1%), `teacher_talk` (~1.7%) is imputed into `linking_idx`.
- Full stats table: `results/phase3/data_summary_stats.csv`

## 2) Cleaning Process (Concise)
Source: `scripts/cleaning/clean_ceps_rescue.py`

1. **Merge multi-source data**: student + parent + teacher + school.
2. **Target variable**: use w2b18 (all-student question) to avoid skip-pattern missingness.
   - `expect_edu_raw = w2b18` (1-10 scale).
   - `expect_college = 1(w2b18 >= 7)`, else 0.
3. **SES (PCA)**: compute PCA index from parent education, family economic status, home books, and household assets.
   - `ses_self` is retained as a legacy composite but not used in the main model.
   - `has_computer` is inverted (1 - value) in PCA due to negative correlation with other SES items.
4. **Cognition**: `cog_score=0` is treated as missing (non-participant) and excluded from modeling.
4. **Hukou rescue**: map w2a18, fill by school location and class mode, then global mode.
5. **Teacher data**: aggregated to class level; student-level linking uses teacher praise + talk.
6. **Bonding index**: peer/class climate items (w2b0605/0606/0607) aggregated and z-scored.
7. **SES**: w2a09 with parent econ (w2be23/w2be25), then class-mean rescue.

This pipeline preserves sample size while keeping the measurement logic transparent.

## 3) Basic Descriptive Figures
Figures in: `figures/report_phase3/`
- `figures/report_phase3/expectation_distribution.png`
- `figures/report_phase3/dist_bonding_idx.png`
- `figures/report_phase3/dist_linking_idx.png`
- `figures/report_phase3/dist_ses_self.png` (legacy)
- `figures/report_phase3/interaction_plot_pca.png`

## 5) PCA Report
- `results/phase3/ses_pca_report.txt`
- `figures/report_phase3/dist_cog_score.png`
- `figures/report_phase3/correlation_matrix.png`

## 4) Quick Data Description (Email-ready)
The cleaned dataset contains 9,827 students with class identifiers and a fully observed educational expectation variable (w2b18). The main predictors capture peer bonding, teacher-student linking, family SES, hukou, and cognitive score. The target retains its original 1-10 ordinal scale, which enables ordered modeling without collapsing information. Figures show a strong concentration around expectation levels 6–9 and moderate correlations among the predictors.
