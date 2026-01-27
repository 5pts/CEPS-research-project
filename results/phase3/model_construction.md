# Model Construction (Rigorous Specification)

This document describes the final modeling strategy, assumptions, and implementation details for educational expectation using CEPS Wave 2 data.

## 1) Outcome Definition
Primary outcome uses the ordinal item w2b18:

`expect_edu_raw ∈ {1,2,3,4,5,6,7,8,9,10}`

where:
1=现在就不要念了, 2=初中毕业, 3=中专/技校, 4=职业高中, 5=普通高中, 6=大学专科, 7=大学本科, 8=研究生, 9=博士, 10=无所谓.

We treat **10=无所谓** as a non-ordinal response and exclude it from the ordered models. This is explicitly justified and tested (see Section 5).

## 2) Predictors
All models include:
- `bonding_idx` (peer bonding index; horizontal ties)
  - Items: w2b0605 (班里大多数同学对我很友好), w2b0606 (我所在的班级班风良好), w2b0607 (我经常参加学校或班级组织的活动).
- `linking_idx` (teacher-student linking index; vertical ties)
  - Items: teacher_praise = mean(w2b0507, w2b0508, w2b0509); teacher_talk = w2c09 (1→1, 2/3→0).
  - Scaling: linking_idx = zscore(zscore(teacher_praise) + zscore(teacher_talk)).
- `ses_pca` (SES composite, PCA index)
  - Items: parent_edu_max, family_econ, home_books, has_desk, has_computer.
  - `has_computer` is inverted (1 - value) to align direction with other SES items.
- `hukou_type` (0=urban, 1=rural)
- `cog_score` (cognitive score; treat 0 as missing/non-participant)

Continuous predictors are standardized (z-score) before modeling, except `hukou_type`.

## 3) Main Model: Ordered Logit with Cluster-Robust SE

Let `Y_i` be the ordinal expectation (1-9 after excluding 10). Let `x_i` be the predictor vector.

**Ordered logit (proportional odds):**

P(Y_i ≤ k | x_i) = Λ(θ_k - x_i'β), for k = 1,...,8

where Λ(z) = 1 / (1 + e^{-z}) and θ_k are cutpoints.

**Estimation:**
- Maximum likelihood (Ordered Logit).
- Cluster-robust standard errors by class (`clsids`) to account for intra-class correlation.

**Cluster-robust variance (sandwich):**

V̂_CR = (X'W X)^{-1} ( Σ_g X_g' u_g u_g' X_g ) (X'W X)^{-1}

where g indexes classes, u_g are score residuals, and X_g is the design matrix for class g.

Implementation: `scripts/analysis/ordinal_analysis.py`

## 4) Proportional Odds (PO) Check
We compare ordered logit to multinomial logit via a likelihood-ratio test:

LR = 2(LL_MNLogit - LL_Ordered)

If p < 0.05, PO assumption may be violated. This is reported in:
`results/phase3/ordinal_model_report.txt`

Recommended interpretation:
- Ordered logit remains the primary model for interpretability.
- Multinomial logit can be added as a robustness check if reviewers require relaxation of PO.

## 5) Handling "10=无所谓": Sensitivity Checks
We test whether students with response 10 differ from 1–9 in key predictors:
- t-tests for continuous predictors (SES, cognition, bonding, linking)
- chi-square for hukou

If differences are significant, we explicitly note the potential selection bias when excluding 10, and report that the ordered model applies to respondents who express a concrete educational path.

See `results/phase3/ordinal_model_report.txt`.

## 6) Exploratory Machine Learning (Random Forest)
Purpose: detect non-linear patterns and variable importance. RF is **exploratory** and not used for causal claims.

Model: RandomForestClassifier on classes 1–9, with:
- `n_estimators=200`, `max_depth=5`, `min_samples_leaf=80`
- Features: bonding, linking, SES, hukou, cognition

Outputs:
- Feature importance: `results/phase3/ordinal_rf_feature_importance.csv`
- Partial dependence plots (PDP): `figures/ordinal_rf/pdp_*.png`

Interpretation: PDPs show average marginal effect of each feature on the probability of a chosen target class (default = 7).

Implementation: `scripts/analysis/ordinal_rf_pca.py`

## 7) Example Code (Reproducible Snippets)

Ordered Logit (main model):
```python
from statsmodels.miscmodels.ordinal_model import OrderedModel

X = df[["bonding_idx_z","linking_idx_z","ses_pca_z","hukou_type","cog_score_z"]]
y = df["expect_edu_raw"]  # 1-9 only
model = OrderedModel(y, X, distr="logit")
res = model.fit(method="bfgs", disp=False)
```

Cluster-robust SE by class:
```python
from statsmodels.stats.sandwich_covariance import cov_cluster
import numpy as np

cov = cov_cluster(res, df["clsids"])
se = np.sqrt(np.diag(cov))
```

Random Forest + PDP:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                            min_samples_leaf=80, random_state=42)
rf.fit(X, y)
PartialDependenceDisplay.from_estimator(rf, X, ["bonding_idx"],
                                       target=target_idx)
```

## 8) Figures to Include in the Report/Email
- Expectation distribution: `figures/report_phase3/expectation_distribution.png`
- Predictor distributions: `figures/report_phase3/dist_*.png`
- Correlation matrix: `figures/report_phase3/correlation_matrix.png`
- RF PDPs: `figures/ordinal_rf/pdp_*.png`

## 9) Files Generated in This Round
- `results/phase3/ordinal_model_report.txt`
- `results/phase3/ordinal_rf_feature_importance.csv`
- `results/phase3/data_summary_stats.csv`
- `results/phase3/data_package_summary.md`
- `results/phase3/model_construction.md`
