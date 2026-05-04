# Model Card — Lewisham Household Arrears Risk Model

## Overview

This model scores households in Lewisham council's housing and benefits caseload by their estimated risk of falling into rent arrears. Its output is a ranked priority list for caseworkers to use in proactive outreach — every prediction is reviewed by a human before any contact or resource allocation decision is made.

Two models are trained in parallel:

| Model | Role |
|---|---|
| Logistic regression (LR) | Interpretable baseline; useful for explaining which features drive a score |
| XGBoost | Primary scoring model; better ranking performance on held-out data |

Both are batch models. They are not designed for real-time serving.

---

## Intended Use

**Primary use:** generating a weekly or monthly ranked list of households to prioritise for caseworker outreach, across Lewisham's housing and benefits caseload.

**Decision context:** the score is an input to a caseworker's judgement, not an automated decision. No household should be contacted, referred, or denied a service solely on the basis of a model score.

**Primary users:** housing and benefits caseworkers; service managers using the list to plan caseload capacity.

---

## Training Data

### Sources

| File | Description |
|---|---|
| `household_data.xlsx` | 17,831 Lewisham households, 21 variables drawn from the council's benefits/housing system. Source: [lb-lewisham/data-scientist-interview-task](https://github.com/lb-lewisham/data-scientist-interview-task). No licence specified — not redistributed. |
| `File_1_IoD2025_Index_of_Multiple_Deprivation.xlsx` | IMD 2025 ranks and deciles for all English LSOAs (MHCLG, Open Government Licence v3.0). Joined on LSOA code to attach deprivation context. |

### Population caveat

The training data covers approximately **14% of Lewisham's ~130,000 dwellings** — specifically those already known to the council's benefits and housing system. The composition (92.5% not in work, 61% on Universal Credit) reflects this selected population. Base rates and feature distributions will differ from the borough population as a whole. Any conclusions drawn apply to this caseload population, not to Lewisham residents generally.

### Target variable

`arrears_flag` — binary indicator of current rent arrears (approximately 25% positive rate in training data). Households with `arrears_flag = 1` also have `arrears_amount > 0`; small inconsistencies between the two fields are logged at training time.

### Features

| Group | Columns | Treatment |
|---|---|---|
| Continuous | `monthly_rent`, `income_after_costs`, `lha_shortfall_weekly`, `ben_cap_amount`, `overpayment_balance`, `imd_rank` | Median imputation (`ben_cap_amount` only); StandardScaler (LR only) |
| Binary | `disability`, `fuel_poverty`, `food_poverty`, `universal_credit`, `ctc_takeup`, `whd_takeup`, `fsm_eligible` | Recoded Yes/No → 1/0; passthrough |
| Categorical | `tenure_type`, `economic_status`, `household_type`, `ward` | OneHotEncoder (LR); OrdinalEncoder (XGB) |
| Ordinal | `age_bracket`, `imd_decile` | OrdinalEncoder with explicit ordering; StandardScaler (LR only) |
| Engineered | `total_shortfall` = `lha_shortfall_weekly` + `ben_cap_amount` | Computed inside the pipeline before encoding |
| Excluded | `reference`, `lsoa21cd`, `arrears_amount` | Identifier, geographic code, and leakage variable — dropped before modelling |

---

## Evaluation

All metrics are computed on a stratified held-out test set (20% of data, withheld before training). Cross-validation (stratified 5-fold) is run on the training set for variance estimates. Full metric values are stored in `metadata.json` inside each model run directory (`models/{timestamp}/`).

| Metric | What it measures in this context |
|---|---|
| ROC-AUC | Overall discrimination — ability to rank households in arrears above those not in arrears |
| PR-AUC | Precision-recall trade-off at the operating point that matters most: identifying true positives in an imbalanced setting |
| F1 (threshold 0.5) | Harmonic mean of precision and recall at a fixed operating threshold |
| Brier score | Calibration of the probability estimate; relevant for the equity overlay (see below) |

No automated deployment gate is defined. Metric values and trends should be reviewed against the monitoring baselines (see [Monitoring](#monitoring)) before each production run.

---

## Equity Overlay

After scoring, a small additive uplift is applied to the model's predicted probability before ranking:

| Criterion | Weight |
|---|---|
| Household has dependent children (`household_type` ∈ {Lone parent, Couple with children} or `fsm_eligible` = 1) | +0.05 |
| Household has a disabled member (`disability` = 1) | +0.05 |

**Composite score = predicted\_probability + sum of applicable weights**

The weights represent a transparent value-judgement — that households with children or disabled members should receive a small systematic boost — not a statistical adjustment or a fairness guarantee. The additive form is only meaningful if the underlying probability estimate is reasonably calibrated; a miscalibrated model can make the composite score misleading. Weights are configured in `config/default.yaml` (`equity.children`, `equity.disability`) and can be adjusted by the council through normal change control.

---

## Fairness

### Approach

No single fairness criterion is used as a deployment gate. This is a deliberate choice, not an oversight. Chouldechova (2017) and Kleinberg et al. (2016) showed that when base rates differ across groups — which they do here — it is mathematically impossible to satisfy demographic parity, equal opportunity, equalised odds, and predictive parity simultaneously. Choosing which criterion to prioritise is a policy decision requiring input from council leadership, legal, and community stakeholders, not a modelling decision.

Instead, the evaluation pipeline computes descriptive sliced metrics for each protected characteristic. These are surfaced in `metadata.json` for review at each training run and should be included in any governance or audit process.

### Sensitive features sliced

- `disability` (0/1)
- `household_type` (Single, Lone parent, Couple with children, Couple without children)
- `age_bracket` (19 bands, 16–17 through 100+)

### Metrics per slice

Selection rate, TPR, FPR, precision, ROC-AUC (where the slice contains both classes).

### Known fairness limitations

- **Ward-level geographic signal** (`ward`, `imd_rank`, `imd_decile`) may act as a proxy for race or ethnicity if residential segregation is present. This is not currently sliced because ethnicity is not in the training data.
- **Universal Credit uptake** is correlated with age bracket (−0.79 Spearman) and several other features. Multicollinearity makes individual feature attribution unreliable; interpretation should be at the prediction level, not feature-importance level.

---

## Known Limitations

1. **Selection bias.** The data describes households already known to the council's benefits and housing system. The model will perform less well — and may be systematically biased — when applied to households newly entering the caseload or from populations under-represented in training.

2. **Static snapshot.** The model is trained on a point-in-time extract. Feature distributions shift over time due to policy changes (e.g. Universal Credit rollout), economic cycles, and seasonal effects. Performance should be expected to degrade without periodic retraining.

3. **No causal inference.** Associations in the training data are not evidence of causal mechanisms. The model should not be used to design interventions (e.g., "reducing benefit cap amount reduces arrears risk") — only to prioritise outreach.

4. **Arrears flag / amount consistency.** A small number of rows have `arrears_flag = 1` with `arrears_amount = 0` or vice versa, likely due to timing differences in the source system. These are logged but not filtered. The flag is treated as the ground truth for modelling.

5. **Calibration.** The model outputs a ranking score. While Brier score is tracked, the output has not been formally calibrated (e.g. via Platt scaling or isotonic regression). The equity overlay assumes adequate calibration; if Brier score is poor, review the composite score interpretation.

---

## Monitoring

The following checks are recommended before each production scoring run and as part of quarterly model review.

### Input drift

| Check | Method | Threshold |
|---|---|---|
| Feature distribution shift | Population Stability Index (PSI) on all continuous and categorical input features, comparing current run against training baseline | PSI > 0.2 → flag for review; PSI > 0.25 → block run pending investigation |
| Score distribution shift | KS test comparing current predicted-probability distribution against training baseline | p < 0.05 → flag for review |

### Concept drift

| Check | Method | Threshold |
|---|---|---|
| Arrears base rate shift | Compare current `arrears_flag` rate in new data against training baseline (~25%) | > 5 percentage point shift → trigger retraining review |
| Model performance on labelled new data | Compute ROC-AUC and PR-AUC on ground-truth labels once available (typically lagged ~3 months) | > 0.05 drop from training held-out value → review |

### Fairness drift

Recompute sliced metrics (selection rate, TPR by disability/household_type/age bracket) at each production run and compare against training baseline. Significant divergence should be reviewed before the priority list is distributed.

---

## Out-of-scope Uses

- Automated decisions without caseworker review
- Real-time or API-based scoring
- Other local authorities without retraining on local data
- Predicting arrears probability for individual households over multi-year horizons
- Enforcement or legal proceedings
- Any purpose outside Lewisham council's housing and benefits service operations
