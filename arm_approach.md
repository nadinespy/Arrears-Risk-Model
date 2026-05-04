# Arrears Risk Model: Analytical Approach

## Overview

This document details the analytical approach for the Lewisham arrears risk task, including the reasoning behind decisions, options considered, and choices made.

Two tasks:
- **Task 1**: Understand patterns of vulnerability across households and small areas
- **Task 2**: Build a predictive model to identify households at risk of arrears, and develop a prioritisation strategy for the top 10%

---

## Data

### Datasets

| Dataset | Description |
|---|---|
| `household_data.xlsx` | 17,831 households in Lewisham, 21 variables |
| `File_1_IoD2025_Index_of_Multiple_Deprivation.xlsx` | IMD rank and decile for all 33,755 LSOAs in England |
| `lewisham_lsoa_to_ward_best-fit_lookup.geojson` | Polygon boundaries for Lewisham's 175 LSOAs, for mapping |

### Population

The dataset does not cover all Lewisham households. Lewisham has ~130,000 dwellings (Census 2021); the dataset contains 17,831 (~14%). The composition — 92.5% not in work, 61% on Universal Credit, and variables drawn from the benefits system (LHA shortfall, benefit cap, overpayment balance) — indicates this is a subset of households known to the council's benefits/housing system, not the borough population as a whole. This is not stated explicitly in the data and remains an inference. Any patterns found describe this population, not Lewisham in general.

### Key variable definitions

Most variable names are self-explanatory (`disability`, `fuel_poverty` etc.). The following are less obvious:

- **`tenure_type`**: The household's housing arrangement. Values:
  - *Council tenant* — rents directly from the local authority (Lewisham council); secure tenancy, lowest rents
  - *Social tenant* — rents from a housing association (registered non-profit); similar rents and security to council housing, different landlord
  - *Private tenant* — rents on the private market; typically higher rents, less security
  - *Owner-occupier* — owns the property
  - *Supported housing* — accommodation with wraparound support/care (e.g. for mental health needs, disabilities, domestic abuse); could be council- or housing association-run
  - *Temporary accommodation* — short-term placement for households accepted as homeless; resident is a licensee not a tenant, intended as transitional

- **`lha_shortfall_weekly`**: Local Housing Allowance (LHA) is the maximum housing benefit paid to private renters. The shortfall is the amount by which actual rent exceeds the LHA rate — i.e. the gap the household must cover themselves
- **`ben_cap_amount`**: The UK government caps total benefits a household can receive. If their benefits would otherwise exceed the cap, this is the amount deducted — a direct income reduction
- **`income_after_costs`**: Estimated household income after essential costs. The task brief says "essential costs" without specifying what's included — likely dominated by rent given this population, but could also include utilities, council tax, etc. Can be negative when costs exceed income
- **`overpayment_balance`**: Outstanding balance of benefits previously overpaid to the household, which they are required to repay — an ongoing financial drag
- **`ctc_takeup`**: Whether the household claims Child Tax Credit — a legacy benefit (being replaced by Universal Credit) for households with children. Very low takeup (0.4%) because most claimants have migrated to UC
- **`whd_takeup`**: Whether the household receives the Warm Home Discount — a £150/year rebate on energy bills for low-income households, automatically applied based on benefit receipt
- **`fsm_eligible`**: Whether the household contains children eligible for Free School Meals — a widely used proxy for child poverty, based on receiving UC below a certain income threshold

### Target variable

`arrears_flag` (binary: 1 = in arrears, 0 = not) is the primary target. `arrears_amount` (continuous, £) is the total amount of arrears across housing and council tax combined. It captures both the flag and severity, and is used as an outcome in some analyses (see below).

Note: it should be verified that all households with `arrears_flag = 1` also have `arrears_amount > 0`. If so, `arrears_amount` subsumes the flag and can be used as the single outcome variable.

### Missing data

Missing values should be checked and handled before any analysis. The appropriate approach depends on the extent of missingness:
- If minimal: for continuous variables, impute with median (robust to outliers); for categorical variables, impute with mode
- If substantial: consider whether missingness is itself informative (add a binary indicator feature for "was this missing?"), or whether the variable should be dropped
- If systematic (e.g. missing only for a specific subgroup): treat with care — imputation may introduce bias

### IMD data join

The IMD file covers all of England. It is joined to household data on `lsoa21cd`, which filters to Lewisham's 175 LSOAs. Every household in the same LSOA receives the same IMD rank and decile — it is an area-level variable, not a household-level one. It captures "where you live" rather than "who you are."

**What IMD measures**: a composite of seven domains — Income (22.5%), Employment (22.5%), Education (13.5%), Health (13.5%), Crime (9.3%), Barriers to Housing & Services (9.3%), and Living Environment (9.3%). The Income and Employment domains together account for 45% of the score and are based on benefit receipt and worklessness — largely the same signals the household-level data already captures directly. This is relevant for interpreting the (lack of) correlation between IMD and arrears within this population.

**What is a decile?** The IMD ranks all 33,755 LSOAs in England from 1 (most deprived) to 33,755 (least deprived), then divides them into 10 equal groups. Decile 1 = most deprived 10% of English LSOAs; decile 10 = least deprived 10%.

---

## Task 1: Vulnerability Analysis

### 1.1 Descriptive overview

A basic portrait of the dataset before any analysis, to understand what the data looks like and flag any anomalies.

- Overall arrears rate (proportion of households in arrears)
- Distribution of `arrears_amount` for households in arrears only — typical severity, skew, outliers
- Distributions and rates for key variables across all households:
  - Binary/categorical: `tenure_type`, `economic_status`, `household_type`, `disability`, `fuel_poverty`, `food_poverty`, `universal_credit`, `ctc_takeup`, `whd_takeup`, `free_school_meals` — bar charts showing proportion of households in each category (for binary variables: proportion with value = 1)
  - Continuous: `monthly_rent`, `income_after_costs`, `lha_shortfall_weekly`, `ben_cap_amount`, `overpayment_balance` — histograms or box plots
  - Ordinal/grouped: `age_bracket` — bar chart of counts per bracket
  - Area-level: distribution of `IMD rank` and `IMD decile` across Lewisham's LSOAs — gives a sense of where Lewisham sits in the national deprivation picture
- Summary table: for each variable, count, mean or proportion, and for binary variables the proportion equal to 1

---

### 1.2 IMD analysis: How does the proportion of households in arrears vary by IMD decile?

**On the outcome variable for this section**

`arrears_amount` is used rather than `arrears_flag` throughout Task 1. It captures both whether a household is in arrears and how severe that arrears is. Two complementary views are reported for each analysis: proportion of households in arrears (% with arrears_amount > 0) and average arrears amount. The latter conditions on the zero values being meaningful (not in arrears), not a data quality issue.

**Visualisations:**

1. **Bar chart**: proportion of households in arrears per IMD decile, bars ranked 1–10 (most to least deprived). Quick overview of the gradient.

2. **Scatterplot**: each of the 175 LSOAs is one point. X-axis = IMD rank; Y-axis = arrears rate for that LSOA. Overlay a trend line. Report **Spearman rank correlation**.

   Spearman is used rather than Pearson because the IMD decile is ordinal (ranked categories with no guarantee that the intervals between deciles are equal), and Spearman makes no such assumption. IMD rank is more granular than decile (175 distinct values rather than 10) but Spearman remains the appropriate choice as the underlying construct is still a ranking.

3. **Two side-by-side maps**: one coloured by arrears rate per LSOA, one coloured by IMD decile per LSOA. If the spatial patterns look similar, this is direct visual evidence of the relationship. Combining both variables in a single map was considered but rejected — LSOAs are small polygons making embedded numbers illegible, and encoding two variables in one colour scale is ambiguous.

4. Average arrears amount per decile/LSOA shown alongside the proportion in arrears — capturing severity, not just prevalence.

---

### 1.3 Feature association analysis: Which factors are most associated with arrears?

**Two complementary approaches:**

**A. Correlation heatmap (broad overview)**

Pairwise Spearman correlations between all features and `arrears_amount`, displayed as a heatmap. Spearman is used throughout — it handles ordinal, continuous, and binary variables without assuming linearity or equal intervals, and avoids having to mix correlation methods across variable types.

This gives a first pass at which variables are most associated with arrears, and also reveals multicollinearity between features (relevant for modelling).

**B. Domain-driven deep dives**

The following variables are substantively motivated as candidates for arrears risk:

`tenure_type`, `economic_status`, `household_type`, `fuel_poverty`, `food_poverty`, `universal_credit`, `disability`, `income_after_costs`, `lha_shortfall_weekly`, `monthly_rent`, `age_bracket`, `ben_cap_amount`, `overpayment_balance`, `ctc_takeup`, `whd_takeup`, `free_school_meals`, `IMD rank`, `IMD decile`

This list is a candidate pool. Based on the correlation heatmap and domain judgement, 4–5 are selected for detailed visualisation. For each selected variable:
- **Categorical/binary variables**: arrears rate and mean arrears amount per category
- **Continuous variables**: distribution of the variable in arrears vs. non-arrears households (overlapping histograms or box plots)

The heatmap serves as a check and complement — it may surface unexpected relationships or confirm that expected predictors are weaker than anticipated — but domain reasoning is not overridden by correlation results alone.

---

### 1.4 Vulnerability profiling: 3–5 groups

**What "vulnerability profiles" means here**

The task asks for 3–5 vulnerability profile groups. This is interpreted as: groups of households defined by shared characteristics (demographics, financial situation, tenure, benefits) — i.e. circumstantial profiles. This seems the most natural reading because (a) severity profiles would only apply to the 25% already in arrears and would not describe the full population, and (b) risk profiles require a predictive model, which belongs to Task 2. Arrears rate and average arrears amount are reported as properties of each group after the fact, not used to form them.

**Algorithm choice: k-prototypes**

Options considered:

- **Rule-based / pre-defined groups**: rejected. Pre-defining groups is harder than it sounds with many variables. A group defined as "employed, private tenant" is reasonable but leaves an overly broad residual group that is essentially a negation — and it is not obvious how to choose which variable combinations to pre-specify without being arbitrary.
- **K-prototypes**: handles mixed continuous + categorical data — the best practical fit for this dataset (K-means is designed for continuous data only, K-modes designed for categorical data only)
- **Hierarchical clustering**: works with custom distance matrices; gives a dendrogram to help choose number of clusters. Rejected for this dataset: computing pairwise distances for 17,831 records is computationally expensive and the resulting dendrogram would be unwieldy at this scale
- **UMAP + clustering**: UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique that compresses many features into 2–3 dimensions while preserving data structure, after which a standard clustering algorithm is applied. Advantage: the 2D representation can be visualised directly. Rejected because the reduced dimensions are not interpretable — describing clusters in terms of original features becomes indirect

**K-prototypes** is chosen as the primary approach.

**Outputs per cluster:**

1. **Summary table**: for each cluster — proportions for binary/categorical variables, means for continuous variables, arrears rate, average arrears amount. The cluster label (e.g. "older, disabled, social tenants") emerges interpretively from reading which features are most distinctive per cluster.

2. **Heatmap 1 (continuous variables)**: standardised means per cluster (subtract mean, divide by std across all households, so variables with large numeric ranges do not dominate the colour scale).

3. **Heatmap 2 (binary/categorical variables)**: proportions per cluster (naturally on a 0–1 scale). Kept separate from heatmap 1 — standardised means and proportions are not on a comparable scale and cannot share a colour axis.

---

## Task 2: Predictive Model

### 2.1 Training and evaluation paradigm

**Stratified k-fold cross-validation** (k=5 or k=10). The dataset is split into k equal folds; the model is trained on k-1 folds and evaluated on the held-out fold, repeated k times. Stratified means each fold preserves the proportion of arrears-positive households from the full dataset — important given the class imbalance.

Optionally, a final held-out test set (~20% of data, set aside before any modelling) can be used for a fully unbiased final performance estimate. Whether this is necessary depends on the degree of hyperparameter tuning involved.

### 2.2 What is being predicted

**Two models, two outcome variables:**

| Model | Outcome | Question answered |
|---|---|---|
| Classification | `arrears_flag` (binary) | Who is at risk of arrears? |
| Regression | `arrears_amount` (continuous) | How severe might the arrears be? |

The classification model outputs a probability per household — a continuous risk score suitable for ranking. The regression model adds a severity dimension.

**Training population**: both models are trained on all households (those in arrears and those not). For the regression model, non-arrears households have `arrears_amount = 0`. An alternative — training the regression model only on arrears-positive households — was considered but rejected: it would model severity conditional on already being in arrears, not predict severity for households not yet there. 

**On zero-inflated regression**: The distribution of `arrears_amount` might be zero-inflated (too many households without arrears). A formal zero-inflated model (e.g. hurdle regression, zero-inflated Poisson) explicitly models the zero-generating process separately from the non-zero values. However, gradient boosting applied to regression can handle zero-inflated outcomes reasonably well without this — it learns the pattern of zeros as part of the data. Whether a formal zero-inflated model adds meaningful performance is an empirical question. It is noted as a potential refinement if the regression model performs poorly.

**Application**: both models are applied to households currently *not* in arrears to generate risk scores and severity estimates for the intervention prioritisation.

### 2.3 Features

**Excluded:**
- `reference`: household identifier, no predictive meaning
- `lsoa21cd`: geographic identifier — geographic information is captured via the IMD rank/decile join; raw LSOA codes are not meaningful as model inputs
- `arrears_flag` and `arrears_amount`: targets; including either as a feature would cause data leakage

**Included:**
- All remaining household variables
- `IMD Rank` and `IMD Decile` joined from the deprivation dataset on `lsoa21cd`

**Engineered features:**

Only one engineered feature is added: `total_shortfall = lha_shortfall_weekly + ben_cap_amount`, capturing the total financial gap from LHA shortfall and benefit cap deduction combined.

More engineered features (ratios, interactions, polynomial terms) were considered but not added for the following reasons:
- `income_after_costs / monthly_rent`: dropped because `income_after_costs` already excludes rent, making the ratio circular
- Interaction terms: for tree-based models, interactions between features are learned automatically through sequential splits — explicitly adding them is redundant. For logistic regression, manually specifying all plausible interactions across 20 features would be arbitrary and add many columns for limited gain
- Polynomial features: similarly redundant for tree-based models and not well-motivated here

**Encoding categoricals:**
- **Logistic regression**: one-hot encoding. Arbitrary integers would imply a false linear ordering (e.g. "social tenant = 3" implies it is numerically between two other categories)
- **Tree-based models**: label encoding (arbitrary integers) is acceptable — trees split on thresholds and do not treat integer values as linearly ordered

**Note on `lsoa21cd` and IMD rank**: `lsoa21cd` is a string code (e.g. "E01003189") with no numeric meaning — it is dropped entirely. IMD rank and decile carry the geographic signal instead. IMD rank is already a meaningful number (a position in a national ranking from 1 to 33,755) with natural order and scale, so it goes directly into the model as a numeric feature — no encoding needed. After joining, Lewisham's 175 LSOAs have 175 distinct IMD rank values spread across that full national range.

**Note on `ward`**: ward is a coarser geographic unit than LSOA — Lewisham has 19 wards, each containing multiple LSOAs. It is a categorical variable with 19 categories, making one-hot encoding straightforward (19 binary columns).

**Standardisation:**
- Required for logistic regression (sensitive to feature scales)
- Not required for tree-based models (scale-invariant)

**Collinearity:**
The correlation heatmap from Task 1 flags correlated features. If two features correlate strongly (r > 0.8), consider dropping one — particularly for logistic regression. For tree-based models, collinearity is less problematic but worth knowing about for interpretability.

### 2.4 Model choice

Three models in total:

| Model | Task | Rationale |
|---|---|---|
| **Logistic regression** | Risk (classification) | Linear, interpretable baseline. Requires standardisation and one-hot encoding. Cannot learn feature interactions without explicit specification. |
| **Gradient boosting (XGBoost/LightGBM)** | Risk (classification) | Ensemble of sequential decision trees, each correcting errors of the previous. Learns non-linear relationships and interactions automatically. Typically highest performance on tabular data. Scale-invariant. |
| **Gradient boosting (XGBoost/LightGBM)** | Severity (regression) | Same algorithm as above, different loss function (MSE rather than cross-entropy), different target variable (`arrears_amount`). Output used as input to prioritisation logic — model internals not presented in depth. |

Logistic regression and gradient boosting are two entirely different model families. Logistic regression is a linear model; gradient boosting builds sequential decision trees. "Boosting" refers to the tree-based approach only.

**Why this combination**: the two risk models tell a coherent story — "here is what a simple linear model finds; here is what a more flexible model adds." The severity model is a practical tool, not a narrative centrepiece.

**On random forest**: considered as a middle ground between logistic regression and gradient boosting, but dropped. It does not add to the narrative, and gradient boosting typically outperforms it on tabular data. Random forest builds trees independently rather than sequentially, which is less efficient at correcting errors.

**Why gradient boosting tends to perform best on tabular data**: it makes no assumptions about the functional form of relationships (unlike logistic regression's linearity assumption), and the sequential error-correction process efficiently captures complex patterns including feature interactions, without those needing to be specified manually.

**Neural networks and SVMs** were not considered: overkill for this dataset size, and poor interpretability.

**Note on implementation**: the two gradient boosting models share the same feature preparation pipeline — the only difference is the target variable and loss function.

### 2.5 Evaluation

**Classification model** — given class imbalance, accuracy is not a useful metric:
- AUC-ROC
- Precision-recall curve
- F1 score

**Regression model:**
- RMSE, MAE

**Explainability — SHAP values:**
SHAP (SHapley Additive exPlanations) quantifies how much each feature contributed to a given household's predicted score, in which direction and by how much. For example: "private tenancy pushed this household's risk score up by +0.15; income_after_costs of £300 pushed it up by +0.22." This makes individual predictions transparent and actionable — not just "high risk" but "high risk because of X and Y." Averaging SHAP values across all households gives overall feature importance.

**Geographic aggregation (small-area risk):**
Aggregate household-level predicted probabilities to LSOA level (mean predicted probability, % above a threshold per LSOA). This produces a small-area risk picture for mapping without requiring a separate area-level model.

### 2.6 Prioritisation: which 10% of households to support?

Apply the classification model to all non-arrears households. Rank by predicted probability. The top 10% (~1,783 households) forms the candidate intervention list.

**On severity:**
The regression model adds a severity estimate. Whether severity is decision-relevant depends on the intervention logic — a question the model cannot answer:

- **Fixed support package** (advice, referral, one-off payment): who to support matters; severity is largely irrelevant
- **Proportional support** (more money to higher-need households): severity informs how much to give, not necessarily who gets support
- **Prevention-focused strategy** (target households with small predicted arrears): maximises impact per pound spent — prevents arrears rather than managing existing cases

These are genuine policy trade-offs without a single correct answer. They should be named explicitly as design choices rather than resolved by the model.

**Equity considerations beyond risk score:**
The model gives a risk probability, but prioritisation can incorporate explicit equity criteria:
- Presence of children: proxied by `household_type` (lone parent, couple with children) and `free_school_meals` eligibility
- Disability status: `disability`
- Depth of poverty: `fuel_poverty`, `food_poverty`

**Proposed composite priority score:**

```
priority = predicted_probability + equity_weights
```

Where equity weights are explicitly defined and auditable. The model informs but does not solely determine who receives support — the value judgements are made transparent.

---

## Future Work / Potential Improvements

- **Ensemble model**: combining logistic regression and gradient boosting predictions may improve performance. Deferred — adds complexity without clear payoff given the two risk models are already compared directly.
- **Zero-inflated regression**: if the gradient boosting severity model performs poorly, a formal zero-inflated model (e.g. hurdle regression) could be explored — explicitly modelling the zero-generating process separately from the non-zero arrears amounts.
- **Interactive widgets**: for the feature deep dives, interactive widgets would allow exploration of all candidate variables. Deferred due to development time cost for a notebook submission.

---

## Notebook Structure

1. **Descriptive overview** — distributions of key variables, arrears rate, severity distribution
2. **IMD analysis** — bar chart, scatterplot (Spearman r), two side-by-side maps, average arrears amount per decile/LSOA
3. **Feature association analysis** — Spearman correlation heatmap, per-variable deep dives
4. **Vulnerability profiling** — k-prototypes clustering, summary table, two heatmaps (continuous / categorical)
5. **Predictive models** — feature preparation, logistic regression (risk) + gradient boosting (risk) + gradient boosting (severity), evaluation, SHAP
6. **Prioritisation** — ranking non-arrears households, composite priority score, equity considerations
