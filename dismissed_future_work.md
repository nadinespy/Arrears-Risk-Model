# Dismissed / Deferred Future Work

Ideas considered for future work but not prioritised in the notebook's main future-work section. Kept here so the reasoning is not lost.

## IMD / spatial analysis (Section 2)

- **Spatial features**: the maps show spatial clustering in arrears rates that is not explained by IMD. A spatial lag feature (e.g. average arrears rate of neighbouring LSOAs) could capture whatever local factor drives this clustering. Care needed to avoid leakage from the outcome variable — e.g. compute spatial lags only from training data, or use non-outcome spatial variables (proximity to services, local labour market conditions).

## Feature association (Section 3)

- **VIF (Variance Inflation Factor)**: the Spearman heatmap reveals pairwise collinearity (e.g., `universal_credit` ↔ `age_bracket` at -0.79), but VIF would add rigour by detecting *joint* collinearity — where a predictor is explained by several others combined, even without a single strong pairwise correlation. Particularly relevant for logistic regression, where collinearity inflates standard errors.
- **Interaction effect methods**: the weak pairwise correlations may mask stronger *joint* effects — combinations of variables predicting arrears in ways no single variable does alone. Several methods could explore this, though all are sophisticated and may not work out in practice given the data constraints (binary/zero-inflated variables, mixed types):
  - **CCA (Canonical Correlation Analysis)**: finds linear combinations of variable groups that maximally correlate with outcomes, revealing structured relationships between clusters of predictors. Assumes linearity; harder to interpret with mixed variable types.
  - **Mutual information (including multivariate extension)**: captures any statistical dependence (non-linear, interaction-based, threshold effects), unlike Spearman which is limited to monotonic associations. Extends naturally to three or more variables to detect joint effects. For discrete/binary variables, MI is straightforward. For continuous variables or mixed discrete-continuous combinations, density estimation is needed (via binning, kernel density, or k-nearest-neighbour estimators), which adds noise and complexity. Multivariate results are also harder to interpret (sign can flip depending on redundancy vs synergy). Only worthwhile if findings need strong empirical backing and time is not a constraint.
- **Partial correlations**: show the unique association between each predictor and arrears after controlling for all other variables, isolating individual predictor effects beyond what pairwise Spearman can reveal. Standard partial correlations assume linear relationships, which is imperfect for binary and zero-inflated variables — a pragmatic step up from pairwise Spearman, but not a methodologically clean solution. For assumption-free unique effects, conditional mutual information would be needed, but falls into the high-effort category above.

## Vulnerability profiling (Section 4)

- **Cluster stability**: how robust are the cluster assignments? Running k-prototypes with different random seeds or subsamples and checking whether households consistently land in the same cluster would indicate whether the profiles represent genuine structure in the data or artefacts of the algorithm's initialisation. *(Partially addressed in the notebook: the k=5 solution was re-run with four different seeds and the resulting profiles compared. A fuller treatment would track individual household assignments across runs rather than comparing aggregate cluster summaries.)*

## Predictive models (Section 5)

- **Hyperparameter tuning**: XGBoost was run with default settings. Grid search or Bayesian optimisation over learning rate, max depth, and regularisation parameters could improve performance, though given the small gap between logistic regression and XGBoost, gains may be limited.
- **Ensemble model**: combining logistic regression and gradient boosting predictions may improve performance. Deferred — adds complexity without clear payoff given the two models already perform similarly.
- **Zero-inflated regression**: if a severity model is pursued (see Section 6.5), a formal zero-inflated model (e.g. hurdle regression) could be explored — explicitly modelling the zero-generating process separately from the non-zero arrears amounts.
