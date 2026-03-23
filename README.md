<div align="center">

# adsat

### Advertising Saturation Analysis Toolkit

*Saturation modelling · Budget optimisation · Benchmarking · Scenario simulation*

[![CI](https://github.com/stefanobandera1/adsat/actions/workflows/ci.yml/badge.svg)](https://github.com/stefanobandera1/adsat/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/adsat.svg)](https://pypi.org/project/adsat/)
[![PyPI downloads](https://img.shields.io/pypi/dm/adsat.svg)](https://pypi.org/project/adsat/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

`adsat` is a Python package for quantitative analysis of advertising campaign
performance.  It provides a complete, composable toolkit — from raw data
exploration through statistical modelling to budget optimisation and HTML
reporting — all behind a consistent, pandas-native API.

**Install from PyPI:**

```bash
pip install adsat
```

**Install from source:**

```bash
git clone https://github.com/stefanobandera1/adsat.git
cd adsat
pip install -e ".[dev]"
```

---

## Table of contents

- [Why adsat?](#why-adsat)
- [Quick start](#quick-start)
- [Package architecture](#package-architecture)
- [Module reference](#module-reference)
  - [exploratory](#exploratory--campaign-explorer)
  - [distribution](#distribution--distribution-analyser)
  - [transformation](#transformation--data-transformer)
  - [modeling](#modeling--saturation-modeller)
  - [evaluation](#evaluation--model-evaluator)
  - [pipeline](#pipeline--saturation-pipeline)
  - [campaign](#campaign--per-campaign-analyser)
  - [diagnostics](#diagnostics--model-diagnostics)
  - [response_curves](#response_curves--response-curve-analyser)
  - [seasonality](#seasonality--seasonal-decomposer)
  - [budget](#budget--budget-optimiser)
  - [simulation](#simulation--scenario-simulator)
  - [benchmark](#benchmark--campaign-benchmarker)
  - [report](#report--html-report-builder)
- [Data contract](#data-contract)
- [Design principles](#design-principles)
- [Dependencies](#dependencies)
- [Changelog](#changelog)

---

## Why adsat?

Every advertising analytics project asks the same four questions.
`adsat` answers each one with a dedicated, well-tested module:

| Question | Module |
|---|---|
| At what spend level does each campaign saturate? | `campaign` · `pipeline` · `modeling` |
| How should I redistribute the budget to maximise outcome? | `budget` |
| Is this campaign performing unusually well or poorly right now? | `benchmark` |
| What would outcome look like under scenario X? | `simulation` |

Supporting modules handle the analytical groundwork — exploratory charts,
distribution fitting, data transformation, residual diagnostics, seasonality
decomposition, response curve analysis, and polished HTML reporting — so the
core modules always receive clean, well-understood inputs.

---

## Quick start

### Five-line end-to-end workflow

```python
import pandas as pd
from adsat.campaign import CampaignSaturationAnalyzer
from adsat.budget   import optimise_budget

df    = pd.read_csv("campaigns_weekly.csv")

# Step 1 — fit a saturation curve for every campaign
batch  = CampaignSaturationAnalyzer(
             campaign_col = "campaign_id",
             x_col        = "impressions",
             y_col        = "conversions",
         ).run(df)
batch.print_summary()

# Step 2 — find the spend allocation that maximises total conversions
result = optimise_budget(batch, total_budget=5_000_000)
result.print_summary()
result.plot()
```

### Benchmark current performance in one call

```python
from adsat.benchmark import benchmark_campaigns

result = benchmark_campaigns(
    df,
    metric_col           = "conversion_rate",
    metric_type          = "proportion",
    date_col             = "week_start",
    volume_col           = "impressions",
    bin_col              = "spend",
    segment_cols         = ["country", "channel"],
    current_period_start = "2024-10-01",
)
result.print_summary()
result.plot()
```

### Build a shareable HTML report

```python
from adsat import ReportBuilder

builder = ReportBuilder(title="Q3 Campaign Report")
builder.add_campaign_batch(batch)
builder.add_budget_allocation(alloc)
builder.add_simulation(sim_result)
builder.save("q3_report.html")
```

---

## Package architecture

```
adsat/
│
├── exploratory.py      Visual exploration of raw campaign data
├── distribution.py     Statistical distribution fitting per column
├── transformation.py   Reversible data transformations (log, sqrt, Yeo-Johnson…)
│
├── modeling.py         Saturation curve models (Hill, NegExp, Power, Logistic, MM)
├── evaluation.py       Model selection and ranking by AIC / R² / MAPE / composite
├── pipeline.py         Orchestrate transform → fit → evaluate in one call
│
├── campaign.py         Run the pipeline for every campaign in a DataFrame
├── diagnostics.py      Residual diagnostics (normality, autocorrelation, Cook's D)
├── response_curves.py  Marginal returns, ROI, and efficiency zone charting
├── seasonality.py      Seasonal decomposition and adjustment
│
├── budget.py           Constrained budget optimisation across campaigns
├── simulation.py       Compare hypothetical spend scenarios
├── benchmark.py        Classify performance vs statistical baseline (M1/M2/M3/P1)
│
└── report.py           Assemble a self-contained HTML report
```

The modules are **independently usable and composable**.  You can use only
`modeling` and `evaluation` if you have a single campaign, or chain all
fourteen modules into a fully automated pipeline.

---

## Module reference

---

### `exploratory` — Campaign Explorer

Visual and statistical exploration of raw campaign data before any modelling.
Produces a suite of publication-quality plots and a descriptive statistics table.

**Main class:** `CampaignExplorer`

```python
from adsat import CampaignExplorer

explorer = CampaignExplorer(
    x_col     = "impressions",
    y_col     = "conversions",
    date_col  = "week_start",   # optional; enables time-series plots
    group_col = "campaign_id",  # optional; colour-codes plots per campaign
)

# All plots in one call
explorer.explore(df)

# Or individual plots
explorer.plot_descriptive_summary(df)   # mean / median / IQR table + distribution strip
explorer.plot_histograms(df)            # per-column histograms with KDE overlay
explorer.plot_qq(df)                    # Q-Q plots for normality assessment
explorer.plot_ecdf(df)                  # empirical cumulative distribution functions
explorer.plot_correlation(df)           # Pearson / Spearman heatmap
explorer.plot_scatter(df)               # x vs y with regression line and confidence band
explorer.plot_time_series(df)           # metric over time, one sub-panel per campaign
explorer.plot_outliers(df)              # IQR and z-score outlier flags
explorer.plot_distribution_fits(df)     # empirical histogram vs best-fit theoretical PDF
```

**Convenience function:**

```python
from adsat import explore
explore(df, x_col="impressions", y_col="conversions")
```

---

### `distribution` — Distribution Analyser

Fit a library of statistical distributions to each column of a DataFrame and
recommend a pre-processing transformation.  Designed to feed `DataTransformer`
automatically.

**Main class:** `DistributionAnalyzer`

```python
from adsat import DistributionAnalyzer

analyzer = DistributionAnalyzer(
    alpha   = 0.05,   # significance level for Shapiro-Wilk normality test
    verbose = True,
)

reports = analyzer.analyze(df, columns=["impressions", "conversions", "spend"])

# Inspect one column
r = reports["impressions"]
print(r.best_fit.name)              # e.g. "lognorm"
print(r.best_fit.aic)               # Akaike Information Criterion
print(r.recommended_transform)      # "log", "sqrt", "yeo_johnson", "none", …
print(r.is_normal)                  # bool — Shapiro-Wilk at alpha
print(r.skewness, r.kurtosis)

# Tabular comparison across all columns
print(analyzer.summary_table(reports))

# Visual — histogram + fitted PDF for every column
analyzer.plot_distributions(reports)
```

**`DistributionFitResult` attributes**

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Distribution name (scipy convention) |
| `params` | `dict` | Fitted shape, location, and scale parameters |
| `aic` | `float` | Akaike Information Criterion (lower = better fit) |
| `ks_pvalue` | `float` | Kolmogorov-Smirnov goodness-of-fit p-value |
| `is_acceptable` | `bool` | KS test passes at alpha |

---

### `transformation` — Data Transformer

Apply and track reversible transformations to DataFrame columns, with full
support for inverting them — critical for returning predictions to the original
scale.

**Main class:** `DataTransformer`

```python
from adsat import DataTransformer

# strategy="auto" reads recommendations from DistributionAnalyzer
transformer = DataTransformer(strategy="auto")

df_t = transformer.fit_transform(
    df,
    columns              = ["impressions", "conversions"],
    distribution_reports = reports,     # from DistributionAnalyzer.analyze()
)

# Apply the same fitted transforms to a held-out set
df_test_t = transformer.transform(df_test, columns=["impressions", "conversions"])

# Reverse — brings predictions back to original units
df_orig = transformer.inverse_transform(df_t, columns=["impressions", "conversions"])

# Audit what was applied
print(transformer.get_transform_summary())
```

**Supported strategies**

| `strategy` | Behaviour |
|---|---|
| `"auto"` | Use `DistributionAnalyzer` recommendations per column |
| `"log"` | `log(x + ε)` applied to all columns |
| `"log1p"` | `log(1 + x)` |
| `"sqrt"` | `sqrt(x + ε)` |
| `"yeo_johnson"` | Yeo-Johnson power transform |
| `"box_cox"` | Box-Cox (requires positive values) |
| `"standard_scaler"` | Zero-mean unit-variance standardisation |
| `"none"` | Pass through unchanged |
| `dict` | Per-column override, e.g. `{"impressions": "log", "revenue": "none"}` |

---

### `modeling` — Saturation Modeller

Fit nonlinear saturation curves to impression/spend vs outcome data and locate
the saturation point — the input level beyond which additional spend yields
diminishing returns.

**Main class:** `SaturationModeler`

```python
from adsat import SaturationModeler
import numpy as np

modeler = SaturationModeler(
    models               = ["hill", "negative_exponential", "michaelis_menten"],
    saturation_threshold = 0.90,         # saturation = 90% of asymptote
    use_bayesian_hill    = False,         # set True if pymc is installed
)

results = modeler.fit(df, x_col="impressions", y_col="conversions")

# Inspect the Hill fit
r = results["hill"]
print(r.r2, r.aic, r.saturation_point)
print(r.params)          # {"a_max": 48200, "k_half": 312000, "n_hill": 1.8}

# Predict at new spend values
y_hat = modeler.predict("hill", np.linspace(0, 1_000_000, 500))

# Compare all models
print(modeler.summary_table())
```

**Supported models**

| Name | Equation | Key property |
|---|---|---|
| `hill` | `a·xⁿ / (kⁿ + xⁿ)` | Flexible sigmoidal; Bayesian variant available |
| `negative_exponential` | `a·(1 − e^{−bx})` | Classic diminishing-returns shape |
| `michaelis_menten` | `Vmax·x / (Km + x)` | Hill with n = 1 |
| `logistic` | `L / (1 + e^{−k(x−x₀)})` | S-curve with explicit inflection |
| `power` | `a·xᵇ` | No asymptote; captures power-law growth |

**Bayesian Hill** (requires `pip install pymc`):

```python
modeler = SaturationModeler(use_bayesian_hill=True)
results = modeler.fit(df, x_col="impressions", y_col="conversions")

r = results["hill_bayesian"]
print(r.credible_intervals)   # {"a_max": (41000, 56000), "k_half": (290000, 335000), …}
```

**`ModelFitResult` attributes**

| Attribute | Type | Description |
|---|---|---|
| `params` | `dict` | Fitted parameter values |
| `r2` / `rmse` / `aic` / `bic` | `float` | Goodness-of-fit metrics |
| `saturation_point` | `float` | x at which y ≥ `saturation_threshold × asymptote` |
| `saturation_y` | `float` | Predicted outcome at the saturation point |
| `converged` | `bool` | Whether the solver converged |

---

### `evaluation` — Model Evaluator

Rank competing models from `SaturationModeler` and select the best one according
to a configurable criterion.

**Main class:** `ModelEvaluator`

```python
from adsat import ModelEvaluator

evaluator = ModelEvaluator(
    primary_metric      = "aic",    # "aic", "bic", "r2", "rmse", "mae", "mape", "composite"
    require_convergence = True,     # exclude models that did not converge
    min_r2              = 0.50,     # exclude models below this R²
)

report = evaluator.evaluate(results)    # results from SaturationModeler.fit()

print(report.best_model)                # "hill"
print(report.saturation_point)          # 820_000
print(report.ranked_models)             # DataFrame sorted best → worst

evaluator.plot_model_comparison(results)
evaluator.print_report(results)
```

---

### `pipeline` — Saturation Pipeline

Orchestrate the full `transform → fit → evaluate` workflow in a single object.
The recommended entry point when you have a single campaign's data.

**Main class:** `SaturationPipeline`

```python
from adsat import SaturationPipeline
import numpy as np

pipeline = SaturationPipeline(
    x_col                = "impressions",
    y_col                = "conversions",
    models               = ["hill", "negative_exponential", "power"],
    transform_strategy   = "auto",
    saturation_threshold = 0.90,
    primary_metric       = "aic",
)

result = pipeline.run(df)

print(result.best_model)           # "hill"
print(result.saturation_point)     # x at 90% of asymptote
result.print_summary()
pipeline.plot(result)

# Predict in original (untransformed) scale
y_hat = pipeline.predict(result, np.linspace(0, 1_000_000, 200))
```

**`PipelineResult` attributes:** `best_model`, `saturation_point`, `saturation_y`,
`model_results` (dict of `ModelFitResult`), `transformer`, `evaluation_report`.

---

### `campaign` — Per-Campaign Analyser

Run the full pipeline independently for every campaign in a multi-campaign
DataFrame.  The primary entry point for production advertising data.

**Main class:** `CampaignSaturationAnalyzer`

```python
from adsat import CampaignSaturationAnalyzer

analyzer = CampaignSaturationAnalyzer(
    campaign_col         = "campaign_id",
    x_col                = "impressions",
    y_col                = "conversions",
    min_observations     = 12,           # skip campaigns with fewer data points
    saturation_threshold = 0.90,
    primary_metric       = "aic",
    verbose              = True,
)

batch = analyzer.run(df)

# Summary and plots
batch.print_summary()
batch.plot_all()                    # grid of fitted saturation curves
batch.plot_saturation_comparison()  # horizontal bars: saturation point per campaign
batch.plot_status_breakdown()       # donut: below / approaching / at / beyond

# Per-campaign access
cr = batch.get("campaign_42")
print(cr.saturation_point, cr.saturation_status, cr.r2, cr.best_model)

# Filter by saturation status
approaching = batch.campaigns_by_status("approaching")
failed_ids  = batch.failed_campaigns()
```

**Saturation statuses**

| Status | Meaning |
|---|---|
| `"below"` | Current median x < 50% of saturation point — plenty of headroom |
| `"approaching"` | 50–80% — entering the diminishing-returns zone |
| `"at"` | 80–110% — effectively at saturation |
| `"beyond"` | > 110% — spend is past the saturation point |

**Convenience function:**

```python
from adsat.campaign import predict_saturation_per_campaign

summary_df = predict_saturation_per_campaign(
    df, "campaign_id", "impressions", "conversions"
)
```

---

### `diagnostics` — Model Diagnostics

Validate that a fitted model's residuals satisfy the statistical assumptions
underpinning the saturation curves.  Detects non-normality, autocorrelation,
heteroscedasticity, and high-influence observations.

**Main class:** `ModelDiagnostics`

```python
from adsat import ModelDiagnostics

diag   = ModelDiagnostics(alpha=0.05)
report = diag.run(results["hill"])   # results from SaturationModeler.fit()

# Structured pass/fail output
report.print_summary()
# [OK]  Shapiro-Wilk                p = 0.2341
# [!!]  Durbin-Watson               DW = 1.12  (ideal ~2.0)
# ...

# Diagnostic plots
diag.plot(report)              # 4-panel: residuals vs fitted, Q-Q, scale-location, Cook's D
diag.plot_comparison(reports)  # heatmap comparing all models side-by-side

# Run across all models at once
reports = diag.run_all(results)
print(diag.summary_table(reports))
```

**Tests performed**

| Test | What it checks | Concern threshold |
|---|---|---|
| Shapiro-Wilk | Normality of residuals | p < 0.05 |
| Kolmogorov-Smirnov | Normality (for larger samples) | p < 0.05 |
| Jarque-Bera | Skewness + kurtosis jointly | p < 0.05 |
| Durbin-Watson | Serial autocorrelation | DW < 1 or DW > 3 |
| Levene | Homoscedasticity (equal variance) | p < 0.05 |
| Cook's Distance | High-influence observations | D > 4/n |

**Convenience function:**

```python
from adsat.diagnostics import run_diagnostics
report = run_diagnostics(results["hill"])
```

---

### `response_curves` — Response Curve Analyser

Compute and visualise the shape of each campaign's saturation curve in detail:
marginal returns, ROI, efficiency zones, and inflection points.

**Main class:** `ResponseCurveAnalyzer`

```python
from adsat import ResponseCurveAnalyzer

analyzer = ResponseCurveAnalyzer(
    n_points              = 500,          # spend grid resolution
    x_max_multiplier      = 1.5,          # grid extends to 1.5× saturation point
    efficiency_thresholds = (20.0, 80.0), # percentiles defining high/low efficiency
)

curves = analyzer.analyse(batch)   # batch from CampaignSaturationAnalyzer.run()

analyzer.plot_curves(curves)                  # overlaid outcome curves
analyzer.plot_marginal_returns(curves)        # d(outcome)/d(spend) per campaign
analyzer.plot_roi_curves(curves)              # outcome / spend ratio
analyzer.plot_efficiency_comparison(curves)   # which campaigns get most per £

print(analyzer.summary_table(curves))
```

**`ResponseCurveResult` attributes:**
`spend_grid`, `outcome_curve`, `marginal_return_curve`, `roi_curve`,
`inflection_point`, `asymptote`, `efficiency_zone_low`, `efficiency_zone_high`.

**Convenience function:**

```python
from adsat.response_curves import analyse_response_curves
curves = analyse_response_curves(batch)
```

---

### `seasonality` — Seasonal Decomposer

Decompose a time series into trend, seasonal, and residual components using
classical (CMA-based) decomposition, then adjust campaign metrics to remove
seasonal confounding before modelling.

**Main class:** `SeasonalDecomposer`

```python
from adsat import SeasonalDecomposer

decomposer = SeasonalDecomposer(
    period = "auto",       # auto-detect via autocorrelation; or pass an integer e.g. 52
    model  = "additive",   # or "multiplicative"
)

# Fit on a metric Series
result = decomposer.fit(df["conversions"])
result.print_summary()
# Model:    additive
# Period:   52 weeks
# Strength: 0.73 (moderate)

# Diagnostic plots
decomposer.plot(result)                      # trend / seasonal / residual panels
decomposer.plot_seasonal_factors(result)     # bar chart of seasonal indices by period
decomposer.plot_adjusted_vs_original(result) # raw vs seasonally-adjusted overlay

# Remove seasonality from the full DataFrame before modelling
df_adj = decomposer.fit_transform(df, metric_col="conversions")
# df_adj["conversions_adj"] — use this column for saturation modelling

# Restore seasonality to predictions (e.g. after forecasting on adjusted data)
preds_with_season = decomposer.inverse_adjust(model_predictions, result)
```

**`SeasonalDecomposition` attributes**

| Attribute | Type | Description |
|---|---|---|
| `trend` | `pd.Series` | Smoothed trend component |
| `seasonal` | `pd.Series` | Repeating seasonal component |
| `residual` | `pd.Series` | What remains after removing trend + seasonal |
| `adjusted` | `pd.Series` | Original series with seasonal component removed |
| `seasonal_factors` | `np.ndarray` | One factor per period (52 values for weekly) |
| `strength_of_seasonality` | `float` | 0–1; proportion of variance explained by seasonality |
| `dominant_period` | `int` | Most prominent seasonal period detected |

**Convenience function:**

```python
from adsat.seasonality import adjust_for_seasonality
df_adj = adjust_for_seasonality(df, metric_col="conversions")
```

---

### `budget` — Budget Optimiser

Find the spend allocation across campaigns that maximises total predicted outcome
subject to a total budget constraint and optional per-campaign spend floors and caps.

**The optimisation problem**

```
maximise   Σ f_i(x_i)
subject to Σ x_i  = total_budget
           lb_i   ≤ x_i ≤ ub_i    for every campaign i
```

where `f_i` is campaign `i`'s fitted saturation curve.

**Main class:** `BudgetOptimizer`

```python
from adsat import BudgetOptimizer

opt = BudgetOptimizer(
    total_budget = 5_000_000,
    min_spend    = 50_000,                    # floor on every campaign
    max_spend    = {"Brand_UK": 2_000_000},   # per-campaign cap (dict or scalar)
    n_restarts   = 10,                        # random starts to avoid local optima
    random_seed  = 42,
)

result = opt.optimise(batch)   # batch from CampaignSaturationAnalyzer.run()

result.print_summary()
# Total budget        :   5,000,000
# Current outcome     :      82,400
# Optimal outcome     :      96,100
# Outcome lift        :      13,700  (+16.6%)

result.plot(response_fns=opt.response_fns)  # 4-panel: spend, lift, saturation, curves

# Full allocation table
print(result.allocations)
# campaign_id | current_spend | optimal_spend | spend_change_pct | outcome_lift_pct

# Marginal return at every spend level — useful before optimising
print(opt.marginal_returns_table(batch))
```

**`BudgetAllocation` attributes**

| Attribute | Description |
|---|---|
| `allocations` | DataFrame — one row per campaign with spend and outcome columns |
| `total_outcome_lift` | Absolute outcome gain over current allocation |
| `total_outcome_lift_pct` | Percentage gain |
| `converged` | Whether the solver satisfied the budget constraint |

**Convenience function:**

```python
from adsat.budget import optimise_budget
result = optimise_budget(batch, total_budget=5_000_000)
```

---

### `simulation` — Scenario Simulator

Compare hypothetical spend allocations side by side, without touching real data.
Ideal for "what if" planning: what happens to total conversions if we shift budget
from one campaign to another?

**Main class:** `ScenarioSimulator`

```python
from adsat import ScenarioSimulator

sim = ScenarioSimulator(batch)   # batch from CampaignSaturationAnalyzer.run()

# Define named scenarios
sim.add_scenario(
    "Current",
    spends = {"Alpha": 2_000_000, "Beta": 800_000, "Gamma": 600_000},
)
sim.add_scenario(
    "+20% Alpha",
    spends = {"Alpha": 2_400_000, "Beta": 800_000, "Gamma": 600_000},
)
sim.add_scenario(
    "Rebalanced",
    spends = {"Alpha": 1_500_000, "Beta": 1_500_000, "Gamma": 800_000},
)

# Sweep a spend range for one campaign automatically
sim.add_budget_sweep(
    campaign_id  = "Alpha",
    spend_range  = (500_000, 4_000_000),
    n_steps      = 20,
)

result = sim.run()
result.print_summary()

sim.plot(result)              # grouped bars: total outcome per scenario
sim.plot_spend_sweep(result)  # outcome vs spend for swept campaigns

# Which campaigns are most sensitive to budget changes?
print(sim.sensitivity_table(result))
```

**Convenience function:**

```python
from adsat.simulation import simulate
result = simulate(batch, scenarios=[...])
```

---

### `benchmark` — Campaign Benchmarker

Classify each campaign observation as **Above / Within / Below** a statistical
baseline using four complementary analytical methods, with automatic change-point
detection and cross-segment comparison.

This module is the most analytically rich in the package.  The four methods are
designed to answer different questions and complement each other:

| Method | Central question | When to trust it |
|---|---|---|
| **M1** Trend + SE bands | Is this week unusual given the campaign's own history and trend? | Regular time series with enough history (≥ `min_history_rows`) |
| **M2** Peer-bin quantiles | Is this week unusual compared to similar-scale campaigns? | When spend/volume levels differ significantly across campaigns |
| **M3** Adaptive selector | Which of M1 or M2 is more appropriate for this specific observation? | Always — M3 is the recommended primary classification |
| **P1** Cross-segment z-score | Which campaigns are genuinely outperforming their scale peers? | When comparing campaigns across segments (country, channel, …) |

**Main class:** `CampaignBenchmarker`

```python
from adsat.benchmark import CampaignBenchmarker

bm = CampaignBenchmarker(
    # ── Metric ───────────────────────────────────────────────────────────────
    metric_col  = "conversion_rate",
    metric_type = "proportion",          # "proportion" or "continuous"

    # ── Time structure ────────────────────────────────────────────────────────
    date_col    = "week_start",
    volume_col  = "impressions",         # denominator for quasi-binomial SE

    # ── Scale binning (enables M2, M3, P1) ───────────────────────────────────
    bin_col     = "spend",               # variable for quartile peer-binning
    n_bins      = 4,                     # L / M / H / VH (default quartiles)

    # ── Segmentation ─────────────────────────────────────────────────────────
    segment_cols = ["country", "channel"],

    # ── Confounder adjustment for M1 ─────────────────────────────────────────
    confounder_cols = ["market_cpi"],    # rows with NaN confounders fall back to
                                        # time-only trend for that row

    # ── Historical / current split ────────────────────────────────────────────
    current_period_start = "2024-10-01",

    # ── Change-point detection ────────────────────────────────────────────────
    refit_after_changepoint = True,      # auto-refit M1 after a detected break
    cusum_h                 = 4.0,       # CUSUM threshold in σ units (higher = less sensitive)
    pettitt_alpha           = 0.05,

    # ── Bootstrap CI around M2 peer thresholds ────────────────────────────────
    bootstrap_m2 = True,                 # opt-in — slower but shows band uncertainty
    n_bootstrap  = 500,
)

result = bm.fit(df)

result.print_summary()   # warnings · change-point alerts · z-score highlights · compact table
result.plot()            # time series / distribution / M3 heatmap

# Access outputs
result.enriched_df           # original df + all classification columns
result.summary_compact       # one row per segment × method with Above/Within/Below counts
result.summary_detail        # one row per current-period observation with explanation
result.changepoint_summary   # one row per segment: CUSUM/Pettitt indices, agreed flag
```

**Output columns added to `enriched_df`**

```
M1:  m1_baseline, m1_lower90, m1_upper90, m1_lower95, m1_upper95, m1_class
M2:  m2_p10, m2_p50, m2_p90, m2_class
     m2_p10_ci_low, m2_p10_ci_high, m2_p90_ci_low, m2_p90_ci_high  (bootstrap only)
M3:  m3_lower, m3_upper, m3_rule, m3_class
P1:  cross_seg_zscore, cross_seg_class
CP:  cp_cusum_idx, cp_pettitt_idx, cp_refit_from
     traffic_bin, bin_idx, fallback_level
```

**Change-point detection**

Both CUSUM and Pettitt tests run automatically on each segment's historical metric
series.  When both agree and `refit_after_changepoint=True`, M1 is automatically
refitted using only post-break data so the baseline reflects the current regime.

```python
from adsat.benchmark import detect_changepoints

cp = detect_changepoints(series)
# {
#   "cusum_index":            45,
#   "pettitt_index":          47,
#   "agreed":                 True,
#   "recommended_refit_from": 47
# }
```

| Test | What it detects |
|---|---|
| CUSUM | Sustained directional drift in the mean — catches slow structural shifts |
| Pettitt | Single most likely abrupt change-point — catches sudden step-changes |

**Undo an automatic log-transform**

When `bin_col` is highly skewed, `adsat` auto-log-transforms it and warns you.
To revert:

```python
result2 = bm.undo_log_transform("spend", result)
```

**Convenience function:**

```python
from adsat.benchmark import benchmark_campaigns

result = benchmark_campaigns(
    df,
    metric_col           = "conversion_rate",
    metric_type          = "proportion",
    date_col             = "week_start",
    volume_col           = "impressions",
    bin_col              = "spend",
    segment_cols         = ["country"],
    confounder_cols      = ["market_cpi"],
    current_period_start = "2024-10-01",
    bootstrap_m2         = True,
)
```

---

### `report` — HTML Report Builder

Assemble a self-contained, styled HTML report from any combination of `adsat`
result objects.  All figures are embedded as base64 so the output is a single
portable file.

**Main class:** `ReportBuilder`

```python
from adsat import ReportBuilder

builder = ReportBuilder(
    title    = "Q3 Campaign Analysis",
    subtitle = "Paid Search & Display — EMEA",
)

# Add any combination of adsat outputs
builder.add_campaign_batch(batch)         # saturation analysis section
builder.add_budget_allocation(alloc)      # budget optimisation section
builder.add_response_curves(curves)       # response curve section
builder.add_diagnostics(diag_reports)     # model diagnostics section
builder.add_seasonality(decomp_result)    # seasonal decomposition section
builder.add_simulation(sim_result)        # scenario simulation section

# Custom content
builder.add_custom_section("Analyst Notes", "<p>Reviewed by Analytics team.</p>")
builder.add_figure("extra_chart.png", caption="Custom deep-dive chart")

# Output
builder.save("q3_report.html")
html = builder.get_html()   # return as string instead
```

**Convenience function:**

```python
from adsat import generate_report

generate_report(
    batch       = batch,
    allocation  = alloc,
    curves      = curves,
    output_path = "report.html",
    title       = "Campaign Report",
)
```

---

## Data contract

Every `adsat` module accepts a standard `pd.DataFrame`.  There are **no
hard-coded column names** — all column identifiers are passed as parameters.

**Typical minimum schema**

| Column | Role | Example |
|---|---|---|
| `campaign_id` | Campaign identifier | `"UK_Brand_Search"` |
| `impressions` or `spend` | Input variable (x) | `245_000` |
| `conversions` or `revenue` | Output variable (y) | `1_842` |
| `week_start` *(optional)* | Date for trend + seasonality | `"2024-01-08"` |

**One row = one observation period** (e.g. one week) for one campaign.

The `campaign` module handles multi-campaign DataFrames where each campaign
occupies multiple rows.  All other modules operate on a single campaign slice.

---

## Design principles

**Composable, not monolithic.**
Each module solves exactly one well-defined problem and returns a typed result
object.  Modules can be used in isolation or chained in any order.  There is no
global state and no hidden coupling between modules.

**Sensible defaults, full control.**
Every parameter has a documented default that produces correct results on typical
advertising data.  Nothing is hidden behind magic behaviour.

**Fail loudly on bad input.**
Required columns are validated before any computation.  Missing columns, invalid
metric types, and unsupported model names all raise `ValueError` with clear,
actionable messages — not silent failures or empty results.

**One-liners for every workflow.**
Every major class ships a corresponding convenience function
(`predict_saturation_per_campaign`, `optimise_budget`, `benchmark_campaigns`,
`run_diagnostics`, `adjust_for_seasonality`, …) for users who want results
without configuring a class instance.

**Reproducible by default.**
Every module with a random component (`bootstrap_m2`, `n_restarts`, Bayesian
sampling) accepts a `random_seed` parameter and defaults to `42`.

**Pandas-native outputs.**
All tabular results are plain `pd.DataFrame` objects.  There are no proprietary
table classes to learn — filter, sort, merge, and export with standard pandas
idioms.

---

## Dependencies

| Package | Used for |
|---|---|
| `numpy` | Numerical core — arrays, linear algebra, random sampling |
| `pandas` | DataFrames and time-series utilities throughout |
| `scipy` | Curve fitting, statistical tests, distribution fitting |
| `scikit-learn` | Evaluation metrics (R², RMSE, MAE) and scalers |
| `matplotlib` | All visualisation |
| `pymc` *(optional)* | Bayesian Hill model only — `use_bayesian_hill=True` |
| `arviz` *(optional)* | Required alongside `pymc` for posterior diagnostics |

```bash
# Core install
pip install numpy pandas scipy scikit-learn matplotlib

# Optional: Bayesian Hill model
pip install pymc arviz
```

---

## Changelog

| Version | Highlights |
|---|---|
| 0.1.0 | Core pipeline: `exploratory` → `distribution` → `transformation` → `modeling` → `evaluation` → `pipeline` |
| 0.2.0 | `campaign` (per-campaign batch analysis) · `budget` optimiser |
| 0.3.0 | `response_curves` · `diagnostics` · `seasonality` decomposer |
| 0.4.0 | `report` HTML builder · `simulation` scenario comparator |
| 0.5.0 | `benchmark`: M1/M2/M3/P1 classification · CUSUM + Pettitt change-point detection · confounder-adjusted baselines · bootstrap CI for peer thresholds · cross-segment z-score |

---

<div align="center">
<sub>Built with ♥ for advertising analytics teams.</sub>
</div>

---

### `attribution` — Multi-Touch Attribution

Model how credit for conversions and revenue should be distributed across
advertising channels, based on user-level journey data.  Supports nine models
from simple rule-based approaches to advanced game-theoretic and data-driven
methods, with a rigorous evaluation framework and direct budget recommendation
output.

---

#### Input data

Unlike the saturation modules which work with aggregate campaign data,
attribution operates on **user-level touchpoint events** — one row per ad
interaction.

**Minimum required columns**

| Column | Type | Description |
|---|---|---|
| `user_id` | str / int | Unique user or cookie identifier |
| `timestamp` | datetime | Time of the touchpoint — used for ordering and time-decay |
| `channel` | str | Channel name (e.g. `"paid_search"`, `"email"`) |
| `interaction_type` | str | `"click"` or `"impression"` |
| `converted` | int (0/1) | Whether this journey ended in a conversion |
| `revenue` | float | Revenue value (0 on non-converting rows) |

**Optional columns** (enable richer analysis)

| Column | Enables |
|---|---|
| `cost` | ROI-weighted budget allocation |
| `session_id` | Session-level journey segmentation |
| `device` | Device-level breakdown |
| `campaign` | Sub-channel / campaign label |

**Generate a realistic test dataset instantly:**

```python
from adsat.attribution import make_sample_events

events = make_sample_events(
    n_users      = 2_000,
    channels     = ["paid_search", "display", "social_paid",
                    "email", "organic_search", "direct"],
    conv_rate    = 0.20,
    avg_revenue  = 85.0,
    random_seed  = 42,
)
# Returns a DataFrame with all required + optional columns
```

---

#### Step 1 — Build journeys from raw events

`JourneyBuilder` converts the raw event log into structured user journeys
(one row per journey) ready for attribution modelling.

```python
from adsat.attribution import JourneyBuilder

builder = JourneyBuilder(
    user_col           = "user_id",
    time_col           = "timestamp",
    channel_col        = "channel",
    interaction_col    = "interaction_type",
    converted_col      = "converted",
    revenue_col        = "revenue",
    cost_col           = "cost",              # optional — enables ROI allocation

    # Journey window: how far back to look before a conversion
    lookback_days      = 30,                  # int, or None for auto-detection

    # Multi-conversion users
    multi_conversion   = "reset",             # "reset" (default) or "rolling"

    # Impression vs click weighting
    interaction_weight = {"click": 1.0, "impression": 0.3},
)

journeys = builder.build(events)
```

**`multi_conversion` options**

| Value | Behaviour |
|---|---|
| `"reset"` | Each conversion starts a fresh journey — prior touchpoints are cut off |
| `"rolling"` | All touchpoints within the lookback window count toward every conversion |

**`lookback_days` options**

| Value | Behaviour |
|---|---|
| Integer (e.g. `30`) | Fixed window — only touchpoints within N days before conversion count |
| `None` | Auto-detect: uses 3× the median inter-event gap per user, clamped 7–90 days |

---

#### Step 2 — Fit attribution models

**Main class:** `AttributionAnalyzer`

```python
from adsat.attribution import AttributionAnalyzer

analyzer = AttributionAnalyzer(
    models               = ["last_click", "shapley", "markov", "ensemble"],
    markov_order         = 1,          # 1 = standard; 2+ = remembers longer context
    time_decay_half_life = 7.0,        # days — credit halves every N days
    position_weights     = {           # override U-shaped defaults
        "first": 0.40, "last": 0.40, "middle": 0.20
    },
    cost_col             = "cost",     # enables ROI column in output
    random_seed          = 42,
)

result = analyzer.fit(journeys)
result.print_summary()
result.plot()
```

**Supported models**

| Model | Type | Key assumption / approach |
|---|---|---|
| `last_click` | Rule | 100% credit to the final touchpoint — industry legacy default |
| `first_click` | Rule | 100% credit to the first touchpoint — upper-funnel bias |
| `linear` | Rule | Equal credit to all touchpoints — neutral baseline |
| `position_based` | Rule | U-shaped: 40% first, 40% last, 20% middle (configurable) |
| `time_decay` | Rule | Exponential decay — recent touchpoints get exponentially more credit |
| `shapley` | Advanced | Game-theoretic marginal contribution — exact for ≤ 12 channels, Monte Carlo above |
| `markov` | Advanced | Removal effect via configurable-order Markov transition matrix |
| `data_driven` | Advanced | Logistic regression + SHAP values — mirrors Google Analytics 4 |
| `ensemble` | Meta | Weighted average across all fitted models — reduces single-model risk |

**Shapley auto-switching**

```python
# With 6 channels: exact Shapley (2^6 = 64 coalitions — fast)
# With 15 channels: auto-switches to Monte Carlo (5 000 permutations)
# Override thresholds:
analyzer = AttributionAnalyzer(
    models               = ["shapley"],
    shapley_exact_limit  = 10,       # switch to MC above this channel count
    shapley_n_iterations = 10_000,   # MC permutation count
)
```

**Markov chain order**

```python
# Order 1 (default): P(next channel | current channel)
# Order 2: P(next channel | current channel, previous channel)
# Higher order = more path context, requires more data
analyzer = AttributionAnalyzer(models=["markov"], markov_order=2)
```

---

#### Step 3 — Inspect results

```python
# Full text summary across all models
result.print_summary()

# Get credits for one model
shapley_df = result.get_credits("shapley")
# channel | attributed_conversions | credit_share | attributed_revenue | roi

# Best model according to composite evaluation score
print(result.best_model())    # e.g. "markov"

# Full model comparison table
print(result.model_comparison)
# model | normalisation_error | path_coverage | conversion_alignment
#       | stability_score | cross_model_agreement | composite_score | rank
```

**`AttributionResult` attributes**

| Attribute | Type | Description |
|---|---|---|
| `channel_credits` | `pd.DataFrame` | One row per channel × model — all credit metrics |
| `journey_credits` | `pd.DataFrame` | Per-journey channel breakdown (for path analysis) |
| `model_comparison` | `pd.DataFrame` | Ranked model evaluation table |
| `channels` | `list` | All channel names in the data |
| `total_conversions` | `int` | Actual conversion count |
| `total_revenue` | `float` | Actual total revenue |
| `models_fitted` | `list` | Models that were successfully fitted |

---

#### Step 4 — Visualise

```python
from adsat.attribution import plot_attribution

plot_attribution(
    result    = result,
    model     = "shapley",     # model to highlight in single-model panels
    save_path = "attribution.png",
)
```

**Six-panel figure:**

| Panel | Content |
|---|---|
| Top-left | Grouped bar: credit share per channel, one bar per model |
| Top-right | Stacked bar: how the highlighted model distributes credit |
| Mid-left | Stacked horizontal bar: attributed revenue per model by channel |
| Mid-right | Path length histogram for converting journeys |
| Bottom-left | Top 10 most common converting channel sequences |
| Bottom-right | Radar chart: model evaluation across all five dimensions |

---

#### Step 5 — Evaluate models

The `AttributionEvaluator` scores each model across five orthogonal dimensions
and produces a composite rank.  It runs automatically inside `analyzer.fit()`
but can also be called standalone.

**Evaluation dimensions**

| Metric | What it measures | Weight |
|---|---|---|
| **Normalisation accuracy** | Does attributed revenue = actual revenue? (should be 0 error) | 25% |
| **Path coverage** | Is the top-credited channel actually in most converting paths? | 25% |
| **Conversion alignment** | Does channel ranking agree with raw conversion signal? | 20% |
| **Stability** | Do credits stay consistent across random data subsamples? | 20% |
| **Cross-model agreement** | Does this model agree with peer models (Spearman rank corr)? | 10% |

---

#### Step 6 — Recommend budget allocation

**Main class:** `AttributionBudgetAdvisor`

Two allocation methods are supported:

**Method B — ROI-weighted** (requires `cost_col`)

```
channel_budget = total_budget × (attributed_revenue / cost) / Σ(attributed_revenue / cost)
```

Channels that generate more attributed revenue per pound of spend receive
proportionally more budget.

**Method C — Saturation-aware** (requires `cost_col` + `CampaignBatchResult`)

Starts from ROI-weighted shares, then discounts channels that are near or
beyond their saturation point using the fitted saturation curves from
`adsat.campaign`.  Channels beyond 90% of their saturation point receive
a progressively smaller allocation.

**Fallback:** when cost data is unavailable, both methods fall back to
revenue-share allocation (budget ∝ attributed revenue share).

```python
from adsat.attribution import AttributionBudgetAdvisor

# Method B — ROI-weighted
advisor = AttributionBudgetAdvisor(
    total_budget  = 500_000,
    method        = "roi_weighted",
    min_spend     = 10_000,                        # floor for every channel
    max_spend     = {"paid_search": 200_000},       # per-channel cap
    current_spend = {                               # for spend_change columns
        "paid_search": 120_000,
        "display":      80_000,
        "social_paid":  90_000,
        "email":        50_000,
    },
)
alloc = advisor.recommend(result, model="shapley")
alloc.print_summary()
alloc.plot()

# Method C — saturation-aware (integrates with adsat.campaign)
from adsat.campaign import CampaignSaturationAnalyzer

batch   = CampaignSaturationAnalyzer(...).run(campaign_df)
advisor = AttributionBudgetAdvisor(total_budget=500_000, method="saturation_aware")
alloc   = advisor.recommend(result, model="ensemble", batch_result=batch)
alloc.print_summary()
```

**`AttributionBudgetAllocation` attributes**

| Attribute | Description |
|---|---|
| `allocations` | DataFrame: channel, current_spend, recommended_spend, spend_change_pct, attributed_revenue, credit_share, roi |
| `total_budget` | Budget that was distributed |
| `method` | Allocation method used |
| `model_used` | Attribution model that drove the allocation |

---

#### One-liner convenience function

```python
from adsat.attribution import attribute_campaigns

result = attribute_campaigns(
    events,
    models              = ["shapley", "markov", "ensemble"],
    lookback_days       = 30,
    multi_conversion    = "reset",
    interaction_weight  = {"click": 1.0, "impression": 0.3},
    markov_order        = 1,
    time_decay_half_life= 7.0,
    cost_col            = "cost",
)
result.print_summary()
```

---

#### Full workflow example

```python
import pandas as pd
from adsat.attribution import (
    JourneyBuilder, AttributionAnalyzer,
    AttributionBudgetAdvisor, plot_attribution, make_sample_events,
)

# 0. Load or generate data
events = pd.read_parquet("touchpoint_events.parquet")
# events = make_sample_events(n_users=5_000)  # for testing

# 1. Build journeys
journeys = JourneyBuilder(
    user_col           = "user_id",
    time_col           = "timestamp",
    channel_col        = "channel",
    interaction_col    = "interaction_type",
    converted_col      = "converted",
    revenue_col        = "revenue",
    cost_col           = "cost",
    lookback_days      = 30,
    multi_conversion   = "reset",
    interaction_weight = {"click": 1.0, "impression": 0.3},
).build(events)

# 2. Fit models
result = AttributionAnalyzer(
    models       = ["last_click", "shapley", "markov", "data_driven", "ensemble"],
    markov_order = 1,
    cost_col     = "cost",
).fit(journeys)

# 3. Review
result.print_summary()
plot_attribution(result, model=result.best_model())

# 4. Allocate budget
alloc = AttributionBudgetAdvisor(
    total_budget  = 1_000_000,
    method        = "roi_weighted",
    current_spend = {"paid_search": 300_000, "display": 200_000,
                     "email": 150_000, "social_paid": 200_000,
                     "organic_search": 100_000, "direct": 50_000},
).recommend(result, model=result.best_model())

alloc.print_summary()
alloc.plot()
```

---

## Contributing

Contributions are welcome — bug reports, feature requests, and pull requests.

- **Issues:** [github.com/stefanobandera1/adsat/issues](https://github.com/stefanobandera1/adsat/issues)
- **Pull requests:** fork the repo, branch from `main`, open a PR
- **Full guide:** see [CONTRIBUTING.md](CONTRIBUTING.md)

```bash
git clone https://github.com/stefanobandera1/adsat.git
cd adsat
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v
```

---

## Changelog

See [CHANGELOG.md](https://github.com/stefanobandera1/adsat/blob/main/CHANGELOG.md) for the full version history.

---

## License

[MIT](LICENSE) © 2025 Stefano Bandera

