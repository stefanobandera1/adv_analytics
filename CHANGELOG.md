# Changelog

All notable changes to **adsat** are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- Interactive HTML report output
- Prophet-based seasonality-aware saturation modeling
- CLI entry-point (`adsat analyse campaign_data.csv`)

---

## [0.5.1] – 2026-03-27

### Added
- `CampaignResult.saturation_extrapolation_warning` — `str | None` field set when the
  fitted saturation point exceeds `extrapolation_threshold × max(x_observed)` (default 3.0×).
  Signals that the saturation estimate relies on extrapolation beyond observed data and
  should be treated with caution in budget decisions.
- `CampaignSaturationAnalyzer.__init__` now accepts `extrapolation_threshold: float = 3.0`.
  Set to `float('inf')` to disable the warning entirely.

### Changed
- Test coverage raised from 66% to 72%; coverage threshold raised to 70% in
  `pyproject.toml` and CI.
- 18 new tests added to `TestAttribution` and `TestCampaignBenchmarker` covering
  `EnsembleModel`, `AttributionEvaluator`, M2 bootstrap CI, M3 class values,
  cross-segment z-score, CUSUM change-point detection, VIF checking, and plot methods.
- `@pytest.mark.slow` marker introduced; Shapley Monte Carlo path gated behind it.
  Local `pytest` skips slow tests; CI runs all with `-m ""`.

### Fixed
- `negative_exponential` model correctly documented (previously listed as "Exponential").
- R² = 0 on fully-saturated campaigns documented as expected behaviour, not a fitting failure.

---

## [0.5.0] – 2026-03-22

### Added
- `adsat.attribution` module — multi-touch attribution with 9 models:
  `last_click`, `first_click`, `linear`, `position_based`, `time_decay`,
  `shapley` (exact below 12 channels, Monte Carlo above), `markov`,
  `data_driven`, `ensemble`
- `JourneyBuilder` — converts raw touchpoint event logs to structured journeys;
  supports configurable lookback window, interaction weights, multi-conversion
  strategies ("reset" / "rolling")
- `AttributionAnalyzer` — fits all configured models and returns a unified
  `AttributionResult` with per-channel credits, ROI, and model comparison
- `AttributionEvaluator` — cross-model comparison metrics
- `AttributionBudgetAdvisor` — ROI-weighted budget allocation from attribution credits
- `make_sample_events()` — synthetic event log generator for testing
- `plot_attribution()` — module-level plot convenience function
- `adsat.benchmark` module — `CampaignBenchmarker` with four analytical methods:
  - **M1** OLS linear trend + quasi-dispersion SE bands (proportion and continuous)
  - **M2** Peer-bin quantile bands (leave-one-out, optional bootstrap CI)
  - **M3** Adaptive selector (M1 default, M2 override on scale shift)
  - **P1** Cross-segment z-score outperformance classification
- CUSUM + Pettitt change-point detection with `refit_after_changepoint` option
- VIF confounder checking (warns when VIF > 5)
- `benchmark_campaigns()` one-liner convenience function
- Full test coverage for all previously untested modules:
  `exploratory`, `diagnostics`, `budget`, `response_curves`, `seasonality`,
  `simulation`, `report`, `attribution`, `benchmark` (89 new tests)
- `tests/conftest.py` with Agg matplotlib backend for headless CI
- `.github/workflows/ci.yml` — matrix CI across Python 3.9–3.13 on Ubuntu/Windows/macOS
- `.github/workflows/publish.yml` — OIDC trusted publishing to PyPI on version tag

### Fixed
- `CampaignSaturationAnalyzer.__init__` was not storing `self.campaign_col`,
  causing `AttributeError` on any call to `.run()` or `.run_single()`

---

## [0.4.0] – 2025-10-01

### Added
- `adsat.budget` module — `BudgetOptimizer` with SLSQP constrained optimisation;
  10 multi-start restarts by default to avoid local optima with tight per-campaign
  constraints; `optimise_budget()` one-liner
- `adsat.response_curves` module — `ResponseCurveAnalyzer` computing marginal returns,
  ROI curves, elasticity, and efficiency zones (high / medium / low);
  `analyse_response_curves()` one-liner
- `adsat.diagnostics` module — `ModelDiagnostics` with six residual diagnostic tests
  (Shapiro-Wilk, KS, Jarque-Bera, Durbin-Watson, Levene, Cook's D);
  `run_diagnostics()` one-liner
- `adsat.seasonality` module — `SeasonalDecomposer` with additive / multiplicative CMA
  decomposition; pure NumPy/SciPy, no statsmodels dependency; `adjust_for_seasonality()`
  one-liner; `inverse_adjust()` for adding seasonality back to predictions
- `adsat.simulation` module — `ScenarioSimulator` for named what-if scenario comparison;
  `sensitivity_table()` for ±% spend impact analysis; `simulate()` one-liner
- `adsat.report` module — `ReportBuilder` generating self-contained HTML reports with
  charts embedded as base64 PNG; `generate_report()` one-liner

### Fixed
- `BudgetOptimizer` second starting point used outcome-space values as spend-space `x0`
  (units mismatch); fixed to use `(lo + hi) / 2.0`

---

## [0.3.0] – 2025-06-10

### Added
- `adsat.exploratory` module with `CampaignExplorer` class and `explore()` one-liner
- `CampaignExplorer.plot_descriptive_summary()` – stats table + boxplots
- `CampaignExplorer.plot_histograms()` – histograms with KDE, mean/median lines, skewness
- `CampaignExplorer.plot_qq()` – Q-Q plots vs Normal, lognormal, exponential
- `CampaignExplorer.plot_ecdf()` – ECDF with Normal CDF overlay and percentile markers
- `CampaignExplorer.plot_correlation()` – Pearson/Spearman heatmap + scatter matrix
- `CampaignExplorer.plot_scatter()` – scatter with OLS regression line + 95% CI band
- `CampaignExplorer.plot_time_series()` – metrics over time, coloured by campaign
- `CampaignExplorer.plot_outliers()` – IQR boxplot + z-score strip plot with outlier table
- `CampaignExplorer.plot_distribution_fits()` – 4-panel: histogram+PDFs, ECDF+CDFs, Q-Q, AIC/BIC
- `CampaignExplorer.explore()` – runs every plot in one call
- `explore()` module-level one-liner function
- Fixed `DistributionAnalyzer.plot_distributions()` raw-data retrieval bug
- Upgraded `plot_distributions()` to full 4-panel layout matching `plot_distribution_fits()`
- `__version__` bumped to `0.3.0`

---

## [0.2.0] – 2025-06-01

### Added
- `adsat.campaign` module with `CampaignSaturationAnalyzer` class
- `predict_saturation_per_campaign()` convenience one-liner
- `CampaignBatchResult` with `.plot_all()`, `.plot_saturation_comparison()`, `.plot_status_breakdown()`
- Per-campaign saturation status classification: `below / approaching / at / beyond`
- `pct_of_saturation` field showing current spend as % of the saturation point
- `run_single()` method for targeted per-campaign re-analysis
- `CampaignResult.print_summary()` for human-readable per-campaign output
- Full end-to-end test script `run_end_to_end.py`

### Changed
- `__version__` bumped to `0.2.0`
- `pyproject.toml` now uses `setuptools.build_meta` backend (PEP 517 compliant)
- README expanded with per-campaign usage examples

---

## [0.1.0] – 2025-05-01

### Added
- `DistributionAnalyzer` – fits 14+ distributions, ranks by AIC/BIC, Shapiro-Wilk normality test, recommends transformation
- `DataTransformer` – supports 13 invertible transformations (log, sqrt, Box-Cox, Yeo-Johnson, quantile, standard, robust, minmax, etc.)
- `SaturationModeler` – Hill, Negative Exponential, Power, Michaelis-Menten, Logistic curve fitting
- Optional Bayesian Hill model via PyMC with posterior credible intervals
- `ModelEvaluator` – ranks models by AIC/BIC/R²/RMSE/MAPE, composite rank score
- `SaturationPipeline` – end-to-end automated pipeline for a single campaign dataset
- `PipelineResult.print_summary()` for structured output
- Saturation point back-transformation to original data scale
- Full unit test suite (`tests/test_adsat.py`)
- Example scripts in `examples/`

---

[Unreleased]: https://github.com/stefanobandera1/adsat/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/stefanobandera1/adsat/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/stefanobandera1/adsat/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/stefanobandera1/adsat/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/stefanobandera1/adsat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/stefanobandera1/adsat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/stefanobandera1/adsat/releases/tag/v0.1.0
