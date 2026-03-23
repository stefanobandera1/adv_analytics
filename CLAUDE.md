# Claude Code — adsat Project Context

> **Also read**: the shared analytics projects `CLAUDE.md` one level above this repo.
> This file covers only what is specific to adsat.

---

## What adsat is

**adsat** (Advertising Saturation Analysis Toolkit) is a Python package for identifying
impression saturation points in advertising campaigns and optimising media spend.

Core use case: given a dataset of impressions → conversions (or any response metric),
fit saturation curves, find the point of diminishing returns, and recommend optimal
budget allocation across campaigns.

The package has two structurally distinct analytical layers that should never be conflated:

1. **Saturation layer** — operates on *aggregated* campaign-level data (one row = one
   time period for one campaign). Modules: `distribution`, `transformation`, `modeling`,
   `evaluation`, `pipeline`, `campaign`, `budget`, `response_curves`, `diagnostics`,
   `seasonality`, `simulation`, `report`, `benchmark`.

2. **Attribution layer** — operates on *journey-level* event data (one row = one user
   touchpoint). Module: `attribution`. Structurally separate by design — see rationale below.

Current version: **0.5.0**

**Status**: code complete, all tests passing, linting clean, ready for GitHub and PyPI publish.

Next steps (in order):
1. Push to GitHub: `https://github.com/stefanobandera1/adsat`
2. Publish to PyPI: tag `v0.5.0` → CI workflow auto-publishes via OIDC

---

## GitHub / PyPI

- **GitHub repo**: `https://github.com/stefanobandera1/adsat`
- **Clone URL**: `https://github.com/stefanobandera1/adsat.git`
- **Issues**: `https://github.com/stefanobandera1/adsat/issues`
- **PyPI page**: `https://pypi.org/project/adsat/` (not yet published as of 2026-03-22)
- **CI badge**: `https://github.com/stefanobandera1/adsat/actions/workflows/ci.yml/badge.svg`

CI/CD workflows exist at `.github/workflows/`:
- `ci.yml` — matrix: Python 3.9–3.13 × Ubuntu/Windows/macOS; runs ruff, black, pytest
- `publish.yml` — triggers on `vX.Y.Z` tag; publishes to PyPI via OIDC trusted publishing

**Publish checklist** (before tagging):
- [ ] `__version__` in `adsat/__init__.py` matches `version` in `pyproject.toml` (both `0.5.0`)
- [ ] `CHANGELOG.md` updated ✓
- [ ] All tests pass: `pytest tests/ -v` → 115 passed ✓
- [ ] Linting clean: `ruff check adsat/ tests/ && black --check adsat/ tests/` ✓ (clean as of 2026-03-23)
- [ ] End-to-end: `python run_end_to_end.py` → all steps completed successfully ✓
- [ ] No placeholder values in any file ✓

---

## Module map

| Module | Class(es) | What it does |
|---|---|---|
| `exploratory` | `CampaignExplorer`, `explore()` | Full EDA — histograms, Q-Q, ECDF, scatter, correlation, time-series, outlier detection |
| `distribution` | `DistributionAnalyzer` | Fits and ranks 14+ statistical distributions; recommends a transformation |
| `transformation` | `DataTransformer` | 13 invertible transforms (log, sqrt, Yeo-Johnson, Box-Cox, etc.); always stores fitted params for inverse |
| `modeling` | `SaturationModeler` | Fits 5 saturation curves: Hill (default), Exponential, Power, Michaelis-Menten, Logistic |
| `evaluation` | `ModelEvaluator` | Ranks models by AIC/BIC/R²/RMSE/MAPE composite score; identifies saturation point |
| `pipeline` | `SaturationPipeline` | Convenience wrapper chaining distribution→transformation→modeling→evaluation |
| `campaign` | `CampaignSaturationAnalyzer`, `CampaignBatchResult` | Loops the pipeline over every campaign in a multi-campaign DataFrame |
| `budget` | `BudgetOptimizer`, `BudgetAllocation`, `optimise_budget()` | SLSQP constrained optimisation: maximise total outcome across campaigns given a fixed budget |
| `response_curves` | `ResponseCurveAnalyzer`, `ResponseCurveResult` | Marginal returns, ROI curves, elasticity, efficiency zones along the fitted curve |
| `diagnostics` | `ModelDiagnostics`, `DiagnosticsReport`, `run_diagnostics()` | Six-panel residual diagnostics: normality, autocorrelation, Cook's D, heteroscedasticity |
| `seasonality` | `SeasonalDecomposer`, `SeasonalDecomposition`, `adjust_for_seasonality()` | CMA-based additive/multiplicative decomposition; pure NumPy/SciPy, no statsmodels |
| `simulation` | `ScenarioSimulator`, `Scenario`, `SimulationResult`, `simulate()` | Named what-if scenario comparison and sensitivity tables |
| `report` | `ReportBuilder`, `generate_report()` | Self-contained HTML report with all charts embedded as base64 PNG |
| `attribution` | `AttributionAnalyzer` + 9 model classes | Multi-touch attribution on journey-level event data |
| `benchmark` | `CampaignBenchmarker`, `BenchmarkResult`, `benchmark_campaigns()` | Statistical benchmarking of campaign metrics against historical baselines (M1/M2/M3) |

### Module dependency order (bottom → top)

```
distribution → transformation → modeling → evaluation → pipeline
                                                              ↓
                                              campaign → budget
                                                         response_curves
                                                         simulation
                                              diagnostics (standalone — consumes modeling output)
                                              seasonality (standalone — pre-processing step)
                                              attribution (standalone — different data structure)
                                              benchmark   (standalone — different data structure)
                                              report      (consumes output from any module)
```

**Hard rule**: no circular imports. `attribution` and `benchmark` do not import from
`pipeline` or `campaign`. `report` imports result dataclasses only, not fitting logic.

---

## Why each module exists — original design rationale

Understanding *why* each module was added prevents duplicating what already exists and
guides where new functionality should live.

### Core pipeline modules (built first, v0.1–0.2)

**`distribution.py`** — First step in the pipeline because the right transformation
depends on knowing the distribution. Fitting 14+ distributions and ranking by AIC/BIC
was deliberately chosen over just computing skewness — it makes the recommendation
defensible and auditable by the user.

**`transformation.py`** — Every transform is *invertible by design*. This was an
explicit original requirement: "saturation points must come back in the original scale."
The `inverse_transform()` method stores fitted parameters (e.g. lambda for Box-Cox) so
the round-trip is always available. **Never add a transformation that cannot be
inverted.** Round-trip accuracy was tested across all 13 transforms with max error = 0.0.

**`modeling.py`** — Five models are included for comparison, but **Hill is the primary
model and the default**. Hill (sigmoid) was chosen because: (1) the EC50 parameter is
literally the saturation point — interpretable without post-processing; (2) n controls
curve steepness and is meaningful; (3) it generalises empirically well across
impression→response relationships. The other four (Power, Exponential, Michaelis-Menten,
Logistic) exist so users can see Hill winning on their data, not to replace it. The
optional Bayesian Hill via PyMC provides credible intervals but is not the main path.

**`evaluation.py`** — Uses a composite rank score across AIC/BIC/R²/RMSE/MAPE because
no single metric is sufficient. A model can have good R² but poor AIC if it overfits.

**`pipeline.py`** — A convenience wrapper, not a core abstraction. Added because users
asked for "one call that does everything." Power users call the individual classes
directly. **Do not add business logic to `pipeline.py`** — it belongs in the upstream
modules.

**`campaign.py`** — Added in v0.2 because the original pipeline had no concept of
campaign identity — it operated on a single DataFrame slice with no campaign column.
This module adds the looping, per-campaign results, and status classification:
`below / approaching / at / beyond saturation`. The `predict_saturation_per_campaign()`
function returns a clean one-row-per-campaign summary DataFrame.

### Extension modules (added at v0.3–0.4)

**`exploratory.py`** — Added because the original EDA was limited and had a bug in
`plot_distributions()` (tried to get `source_data` from `descriptive_stats` — it wasn't
there). Rebuilt from scratch with 9 dedicated plot methods. `explore()` runs all nine.

**`budget.py`** — The natural next question after knowing saturation points: "how should
I reallocate budget across campaigns to maximise total conversions?" Uses SLSQP with
**10 restarts by default** (`n_restarts` is user-configurable). Why 10 restarts: the
saturation curves are concave so SLSQP should find the global optimum in theory, but
tight per-campaign min/max constraints and large scale differences (e.g. one campaign at
£50k, another at £5M) create practical local optima. An earlier version had a bug in the
second starting point (outcome-space values used as spend-space x0 — units mismatch);
the fix uses `(lo + hi) / 2.0`. Do not revert this.

**`response_curves.py`** — Answers "how fast are we getting to saturation?" Marginal
returns, ROI curves, elasticity, and efficiency zones with shading were identified as
the most commonly requested insights in MMM contexts.

**`diagnostics.py`** — Added because there was no way to inspect model quality beyond
R² and AIC. The six-panel output (Shapiro-Wilk, KS, Jarque-Bera, Durbin-Watson,
Levene, Cook's D) is the standard set a data scientist would check after fitting.

**`seasonality.py`** — Advertising data has weekly and seasonal patterns (Christmas,
Black Friday) that distort saturation estimates if not removed first. Uses CMA
decomposition in pure NumPy/SciPy. **No statsmodels dependency** — a deliberate choice
because statsmodels is not available in all target environments. Both additive and
multiplicative decomposition are supported. `inverse_adjust()` adds seasonality back
to predictions.

**`report.py`** — The "last mile" for non-technical stakeholders. Generates a
self-contained HTML file with all charts embedded as base64 PNG — one file, no
external assets. Accepts results from any module via method chaining.

**`simulation.py`** — What-if scenario planning built on top of `campaign.py` results.
Named scenarios allow side-by-side comparison. `sensitivity_table()` shows outcome
impact of ±% spend changes per campaign.

### Standalone analytical modules

**`benchmark.py`** — The most complex module. Built to answer: "is this campaign
performing above or below expectation, given its scale and history?" Directly inspired
by a real web analytics notebook (BBC-style visit CVR benchmarking) and generalised.

Three methods exposed separately (M1/M2/M3):
- **M1** — OLS linear trend + quasi-dispersion-inflated SE bands. For proportions (CVR,
  CTR): quasi-binomial formula `sqrt(phi × p × (1-p) / n)`. For continuous metrics:
  `phi × std(residuals)` — does NOT divide by volume (would produce meaninglessly tight
  bands when impressions are in the hundreds of thousands).
- **M2** — Peer-bin quantile bands. Configurable quartile binning on any numerical
  variable. Leave-one-out comparison against historical peers in the same bin.
  Optional `bootstrap_m2=True` adds 95% CI around p10/p90 (opt-in because it's O(500n²)).
- **M3** — Adaptive selector: defaults to M1, overrides to M2 when the observation's
  scale is unusually far from the typical. When `use_seasonality=False`, M3 uses
  overall median bin distance instead of seasonal typical — preserves the spirit of M3
  without requiring time structure.

Additional features: cross-segment z-score (UK_paid vs DE_paid comparison), CUSUM +
Pettitt change-point detection per segment with `refit_after_changepoint=True`, VIF
checking for confounder columns (warn if VIF > 5), graceful aggregation fallback when
a segment group is below `min_history_rows`.

**Key invariant**: M1/M2/M3 are always fitted on *historical data only*. Current-period
rows are scored but never influence the baseline. This was a deliberate design decision
to prevent leakage.

**`attribution.py`** — See dedicated section below.

---

## Key design decisions (don't change without discussion)

### Hill function as default
EC50 is the saturation point directly. n is the steepness. Both are interpretable
without post-processing. The other four models exist for comparison, not replacement.

### All transformations must be invertible
Saturation points must always be reportable in original units. `inverse_transform()`
is not optional — it's the point of the module. Do not add any transform that cannot
be inverted.

### Attribution module uses different input data — keep it separate
`attribution.py` takes journey-level event data. The saturation modules take aggregated
campaign data. These structures are fundamentally different. The `JourneyBuilder` class
converts raw event logs to structured journeys. Non-converting journeys must be included
— Shapley and Markov are severely biased without them (they need non-converting paths to
estimate baseline conversion probability correctly).

### Shapley auto-switch at 12 channels
Exact Shapley is O(2^n). Below 12 channels: exact computation (all 2^n coalitions).
Above 12: Monte Carlo permutation sampling. Threshold based on 2^12 = 4,096 coalitions
being tractable in under a second; 2^13 starts to be slow on normal hardware.

### Benchmark metric type must be declared by user
Two types: `"proportion"` (CVR, CTR — bounded 0–1) and `"continuous"` (ROAS, revenue
per impression — unbounded). The SE formula is different for each. The user must
declare which type they're using — the module does not auto-detect.

### Seasonality: no statsmodels
Pure NumPy/SciPy CMA decomposition was chosen deliberately because statsmodels is not
available in all environments. Do not add a statsmodels dependency to `seasonality.py`.

---

## Attribution module — input data schema

Minimum required schema for `attribution.py`:

| Column | Type | Required | Notes |
|---|---|---|---|
| `user_id` | str/int | ✓ | Unique user or cookie identifier |
| `timestamp` | datetime | ✓ | Touchpoint time — ordering and time-decay depend on this |
| `channel` | str | ✓ | Channel name (e.g. "paid_search", "email") |
| `interaction_type` | str | ✓ | "impression" or "click" |
| `converted` | int (0/1) | ✓ | Whether this touchpoint's journey ended in conversion |
| `revenue` | float | ✓ | Revenue at conversion (0 for non-conversion rows) |
| `cost` | float | optional | Needed for ROI-based budget allocation |
| `campaign` | str | optional | Sub-channel grouping |

`converted` and `revenue` should be set only on the conversion touchpoint row.
`JourneyBuilder` propagates these to the journey level. Interaction weights are
user-configurable (e.g. `{"click": 1.0, "impression": 0.3}`). Journey window is
user-configurable or auto-detected (3× median inter-event gap, clamped 7–90 days).
Multi-conversion users: "reset" (each conversion restarts the journey) or "rolling"
(all touchpoints within window count toward every conversion) — user declares which.

### `AttributionResult.channel_credits` is a DataFrame, not a dict

`result.channel_credits` is a **long-format DataFrame** with columns:
`channel`, `model`, `attributed_conversions`, `attributed_revenue`, `credit_share`
(and optionally `roi`, `attributed_revenue_share`).

To get credits for a specific model, use `result.get_credits("model_name")` which
returns a filtered DataFrame. Do not subscript with a model name directly
(`result.channel_credits["last_click"]` will fail — it tries a column lookup).
To check which models were fitted use `result.models_fitted` (a list), not
`"model_name" in result.channel_credits`.

---

## Bugs found and fixed — do not reintroduce

| Module | Bug pattern | Fix |
|---|---|---|
| `modeling.py` | `y_fine[-1]` used as asymptote for all models (wrong for Power) | Model-specific analytical asymptote extraction |
| `modeling.py` | `n*log(ss_res/n)` → -inf on near-perfect fits | Clamp `ss_res` to 1e-10 |
| `modeling.py` | All-zero y collapses bounds: `a_init=0` → `[0,0]` bounds crash `curve_fit` | `a_init = max(y.max() * 1.2, 1.0)` |
| `modeling.py` | `MODEL_REGISTRY['hill_bayesian']` KeyError in predict() | Fallback to `'hill'` key |
| `modeling.py` | `import arviz as az` inside Bayesian fit — `az` never used, causing F401 ruff error | Removed the unused import (fixed 2026-03-23) |
| `campaign.py` | `if self.saturation_point:` is False when value is 0.0 | Use `is not None` explicitly |
| `campaign.py` | Unknown campaign ID passes empty DataFrame to pipeline (confusing crash deep in SciPy) | Validate campaign ID upfront with clear KeyError |
| `campaign.py` | `self.campaign_col` never assigned in `__init__` — `AttributeError` on every `.run()` call | Add `self.campaign_col = campaign_col` to `__init__` (fixed 2026-03-22) |
| `evaluation.py` | Dummy zero-line drawn before real curve in plot | Draw real curve directly with label |
| `exploratory.py` | `plot_correlation()` crashes with 1 numeric column (scalar Axes not subscriptable) | `squeeze=False` + early guard |
| `budget.py` | Second starting point used outcome-space values as spend-space x0 | Replace with `(lo + hi) / 2.0` |
| `distribution.py` | `_recommend_transform` skewness thresholds fired after lognorm check | Fix priority: skewness thresholds first |
| `attribution.py` | Ambiguous variable `l` in legend dedup loop (E741) | Renamed to `lbl` (fixed 2026-03-23) |
| `benchmark.py` | Ambiguous variable `l` in categorical order list comprehension (E741) | Renamed to `lbl` (fixed 2026-03-23) |

---

## Known issues

All previously identified issues have been resolved. The package is clean for initial release.

- ~~Test coverage gaps~~ — resolved: 89 new tests added, all 115 tests pass
- ~~`__init__.py` import ordering~~ — resolved: `benchmark` and `attribution` imports are in place
- ~~ruff/black lint failures~~ — resolved 2026-03-23: all 17 files reformatted by black; 374+ ruff
  issues auto-fixed (deprecated `typing.Dict/List/Tuple/Optional/Union` modernised to built-in
  equivalents); uppercase convention variables (N803/N806) added to `pyproject.toml` ignore list
  since they are standard mathematical notation in stats code (`X` feature matrix, `U` test stat,
  `S_pos`/`S_neg` CUSUM accumulators, `K` changepoint index, `L` logistic maximum)
- `generate_report()` does not accept a `verbose` parameter (unlike most other convenience
  functions). This is intentional — the function has no logging path. Do not add `verbose=False`
  when calling it.

---

## Test suite status (as of 2026-03-23)

- **Total**: 115 tests, all passing
- **Runtime**: ~70 seconds (dominated by attribution Shapley and benchmark fitting)
- **File**: `tests/test_adsat.py` — single file, one class per module
- **Conftest**: `tests/conftest.py` — sets `matplotlib.use("Agg")` before any import
  to prevent display errors in headless/CI environments

### Test class inventory

| Class | Module | Tests |
|---|---|---|
| `TestDistributionAnalyzer` | `distribution` | 4 |
| `TestDataTransformer` | `transformation` | 3 |
| `TestSaturationModeler` | `modeling` | 5 |
| `TestModelEvaluator` | `evaluation` | 2 |
| `TestSaturationPipeline` | `pipeline` | 2 |
| `TestCampaignSaturationAnalyzer` | `campaign` | 10 |
| `TestPredictSaturationPerCampaign` | `campaign` | 3 |
| `TestCampaignExplorer` | `exploratory` | 7 |
| `TestModelDiagnostics` | `diagnostics` | 10 |
| `TestBudgetOptimizer` | `budget` | 8 |
| `TestResponseCurveAnalyzer` | `response_curves` | 7 |
| `TestSeasonalDecomposer` | `seasonality` | 13 |
| `TestScenarioSimulator` | `simulation` | 9 |
| `TestReportBuilder` | `report` | 6 |
| `TestAttribution` | `attribution` | 11 |
| `TestCampaignBenchmarker` | `benchmark` | 7 |

---

## Test data conventions

Standard fixture for saturation tests:
```python
@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 80
    x = np.linspace(100_000, 3_000_000, n)
    y = 30_000 * (x ** 1.5) / (1_200_000 ** 1.5 + x ** 1.5) + np.random.normal(0, 300, n)
    return pd.DataFrame({
        "impressions": x.astype(int),
        "conversions": np.maximum(y, 0).round(0)
    })
```

Multi-campaign fixture uses `_make_campaign()` helper with campaigns `"A"`, `"B"`, `"C"`
(C has only 3 rows and is expected to fail `min_observations=10` check).

For attribution tests: use `make_sample_events(n_users=200, random_seed=42)` — exported
from the package. Build journeys with `JourneyBuilder(lookback_days=30).build(events_df)`.

For benchmark tests: include a `week_start` date column, at least one categorical
segment column (e.g. `"segment"`), 40 historical rows + 4 current rows per segment,
and pass `use_seasonality=False` to avoid week-of-year lookups on short synthetic data.

---

## What already exists — check before adding

- Marginal ROI / elasticity → `response_curves.py`
- Spend scenarios → `simulation.py`
- Cross-campaign comparison → `benchmark.py`
- Budget optimisation → `budget.py`
- Residual / model quality → `diagnostics.py`
- Seasonal adjustment → `seasonality.py`
- Distribution fitting → `distribution.py` (14+ distributions)
- Attribution / channel credit → `attribution.py` (9 models)

---

## Running locally

```bash
pip install -e ".[dev]"
pytest tests/ -v                        # 115 tests, ~70s
python run_end_to_end.py                # smoke test, all steps complete cleanly
ruff check adsat/ tests/
black --check adsat/ tests/
```

## CI/CD

- **CI** (`ci.yml`): push/PR → ruff + black + pytest across Python 3.9–3.13 on Ubuntu/Windows/macOS
- **Publish** (`publish.yml`): `vX.Y.Z` tag → auto-publish to PyPI via OIDC (no stored secrets)

```bash
# Release process
# 1. Bump version in pyproject.toml AND adsat/__init__.py
# 2. Update CHANGELOG.md — move [Unreleased] items to new version heading
# 3. git commit -m "chore: bump version to X.Y.Z"
# 4. git tag vX.Y.Z && git push origin main --tags
# 5. GitHub Actions publishes to PyPI automatically
```
