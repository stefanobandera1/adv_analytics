"""
adsat.benchmark
===============
Campaign performance benchmarking: classify observations as Above / Within / Below
a statistically-derived baseline, controlling for scale (volume/spend) and seasonality.

Methods
-------
M1  Trend + seasonal SE bands
    Fits a linear OLS trend over time per segment (optionally with external
    confounders via `confounder_cols`), adds a week-of-year seasonal correction,
    then builds quasi-dispersion-inflated confidence bands at 90 % and 95 %.
    Automatically checks for structural breaks (CUSUM + Pettitt) and, when
    `refit_after_changepoint=True`, refits using only post-break data.

M2  Peer-bin quantile bands  (bootstrap-aware)
    Segments observations into n_bins quantile bins on a numerical variable
    (e.g. spend, impressions).  Each observation is compared leave-one-out
    against peers in the same bin.  With `bootstrap_m2=True` the module also
    computes 95 % bootstrap confidence intervals around the p10/p90 thresholds
    so users can judge how stable the peer comparison is.

M3  Adaptive selector
    Defaults to M1.  Overrides to M2 when the observation's scale bin is
    unusually far from the typical bin for that time of year (or overall
    median when seasonality is disabled).

P1  Cross-segment z-score  (added columns: cross_seg_zscore, cross_seg_class)
    Standardises each observation's metric within its spend/impression bin
    across ALL segments using historical data only.  This answers "which
    campaign is outperforming its peers right now?" — a question the within-
    segment methods (M1–M3) cannot answer on their own.

Quick-start
-----------
>>> from adsat.benchmark import CampaignBenchmarker
>>> bm = CampaignBenchmarker(
...     metric_col           = "conversion_rate",
...     metric_type          = "proportion",
...     date_col             = "week_start",
...     volume_col           = "impressions",
...     bin_col              = "spend",
...     segment_cols         = ["country", "channel"],
...     confounder_cols      = ["market_cpi"],          # optional
...     current_period_start = "2024-10-01",
... )
>>> result = bm.fit(df)
>>> result.print_summary()
>>> result.plot()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# ── palette ───────────────────────────────────────────────────────────────────
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_GREEN = "#3BB273"
_RED = "#E84855"
_GREY = "#6C757D"
_AMBER = "#F4A261"
_PURPLE = "#7B2D8B"
_TEAL = "#2EC4B6"

_CLASS_COLOURS = {
    "Above (Act)": _RED,
    "Above (Watch)": _AMBER,
    "Above": _AMBER,
    "Within": _GREEN,
    "Below (Watch)": _AMBER,
    "Below (Act)": _BLUE,
    "Below": _BLUE,
    "Outperforming": _RED,
    "Typical": _GREEN,
    "Underperforming": _BLUE,
}

# ── constants ─────────────────────────────────────────────────────────────────
_Z90, _Z95 = 1.645, 1.960
_BIN_LABELS = ["L", "M", "H", "VH"]
_GAP_MULTIPLE = 3.0  # discrete burst detection: gap > 3× median interval
_SKEW_THRESH = 1.0  # auto log-transform when |skewness| > this
_VIF_THRESH = 5.0  # warn when VIF of a confounder exceeds this
_CUSUM_H = 4.0  # CUSUM decision threshold (in σ units)
_PETTITT_ALPHA = 0.05  # significance level for Pettitt change-point test


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────


def _safe_log(series: pd.Series) -> pd.Series:
    """
    Return log(x) as a float Series, replacing zeros and negative values with NaN.

    Used to safely log-transform spend / impression columns that may contain zeros.
    """
    return np.log(series.where(series > 0, other=np.nan))


def _circular_roll(series: pd.Series, window: int = 3) -> pd.Series:
    """Wrap-around moving average for week-of-year smoothing."""
    if window <= 1 or len(series) == 0:
        return series
    k = window // 2
    vals = series.sort_index().values
    padded = np.r_[vals[-k:], vals, vals[:k]]
    smoothed = np.convolve(padded, np.ones(window) / window, mode="valid")
    return pd.Series(smoothed, index=series.sort_index().index)


def _ols(
    t: np.ndarray,
    y: np.ndarray,
    X_extra: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    OLS fit.  Returns (fitted_values, coef_array).
    t        : 1-D time index
    y        : 1-D target
    X_extra  : optional extra columns to add to the design matrix (n × k)
    Rows where any value is NaN are dropped from fitting but still scored
    using row-wise available columns.
    """
    # Build design matrix [1, t, X_extra...]
    cols = [np.ones_like(t), t]
    if X_extra is not None and X_extra.ndim == 2 and X_extra.shape[1] > 0:
        cols.append(X_extra)
    X_full = np.column_stack(cols)

    # Row-wise NaN mask
    nan_in_X = np.isnan(X_full).any(axis=1)
    nan_in_y = np.isnan(y)
    fit_mask = ~(nan_in_X | nan_in_y)

    if fit_mask.sum() < 2:
        mu = float(np.nanmean(y))
        return np.full_like(y, mu, dtype=float), np.array([mu] + [0.0] * (X_full.shape[1] - 1))

    coef, _, _, _ = np.linalg.lstsq(X_full[fit_mask], y[fit_mask], rcond=None)

    # Score all rows; rows with NaN confounders fall back to intercept + time only
    fitted = np.full_like(y, np.nan, dtype=float)
    for i in range(len(y)):
        if nan_in_y[i]:
            continue
        row = X_full[i]
        if np.isnan(row).any():
            # use only intercept + time columns for this row
            row_safe = np.array([1.0, t[i]])
            fitted[i] = float(coef[:2] @ row_safe)
        else:
            fitted[i] = float(coef @ row)

    return fitted, coef


def _vif(X: np.ndarray) -> np.ndarray:
    """
    Compute VIF for each column of X (no intercept column assumed).
    Returns array of VIF values in the same column order.
    """
    n, p = X.shape
    vifs = np.zeros(p)
    for j in range(p):
        y_j = X[:, j]
        X_other = np.column_stack([np.ones(n), X[:, [i for i in range(p) if i != j]]])
        mask = ~np.isnan(y_j) & ~np.isnan(X_other).any(axis=1)
        if mask.sum() < 3:
            vifs[j] = np.nan
            continue
        coef, _, _, _ = np.linalg.lstsq(X_other[mask], y_j[mask], rcond=None)
        y_hat = X_other[mask] @ coef
        ss_res = np.sum((y_j[mask] - y_hat) ** 2)
        ss_tot = np.sum((y_j[mask] - y_j[mask].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[j] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    return vifs


def _woy_circ_dist(a: int, b: int, cycle: int = 53) -> int:
    """
    Circular distance between two week-of-year integers on a 53-week cycle.

    E.g. week 1 and week 52 are 2 weeks apart, not 51.
    """
    d = abs(int(a) - int(b))
    return min(d, cycle - d)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median of a 1-D array.

    Parameters
    ----------
    values  : 1-D float array
    weights : 1-D float array, same length

    Returns float, or nan when values is empty.
    """
    if len(values) == 0:
        return np.nan
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    idx = np.searchsorted(cw, 0.5 * w.sum(), side="left")
    return float(v[min(idx, len(v) - 1)])


def _weighted_mode_bin(
    bin_idxs: np.ndarray,
    weights: np.ndarray,
    valid_bins: tuple = (1, 2, 3, 4),
) -> int:
    """
    Return the bin index with the highest cumulative weight.

    Ties are broken by choosing the candidate closest to the weighted median.

    Parameters
    ----------
    bin_idxs   : integer array of bin labels
    weights    : float array, same length
    valid_bins : tuple of valid integer bin labels
    """
    if len(bin_idxs) == 0:
        return int(np.median(valid_bins))
    max_bin = max(valid_bins)
    wc = np.zeros(max_bin + 1, dtype=float)
    for b, w in zip(bin_idxs, weights):
        if int(b) in valid_bins:
            wc[int(b)] += float(w)
    top = wc.max()
    candidates = [b for b in valid_bins if wc[b] == top]
    if len(candidates) == 1:
        return int(candidates[0])
    med = _weighted_median(np.array(valid_bins, dtype=float), wc[list(valid_bins)])
    return int(min(candidates, key=lambda b: abs(b - med)))


def _detect_series_type(dates: pd.Series, gap_multiple: float = _GAP_MULTIPLE) -> str:
    """
    Infer whether a date series represents a continuous or discrete-burst campaign.

    Returns "discrete" when at least one inter-observation gap exceeds
    gap_multiple × the median gap; returns "continuous" otherwise.

    Parameters
    ----------
    dates        : pd.Series of datetime values
    gap_multiple : threshold multiplier (default 3.0)
    """
    if len(dates) < 3:
        return "continuous"
    sorted_d = dates.dropna().sort_values()
    diffs = sorted_d.diff().dropna()
    if len(diffs) == 0 or diffs.median() == pd.Timedelta(0):
        return "continuous"
    return "discrete" if diffs.max() > gap_multiple * diffs.median() else "continuous"


# ─────────────────────────────────────────────────────────────────────────────
# Change-point detection helpers
# ─────────────────────────────────────────────────────────────────────────────


def _cusum_changepoint(y: np.ndarray, h: float = _CUSUM_H) -> int | None:
    """
    Two-sided CUSUM on a 1-D series y.
    Returns the index of the first crossing of threshold h*sigma, or None.
    h   : decision threshold in sigma units (default 4 — conservative).
    """
    clean = y[~np.isnan(y)]
    if len(clean) < 6:
        return None
    mu = float(np.mean(clean))
    sigma = float(np.std(clean))
    if sigma == 0:
        return None

    z = (y - mu) / sigma
    S_pos = np.zeros(len(z))
    S_neg = np.zeros(len(z))
    k = 0.5  # allowance parameter (half a sigma)

    for i in range(1, len(z)):
        S_pos[i] = max(0.0, S_pos[i - 1] + z[i] - k)
        S_neg[i] = max(0.0, S_neg[i - 1] - z[i] - k)

    triggered = np.where((S_pos > h) | (S_neg > h))[0]
    if len(triggered) == 0:
        return None

    # Walk back from first alarm to find where accumulation started
    alarm = int(triggered[0])
    # Backtrack: find the last point before the alarm where CUSUM reset to 0
    for i in range(alarm, 0, -1):
        if S_pos[i] == 0 and S_neg[i] == 0:
            return i
    return 0


def _pettitt_changepoint(y: np.ndarray, alpha: float = _PETTITT_ALPHA) -> int | None:
    """
    Pettitt change-point test (non-parametric).
    Returns the most likely change-point index if significant at `alpha`, else None.

    Reference: Pettitt (1979) Applied Statistics 28(2) 126-135.
    The p-value is approximated via the formula in that paper.
    """
    y = np.array(y, dtype=float)
    mask = ~np.isnan(y)
    y_c = y[mask]
    n = len(y_c)
    if n < 6:
        return None

    # U statistic: U_t = sum_{i<=t, j>t} sign(y_j - y_i)
    U = np.zeros(n)
    for t in range(1, n):
        U[t] = U[t - 1] + np.sum(np.sign(y_c[t:] - y_c[t - 1]))

    K = int(np.argmax(np.abs(U)))
    K_val = float(np.abs(U[K]))

    # Approximate p-value
    p_val = 2.0 * np.exp(-6.0 * K_val**2 / (n**3 + n**2))
    p_val = min(p_val, 1.0)

    if p_val >= alpha:
        return None

    # Map back to original (possibly NaN-filtered) index
    clean_positions = np.where(mask)[0]
    return int(clean_positions[min(K, len(clean_positions) - 1)])


def detect_changepoints(
    series: pd.Series,
    cusum_h: float = _CUSUM_H,
    pettitt_alpha: float = _PETTITT_ALPHA,
) -> dict[str, Any]:
    """
    Run both CUSUM and Pettitt change-point tests on a metric series.

    Returns a dict with keys:
        cusum_index    : int or None — position of CUSUM alarm
        pettitt_index  : int or None — position of Pettitt change-point
        agreed         : bool — both tests agree on a change-point
        recommended_refit_from : int or None — index to refit from
            (uses Pettitt if available, falls back to CUSUM)

    Parameters
    ----------
    series : pd.Series
        The metric values in time order.
    cusum_h : float
        CUSUM decision threshold in sigma units.  Higher = less sensitive.
        Default 4.0 (conservative; reduces false positives).
    pettitt_alpha : float
        Significance level for Pettitt test.  Default 0.05.
    """
    y = series.values.astype(float)
    cusum_idx = _cusum_changepoint(y, h=cusum_h)
    pettitt_idx = _pettitt_changepoint(y, alpha=pettitt_alpha)

    agreed = (cusum_idx is not None) and (pettitt_idx is not None)

    refit_from = pettitt_idx if pettitt_idx is not None else cusum_idx

    return {
        "cusum_index": cusum_idx,
        "pettitt_index": pettitt_idx,
        "agreed": agreed,
        "recommended_refit_from": refit_from,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """
    Output of ``CampaignBenchmarker.fit()``.

    Key attributes
    --------------
    enriched_df : pd.DataFrame
        Original data plus all classification columns:

        M1 columns  : m1_baseline, m1_lower90, m1_upper90,
                      m1_lower95, m1_upper95, m1_class
        M2 columns  : m2_p10, m2_p50, m2_p90, m2_class
                      m2_p10_ci_low, m2_p10_ci_high  (if bootstrap_m2=True)
                      m2_p90_ci_low, m2_p90_ci_high  (if bootstrap_m2=True)
        M3 columns  : m3_lower, m3_upper, m3_rule, m3_class
        P1 columns  : cross_seg_zscore, cross_seg_class
                      ("Outperforming" / "Typical" / "Underperforming")
        Meta        : traffic_bin, bin_idx, fallback_level,
                      cp_cusum_idx, cp_pettitt_idx, cp_refit_from

    summary_compact : pd.DataFrame   — one row per segment × method
    summary_detail  : pd.DataFrame   — one row per current-period observation
    changepoint_summary : pd.DataFrame — one row per segment with CP findings
    """

    enriched_df: pd.DataFrame
    summary_compact: pd.DataFrame
    summary_detail: pd.DataFrame
    changepoint_summary: pd.DataFrame
    segment_cols: list[str]
    metric_col: str
    metric_type: str
    log_transformed_cols: dict[str, str]
    warnings: list[str]

    _benchmarker: Any = field(default=None, repr=False)

    def print_summary(self) -> None:
        """
        Print a structured benchmark summary to stdout.

        Sections printed:
        - Metric name and type
        - Auto log-transform notifications
        - Warnings emitted during fitting
        - Change-point detection alerts (segments where both CUSUM and Pettitt agreed)
        - Cross-segment z-score highlights (top and bottom 3 current-period observations)
        - M1 / M2 / M3 classification counts per segment (summary_compact table)
        """
        sep = "=" * 76
        print(sep)
        print("  ADSAT — BENCHMARK RESULTS")
        print(sep)
        print(f"  Metric         : {self.metric_col}  ({self.metric_type})")

        if self.log_transformed_cols:
            for col, orig in self.log_transformed_cols.items():
                print(f"  Log-transform  : {orig} → {col}  (auto-detected skew)")

        if self.warnings:
            print(f"\n  ⚠  {len(self.warnings)} warning(s):")
            for w in self.warnings:
                print(f"     • {w}")

        # Change-point highlights
        if len(self.changepoint_summary) > 0:
            triggered = self.changepoint_summary[self.changepoint_summary["agreed"]]
            if len(triggered) > 0:
                print(f"\n  🔴  Change-points detected in " f"{len(triggered)} segment(s):")
                for _, row in triggered.iterrows():
                    print(
                        f"     • {row['segment']}: "
                        f"CUSUM at position {row['cusum_index']}, "
                        f"Pettitt at position {row['pettitt_index']} — "
                        f"baseline refitted from position "
                        f"{row['refit_from']}"
                    )

        # Cross-segment z-score highlights
        edf = self.enriched_df
        if "cross_seg_class" in edf.columns:
            curr_mask = (
                self._benchmarker._is_current(edf)
                if self._benchmarker
                else pd.Series(True, index=edf.index)
            )
            curr = edf[curr_mask]
            out = curr[curr["cross_seg_class"] == "Outperforming"]
            und = curr[curr["cross_seg_class"] == "Underperforming"]
            if len(out) > 0 or len(und) > 0:
                print("\n  📊  Cross-segment z-score (current period):")
                if len(out) > 0:
                    top = curr.nlargest(3, "cross_seg_zscore")[["_segment_key", "cross_seg_zscore"]]
                    for _, r in top.iterrows():
                        print(
                            f"     ↑ {r['_segment_key']}: z={r['cross_seg_zscore']:+.2f}  (Outperforming)"
                        )
                if len(und) > 0:
                    bot = curr.nsmallest(3, "cross_seg_zscore")[
                        ["_segment_key", "cross_seg_zscore"]
                    ]
                    for _, r in bot.iterrows():
                        print(
                            f"     ↓ {r['_segment_key']}: z={r['cross_seg_zscore']:+.2f}  (Underperforming)"
                        )

        print(f"\n{'─'*76}")
        print(self.summary_compact.to_string(index=False))
        print(sep)

    def plot(
        self,
        segment: str | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Render all plots for a given segment (or the first if None).

        Parameters
        ----------
        segment : str, optional
        save_path : str, optional
        """
        if self._benchmarker is not None:
            self._benchmarker._plot_all(self, segment=segment, save_path=save_path)
        else:
            print("No benchmarker reference available for plotting.")


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class CampaignBenchmarker:
    """
    Classify campaign observations as Above / Within / Below a statistical
    baseline, with four complementary analytical lenses:

      M1  Trend + SE bands  (optionally confounder-adjusted + change-point aware)
      M2  Peer-bin quantile bands  (optionally bootstrapped)
      M3  Adaptive selector  (M1 default, M2 override when scale shifts)
      P1  Cross-segment z-score  (who is outperforming peers across segments?)

    Parameters
    ----------
    metric_col : str
        Success metric column (e.g. "conversion_rate", "roas").
    metric_type : str
        "proportion"  — bounded [0, 1]; quasi-binomial SE.
        "continuous"  — unbounded; std-based SE.
    date_col : str, optional
    volume_col : str, optional
        Denominator for proportion SE (impressions, visits, …).
    bin_col : str, optional
        Column for quartile-binning in M2 (can differ from volume_col).
    bin_col_log : bool or None
        True/False/None (auto-detect via skewness).
    n_bins : int
        Number of quantile bins.  Default 4.
    segment_cols : list of str, optional
        Columns defining independent groups.
    confounder_cols : list of str, optional
        External variable columns to include in the M1 OLS trend fit
        (e.g. ["market_cpi", "platform_cpm"]).  Rows with missing confounder
        values are dropped from fitting but still scored via time-only trend.
        Multicollinearity is checked automatically; columns with VIF > 5 trigger
        a warning recommending removal.
    current_period_start : str or Timestamp, optional
        Baseline fitted on data before this date; current period scored only.
    series_type : str or None
        "continuous" / "discrete" / None (auto-detect).
    use_seasonality : bool
        Week-of-year seasonal correction in M1/M3.  Default True.
    refit_after_changepoint : bool
        If True (default), automatically refit M1 using only data after the
        most recently detected structural break.  Requires date_col.
    cusum_h : float
        CUSUM sensitivity.  Higher = less sensitive.  Default 4.0.
    pettitt_alpha : float
        Pettitt test significance level.  Default 0.05.
    bootstrap_m2 : bool
        If True, compute 95 % bootstrap CI around M2 p10/p90 thresholds.
        Adds m2_p10_ci_low/high and m2_p90_ci_low/high columns.  Default False
        (opt-in because it is slower for large datasets).
    n_bootstrap : int
        Bootstrap iterations.  Default 500.
    neighbour_weeks : int
        Seasonal neighbourhood for M3.  Default 3.
    neighbour_weighting : str  "triangular" or "uniform".
    spike_filter : bool  Filter traffic outliers in M3.  Default True.
    spike_quantiles : tuple  Default (0.10, 0.90).
    distance_threshold : int  M3 bin-distance trigger.  Default 2.
    traffic_log_delta : float  M3 log-scale trigger.  Default 0.25.
    min_history_rows : int  Minimum rows for a segment to trust its own data.
    verbose : bool

    Examples
    --------
    >>> bm = CampaignBenchmarker(
    ...     metric_col           = "cvr",
    ...     metric_type          = "proportion",
    ...     date_col             = "week_start",
    ...     volume_col           = "impressions",
    ...     bin_col              = "spend",
    ...     confounder_cols      = ["market_cpi"],
    ...     segment_cols         = ["country", "channel"],
    ...     current_period_start = "2024-10-01",
    ...     bootstrap_m2         = True,
    ... )
    >>> result = bm.fit(df)
    >>> result.print_summary()
    >>> result.plot(segment="UK_paid")
    """

    def __init__(
        self,
        metric_col: str,
        metric_type: str,
        date_col: str | None = None,
        volume_col: str | None = None,
        bin_col: str | None = None,
        bin_col_log: bool | None = None,
        n_bins: int = 4,
        segment_cols: list[str] | None = None,
        confounder_cols: list[str] | None = None,
        current_period_start: str | pd.Timestamp | None = None,
        series_type: str | None = None,
        use_seasonality: bool = True,
        refit_after_changepoint: bool = True,
        cusum_h: float = _CUSUM_H,
        pettitt_alpha: float = _PETTITT_ALPHA,
        bootstrap_m2: bool = False,
        n_bootstrap: int = 500,
        neighbour_weeks: int = 3,
        neighbour_weighting: str = "triangular",
        spike_filter: bool = True,
        spike_quantiles: tuple[float, float] = (0.10, 0.90),
        distance_threshold: int = 2,
        traffic_log_delta: float = 0.25,
        min_history_rows: int = 8,
        verbose: bool = True,
    ):
        """
        Configure the benchmarker with metric definition, column names, and method parameters.

        No computation occurs here.  Call fit(df) to run the full pipeline.

        Parameters
        ----------
        metric_col              : Column containing the success metric.
        metric_type             : "proportion" (bounded 0-1) or "continuous".
        date_col                : Date column for trend fitting and seasonality.
        volume_col              : Denominator for proportion SE (impressions, visits, …).
        bin_col                 : Column for quartile peer-binning (M2/M3/P1).
        bin_col_log             : True/False/None — log-transform bin_col before binning.
        n_bins                  : Number of quantile bins (default 4 = quartiles).
        segment_cols            : Columns defining independent analytical groups.
        confounder_cols         : External regressors added to the M1 OLS trend.
                                  Rows with missing confounder values are dropped from
                                  the OLS fit but still scored via the time-only trend.
        current_period_start    : Observations on/after this date are "current period".
                                  Earlier rows are used for baseline fitting only.
        series_type             : "continuous"/"discrete"/None (auto-detect).
        use_seasonality         : Week-of-year seasonal correction in M1 and M3.
        refit_after_changepoint : Auto-refit M1 using only post-break history.
        cusum_h                 : CUSUM decision threshold in σ units (default 4.0).
        pettitt_alpha           : Significance level for the Pettitt test (default 0.05).
        bootstrap_m2            : Add 95% bootstrap CI around M2 p10/p90 thresholds.
        n_bootstrap             : Bootstrap resampling iterations (default 500).
        neighbour_weeks         : Seasonal neighbourhood half-width for M3 (default 3).
        neighbour_weighting     : "triangular" (default) or "uniform".
        spike_filter            : Filter traffic outliers in M3 typical-bin calculation.
        spike_quantiles         : (low, high) quantiles defining outliers (default 0.10/0.90).
        distance_threshold      : M3 bin-distance trigger for M2 override (default 2).
        traffic_log_delta       : M3 log-scale trigger for M2 override (default 0.25).
        min_history_rows        : Minimum historical rows for a segment to use its own data.
        verbose                 : Print warnings and progress messages.
        """
        if metric_type not in ("proportion", "continuous"):
            raise ValueError(
                f"metric_type must be 'proportion' or 'continuous', got '{metric_type}'."
            )
        if series_type is not None and series_type not in ("continuous", "discrete"):
            raise ValueError("series_type must be 'continuous', 'discrete', or None.")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}.")
        if min_history_rows < 1:
            raise ValueError(f"min_history_rows must be >= 1, got {min_history_rows}.")

        self.metric_col = metric_col
        self.metric_type = metric_type
        self.date_col = date_col
        self.volume_col = volume_col
        self.bin_col = bin_col
        self.bin_col_log = bin_col_log
        self.n_bins = n_bins
        self.segment_cols = list(segment_cols) if segment_cols else []
        self.confounder_cols = list(confounder_cols) if confounder_cols else []
        self.current_period_start = (
            pd.Timestamp(current_period_start) if current_period_start else None
        )
        self.series_type = series_type
        self.use_seasonality = use_seasonality
        self.refit_after_changepoint = refit_after_changepoint
        self.cusum_h = cusum_h
        self.pettitt_alpha = pettitt_alpha
        self.bootstrap_m2 = bootstrap_m2
        self.n_bootstrap = n_bootstrap
        self.neighbour_weeks = neighbour_weeks
        self.neighbour_weighting = neighbour_weighting
        self.spike_filter = spike_filter
        self.spike_quantiles = spike_quantiles
        self.distance_threshold = distance_threshold
        self.traffic_log_delta = traffic_log_delta
        self.min_history_rows = min_history_rows
        self.verbose = verbose

        # populated during fit()
        self._log_transformed: dict[str, str] = {}
        self._fit_warnings: list[str] = []
        self._cp_records: list[dict] = []

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> BenchmarkResult:
        """Run the full benchmarking pipeline."""
        self._log_transformed = {}
        self._fit_warnings = []
        self._cp_records = []

        df = df.copy()
        df = self._validate_and_prepare(df)

        # ── grouping keys ─────────────────────────────────────────────────────
        if self.segment_cols:
            df["_segment_key"] = df[self.segment_cols].astype(str).agg("_".join, axis=1)
        else:
            df["_segment_key"] = "__all__"
            self._warn(
                "No segment_cols specified — analysing all data as a single group. "
                "Consider passing segment_cols=['country', 'channel'] for more "
                "meaningful peer comparisons."
            )

        # ── per-segment pipeline ──────────────────────────────────────────────
        results: list[pd.DataFrame] = []
        for seg_key, seg_df in df.groupby("_segment_key", sort=False):
            results.append(self._process_segment(seg_df.copy(), str(seg_key)))

        enriched = pd.concat(results, ignore_index=True)

        # ── P1: cross-segment z-score (needs all segments in one pass) ────────
        enriched = self._run_cross_segment_zscore(enriched)

        # ── summaries ─────────────────────────────────────────────────────────
        cp_summary = self._build_changepoint_summary()
        summary_compact = self._build_summary_compact(enriched)
        summary_detail = self._build_summary_detail(enriched)

        return BenchmarkResult(
            enriched_df=enriched,
            summary_compact=summary_compact,
            summary_detail=summary_detail,
            changepoint_summary=cp_summary,
            segment_cols=self.segment_cols,
            metric_col=self.metric_col,
            metric_type=self.metric_type,
            log_transformed_cols=dict(self._log_transformed),
            warnings=list(self._fit_warnings),
            _benchmarker=self,
        )

    def undo_log_transform(self, col: str, result: BenchmarkResult) -> BenchmarkResult:
        """
        Reverse an automatic log-transform on bin_col and refit.

        Parameters
        ----------
        col : str   The *original* column name (before log transformation).
        result : BenchmarkResult

        Returns
        -------
        BenchmarkResult  — refitted without the log transform.
        """
        log_col = next((lc for lc, oc in self._log_transformed.items() if oc == col), None)
        if log_col is None:
            raise ValueError(
                f"Column '{col}' was not auto-log-transformed. "
                f"Transformed columns: {list(self._log_transformed.values())}"
            )
        if log_col != f"_log_{self.bin_col}":
            raise ValueError("Log-transform reversal is only supported for bin_col.")

        self._log("Re-fitting with log-transform disabled for bin_col.")
        self.bin_col_log = False
        del self._log_transformed[log_col]

        drop_cols = [
            c
            for c in result.enriched_df.columns
            if c.startswith("_")
            or c
            in (
                "m1_baseline",
                "m1_lower90",
                "m1_upper90",
                "m1_lower95",
                "m1_upper95",
                "m1_class",
                "m2_p10",
                "m2_p50",
                "m2_p90",
                "m2_class",
                "m2_p10_ci_low",
                "m2_p10_ci_high",
                "m2_p90_ci_low",
                "m2_p90_ci_high",
                "m3_lower",
                "m3_upper",
                "m3_rule",
                "m3_class",
                "cross_seg_zscore",
                "cross_seg_class",
                "trend",
                "resid",
                "r0",
                "expected",
                "p0",
                "bin_idx",
                "traffic_bin",
                "fallback_level",
                "t_days",
                "weekofyear",
                "cp_cusum_idx",
                "cp_pettitt_idx",
                "cp_refit_from",
            )
        ]
        return self.fit(result.enriched_df.drop(columns=drop_cols, errors="ignore"))

    # ── validation & preparation ──────────────────────────────────────────────

    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate required columns, parse dates, build t_days and weekofyear helpers,
        auto-log-transform the bin_col when skewness exceeds the threshold, and run
        VIF checks on confounder columns.

        Returns the prepared DataFrame (sorted by date if date_col is provided).
        Raises ValueError for any missing required column.
        """
        if self.metric_col not in df.columns:
            raise ValueError(f"Column '{self.metric_col}' not found in dataframe.")
        for col in self.segment_cols:
            if col not in df.columns:
                raise ValueError(f"Segment column '{col}' not found in dataframe.")
        for col in self.confounder_cols:
            if col not in df.columns:
                raise ValueError(f"Confounder column '{col}' not found in dataframe.")

        if self.date_col:
            if self.date_col not in df.columns:
                raise ValueError(f"date_col '{self.date_col}' not found.")
            df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
            df = df.sort_values(self.date_col).reset_index(drop=True)
            df["weekofyear"] = df[self.date_col].dt.isocalendar().week.astype(float)
            df["t_days"] = (df[self.date_col] - df[self.date_col].min()).dt.days.astype(float)

        # Auto log-transform bin_col
        if self.bin_col and self.bin_col in df.columns:
            do_log = self._should_log_transform(df[self.bin_col], self.bin_col)
            if do_log:
                log_col = f"_log_{self.bin_col}"
                df[log_col] = _safe_log(df[self.bin_col])
                self._log_transformed[log_col] = self.bin_col
                self._active_bin_col = log_col
            else:
                self._active_bin_col = self.bin_col
        else:
            self._active_bin_col = self.bin_col

        # VIF check on confounders
        if len(self.confounder_cols) >= 2:
            self._check_confounder_vif(df)

        return df

    def _should_log_transform(self, series: pd.Series, name: str) -> bool:
        """
        Decide whether to log-transform the named column before binning.

        Resolution order:
          1. bin_col_log=True  → always transform.
          2. bin_col_log=False → never transform.
          3. bin_col_log=None  → auto-detect: transform when |skewness| > _SKEW_THRESH.

        Emits a UserWarning when auto-transform is applied so the user can revert
        via undo_log_transform() if desired.
        """
        if self.bin_col_log is True:
            return True
        if self.bin_col_log is False:
            return False
        clean = series.dropna()
        if len(clean) < 4 or (clean <= 0).any():
            return False
        skew = float(scipy_stats.skew(clean))
        if abs(skew) > _SKEW_THRESH:
            self._warn(
                f"Column '{name}' has skewness={skew:.2f} (threshold |skew|>{_SKEW_THRESH}). "
                f"Auto-applying log-transform before binning. "
                f"Call undo_log_transform('{name}', result) to revert."
            )
            return True
        return False

    def _check_confounder_vif(self, df: pd.DataFrame) -> None:
        """Compute VIF for confounder columns and warn if any exceed the threshold."""
        X = df[self.confounder_cols].values.astype(float)
        mask = ~np.isnan(X).any(axis=1)
        if mask.sum() < len(self.confounder_cols) + 2:
            return
        vifs = _vif(X[mask])
        for col, v in zip(self.confounder_cols, vifs):
            if np.isfinite(v) and v > _VIF_THRESH:
                self._warn(
                    f"Confounder '{col}' has VIF={v:.1f} (threshold {_VIF_THRESH}), "
                    f"indicating high multicollinearity with other confounders. "
                    f"Consider removing '{col}' from confounder_cols to stabilise "
                    f"the M1 trend estimate."
                )

    # ── segment processing ────────────────────────────────────────────────────

    def _process_segment(self, df: pd.DataFrame, seg_key: str) -> pd.DataFrame:
        """
        Run the complete M1 → M2 → M3 → change-point pipeline for one segment.

        Steps:
          1. Split historical / current rows.
          2. Detect series type (continuous vs discrete bursts).
          3. Determine fallback level (segment vs global).
          4. Run change-point detection (CUSUM + Pettitt).
          5. Optionally trim historical data to post-change-point window.
          6. Fit M1 trend + SE bands.
          7. Fit M2 peer-bin quantile bands (when bin_col is available).
          8. Compute M3 adaptive selector.

        Returns the segment DataFrame with all classification columns appended.
        """
        # Historical / current split
        if self.current_period_start is not None and self.date_col:
            is_curr = df[self.date_col] >= self.current_period_start
            hist_df = df[~is_curr].copy()
        else:
            hist_df = df.copy()

        # Series type detection
        if self.series_type is not None:
            detected_type = self.series_type
        elif self.date_col:
            detected_type = _detect_series_type(df[self.date_col])
            if detected_type == "discrete":
                self._warn(
                    f"Segment '{seg_key}': auto-detected DISCRETE burst pattern "
                    f"(gap > {_GAP_MULTIPLE}× median interval). "
                    f"Trend will be fitted per burst. Set series_type='continuous' to override."
                )
        else:
            detected_type = "continuous"

        df["_series_type"] = detected_type

        # Fallback level
        fallback = self._determine_fallback_level(hist_df, seg_key)
        df["fallback_level"] = fallback

        # ── Change-point detection ────────────────────────────────────────────
        df = self._run_changepoint_detection(df, hist_df, seg_key)

        # Potentially trim hist_df to post-change-point data for M1
        hist_for_m1 = self._get_hist_for_m1(df, hist_df, seg_key)

        # ── M1 ────────────────────────────────────────────────────────────────
        df = self._run_m1(df, hist_for_m1, seg_key, detected_type)

        # ── M2 ────────────────────────────────────────────────────────────────
        if self._active_bin_col:
            df = self._run_m2(df, hist_df, seg_key)
        else:
            df["m2_p10"] = np.nan
            df["m2_p50"] = np.nan
            df["m2_p90"] = np.nan
            df["m2_class"] = "N/A (no bin_col)"
            if self.bootstrap_m2:
                df["m2_p10_ci_low"] = np.nan
                df["m2_p10_ci_high"] = np.nan
                df["m2_p90_ci_low"] = np.nan
                df["m2_p90_ci_high"] = np.nan
            df["traffic_bin"] = np.nan
            df["bin_idx"] = np.nan

        # ── M3 ────────────────────────────────────────────────────────────────
        df = self._run_m3(df, hist_df, seg_key)

        return df

    def _determine_fallback_level(self, hist_df: pd.DataFrame, seg_key: str) -> str:
        """
        Decide whether this segment has enough history to use its own baseline.

        Returns "segment" when len(hist_df) >= min_history_rows, otherwise "global".
        Emits a warning when falling back so the user knows which segments were affected.
        """
        n = len(hist_df)
        if n >= self.min_history_rows:
            return "segment"
        if n > 0:
            self._warn(
                f"Segment '{seg_key}' has only {n} historical rows "
                f"(min_history_rows={self.min_history_rows}). "
                f"Falling back to global baseline for SE bands."
            )
            return "global"
        self._warn(f"Segment '{seg_key}' has NO historical rows. Using global baseline.")
        return "global"

    # ── Change-point detection ────────────────────────────────────────────────

    def _run_changepoint_detection(
        self,
        df: pd.DataFrame,
        hist_df: pd.DataFrame,
        seg_key: str,
    ) -> pd.DataFrame:
        """
        Run CUSUM + Pettitt on the historical metric series.
        Stores results in df columns and in self._cp_records for the summary.
        """
        df["cp_cusum_idx"] = None
        df["cp_pettitt_idx"] = None
        df["cp_refit_from"] = None

        if not self.date_col or len(hist_df) < 6:
            self._cp_records.append(
                {
                    "segment": seg_key,
                    "cusum_index": None,
                    "pettitt_index": None,
                    "agreed": False,
                    "refit_from": None,
                }
            )
            return df

        # Sort historical by date, run on the metric
        hist_sorted = hist_df.sort_values(self.date_col)
        cp = detect_changepoints(
            hist_sorted[self.metric_col],
            cusum_h=self.cusum_h,
            pettitt_alpha=self.pettitt_alpha,
        )

        # Store summary record
        self._cp_records.append(
            {
                "segment": seg_key,
                "cusum_index": cp["cusum_index"],
                "pettitt_index": cp["pettitt_index"],
                "agreed": cp["agreed"],
                "refit_from": cp["recommended_refit_from"],
            }
        )

        # Fill scalar values into ALL rows of this segment
        df["cp_cusum_idx"] = cp["cusum_index"]
        df["cp_pettitt_idx"] = cp["pettitt_index"]
        df["cp_refit_from"] = cp["recommended_refit_from"]

        # Warn when change-point detected
        if cp["recommended_refit_from"] is not None:
            n_hist = len(hist_sorted)
            refit_pos = cp["recommended_refit_from"]
            pct_kept = round(100.0 * (n_hist - refit_pos) / n_hist, 1)
            refit_date = (
                str(hist_sorted[self.date_col].iloc[refit_pos].date())
                if refit_pos < n_hist
                else "end"
            )
            both = " (both CUSUM and Pettitt agree)" if cp["agreed"] else ""
            action = (
                f"Baseline refitted using post-break data ({pct_kept}% of history kept)."
                if self.refit_after_changepoint
                else "Set refit_after_changepoint=True to automatically refit from this point."
            )
            self._warn(
                f"Segment '{seg_key}': structural break detected{both} "
                f"around position {refit_pos} ({refit_date}). {action}"
            )

        return df

    def _get_hist_for_m1(
        self,
        df: pd.DataFrame,
        hist_df: pd.DataFrame,
        seg_key: str,
    ) -> pd.DataFrame:
        """Return the historical data to use for M1, trimmed to post-CP if applicable."""
        if not self.refit_after_changepoint or not self.date_col:
            return hist_df

        refit_pos = df["cp_refit_from"].iloc[0]
        if refit_pos is None:
            return hist_df

        refit_pos = int(refit_pos)
        hist_sorted = hist_df.sort_values(self.date_col)
        trimmed = hist_sorted.iloc[refit_pos:]

        if len(trimmed) < self.min_history_rows:
            self._warn(
                f"Segment '{seg_key}': post-change-point history has only "
                f"{len(trimmed)} rows (need {self.min_history_rows}). "
                f"Keeping full history for M1."
            )
            return hist_df

        return trimmed

    # ── M1: trend + SE bands ──────────────────────────────────────────────────

    def _run_m1(
        self,
        df: pd.DataFrame,
        hist_df: pd.DataFrame,
        seg_key: str,
        detected_type: str,
    ) -> pd.DataFrame:
        """
        Fit the M1 trend + seasonal SE bands for one segment.

        Steps:
          1. OLS trend on t_days (+ optional confounder columns).
          2. Week-of-year circular-smoothed seasonal correction on residuals.
          3. Compute expected value = trend + seasonal correction.
          4. Estimate quasi-dispersion phi from historical residuals.
          5. Build 90% and 95% confidence bands:
               - Proportions: quasi-binomial SE = sqrt(phi*p*(1-p)/n).
               - Continuous:  std-based SE = phi * residual_std (no volume denominator).
          6. Classify each row as Above (Act/Watch) / Within / Below (Watch/Act).

        Discrete-burst series are handled by fitting OLS independently per burst.
        """
        prop = self.metric_type == "proportion"
        metric = self.metric_col

        # Build confounder matrix (historical rows)
        X_conf_hist = self._get_confounder_matrix(hist_df) if self.confounder_cols else None
        X_conf_all = self._get_confounder_matrix(df) if self.confounder_cols else None

        # ── trend ─────────────────────────────────────────────────────────────
        if self.date_col and "t_days" in df.columns and len(hist_df) >= 2:
            if detected_type == "discrete":
                df = self._fit_trend_discrete(df, hist_df, metric, X_conf_all)
            else:
                t_h = hist_df["t_days"].values
                y_h = hist_df[metric].values
                _, coef = _ols(t_h, y_h, X_conf_hist)
                t_all = df["t_days"].values
                X_all_with_t = np.column_stack(
                    [np.ones_like(t_all), t_all] + ([X_conf_all] if X_conf_all is not None else [])
                )
                fitted = np.full(len(df), np.nan)
                for i in range(len(df)):
                    row = X_all_with_t[i]
                    if np.isnan(row).any():
                        fitted[i] = float(coef[0] + coef[1] * t_all[i])
                    else:
                        fitted[i] = float(coef @ row)
                df["trend"] = fitted
        else:
            mu = hist_df[metric].mean() if len(hist_df) > 0 else df[metric].mean()
            df["trend"] = float(mu) if pd.notna(mu) else 0.0

        df["trend"] = df["trend"].clip(lower=0.0, upper=1.0 if prop else None)
        df["resid"] = df[metric] - df["trend"]

        # ── seasonal correction on residuals ──────────────────────────────────
        if self.use_seasonality and "weekofyear" in df.columns:
            hist_resid = hist_df[metric].values - df.loc[hist_df.index, "trend"].values
            r0_raw = pd.Series(hist_resid, index=hist_df["weekofyear"].values)
            r0_raw = r0_raw.groupby(r0_raw.index).median()
            r0_sm = _circular_roll(r0_raw, window=3)
            df["r0"] = df["weekofyear"].map(r0_sm).fillna(0.0)
        else:
            df["r0"] = 0.0

        # ── expected value ────────────────────────────────────────────────────
        if prop:
            df["expected"] = (df["trend"] + df["r0"]).clip(0.0, 1.0)
        else:
            df["expected"] = df["trend"] + df["r0"]
        df["m1_baseline"] = df["expected"]

        # ── quasi-dispersion phi ──────────────────────────────────────────────
        if len(hist_df) > 0:
            exp_h = df.loc[hist_df.index, "expected"].values
            resid_h = hist_df[metric].values - exp_h
        else:
            exp_h = df["expected"].values
            resid_h = df[metric].values - exp_h

        vol_h = self._get_volume(hist_df if len(hist_df) > 0 else df)

        if prop:
            num = np.nansum(vol_h * resid_h**2)
            den = np.nansum(exp_h * (1 - exp_h) + 1e-12)
        else:
            exp_std = float(np.nanstd(resid_h)) if len(resid_h) > 1 else 1.0
            num = np.nansum(vol_h * resid_h**2)
            den = np.nansum(vol_h * exp_std**2 + 1e-12)

        phi = max(float(num / den) if den > 0 else 1.0, 1.0)

        # ── per-row SE ────────────────────────────────────────────────────────
        vol_all = self._get_volume(df)
        exp_all = df["expected"].values
        scale = 100.0 if prop else 1.0

        if prop:
            se = np.sqrt(phi * exp_all * (1 - exp_all) / np.where(vol_all > 0, vol_all, np.nan))
            df["m1_lower90"] = np.clip(exp_all - _Z90 * se, 0, 1) * scale
            df["m1_upper90"] = np.clip(exp_all + _Z90 * se, 0, 1) * scale
            df["m1_lower95"] = np.clip(exp_all - _Z95 * se, 0, 1) * scale
            df["m1_upper95"] = np.clip(exp_all + _Z95 * se, 0, 1) * scale
        else:
            resid_std = float(np.nanstd(resid_h)) if len(resid_h) > 1 else 1.0
            se = np.full(len(df), np.sqrt(phi) * resid_std)
            df["m1_lower90"] = (exp_all - _Z90 * se) * scale
            df["m1_upper90"] = (exp_all + _Z90 * se) * scale
            df["m1_lower95"] = (exp_all - _Z95 * se) * scale
            df["m1_upper95"] = (exp_all + _Z95 * se) * scale

        df["m1_baseline"] = exp_all * scale
        df["_metric_scaled"] = df[metric] * scale if prop else df[metric]

        def _m1_class(row):
            """
            Classify one row against its M1 confidence bands.

            Returns one of: "Above (Act)", "Above (Watch)", "Within",
            "Below (Watch)", "Below (Act)", or "N/A" when the metric value is missing.
            """
            c = row["_metric_scaled"]
            if pd.isna(c):
                return "N/A"
            if c < row["m1_lower95"]:
                return "Below (Act)"
            if c > row["m1_upper95"]:
                return "Above (Act)"
            if c < row["m1_lower90"]:
                return "Below (Watch)"
            if c > row["m1_upper90"]:
                return "Above (Watch)"
            return "Within"

        df["m1_class"] = df.apply(_m1_class, axis=1)
        return df

    def _fit_trend_discrete(
        self,
        df: pd.DataFrame,
        hist_df: pd.DataFrame,
        metric: str,
        X_conf_all: np.ndarray | None,
    ) -> pd.DataFrame:
        """Fit OLS trend independently per burst."""
        dates = df[self.date_col].sort_values()
        diffs = dates.diff()
        med_gap = diffs.median()
        burst_id = (diffs > _GAP_MULTIPLE * med_gap).cumsum()
        df["_burst"] = burst_id.values

        X_conf_hist = self._get_confounder_matrix(hist_df) if self.confounder_cols else None
        t_h = hist_df["t_days"].values
        y_h = hist_df[metric].values
        _, coef_global = _ols(t_h, y_h, X_conf_hist)

        idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}
        trends = np.full(len(df), np.nan)

        for bid, burst_df in df.groupby("_burst"):
            t_b = burst_df["t_days"].values
            y_b = burst_df[metric].values
            positions = [idx_to_pos[i] for i in burst_df.index]
            X_b = (
                X_conf_all[[idx_to_pos[i] for i in burst_df.index]]
                if X_conf_all is not None
                else None
            )

            if len(burst_df) >= 2:
                fitted_b, _ = _ols(t_b, y_b, X_b)
                for pos, val in zip(positions, fitted_b):
                    trends[pos] = val
            else:
                X_row = np.array([1.0, t_b[0]] + (list(X_b[0]) if X_b is not None else []))
                trends[positions[0]] = float(coef_global[: len(X_row)] @ X_row)

        df["trend"] = trends
        return df

    def _get_confounder_matrix(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Return the confounder columns as a float numpy matrix (n_rows × n_confounders),
        or None when no confounder_cols are configured or present in df.
        """
        if not self.confounder_cols:
            return None
        cols = [c for c in self.confounder_cols if c in df.columns]
        if not cols:
            return None
        return df[cols].values.astype(float)

    # ── M2: peer-bin quantile bands ───────────────────────────────────────────

    def _run_m2(
        self,
        df: pd.DataFrame,
        hist_df: pd.DataFrame,
        seg_key: str,
    ) -> pd.DataFrame:
        """
        Fit M2 peer-bin quantile bands for one segment.

        For each observation:
          1. Identify its spend/impression bin.
          2. Build a leave-one-out peer pool from the same historical bin.
             Falls back to the full historical pool when the bin has fewer than
             min_history_rows observations.
          3. Compute p10 / p50 / p90 of peer residuals (or raw metric when no trend).
          4. Classify as Above / Within / Below relative to p10/p90.
          5. Optionally bootstrap the p10/p90 to produce 95% CI columns
             (when bootstrap_m2=True).
        """
        bin_col = self._active_bin_col
        prop = self.metric_type == "proportion"
        scale = 100.0 if prop else 1.0
        n_bins = min(self.n_bins, df[bin_col].nunique())
        bl = [f"Q{i+1}" for i in range(n_bins)] if n_bins != 4 else _BIN_LABELS

        try:
            df["traffic_bin"] = pd.qcut(df[bin_col], q=n_bins, labels=bl, duplicates="drop")
        except Exception:
            df["traffic_bin"] = bl[0]
            self._warn(f"Segment '{seg_key}': could not create {n_bins} bins on '{bin_col}'.")

        cat_order = (
            [lbl for lbl in bl if lbl in df["traffic_bin"].cat.categories]
            if hasattr(df["traffic_bin"], "cat")
            else bl
        )
        df["traffic_bin"] = pd.Categorical(df["traffic_bin"], categories=cat_order, ordered=True)
        df["bin_idx"] = df["traffic_bin"].cat.codes + 1

        # Sync bins to hist_df
        hist_binned = hist_df.copy()
        for col in ("traffic_bin", "bin_idx", "resid", "trend"):
            if col in df.columns and col not in hist_binned.columns:
                hist_binned[col] = df.loc[df.index.intersection(hist_binned.index), col]

        has_resid = "resid" in df.columns
        rng = np.random.default_rng(42)

        p10_l, p50_l, p90_l, cls_l = [], [], [], []
        p10_ci_lo, p10_ci_hi = [], []
        p90_ci_lo, p90_ci_hi = [], []

        for i, row in df.iterrows():
            b = row["traffic_bin"]
            hist_pool = hist_binned if len(hist_binned) >= self.min_history_rows else df
            same = hist_pool[(hist_pool["traffic_bin"] == b) & (hist_pool.index != i)]
            if len(same) < self.min_history_rows:
                same = hist_pool[hist_pool.index != i]
            if len(same) == 0:
                same = df[df.index != i]

            trend_val = float(row.get("trend", 0.0))

            if has_resid and "resid" in same.columns:
                vals = same["resid"].dropna().values
                p10_r = np.quantile(vals, 0.10) if len(vals) else 0.0
                p50_r = np.quantile(vals, 0.50) if len(vals) else 0.0
                p90_r = np.quantile(vals, 0.90) if len(vals) else 0.0

                p10 = np.clip((trend_val + p10_r) * scale, 0, scale)
                p50 = np.clip((trend_val + p50_r) * scale, 0, scale)
                p90 = np.clip((trend_val + p90_r) * scale, 0, scale)

                row_resid = float(row.get("resid", row[self.metric_col] - trend_val))
                cls = "Below" if row_resid < p10_r else ("Above" if row_resid > p90_r else "Within")

                if self.bootstrap_m2 and len(vals) >= 4:
                    bs_p10 = [
                        np.quantile(rng.choice(vals, len(vals), replace=True), 0.10)
                        for _ in range(self.n_bootstrap)
                    ]
                    bs_p90 = [
                        np.quantile(rng.choice(vals, len(vals), replace=True), 0.90)
                        for _ in range(self.n_bootstrap)
                    ]
                    p10_ci_lo.append(
                        float(np.clip((trend_val + np.quantile(bs_p10, 0.025)) * scale, 0, scale))
                    )
                    p10_ci_hi.append(
                        float(np.clip((trend_val + np.quantile(bs_p10, 0.975)) * scale, 0, scale))
                    )
                    p90_ci_lo.append(
                        float(np.clip((trend_val + np.quantile(bs_p90, 0.025)) * scale, 0, scale))
                    )
                    p90_ci_hi.append(
                        float(np.clip((trend_val + np.quantile(bs_p90, 0.975)) * scale, 0, scale))
                    )
                else:
                    p10_ci_lo.append(np.nan)
                    p10_ci_hi.append(np.nan)
                    p90_ci_lo.append(np.nan)
                    p90_ci_hi.append(np.nan)
            else:
                m_col = "_metric_scaled" if "_metric_scaled" in df.columns else self.metric_col
                mv = (
                    same[m_col].dropna().values
                    if m_col in same.columns
                    else same[self.metric_col].dropna().values
                )
                p10 = float(np.quantile(mv, 0.10)) if len(mv) else np.nan
                p50 = float(np.quantile(mv, 0.50)) if len(mv) else np.nan
                p90 = float(np.quantile(mv, 0.90)) if len(mv) else np.nan
                rv = float(row.get("_metric_scaled", row[self.metric_col]))
                cls = "Below" if rv < p10 else ("Above" if rv > p90 else "Within")

                if self.bootstrap_m2 and len(mv) >= 4:
                    bs_p10 = [
                        np.quantile(rng.choice(mv, len(mv), replace=True), 0.10)
                        for _ in range(self.n_bootstrap)
                    ]
                    bs_p90 = [
                        np.quantile(rng.choice(mv, len(mv), replace=True), 0.90)
                        for _ in range(self.n_bootstrap)
                    ]
                    p10_ci_lo.append(float(np.quantile(bs_p10, 0.025)))
                    p10_ci_hi.append(float(np.quantile(bs_p10, 0.975)))
                    p90_ci_lo.append(float(np.quantile(bs_p90, 0.025)))
                    p90_ci_hi.append(float(np.quantile(bs_p90, 0.975)))
                else:
                    p10_ci_lo.append(np.nan)
                    p10_ci_hi.append(np.nan)
                    p90_ci_lo.append(np.nan)
                    p90_ci_hi.append(np.nan)

            p10_l.append(float(p10))
            p50_l.append(float(p50))
            p90_l.append(float(p90))
            cls_l.append(cls)

        df["m2_p10"] = p10_l
        df["m2_p50"] = p50_l
        df["m2_p90"] = p90_l
        df["m2_class"] = cls_l

        if self.bootstrap_m2:
            df["m2_p10_ci_low"] = p10_ci_lo
            df["m2_p10_ci_high"] = p10_ci_hi
            df["m2_p90_ci_low"] = p90_ci_lo
            df["m2_p90_ci_high"] = p90_ci_hi

        return df

    # ── M3: adaptive selector ─────────────────────────────────────────────────

    def _run_m3(
        self,
        df: pd.DataFrame,
        hist_df: pd.DataFrame,
        seg_key: str,
    ) -> pd.DataFrame:
        """
        Compute the M3 adaptive classification for one segment.

        Uses M1 as the default.  Overrides to M2 when either:
          - bin_distance >= distance_threshold (spend bin is unusually far from typical), or
          - |Δlog(bin_col)| >= traffic_log_delta (continuous scale trigger).

        "Typical bin" is the seasonal weighted mode when use_seasonality=True,
        otherwise the overall historical median bin.
        """
        has_m2 = self._active_bin_col is not None and "bin_idx" in df.columns
        has_date = self.date_col is not None and "weekofyear" in df.columns

        if not has_m2:
            df["m3_lower"] = df["m1_lower90"]
            df["m3_upper"] = df["m1_upper90"]
            df["m3_rule"] = "M1 (no bin_col)"
            df["m3_class"] = df["m1_class"].map(
                lambda x: (
                    "Above" if "Above" in str(x) else ("Below" if "Below" in str(x) else "Within")
                )
            )
            return df

        if self.use_seasonality and has_date:
            df = self._compute_typical_bin_seasonal(df, hist_df, seg_key)
        else:
            hist_binned = hist_df.copy()
            if "bin_idx" not in hist_binned.columns and "bin_idx" in df.columns:
                hist_binned["bin_idx"] = df.loc[df.index.intersection(hist_binned.index), "bin_idx"]
            global_med = (
                int(round(hist_binned["bin_idx"].median()))
                if len(hist_binned) > 0 and "bin_idx" in hist_binned.columns
                else 2
            )
            df["m3_typical_bin"] = global_med
            df["m3_typical_logv"] = (
                float(hist_df[self._active_bin_col].median())
                if len(hist_df) > 0
                else float(df[self._active_bin_col].median())
            )

        df["m3_bin_distance"] = (df["bin_idx"] - df["m3_typical_bin"]).abs()
        df["m3_logv_delta"] = (df[self._active_bin_col] - df["m3_typical_logv"]).abs()

        trigger_bin = df["m3_bin_distance"] >= self.distance_threshold
        trigger_cont = df["m3_logv_delta"] >= self.traffic_log_delta
        use_m2 = trigger_bin | trigger_cont

        def _rule_label(b, c):
            """
            Build a human-readable label explaining why M3 chose M1 or M2 for this row.

            Describes which trigger fired (bin-distance, continuous log-delta, or default)
            and whether seasonality was in use.
            """
            if b and c:
                return f"Traffic-shift override: bin_dist≥{self.distance_threshold} & |Δbin_col|≥{self.traffic_log_delta:.2f}"
            if b:
                return f"Traffic-shift override: bin_dist≥{self.distance_threshold}"
            if c:
                return f"Traffic-shift override: |Δbin_col|≥{self.traffic_log_delta:.2f}"
            return "Seasonal default" if self.use_seasonality else "M1 default"

        df["m3_lower"] = pd.to_numeric(
            np.where(use_m2, df["m2_p10"], df["m1_lower90"]), errors="coerce"
        )
        df["m3_upper"] = pd.to_numeric(
            np.where(use_m2, df["m2_p90"], df["m1_upper90"]), errors="coerce"
        )
        df["m3_rule"] = [_rule_label(b, c) for b, c in zip(trigger_bin, trigger_cont)]
        df["m3_class"] = np.where(
            df["_metric_scaled"] < df["m3_lower"],
            "Below",
            np.where(df["_metric_scaled"] > df["m3_upper"], "Above", "Within"),
        )
        return df

    def _compute_typical_bin_seasonal(
        self,
        df: pd.DataFrame,
        hist_df: pd.DataFrame,
        seg_key: str,
    ) -> pd.DataFrame:
        """
        For each observation, compute the seasonally-typical spend bin and scale value
        from historical data in the same week-of-year neighbourhood.

        Uses triangular or uniform weights by week distance, spike filtering on extreme
        traffic values, and partial shrinkage toward the global median when the local
        neighbourhood has fewer than min_history_rows observations.
        """
        bin_col = self._active_bin_col
        hist_df = hist_df.copy()
        if "bin_idx" not in hist_df.columns and "bin_idx" in df.columns:
            hist_df["bin_idx"] = df.loc[df.index.intersection(hist_df.index), "bin_idx"]

        global_med_bin = (
            int(round(hist_df["bin_idx"].median()))
            if len(hist_df) > 0 and "bin_idx" in hist_df.columns
            else 2
        )
        global_med_logv = (
            float(hist_df[bin_col].median()) if len(hist_df) > 0 else float(df[bin_col].median())
        )

        typical_bin, typical_logv = [], []

        for _, row in df.iterrows():
            woy = int(row["weekofyear"])
            cand = (
                hist_df[hist_df[self.date_col] < row[self.date_col]].copy()
                if self.date_col
                else hist_df.copy()
            )
            cand = cand[
                cand["weekofyear"].apply(
                    lambda w: _woy_circ_dist(int(w), woy) <= self.neighbour_weeks
                )
            ].copy()

            if len(cand) == 0:
                typical_bin.append(global_med_bin)
                typical_logv.append(global_med_logv)
                continue

            if self.spike_filter:
                ql = cand[bin_col].quantile(self.spike_quantiles[0])
                qh = cand[bin_col].quantile(self.spike_quantiles[1])
                cand = cand[(cand[bin_col] >= ql) & (cand[bin_col] <= qh)].copy()

            if len(cand) == 0:
                typical_bin.append(global_med_bin)
                typical_logv.append(global_med_logv)
                continue

            dists = cand["weekofyear"].apply(lambda w: _woy_circ_dist(int(w), woy)).to_numpy()
            weights = (
                (self.neighbour_weeks - dists + 1).clip(min=1).astype(float)
                if self.neighbour_weighting.lower() == "triangular"
                else np.ones_like(dists, dtype=float)
            )

            valid_bins = tuple(range(1, self.n_bins + 1))
            local_mode = _weighted_mode_bin(cand["bin_idx"].to_numpy(), weights, valid_bins)
            local_logv = _weighted_median(cand[bin_col].to_numpy(), weights)

            alpha = min(1.0, len(cand) / float(self.min_history_rows))
            shrink_bin = int(round(alpha * local_mode + (1 - alpha) * global_med_bin))
            shrink_logv = alpha * local_logv + (1 - alpha) * global_med_logv

            typical_bin.append(int(shrink_bin))
            typical_logv.append(float(shrink_logv))

        df["m3_typical_bin"] = typical_bin
        df["m3_typical_logv"] = typical_logv
        return df

    # ── P1: cross-segment z-score ─────────────────────────────────────────────

    def _run_cross_segment_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        P1 — Cross-segment z-score benchmarking.

        For each observation, compute a z-score relative to all *historical*
        observations in the same spend/impression bin across ALL segments.
        This answers: "controlling for scale, which campaign is outperforming
        its peers right now?"

        Adds columns:
          cross_seg_zscore : float  — standardised score (positive = above peers)
          cross_seg_class  : str    — "Outperforming" / "Typical" / "Underperforming"
        """
        if not self._active_bin_col or "bin_idx" not in df.columns:
            df["cross_seg_zscore"] = np.nan
            df["cross_seg_class"] = "N/A (no bin_col)"
            return df

        is_hist = ~self._is_current(df)
        hist_df = df[is_hist]
        metric = "_metric_scaled" if "_metric_scaled" in df.columns else self.metric_col

        zscores = np.full(len(df), np.nan)
        classes = ["N/A"] * len(df)
        idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

        # Pre-compute global bin statistics from historical data
        bin_stats: dict[Any, tuple[float, float]] = {}
        for bin_val in df["bin_idx"].dropna().unique():
            bin_hist = hist_df[hist_df["bin_idx"] == bin_val][metric].dropna()
            if len(bin_hist) >= 3:
                mu = float(bin_hist.mean())
                sig = float(bin_hist.std())
                bin_stats[bin_val] = (mu, sig)
            else:
                # Fall back to global historical statistics
                all_hist = hist_df[metric].dropna()
                mu = float(all_hist.mean()) if len(all_hist) >= 3 else np.nan
                sig = float(all_hist.std()) if len(all_hist) >= 3 else np.nan
                bin_stats[bin_val] = (mu, sig)

        for i, row in df.iterrows():
            pos = idx_to_pos[i]
            bin_val = row.get("bin_idx")
            val = float(row.get(metric, np.nan))

            if pd.isna(bin_val) or pd.isna(val):
                continue

            mu, sig = bin_stats.get(bin_val, (np.nan, np.nan))
            if np.isnan(mu) or np.isnan(sig) or sig == 0:
                continue

            z = (val - mu) / sig
            zscores[pos] = round(float(z), 4)

            # Thresholds: |z| > 1.28 ≈ top/bottom 10%, |z| > 1.96 ≈ top/bottom 5%
            if z > 1.96:
                classes[pos] = "Outperforming"
            elif z < -1.96:
                classes[pos] = "Underperforming"
            elif z > 1.28:
                classes[pos] = "Outperforming"
            elif z < -1.28:
                classes[pos] = "Underperforming"
            else:
                classes[pos] = "Typical"

        df["cross_seg_zscore"] = zscores
        df["cross_seg_class"] = classes
        return df

    # ── summaries ─────────────────────────────────────────────────────────────

    def _is_current(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a boolean Series marking rows that belong to the current scoring period.

        All rows are considered current when current_period_start is None.
        """
        if self.current_period_start is not None and self.date_col:
            return df[self.date_col] >= self.current_period_start
        return pd.Series(True, index=df.index)

    def _build_changepoint_summary(self) -> pd.DataFrame:
        """
        Assemble a DataFrame with one row per segment summarising change-point findings:
        segment name, CUSUM index, Pettitt index, whether both tests agreed,
        and the recommended refit-from position.
        """
        rows = []
        for r in self._cp_records:
            rows.append(
                {
                    "segment": r["segment"],
                    "cusum_index": r["cusum_index"],
                    "pettitt_index": r["pettitt_index"],
                    "agreed": r["agreed"],
                    "refit_from": r["refit_from"],
                }
            )
        return pd.DataFrame(rows)

    def _build_summary_compact(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the compact summary table: one row per segment × method (M1, M2, M3)
        for current-period observations, showing Above/Within/Below counts,
        fallback level, M3 override percentage, and whether a change-point was detected.
        """
        curr = df[self._is_current(df)]
        rows = []
        for seg_key, seg in curr.groupby("_segment_key"):
            n = len(seg)
            for method, col in [("M1", "m1_class"), ("M2", "m2_class"), ("M3", "m3_class")]:
                if col not in seg.columns:
                    continue
                vc = seg[col].value_counts()
                rows.append(
                    {
                        "segment": seg_key,
                        "method": method,
                        "n_observations": n,
                        "n_above": int(
                            sum(vc.get(k, 0) for k in ["Above", "Above (Act)", "Above (Watch)"])
                        ),
                        "n_within": int(vc.get("Within", 0)),
                        "n_below": int(
                            sum(vc.get(k, 0) for k in ["Below", "Below (Act)", "Below (Watch)"])
                        ),
                        "fallback_level": (
                            seg["fallback_level"].iloc[0]
                            if "fallback_level" in seg.columns
                            else "segment"
                        ),
                        "m3_overrides_pct": round(
                            (
                                seg["m3_rule"].str.contains("override").sum() / n * 100
                                if method == "M3" and "m3_rule" in seg.columns
                                else 0
                            ),
                            1,
                        ),
                        "cp_detected": (
                            bool(seg["cp_refit_from"].iloc[0] is not None)
                            if method == "M1" and "cp_refit_from" in seg.columns
                            else False
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def _build_summary_detail(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the detailed summary table: one row per current-period observation
        with all classification columns, band values, z-scores, and a plain-English
        explanation string that describes why the observation was classified as it was.
        """
        curr = df[self._is_current(df)].copy()
        keep = ["_segment_key", self.metric_col]
        if self.date_col:
            keep.append(self.date_col)
        for col in (
            "m1_class",
            "m1_baseline",
            "m1_lower90",
            "m1_upper90",
            "m2_class",
            "m2_p10",
            "m2_p50",
            "m2_p90",
            "m2_p10_ci_low",
            "m2_p10_ci_high",
            "m2_p90_ci_low",
            "m2_p90_ci_high",
            "m3_class",
            "m3_lower",
            "m3_upper",
            "m3_rule",
            "cross_seg_zscore",
            "cross_seg_class",
            "traffic_bin",
            "bin_idx",
            "fallback_level",
            "cp_cusum_idx",
            "cp_pettitt_idx",
            "cp_refit_from",
        ):
            if col in curr.columns:
                keep.append(col)

        detail = curr[[c for c in keep if c in curr.columns]].copy()

        def _explain(row):
            """
            Build the plain-English explanation string for one summary_detail row.

            Includes: M3 classification, rule applied, band values, cross-segment z-score,
            fallback level note (when history was thin), and change-point warning if relevant.
            """
            cls = row.get("m3_class", "N/A")
            rule = row.get("m3_rule", "")
            level = row.get("fallback_level", "segment")
            bnd = f"[{row.get('m3_lower','?'):.2f}, {row.get('m3_upper','?'):.2f}]"
            zscore = row.get("cross_seg_zscore", np.nan)
            zcls = row.get("cross_seg_class", "")
            base = f"Classified as '{cls}' using {rule}. Band: {bnd}."
            if not np.isnan(zscore if isinstance(zscore, float) else np.nan):
                base += f" Cross-segment z-score={zscore:+.2f} ({zcls})."
            if level != "segment":
                base += f" (Fitted on {level}-level data due to thin history.)"
            cp = row.get("cp_refit_from")
            if cp is not None:
                base += f" ⚠ Change-point detected; baseline refitted from position {cp}."
            return base

        detail["explanation"] = detail.apply(_explain, axis=1)
        return detail.rename(columns={"_segment_key": "segment"})

    # ── volume helper ─────────────────────────────────────────────────────────

    def _get_volume(self, df: pd.DataFrame) -> np.ndarray:
        """
        Return a float array of volume (denominator) values for SE computation.

        Uses volume_col when available; falls back to 1.0 per row (treating each
        observation as a single unit) when volume_col is absent or not configured.
        """
        if self.volume_col and self.volume_col in df.columns:
            return df[self.volume_col].fillna(1.0).clip(lower=1.0).values
        return np.ones(len(df), dtype=float)

    # ── plots ─────────────────────────────────────────────────────────────────

    def _plot_all(
        self,
        result: BenchmarkResult,
        segment: str | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Render the full three-panel diagnostic figure for one segment and display it.

        Panels:
          Top   (wide): time series with M3 bands, change-point marker, M2 bootstrap ribbon.
          Bottom-left:  metric distribution by spend bin.
          Bottom-right: M3 classification heatmap across segments × calendar months.

        Parameters
        ----------
        result   : BenchmarkResult from fit().
        segment  : combined segment key to plot (defaults to first available).
        save_path: if provided, saves the figure to this path before displaying.
        """
        df = result.enriched_df
        seg_keys = df["_segment_key"].unique()
        seg_key = seg_keys[0] if segment is None else segment
        if seg_key not in seg_keys:
            raise ValueError(f"Segment '{seg_key}' not found. Available: {list(seg_keys)}")

        seg_df = df[df["_segment_key"] == seg_key].copy()
        is_curr = self._is_current(seg_df)

        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            f"Benchmark Analysis — {seg_key}  |  {self.metric_col} ({self.metric_type})",
            fontsize=13,
            fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
        ax_ts = fig.add_subplot(gs[0, :])
        ax_dist = fig.add_subplot(gs[1, 0])
        ax_heat = fig.add_subplot(gs[1, 1])

        self._plot_timeseries(ax_ts, seg_df, seg_key, is_curr, result)
        self._plot_distribution(ax_dist, seg_df, seg_key)
        self._plot_heatmap(ax_heat, result)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def _plot_timeseries(
        self,
        ax: plt.Axes,
        seg_df: pd.DataFrame,
        seg_key: str,
        is_curr: pd.Series,
        result: BenchmarkResult,
    ) -> None:
        """
        Draw the time series panel: metric over time with M1/M3 confidence bands,
        optional M2 bootstrap CI ribbon, baseline trend line, change-point marker,
        current-period start marker, historical scatter, and current-period scatter
        coloured by M3 classification.
        """
        prop = self.metric_type == "proportion"
        scale = 100.0 if prop else 1.0
        has_date = self.date_col and self.date_col in seg_df.columns
        x = seg_df[self.date_col] if has_date else np.arange(len(seg_df))
        metric_vals = seg_df[self.metric_col] * scale if prop else seg_df[self.metric_col]

        # Bands
        if "m3_lower" in seg_df.columns:
            ax.fill_between(
                x,
                seg_df["m3_lower"],
                seg_df["m3_upper"],
                alpha=0.18,
                color=_GREEN,
                label="M3 band (90%)",
            )
        if "m1_lower95" in seg_df.columns:
            ax.fill_between(
                x,
                seg_df["m1_lower95"],
                seg_df["m1_upper95"],
                alpha=0.08,
                color=_BLUE,
                label="M1 band (95%)",
            )

        # M2 bootstrap CI ribbon (if available)
        if self.bootstrap_m2 and "m2_p10_ci_low" in seg_df.columns:
            ax.fill_between(
                x,
                seg_df["m2_p10_ci_low"],
                seg_df["m2_p10_ci_high"],
                alpha=0.12,
                color=_TEAL,
                label="M2 p10 95% CI",
            )
            ax.fill_between(
                x,
                seg_df["m2_p90_ci_low"],
                seg_df["m2_p90_ci_high"],
                alpha=0.12,
                color=_TEAL,
                label="M2 p90 95% CI",
            )

        # Baseline
        if "m1_baseline" in seg_df.columns:
            ax.plot(x, seg_df["m1_baseline"], color=_GREY, lw=1.5, ls="--", label="M1 baseline")

        # Change-point marker
        cp_pos = seg_df["cp_refit_from"].iloc[0] if "cp_refit_from" in seg_df.columns else None
        if cp_pos is not None and has_date:
            cp_dates = seg_df[self.date_col].sort_values()
            if int(cp_pos) < len(cp_dates):
                cp_date = cp_dates.iloc[int(cp_pos)]
                ax.axvline(cp_date, color=_RED, lw=1.5, ls="--", alpha=0.7, label="Change-point")

        # Current period start
        if has_date and self.current_period_start:
            ax.axvline(
                pd.Timestamp(self.current_period_start),
                color=_ORANGE,
                lw=1.5,
                ls=":",
                label="Current period start",
            )

        # Points
        ax.scatter(
            x[~is_curr],
            metric_vals[~is_curr],
            s=22,
            color=_GREY,
            alpha=0.5,
            zorder=3,
            label="Historical",
        )

        if "m3_class" in seg_df.columns:
            for cls, colour in _CLASS_COLOURS.items():
                mask = is_curr & (seg_df["m3_class"] == cls)
                if mask.any():
                    ax.scatter(
                        x[mask],
                        metric_vals[mask],
                        s=55,
                        color=colour,
                        zorder=5,
                        label=f"Current: {cls}",
                    )

        ylabel = f"{self.metric_col} (%)" if prop else self.metric_col
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"Time Series  |  {seg_key}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="best", ncol=2)
        if has_date:
            ax.tick_params(axis="x", rotation=20, labelsize=8)

    def _plot_distribution(
        self,
        ax: plt.Axes,
        seg_df: pd.DataFrame,
        seg_key: str,
    ) -> None:
        """
        Draw per-bin metric histograms for one segment.

        When traffic_bin is available, renders one overlapping histogram per bin
        with a separate colour.  Falls back to a single histogram when no binning
        has been applied.
        """
        prop = self.metric_type == "proportion"
        scale = 100.0 if prop else 1.0
        metric_vals = seg_df[self.metric_col] * scale if prop else seg_df[self.metric_col]

        if "traffic_bin" in seg_df.columns and seg_df["traffic_bin"].notna().any():
            bins = seg_df["traffic_bin"].dropna().unique()
            for i, b in enumerate(sorted(bins, key=str)):
                sub = metric_vals[seg_df["traffic_bin"] == b].dropna()
                if len(sub) == 0:
                    continue
                color = ([_BLUE, _GREEN, _ORANGE, _RED, _PURPLE] * 4)[i]
                ax.hist(sub, bins=12, alpha=0.5, color=color, label=str(b), density=True)
            ax.set_title(
                f"Metric Distribution by Bin  |  {seg_key}", fontsize=10, fontweight="bold"
            )
            ax.legend(fontsize=7)
        else:
            ax.hist(metric_vals.dropna(), bins=15, color=_BLUE, alpha=0.75)
            ax.set_title(f"Metric Distribution  |  {seg_key}", fontsize=10, fontweight="bold")

        xlabel = f"{self.metric_col} (%)" if prop else self.metric_col
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)

    def _plot_heatmap(
        self,
        ax: plt.Axes,
        result: BenchmarkResult,
    ) -> None:
        """
        Draw the summary heatmap: segments (rows) × calendar months (columns),
        coloured by average M3 classification score (green = Above, red = Below).

        Requires date_col to be configured; renders a placeholder when date is absent.
        """
        df = result.enriched_df.copy()
        if not self.date_col or self.date_col not in df.columns:
            ax.text(
                0.5,
                0.5,
                "No date column — heatmap not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=_GREY,
                fontsize=9,
            )
            ax.set_title("Summary Heatmap", fontsize=10, fontweight="bold")
            return

        df["_period"] = df[self.date_col].dt.to_period("M").astype(str)
        periods = sorted(df["_period"].unique())
        segments = sorted(df["_segment_key"].unique())
        class_map = {
            "Above": 2,
            "Above (Act)": 2,
            "Above (Watch)": 1.5,
            "Within": 0,
            "Below (Watch)": -1.5,
            "Below": -2,
            "Below (Act)": -2,
        }

        mat = np.full((len(segments), len(periods)), np.nan)
        for si, seg in enumerate(segments):
            for pi, per in enumerate(periods):
                sub = df[(df["_segment_key"] == seg) & (df["_period"] == per)]
                if len(sub) and "m3_class" in sub.columns:
                    vals = sub["m3_class"].map(class_map).dropna()
                    if len(vals):
                        mat[si, pi] = vals.mean()

        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2, interpolation="nearest")
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([p[-5:] for p in periods], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(segments)))
        ax.set_yticklabels(segments, fontsize=7)
        ax.set_title(
            "M3 Classification Heatmap\n(green=Above, red=Below)", fontsize=10, fontweight="bold"
        )
        plt.colorbar(im, ax=ax, shrink=0.8, label="Above(+) / Below(-)")

    # ── logging ───────────────────────────────────────────────────────────────

    def _warn(self, msg: str) -> None:
        """
        Append msg to the internal warnings list and, when verbose=True,
        emit it as a UserWarning so it appears in the user's console.
        """
        self._fit_warnings.append(msg)
        if self.verbose:
            warnings.warn(f"[CampaignBenchmarker] {msg}", UserWarning, stacklevel=3)

    def _log(self, msg: str) -> None:
        """
        Print msg to stdout when verbose=True.  Used for non-warning informational messages
        such as "Re-fitting with log-transform disabled".
        """
        if self.verbose:
            print(f"[CampaignBenchmarker] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def benchmark_campaigns(
    df: pd.DataFrame,
    metric_col: str,
    metric_type: str,
    date_col: str | None = None,
    volume_col: str | None = None,
    bin_col: str | None = None,
    segment_cols: list[str] | None = None,
    confounder_cols: list[str] | None = None,
    current_period_start: str | None = None,
    use_seasonality: bool = True,
    refit_after_changepoint: bool = True,
    bootstrap_m2: bool = False,
    n_bootstrap: int = 500,
    min_history_rows: int = 8,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    One-function benchmarking shortcut.

    Parameters
    ----------
    df : pd.DataFrame
    metric_col : str
    metric_type : str  — "proportion" or "continuous"
    date_col : str, optional
    volume_col : str, optional  — denominator for proportion SE
    bin_col : str, optional     — variable for quartile binning (M2/M3/P1)
    segment_cols : list, optional
    confounder_cols : list, optional — external variables for M1 trend adjustment
    current_period_start : str, optional
    use_seasonality : bool
    refit_after_changepoint : bool  — auto-refit M1 after detected structural break
    bootstrap_m2 : bool             — add 95% CI around M2 peer thresholds
    n_bootstrap : int               — bootstrap iterations (default 500)
    min_history_rows : int
    verbose : bool

    Returns
    -------
    BenchmarkResult

    Examples
    --------
    >>> result = benchmark_campaigns(
    ...     df,
    ...     metric_col           = "conversion_rate",
    ...     metric_type          = "proportion",
    ...     date_col             = "week_start",
    ...     volume_col           = "impressions",
    ...     bin_col              = "spend",
    ...     segment_cols         = ["country"],
    ...     confounder_cols      = ["market_cpi"],
    ...     current_period_start = "2024-10-01",
    ...     bootstrap_m2         = True,
    ... )
    >>> result.print_summary()
    """
    return CampaignBenchmarker(
        metric_col=metric_col,
        metric_type=metric_type,
        date_col=date_col,
        volume_col=volume_col,
        bin_col=bin_col,
        segment_cols=segment_cols,
        confounder_cols=confounder_cols,
        current_period_start=current_period_start,
        use_seasonality=use_seasonality,
        refit_after_changepoint=refit_after_changepoint,
        bootstrap_m2=bootstrap_m2,
        n_bootstrap=n_bootstrap,
        min_history_rows=min_history_rows,
        verbose=verbose,
    ).fit(df)
