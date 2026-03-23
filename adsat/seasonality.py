"""
adsat.seasonality
=================
Seasonality decomposition and adjustment for advertising time-series data.

Advertising campaigns exhibit strong seasonal patterns — weekly day-of-week
effects, monthly periodicity, and annual cycles (Black Friday, Christmas, etc.).
Fitting saturation models on raw data that contains these patterns produces
biased saturation estimates, because the model confuses the seasonal surge with
a genuine response to increased spend.

This module:
  1. Detects and decomposes seasonality from your campaign metric time series.
  2. Produces a seasonally-adjusted series for use in saturation modeling.
  3. Lets you add back the seasonal component to predictions so that
     forecasts are still expressed in the original "seasonal" space.
  4. Visualises the decomposition clearly.

Method
------
Classical additive / multiplicative decomposition using a centred moving
average (CMA) to estimate trend, iterative seasonal averaging for seasonal
factors, and a cleaning step that removes outliers from the seasonal averages.
This approach requires no external dependencies (pure NumPy / SciPy).

For longer series (≥ 2 full periods) it automatically uses an iterative
approach that refines trend and seasonal components together.

Key classes & functions
-----------------------
SeasonalDecomposer       – main class
SeasonalDecomposition    – result dataclass
adjust_for_seasonality() – one-liner shortcut

Typical workflow
----------------
>>> import pandas as pd
>>> from adsat.seasonality import SeasonalDecomposer
>>>
>>> df = pd.read_csv("weekly_campaign.csv")
>>> decomp = SeasonalDecomposer(period=52, model='additive')
>>> result = decomp.fit(df['conversions'])
>>> result.print_summary()
>>> decomp.plot(result)
>>>
>>> # Use the adjusted series for saturation modeling
>>> df['conversions_adj'] = result.adjusted.values
>>> # ... then fit SaturationModeler on df with 'conversions_adj'
>>>
>>> # Later: add seasonality back to model predictions
>>> raw_preds = decomp.inverse_adjust(predictions, result)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import periodogram

# ── colour palette ────────────────────────────────────────────────────────────
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_GREEN = "#3BB273"
_RED = "#E84855"
_GREY = "#6C757D"
_PURPLE = "#7B2D8B"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SeasonalDecomposition:
    """
    Result of a seasonal decomposition.

    Attributes
    ----------
    model : str
        'additive' or 'multiplicative'.
    period : int
        Seasonal period used (e.g. 52 for weekly data with yearly seasonality).
    original : pd.Series
        The input series (with original index).
    trend : pd.Series
        Smoothed trend component.
    seasonal : pd.Series
        Seasonal component (repeating pattern).
    residual : pd.Series
        What is left after removing trend and seasonal.
    adjusted : pd.Series
        Seasonally-adjusted series (original minus / divided by seasonal).
    seasonal_factors : np.ndarray
        One factor per position in the period (length = period).
    period_labels : list of str
        Human-readable labels for each position in the period.
    strength_of_seasonality : float
        Variance of seasonal component / variance of original (0–1).
        Higher means seasonality explains more of the variation.
    dominant_period : int or None
        Period detected by spectral analysis. None if detection failed.
    n : int
        Number of observations.
    """

    model: str
    period: int
    original: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    residual: pd.Series
    adjusted: pd.Series
    seasonal_factors: np.ndarray
    period_labels: list[str]
    strength_of_seasonality: float
    dominant_period: int | None
    n: int

    def print_summary(self) -> None:
        """
        Print a structured decomposition summary to stdout: model type, period,
        strength of seasonality, and the top 5 highest/lowest seasonal factors.
        """
        sep = "=" * 60
        print(sep)
        print("  ADSAT – SEASONAL DECOMPOSITION SUMMARY")
        print(sep)
        print(f"  Model              : {self.model}")
        print(f"  Period             : {self.period}")
        print(f"  Observations       : {self.n}")
        if self.dominant_period:
            print(f"  Detected period    : {self.dominant_period}")
        print(f"  Seasonal strength  : {self.strength_of_seasonality:.1%}")
        print()
        print("  Seasonal factors (mean ± std per period position):")
        for i, (label, factor) in enumerate(zip(self.period_labels, self.seasonal_factors)):
            bar_len = int(abs(factor) / max(abs(self.seasonal_factors).max(), 1e-9) * 20)
            sign = "+" if factor >= 0 else "-"
            bar = sign + "█" * bar_len
            print(f"    {label:>8s}  {factor:+8.3f}  {bar}")
        print(sep)

    def as_dataframe(self) -> pd.DataFrame:
        """Return all components in a single tidy DataFrame."""
        return pd.DataFrame(
            {
                "original": self.original.values,
                "trend": self.trend.values,
                "seasonal": self.seasonal.values,
                "residual": self.residual.values,
                "adjusted": self.adjusted.values,
            },
            index=self.original.index,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class SeasonalDecomposer:
    """
    Decompose advertising time-series data into trend, seasonal, and residual
    components and produce a seasonally-adjusted series.

    Parameters
    ----------
    period : int or 'auto'
        Seasonal period length in data points.
        Common values:
          * 7   — weekly data with a daily-of-week effect
          * 4   — monthly data with quarterly effect
          * 12  — monthly data with annual effect
          * 52  — weekly data with annual effect
        Use 'auto' to detect automatically from the series spectrum.
    model : str
        'additive'       : original = trend + seasonal + residual
        'multiplicative' : original = trend × seasonal × residual
        Use additive for series that do not grow over time.
        Use multiplicative when the seasonal amplitude grows with the level.
    trend_smoother : str
        'cma'     — centred moving average (default, robust)
        'lowess'  — locally-weighted smoothing (smoother but slower)
    min_periods_for_trend : int
        Minimum number of complete periods required to estimate trend.
        Default 2.
    verbose : bool

    Examples
    --------
    >>> decomp  = SeasonalDecomposer(period=52, model='additive')
    >>> result  = decomp.fit(df['conversions'])
    >>> result.print_summary()
    >>> decomp.plot(result)
    >>>
    >>> # Seasonally adjust for saturation modeling
    >>> df['conversions_adj'] = result.adjusted.values
    >>>
    >>> # Forecasting: add seasonality back
    >>> raw_preds = decomp.inverse_adjust(model_preds, result)
    """

    def __init__(
        self,
        period: int | str = "auto",
        model: str = "additive",
        trend_smoother: str = "cma",
        min_periods_for_trend: int = 2,
        verbose: bool = True,
    ):
        """
        Configure the decomposer with period, model type (additive/multiplicative),
        trend smoother, and minimum period length.
        No fitting occurs here; call fit() or fit_transform() to run the decomposition.
        """
        if model not in ("additive", "multiplicative"):
            raise ValueError(f"model must be 'additive' or 'multiplicative', got '{model}'.")
        if trend_smoother not in ("cma", "lowess"):
            raise ValueError(f"trend_smoother must be 'cma' or 'lowess', got '{trend_smoother}'.")

        self.period = period
        self.model = model
        self.trend_smoother = trend_smoother
        self.min_periods_for_trend = min_periods_for_trend
        self.verbose = verbose

        # Populated during fit()
        self._fitted_period: int | None = None
        self._seasonal_factors: np.ndarray | None = None

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(
        self,
        series: pd.Series | np.ndarray | list[float],
        index: Any | None = None,
    ) -> SeasonalDecomposition:
        """
        Fit the decomposition and return a SeasonalDecomposition result.

        Parameters
        ----------
        series : pd.Series, np.ndarray, or list
            The time series to decompose.  Must be ordered chronologically.
        index : array-like, optional
            Index labels.  Ignored if series is already a pd.Series with an index.

        Returns
        -------
        SeasonalDecomposition
        """
        # Convert to pd.Series
        if isinstance(series, pd.Series):
            s = series.copy()
        else:
            idx = index if index is not None else np.arange(len(series))
            s = pd.Series(np.asarray(series, dtype=float), index=idx)

        n = len(s)
        y = s.values.astype(float)

        if n < 4:
            raise ValueError(
                f"Series too short for decomposition (n={n}). Need at least 4 observations."
            )

        # ── Determine period ──────────────────────────────────────────────────
        if self.period == "auto":
            detected = self._detect_period(y)
            period = detected if detected is not None else min(7, n // 2)
            if detected is None:
                warnings.warn(
                    "Could not auto-detect seasonal period. "
                    f"Defaulting to {period}. "
                    "Set period explicitly for better results.",
                    UserWarning,
                )
        else:
            period = int(self.period)
            detected = self._detect_period(y)

        if period < 2:
            period = 2
        if period >= n:
            period = max(2, n // 2)
            warnings.warn(f"period ({period}) >= n ({n}). Clamped to {period}.", UserWarning)

        self._fitted_period = period
        self._log(f"Period = {period}  |  model = {self.model}  |  n = {n}")

        # ── Trend ─────────────────────────────────────────────────────────────
        trend_vals = self._estimate_trend(y, period)

        # ── Detrend ───────────────────────────────────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.model == "additive":
                detrended = y - trend_vals
            else:
                safe_trend = np.where(np.abs(trend_vals) > 1e-9, trend_vals, 1e-9)
                detrended = y / safe_trend

        # ── Seasonal factors ──────────────────────────────────────────────────
        seasonal_factors = self._estimate_seasonal_factors(detrended, period)
        self._seasonal_factors = seasonal_factors

        # Broadcast seasonal pattern over the full series length
        reps = int(np.ceil(n / period))
        seasonal_full = np.tile(seasonal_factors, reps)[:n]

        # Normalise: additive factors should sum to 0 over a period;
        # multiplicative should average to 1
        if self.model == "additive":
            seasonal_full -= seasonal_full.mean()
        else:
            safe_mean = seasonal_full.mean()
            seasonal_full = seasonal_full / safe_mean if safe_mean != 0 else seasonal_full

        # ── Residual ──────────────────────────────────────────────────────────
        if self.model == "additive":
            residual_vals = y - trend_vals - seasonal_full
        else:
            safe_seasonal = np.where(np.abs(seasonal_full) > 1e-9, seasonal_full, 1e-9)
            safe_trend2 = np.where(np.abs(trend_vals) > 1e-9, trend_vals, 1e-9)
            residual_vals = y / (safe_trend2 * safe_seasonal)

        # ── Seasonally-adjusted series ────────────────────────────────────────
        if self.model == "additive":
            adjusted_vals = y - seasonal_full
        else:
            safe_seas = np.where(np.abs(seasonal_full) > 1e-9, seasonal_full, 1e-9)
            adjusted_vals = y / safe_seas

        # ── Seasonal strength ─────────────────────────────────────────────────
        var_orig = np.var(y)
        var_seasonal = np.var(seasonal_full)
        strength = float(var_seasonal / var_orig) if var_orig > 0 else 0.0
        strength = min(max(strength, 0.0), 1.0)

        # ── Period labels ─────────────────────────────────────────────────────
        period_labels = self._make_period_labels(period, s.index)

        result = SeasonalDecomposition(
            model=self.model,
            period=period,
            original=pd.Series(y, index=s.index, name="original"),
            trend=pd.Series(trend_vals, index=s.index, name="trend"),
            seasonal=pd.Series(seasonal_full, index=s.index, name="seasonal"),
            residual=pd.Series(residual_vals, index=s.index, name="residual"),
            adjusted=pd.Series(adjusted_vals, index=s.index, name="adjusted"),
            seasonal_factors=seasonal_factors,
            period_labels=period_labels,
            strength_of_seasonality=strength,
            dominant_period=detected if isinstance(detected, int) else None,
            n=n,
        )
        self._log(f"Seasonal strength = {strength:.1%}")
        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: list[str],
        date_col: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, SeasonalDecomposition]]:
        """
        Decompose multiple columns of a DataFrame at once.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str
            Column names to decompose.
        date_col : str, optional
            If provided, uses this column as the Series index.

        Returns
        -------
        df_adjusted : pd.DataFrame
            Copy of df with extra columns ``{col}_adj`` for each decomposed column.
        decompositions : dict {col: SeasonalDecomposition}
        """
        df_out = df.copy()
        decompositions: dict[str, SeasonalDecomposition] = {}

        for col in columns:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame — skipping.")
                continue
            if date_col and date_col in df.columns:
                series = pd.Series(df[col].values, index=df[date_col].values, name=col)
            else:
                series = df[col].copy()

            try:
                result = self.fit(series)
                df_out[f"{col}_adj"] = result.adjusted.values
                decompositions[col] = result
            except Exception as e:
                warnings.warn(f"Decomposition of '{col}' failed: {e}")

        return df_out, decompositions

    def inverse_adjust(
        self,
        adjusted_predictions: np.ndarray | pd.Series,
        decomposition: SeasonalDecomposition,
        start_index: int = 0,
    ) -> np.ndarray:
        """
        Add the seasonal component back to seasonally-adjusted predictions.

        Use this to convert model predictions (made on the adjusted series)
        back into the original seasonal scale for reporting or evaluation.

        Parameters
        ----------
        adjusted_predictions : array-like
            Model predictions in the seasonally-adjusted space.
        decomposition : SeasonalDecomposition
            The decomposition object returned by fit().
        start_index : int
            Position in the seasonal cycle to start from.
            0 = first period position. Use this if your predictions start
            at a different point in the season than the training data.

        Returns
        -------
        np.ndarray  — predictions in the original (seasonal) space.
        """
        preds = np.asarray(adjusted_predictions, dtype=float)
        period = decomposition.period
        factors = decomposition.seasonal_factors

        # Normalise factors
        if self.model == "additive":
            normed = factors - factors.mean()
        else:
            mean_f = factors.mean()
            normed = factors / mean_f if mean_f != 0 else factors

        n_preds = len(preds)
        seasonal = np.array([normed[(start_index + i) % period] for i in range(n_preds)])

        if self.model == "additive":
            return preds + seasonal
        else:
            return preds * seasonal

    # ── plots ──────────────────────────────────────────────────────────────────

    def plot(
        self,
        result: SeasonalDecomposition,
        figsize: tuple[int, int] = (13, 11),
        save_path: str | None = None,
    ) -> None:
        """
        Four-panel decomposition chart.

        Panels
        ------
        1. Original series with trend overlay
        2. Trend component
        3. Seasonal component
        4. Residual (what's left after removing trend + seasonal)

        Parameters
        ----------
        result : SeasonalDecomposition
        figsize : tuple
        save_path : str, optional
        """
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        fig.suptitle(
            f"Seasonal Decomposition  |  model={result.model}  "
            f"period={result.period}  |  seasonal strength={result.strength_of_seasonality:.1%}",
            fontsize=12,
            fontweight="bold",
        )

        idx = np.arange(result.n)

        def _use_date_index(ax):
            """Use the series index for x labels when it is date-like."""
            try:
                if hasattr(result.original.index, "strftime"):
                    labels = [str(t)[:10] for t in result.original.index]
                    step = max(1, len(labels) // 8)
                    ax.set_xticks(idx[::step])
                    ax.set_xticklabels(labels[::step], rotation=30, ha="right", fontsize=7)
            except Exception:
                pass

        # ── Panel 1: original + trend ──────────────────────────────────────
        axes[0].plot(idx, result.original, color=_BLUE, lw=1.5, alpha=0.7, label="Original")
        axes[0].plot(idx, result.trend, color=_RED, lw=2.2, label="Trend")
        axes[0].set_title("Original Series + Trend", fontsize=9)
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("Value", fontsize=8)
        _use_date_index(axes[0])

        # ── Panel 2: trend ────────────────────────────────────────────────
        axes[1].plot(idx, result.trend, color=_RED, lw=2)
        axes[1].fill_between(idx, result.trend, alpha=0.15, color=_RED)
        axes[1].set_title("Trend", fontsize=9)
        axes[1].set_ylabel("Trend", fontsize=8)

        # ── Panel 3: seasonal ─────────────────────────────────────────────
        axes[2].plot(idx, result.seasonal, color=_GREEN, lw=1.8)
        axes[2].fill_between(idx, result.seasonal, alpha=0.2, color=_GREEN)
        if self.model == "additive":
            axes[2].axhline(0, color=_GREY, lw=0.8, ls="--")
        else:
            axes[2].axhline(1, color=_GREY, lw=0.8, ls="--", label="Factor=1")
        axes[2].set_title(f"Seasonal  (period={result.period})", fontsize=9)
        axes[2].set_ylabel("Seasonal", fontsize=8)

        # ── Panel 4: residual ─────────────────────────────────────────────
        axes[3].plot(idx, result.residual, color=_GREY, lw=1.2, alpha=0.8)
        axes[3].scatter(idx, result.residual, s=8, color=_GREY, alpha=0.6)
        if self.model == "additive":
            axes[3].axhline(0, color=_RED, lw=1, ls="--")
        else:
            axes[3].axhline(1, color=_RED, lw=1, ls="--")
        axes[3].set_title("Residual", fontsize=9)
        axes[3].set_ylabel("Residual", fontsize=8)
        axes[3].set_xlabel("Time index", fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_seasonal_factors(
        self,
        result: SeasonalDecomposition,
        figsize: tuple[int, int] = (10, 5),
        save_path: str | None = None,
    ) -> None:
        """
        Bar chart of the seasonal factors for each period position.

        Shows how much each position (day/week/month) deviates from the average.
        Positive bars = above average; negative bars = below average.

        Parameters
        ----------
        result : SeasonalDecomposition
        figsize : tuple
        save_path : str, optional
        """
        factors = result.seasonal_factors
        labels = result.period_labels
        colors = [_GREEN if f >= 0 else _RED for f in factors]

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(labels, factors, color=colors, alpha=0.85, edgecolor="white")

        ref = 0.0 if self.model == "additive" else 1.0
        ax.axhline(ref, color=_GREY, lw=1.2, ls="--", label="No seasonal effect")

        y_pad = max(np.abs(factors).max() * 0.04, 1e-6)
        for bar, f in zip(bars, factors):
            sign = "+" if f >= 0 else ""
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                f + (y_pad if f >= 0 else -y_pad * 4),
                f"{sign}{f:.2f}",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )

        ax.set_xlabel(f"Period position (period={result.period})", fontsize=9)
        ylabel = (
            "Additive seasonal factor"
            if self.model == "additive"
            else "Multiplicative seasonal factor"
        )
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(
            f"Seasonal Factors  |  strength={result.strength_of_seasonality:.1%}",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        plt.xticks(rotation=40, ha="right", fontsize=8)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_adjusted_vs_original(
        self,
        result: SeasonalDecomposition,
        figsize: tuple[int, int] = (12, 5),
        save_path: str | None = None,
    ) -> None:
        """
        Overlay plot comparing the original series with the seasonally-adjusted one.

        Parameters
        ----------
        result : SeasonalDecomposition
        figsize : tuple
        save_path : str, optional
        """
        idx = np.arange(result.n)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(idx, result.original, color=_BLUE, lw=1.6, alpha=0.65, label="Original", zorder=2)
        ax.plot(idx, result.adjusted, color=_ORANGE, lw=2, label="Seasonally adjusted", zorder=3)
        ax.fill_between(
            idx,
            result.original,
            result.adjusted,
            alpha=0.10,
            color=_GREY,
            label="Seasonal adjustment",
        )
        ax.set_xlabel("Time index", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)
        ax.set_title(
            "Original vs Seasonally Adjusted Series\n"
            f"(seasonal strength={result.strength_of_seasonality:.1%})",
            fontsize=11,
            fontweight="bold",
        )
        ax.legend(fontsize=9)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _estimate_trend(self, y: np.ndarray, period: int) -> np.ndarray:
        """
        Estimate trend using a centred moving average (CMA) or lowess.
        Boundary values are extrapolated linearly.
        """
        n = len(y)

        if self.trend_smoother == "lowess":
            # Lightweight LOWESS via uniform filter at different bandwidths
            bw = max(period, int(n * 0.2))
            trend = uniform_filter1d(y.astype(float), size=min(bw, n), mode="nearest")
            return trend

        # Centred moving average
        period // 2
        if period % 2 == 0:
            # Even period: average two adjacent CMAs to centre properly
            k = period
            cum = np.cumsum(np.concatenate([[0.0], y]))
            trend_inner = (cum[k:] - cum[:-k]) / k
            # Centre: average adjacent values
            trend_centre = (trend_inner[:-1] + trend_inner[1:]) / 2
            # trend_centre has length n - k; pad symmetrically
            pad_left = k // 2
            pad_right = n - len(trend_centre) - pad_left
        else:
            # Odd period: single CMA centres naturally
            k = period
            cum = np.cumsum(np.concatenate([[0.0], y]))
            trend_centre = (cum[k:] - cum[:-k]) / k
            pad_left = k // 2
            pad_right = n - len(trend_centre) - pad_left

        # Linear extrapolation for boundary
        trend = np.empty(n)
        trend[pad_left : pad_left + len(trend_centre)] = trend_centre

        # Left boundary
        if pad_left > 0 and len(trend_centre) >= 2:
            slope_l = trend_centre[1] - trend_centre[0]
            for i in range(pad_left):
                trend[pad_left - 1 - i] = trend_centre[0] - slope_l * (i + 1)

        # Right boundary
        if pad_right > 0 and len(trend_centre) >= 2:
            slope_r = trend_centre[-1] - trend_centre[-2]
            for i in range(pad_right):
                trend[pad_left + len(trend_centre) + i] = trend_centre[-1] + slope_r * (i + 1)

        return trend

    def _estimate_seasonal_factors(self, detrended: np.ndarray, period: int) -> np.ndarray:
        """
        Compute one seasonal factor per period position by averaging across
        all complete cycles, with outlier rejection.
        """
        len(detrended)
        # For each position in 0..period-1, collect all observed values
        by_pos: list[list[float]] = [[] for _ in range(period)]
        for i, v in enumerate(detrended):
            if np.isfinite(v):
                by_pos[i % period].append(float(v))

        factors = np.zeros(period)
        for pos, vals in enumerate(by_pos):
            if not vals:
                factors[pos] = 0.0 if self.model == "additive" else 1.0
            elif len(vals) == 1:
                factors[pos] = vals[0]
            else:
                arr = np.array(vals)
                # Trim outliers: remove values > 3 IQR from median
                q25, q75 = np.percentile(arr, [25, 75])
                iqr = q75 - q25
                if iqr > 0:
                    mask = np.abs(arr - np.median(arr)) <= 3 * iqr
                    arr = arr[mask]
                factors[pos] = float(arr.mean()) if len(arr) > 0 else 0.0

        return factors

    def _detect_period(self, y: np.ndarray) -> int | None:
        """
        Use Lomb-Scargle-style periodogram to detect dominant seasonal period.
        Falls back to None if no clear peak is found.
        """
        n = len(y)
        if n < 8:
            return None
        try:
            freqs, power = periodogram(y - y.mean())
            # Exclude DC component and very long periods
            min_period, max_period = 2, n // 2
            valid = (
                (freqs > 0)
                & (1 / np.maximum(freqs, 1e-12) <= max_period)
                & (1 / np.maximum(freqs, 1e-12) >= min_period)
            )
            if not valid.any():
                return None
            peak_freq = freqs[valid][np.argmax(power[valid])]
            period = int(round(1.0 / peak_freq))
            return max(2, min(period, n // 2))
        except Exception:
            return None

    @staticmethod
    def _make_period_labels(period: int, index: Any) -> list[str]:
        """
        Generate human-readable labels for each position in the period.
        Uses 'Wk1'…'Wk52', 'Mon'…'Sun', 'Jan'…'Dec', or 'P1'…'Pk'.
        """
        if period == 7:
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if period == 12:
            return [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        if period == 4:
            return ["Q1", "Q2", "Q3", "Q4"]
        if period <= 53:
            return [f"Wk{i+1}" for i in range(period)]
        return [f"P{i+1}" for i in range(period)]

    def _log(self, msg: str) -> None:
        """
        Print msg to stdout when verbose=True.
        """
        if self.verbose:
            print(f"[SeasonalDecomposer] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def adjust_for_seasonality(
    series: pd.Series | np.ndarray,
    period: int | str = "auto",
    model: str = "additive",
    verbose: bool = False,
) -> tuple[pd.Series, SeasonalDecomposition]:
    """
    One-liner: decompose a series and return the seasonally-adjusted version.

    Parameters
    ----------
    series : pd.Series or array-like
    period : int or 'auto'
    model  : 'additive' or 'multiplicative'
    verbose : bool

    Returns
    -------
    adjusted : pd.Series
        Seasonally-adjusted series (same length and index as input).
    decomposition : SeasonalDecomposition
        Full decomposition result.

    Examples
    --------
    >>> from adsat.seasonality import adjust_for_seasonality
    >>> adjusted, decomp = adjust_for_seasonality(df['conversions'], period=52)
    >>> df['conversions_adj'] = adjusted.values
    """
    decomposer = SeasonalDecomposer(period=period, model=model, verbose=verbose)
    result = decomposer.fit(series)
    return result.adjusted, result
