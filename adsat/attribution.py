"""
adsat.attribution
=================
Multi-touch attribution modelling for advertising journeys.

Given a user-level event log — one row per ad touchpoint — this module
attributes conversion credit and revenue across channels using nine
different models, evaluates them with a rigorous comparison framework,
and recommends a budget reallocation based on the chosen model.

Attribution models available
-----------------------------
Simple (rule-based):
  LastClick          – 100% credit to the final touchpoint before conversion
  FirstClick         – 100% credit to the first touchpoint in the journey
  Linear             – equal credit split across all touchpoints
  PositionBased      – configurable weights to first / last / middle touchpoints
  TimeDecay          – exponentially higher credit to more recent touchpoints

Advanced (data-driven):
  Shapley            – game-theoretic marginal contribution per channel
                       (exact for ≤ 12 channels; Monte Carlo sampling above)
  MarkovChain        – removal-effect via configurable-order Markov transition matrix
  DataDriven         – logistic regression + SHAP values; mirrors Google Analytics 4
  Ensemble           – weighted average of any subset of the above models

Key classes & functions
------------------------
JourneyBuilder           – convert a raw event log to a journey-level DataFrame
AttributionAnalyzer      – fit one or more models; produce channel credit tables
AttributionResult        – result dataclass (channel_credits, journey_credits, …)
AttributionEvaluator     – compare models and select the best one
AttributionBudgetAdvisor – translate credits → spend recommendations
attribute_campaigns()    – convenience one-liner

Minimum required input columns
--------------------------------
  user_id          : unique user / cookie identifier
  timestamp        : datetime of the touchpoint
  channel          : channel name  (e.g. "paid_search", "email")
  interaction_type : "click" or "impression"
  converted        : 1 if this journey ended in conversion, else 0
  revenue          : revenue at conversion (0 on non-conversion rows)

Optional columns (enable richer analysis):
  cost             : channel cost per touchpoint  (needed for ROI allocation)
  session_id       : groups touchpoints into sessions
  campaign         : sub-channel label
  device           : device type

Typical workflow
-----------------
>>> from adsat.attribution import AttributionAnalyzer, JourneyBuilder
>>>
>>> builder  = JourneyBuilder(
...     user_col          = "user_id",
...     time_col          = "timestamp",
...     channel_col       = "channel",
...     interaction_col   = "interaction_type",
...     converted_col     = "converted",
...     revenue_col       = "revenue",
...     lookback_days     = 30,
...     interaction_weight = {"click": 1.0, "impression": 0.3},
... )
>>> journeys = builder.build(events_df)
>>>
>>> analyzer = AttributionAnalyzer(models=["last_click", "shapley", "markov"])
>>> result   = analyzer.fit(journeys)
>>> result.print_summary()
>>> result.plot()
>>>
>>> # One-liner
>>> from adsat.attribution import attribute_campaigns
>>> result = attribute_campaigns(events_df, models=["shapley", "markov"])
"""

from __future__ import annotations

import itertools
import math
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# scipy / sklearn are optional-ish — guard in each usage

# ── colour palette (consistent with adsat) ────────────────────────────────────
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_GREEN = "#3BB273"
_RED = "#E84855"
_GREY = "#6C757D"
_PURPLE = "#7B2D8B"
_TEAL = "#17BEBB"
_GOLD = "#E9C46A"
PALETTE = [_BLUE, _ORANGE, _GREEN, _RED, _PURPLE, _GREY, _TEAL, _GOLD, "#264653", "#F4A261"]

# ── constants ─────────────────────────────────────────────────────────────────
_EXACT_SHAPLEY_LIMIT = 12  # channels — above this switch to Monte Carlo
_MC_SHAPLEY_ITERATIONS = 5_000
_SUPPORTED_MODELS = [
    "last_click",
    "first_click",
    "linear",
    "position_based",
    "time_decay",
    "shapley",
    "markov",
    "data_driven",
    "ensemble",
]

# ── colour helpers ─────────────────────────────────────────────────────────────


def _ch_colour(channels: list[str]) -> dict[str, str]:
    """Assign a consistent colour from PALETTE to each channel name."""
    return {ch: PALETTE[i % len(PALETTE)] for i, ch in enumerate(sorted(channels))}


def _fmt_pct(x, _=None) -> str:
    """Format a decimal fraction as a percentage string for axis labels."""
    return f"{x * 100:.0f}%"


def _fmt_money(x, _=None) -> str:
    """Format a monetary value with k/M suffix for axis labels."""
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA MODEL  (JourneyBuilder + schema validation)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class JourneyConfig:
    """
    Column-name map and journey-building parameters for a raw event log.

    All field names default to the canonical names described in the module
    docstring, but every name is overridable so the module works with any
    DataFrame without renaming columns.

    Parameters
    ----------
    user_col           : Column uniquely identifying a user / cookie.
    time_col           : Datetime column — used for ordering and time-decay.
    channel_col        : Channel name column.
    interaction_col    : Touchpoint type column (e.g. "click" / "impression").
    converted_col      : Binary (0/1) conversion flag column.
    revenue_col        : Revenue column (0 on non-converting rows).
    cost_col           : Optional cost per touchpoint column.
    session_col        : Optional session grouping column.
    campaign_col       : Optional sub-channel / campaign label column.
    device_col         : Optional device type column.
    lookback_days      : Journey window in days.  Touchpoints older than
                         ``lookback_days`` before conversion are excluded.
                         Pass None to use auto-detection (median inter-event gap × 3).
    multi_conversion   : How to handle users who convert more than once.
                         "reset"  — each conversion starts a fresh journey.
                         "rolling" — all touchpoints within the window count.
    interaction_weight : Dict mapping interaction_type values to numeric weights
                         (e.g. {"click": 1.0, "impression": 0.3}).
                         Touchpoints with unlisted types receive weight 0.0.
    """

    user_col: str = "user_id"
    time_col: str = "timestamp"
    channel_col: str = "channel"
    interaction_col: str = "interaction_type"
    converted_col: str = "converted"
    revenue_col: str = "revenue"
    cost_col: str | None = None
    session_col: str | None = None
    campaign_col: str | None = None
    device_col: str | None = None
    lookback_days: int | None = 30
    multi_conversion: str = "reset"  # "reset" | "rolling"
    interaction_weight: dict[str, float] = field(
        default_factory=lambda: {"click": 1.0, "impression": 0.3}
    )


class JourneyBuilder:
    """
    Convert a raw touchpoint event log into a structured journey DataFrame.

    Each row in the output represents one complete user journey, with the
    ordered sequence of channel touchpoints stored as a list.  Non-converting
    journeys are included because Shapley and Markov models require them to
    estimate baseline conversion probability.

    Parameters
    ----------
    config : JourneyConfig, or pass kwargs directly that map to JourneyConfig fields.

    Examples
    --------
    >>> builder = JourneyBuilder(
    ...     user_col           = "user_id",
    ...     time_col           = "timestamp",
    ...     channel_col        = "channel",
    ...     interaction_col    = "interaction_type",
    ...     converted_col      = "converted",
    ...     revenue_col        = "revenue",
    ...     lookback_days      = 30,
    ...     interaction_weight = {"click": 1.0, "impression": 0.3},
    ... )
    >>> journeys = builder.build(events_df)
    >>> print(journeys.head())
    """

    def __init__(self, config: JourneyConfig | None = None, **kwargs):
        """
        Initialise with a JourneyConfig object or keyword arguments.

        Keyword arguments are forwarded to JourneyConfig so the class can be
        constructed without explicitly creating a JourneyConfig instance.
        """
        if config is not None:
            self.cfg = config
        else:
            self.cfg = JourneyConfig(**kwargs)

    # ── public API ─────────────────────────────────────────────────────────────

    def build(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a raw event log into a journey-level DataFrame.

        Each output row represents one complete user journey (converting or
        non-converting).  The output is ready to pass directly to
        ``AttributionAnalyzer.fit()``.

        Parameters
        ----------
        events : pd.DataFrame
            Raw touchpoint event log.  Must contain at minimum the columns
            named in JourneyConfig (user_col, time_col, channel_col,
            interaction_col, converted_col, revenue_col).

        Returns
        -------
        pd.DataFrame with columns:
            journey_id, user_id, path, weighted_path, n_touchpoints,
            converted, revenue, [cost], first_touch_time, last_touch_time,
            journey_days, channels_in_path (set)
        """
        cfg = self.cfg
        self._validate(events)

        df = events.copy()
        df[cfg.time_col] = pd.to_datetime(df[cfg.time_col])
        df = df.sort_values([cfg.user_col, cfg.time_col]).reset_index(drop=True)

        # Assign touchpoint weight from interaction_type
        df["_tp_weight"] = df[cfg.interaction_col].map(cfg.interaction_weight).fillna(0.0)

        # Auto-detect lookback when lookback_days is None
        lookback = cfg.lookback_days
        if lookback is None:
            lookback = self._auto_lookback(df)

        journeys = []
        journey_id = 0

        for uid, user_df in df.groupby(cfg.user_col, sort=False):
            user_df = user_df.sort_values(cfg.time_col).reset_index(drop=True)

            if cfg.multi_conversion == "reset":
                sub_journeys = self._split_by_conversion(user_df)
            else:
                sub_journeys = [user_df]  # rolling: one big journey per user

            for jdf in sub_journeys:
                row = self._build_journey_row(jdf, uid, journey_id, lookback)
                if row is not None:
                    journeys.append(row)
                    journey_id += 1

        if not journeys:
            raise ValueError(
                "No valid journeys could be built.  Check that the event log "
                "contains rows with converted=1 and that lookback_days is large "
                "enough to include at least one touchpoint before conversion."
            )

        result = pd.DataFrame(journeys)
        return result

    def validate_schema(self, events: pd.DataFrame) -> list[str]:
        """
        Check the event DataFrame for missing required columns and return a list
        of human-readable error messages.  Returns an empty list when the schema
        is valid.
        """
        cfg = self.cfg
        required = [
            cfg.user_col,
            cfg.time_col,
            cfg.channel_col,
            cfg.interaction_col,
            cfg.converted_col,
            cfg.revenue_col,
        ]
        errors = []
        for col in required:
            if col not in events.columns:
                errors.append(f"Required column '{col}' not found in DataFrame.")
        if errors:
            return errors
        if not pd.api.types.is_numeric_dtype(events[cfg.converted_col]):
            errors.append(f"Column '{cfg.converted_col}' must be numeric (0/1).")
        if not pd.api.types.is_numeric_dtype(events[cfg.revenue_col]):
            errors.append(f"Column '{cfg.revenue_col}' must be numeric.")
        return errors

    # ── internal helpers ───────────────────────────────────────────────────────

    def _validate(self, events: pd.DataFrame) -> None:
        """Raise ValueError if the event DataFrame fails schema validation."""
        errors = self.validate_schema(events)
        if errors:
            raise ValueError(
                "Event DataFrame failed schema validation:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

    def _auto_lookback(self, df: pd.DataFrame) -> int:
        """
        Estimate a sensible lookback window when lookback_days=None.

        Uses 3× the median gap between consecutive touchpoints per user,
        clamped between 7 and 90 days.
        """
        gaps = []
        for _, udf in df.groupby(self.cfg.user_col):
            times = udf[self.cfg.time_col].sort_values()
            if len(times) > 1:
                deltas = times.diff().dropna().dt.days.values
                gaps.extend(deltas[deltas > 0].tolist())
        if not gaps:
            return 30
        median_gap = float(np.median(gaps))
        return int(np.clip(3 * median_gap, 7, 90))

    def _split_by_conversion(self, user_df: pd.DataFrame) -> list[pd.DataFrame]:
        """
        Split a user's event log into independent journeys by resetting at each
        conversion.  Non-converting trailing events form a final non-converting
        journey.
        """
        cfg = self.cfg
        journeys = []
        current: list[int] = []

        for idx, row in user_df.iterrows():
            current.append(idx)
            if int(row[cfg.converted_col]) == 1:
                journeys.append(user_df.loc[current])
                current = []

        # Trailing non-converting events
        if current:
            journeys.append(user_df.loc[current])

        return journeys

    def _build_journey_row(
        self,
        jdf: pd.DataFrame,
        uid: Any,
        journey_id: int,
        lookback: int,
    ) -> dict | None:
        """
        Build a single journey row dict from the events in jdf.

        Applies the lookback window (cutting touchpoints older than lookback
        days before the last event in the journey), then assembles the
        ordered channel path and weighted path.

        Returns None when no touchpoints remain after applying the window.
        """
        cfg = self.cfg
        jdf = jdf.copy().sort_values(cfg.time_col).reset_index(drop=True)

        last_time = jdf[cfg.time_col].iloc[-1]
        cutoff = last_time - pd.Timedelta(days=lookback)
        jdf = jdf[jdf[cfg.time_col] >= cutoff]

        if len(jdf) == 0:
            return None

        # Zero-weight touchpoints are excluded from the path representation
        # but preserved in the raw count
        weighted = jdf[jdf["_tp_weight"] > 0]
        if len(weighted) == 0:
            return None

        path = weighted[cfg.channel_col].tolist()
        weights = weighted["_tp_weight"].tolist()
        revenue = float(jdf[cfg.revenue_col].max())
        converted = int(jdf[cfg.converted_col].max())

        row: dict[str, Any] = {
            "journey_id": journey_id,
            "user_id": uid,
            "path": path,
            "weights": weights,
            "n_touchpoints": len(path),
            "converted": converted,
            "revenue": revenue,
            "first_touch_time": jdf[cfg.time_col].iloc[0],
            "last_touch_time": last_time,
            "journey_days": max((last_time - jdf[cfg.time_col].iloc[0]).days, 0),
            "channels_in_path": set(path),
        }

        # Optional columns
        if cfg.cost_col and cfg.cost_col in jdf.columns:
            row["cost"] = float(jdf[cfg.cost_col].sum())
        if cfg.session_col and cfg.session_col in jdf.columns:
            row["n_sessions"] = jdf[cfg.session_col].nunique()
        if cfg.device_col and cfg.device_col in jdf.columns:
            row["devices"] = list(jdf[cfg.device_col].unique())

        return row

    @property
    def channels(self) -> list[str]:
        """Return a sorted list of all channel names seen by the last build() call."""
        return sorted(self._channels) if hasattr(self, "_channels") else []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BASE MODEL + RESULT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AttributionResult:
    """
    Container for all outputs from one or more attribution models.

    Attributes
    ----------
    channel_credits : pd.DataFrame
        One row per channel × model.  Columns: channel, model,
        attributed_conversions, attributed_revenue, credit_share.
    journey_credits : pd.DataFrame
        One row per journey × model.  Contains per-touchpoint credit
        breakdown (used for path-level analysis and evaluation).
    model_comparison : pd.DataFrame
        One row per model with summary evaluation metrics.
    channels : list of str
        Sorted list of all channels in the data.
    total_conversions : int
    total_revenue : float
    models_fitted : list of str
    metadata : dict
        Lookback, multi_conversion strategy, interaction weights, etc.
    """

    channel_credits: pd.DataFrame
    journey_credits: pd.DataFrame
    model_comparison: pd.DataFrame
    channels: list[str]
    total_conversions: int
    total_revenue: float
    models_fitted: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    # ── human-readable summary ─────────────────────────────────────────────────

    def print_summary(self) -> None:
        """
        Print a structured attribution summary to stdout.

        Shows: total KPIs, per-model channel credit comparison table,
        top converting paths, and model evaluation metrics.
        """
        sep = "=" * 72
        sep2 = "-" * 72
        print(sep)
        print("  ADSAT — ATTRIBUTION RESULTS")
        print(sep)
        print(f"  Total conversions : {self.total_conversions:>10,d}")
        print(f"  Total revenue     : {self.total_revenue:>10,.2f}")
        print(f"  Channels          : {', '.join(self.channels)}")
        print(f"  Models fitted     : {', '.join(self.models_fitted)}")
        print(sep2)
        print()

        # Per-model channel credit table
        for model in self.models_fitted:
            mdf = (
                self.channel_credits[self.channel_credits["model"] == model]
                .sort_values("attributed_conversions", ascending=False)
                .reset_index(drop=True)
            )
            print(f"  [{model.upper().replace('_', ' ')}]")
            display_cols = [
                c
                for c in [
                    "channel",
                    "attributed_conversions",
                    "credit_share",
                    "attributed_revenue",
                    "roi",
                ]
                if c in mdf.columns
            ]
            fmt = mdf[display_cols].copy()
            if "credit_share" in fmt.columns:
                fmt["credit_share"] = fmt["credit_share"].map("{:.1%}".format)
            if "attributed_revenue" in fmt.columns:
                fmt["attributed_revenue"] = fmt["attributed_revenue"].map(
                    lambda v: f"{v:,.2f}" if pd.notna(v) else "N/A"
                )
            if "attributed_conversions" in fmt.columns:
                fmt["attributed_conversions"] = fmt["attributed_conversions"].map(
                    lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
                )
            if "roi" in fmt.columns:
                fmt["roi"] = fmt["roi"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")
            print(fmt.to_string(index=False))
            print()

        # Model comparison table
        if not self.model_comparison.empty:
            print("  [MODEL COMPARISON]")
            print(self.model_comparison.to_string(index=False))
            print()
        print(sep)

    def get_credits(self, model: str) -> pd.DataFrame:
        """
        Return the channel_credits table filtered to one model.

        Parameters
        ----------
        model : str  — e.g. "shapley", "markov", "last_click"

        Raises KeyError if the model was not fitted.
        """
        if model not in self.models_fitted:
            raise KeyError(f"Model '{model}' was not fitted.  " f"Available: {self.models_fitted}")
        return self.channel_credits[self.channel_credits["model"] == model].reset_index(drop=True)

    def best_model(self) -> str:
        """
        Return the name of the highest-ranked model according to the
        composite evaluation score in model_comparison.

        Returns 'ensemble' when multiple models were fitted and the
        comparison table is empty (i.e. evaluation was not run).
        """
        if self.model_comparison.empty or "composite_score" not in self.model_comparison.columns:
            return self.models_fitted[0]
        return str(
            self.model_comparison.sort_values("composite_score", ascending=False).iloc[0]["model"]
        )


class _BaseAttributionModel:
    """
    Abstract base class for all attribution models.

    Subclasses must implement ``_compute(journeys, channels) -> pd.DataFrame``.
    The return value is a DataFrame with columns: channel, attributed_conversions,
    attributed_revenue — one row per channel.
    """

    name: str = "base"

    def fit(
        self,
        journeys: pd.DataFrame,
        channels: list[str],
    ) -> pd.DataFrame:
        """
        Fit the model on a journey DataFrame and return channel-level credits.

        Parameters
        ----------
        journeys : pd.DataFrame  — output of JourneyBuilder.build().
        channels : list of str   — full list of channels (including those
                                   that may not appear in every journey).

        Returns
        -------
        pd.DataFrame with columns:
            channel, attributed_conversions, attributed_revenue
        """
        credits = self._compute(journeys, channels)
        credits["model"] = self.name
        return credits

    def _compute(
        self,
        journeys: pd.DataFrame,
        channels: list[str],
    ) -> pd.DataFrame:
        """Subclasses override this method to implement the attribution logic."""
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SIMPLE RULE-BASED MODELS
# ═══════════════════════════════════════════════════════════════════════════════


class LastClickModel(_BaseAttributionModel):
    """
    Last-click attribution: 100% of conversion credit goes to the final
    touchpoint before conversion.

    This is the industry default in many legacy analytics platforms.  It
    tends to over-credit lower-funnel channels (paid search, retargeting)
    and under-credit awareness channels (display, social).

    Parameters
    ----------
    None — this model has no configurable parameters.
    """

    name = "last_click"

    def _compute(self, journeys, channels):
        """Assign 100% credit to the last channel in each converting journey."""
        acc = defaultdict(lambda: [0.0, 0.0])
        for _, row in journeys.iterrows():
            if not row["converted"] or not row["path"]:
                continue
            ch = row["path"][-1]
            acc[ch][0] += 1.0
            acc[ch][1] += float(row["revenue"])
        return _acc_to_df(acc, channels)


class FirstClickModel(_BaseAttributionModel):
    """
    First-click attribution: 100% of conversion credit goes to the first
    touchpoint in the journey.

    Useful as a upper-funnel benchmark.  Tends to over-credit awareness
    channels and ignore channels that close the sale.

    Parameters
    ----------
    None — this model has no configurable parameters.
    """

    name = "first_click"

    def _compute(self, journeys, channels):
        """Assign 100% credit to the first channel in each converting journey."""
        acc = defaultdict(lambda: [0.0, 0.0])
        for _, row in journeys.iterrows():
            if not row["converted"] or not row["path"]:
                continue
            ch = row["path"][0]
            acc[ch][0] += 1.0
            acc[ch][1] += float(row["revenue"])
        return _acc_to_df(acc, channels)


class LinearModel(_BaseAttributionModel):
    """
    Linear attribution: conversion credit divided equally among all
    touchpoints in the journey, weighted by interaction_type weight.

    Provides a neutral baseline that avoids the positional bias of
    first/last-click models.

    Parameters
    ----------
    None — interaction weights were applied at journey-building time.
    """

    name = "linear"

    def _compute(self, journeys, channels):
        """Distribute credit equally across all weighted touchpoints."""
        acc = defaultdict(lambda: [0.0, 0.0])
        for _, row in journeys.iterrows():
            if not row["converted"] or not row["path"]:
                continue
            path = row["path"]
            weights = row["weights"]
            total_w = sum(weights)
            if total_w <= 0:
                continue
            rev = float(row["revenue"])
            for ch, w in zip(path, weights):
                share = w / total_w
                acc[ch][0] += share
                acc[ch][1] += share * rev
        return _acc_to_df(acc, channels)


class PositionBasedModel(_BaseAttributionModel):
    """
    Position-based (U-shaped / V-shaped) attribution.

    Assigns configurable weights to the first touchpoint, the last
    touchpoint, and all middle touchpoints.

    Default (U-shaped / "bathtub"):
      first = 0.40, last = 0.40, middle = 0.20 (split equally)

    Linear decay variant (V-shaped): pass first=1.0, last=0.0, middle="auto"
    and the middle weights will decrease linearly from first toward last.

    Parameters
    ----------
    first_weight  : Credit share for the first touchpoint. Default 0.40.
    last_weight   : Credit share for the last touchpoint. Default 0.40.
    middle_weight : Credit share for all middle touchpoints combined.
                    Distributed equally among them.  Default 0.20.
    """

    name = "position_based"

    def __init__(
        self,
        first_weight: float = 0.40,
        last_weight: float = 0.40,
        middle_weight: float = 0.20,
    ):
        """
        Store position weights.  The three values are re-normalised to sum to 1
        so arbitrary values are accepted (e.g. passing 40, 40, 20 also works).
        """
        total = first_weight + last_weight + middle_weight
        if total <= 0:
            raise ValueError("Sum of position weights must be > 0.")
        self.first_w = first_weight / total
        self.last_w = last_weight / total
        self.middle_w = middle_weight / total

    def _compute(self, journeys, channels):
        """
        Apply position-based weights to each converting journey.

        Single-touchpoint journeys get 100% credit.
        Two-touchpoint journeys split (first_w + middle_w) and last_w.
        """
        acc = defaultdict(lambda: [0.0, 0.0])
        for _, row in journeys.iterrows():
            if not row["converted"] or not row["path"]:
                continue
            path = row["path"]
            n = len(path)
            rev = float(row["revenue"])

            if n == 1:
                acc[path[0]][0] += 1.0
                acc[path[0]][1] += rev
            elif n == 2:
                # first gets first_w + half of middle_w; last gets last_w + half
                w0 = self.first_w + self.middle_w / 2
                w1 = self.last_w + self.middle_w / 2
                for ch, w in zip(path, [w0, w1]):
                    acc[ch][0] += w
                    acc[ch][1] += w * rev
            else:
                middle_per = self.middle_w / (n - 2)
                ws = [self.first_w] + [middle_per] * (n - 2) + [self.last_w]
                for ch, w in zip(path, ws):
                    acc[ch][0] += w
                    acc[ch][1] += w * rev
        return _acc_to_df(acc, channels)


class TimeDecayModel(_BaseAttributionModel):
    """
    Time-decay attribution: touchpoints closer to conversion receive
    exponentially higher credit.

    Credit for touchpoint i (counting back from conversion) is proportional
    to ``exp(-decay_rate * days_before_conversion)``.

    Parameters
    ----------
    half_life_days : float
        The number of days before conversion at which a touchpoint receives
        half the credit of a same-day touchpoint.  Default 7 days.
        Smaller values = steeper decay (stronger recency bias).
    """

    name = "time_decay"

    def __init__(self, half_life_days: float = 7.0):
        """Store the half-life parameter and compute the equivalent decay rate."""
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive.")
        self.half_life_days = half_life_days
        self._decay = math.log(2) / half_life_days  # λ in exp(-λt)

    def _compute(self, journeys, channels):
        """
        Assign exponentially decaying weights based on days_before_conversion.

        When timestamps are not available (journey_days = 0), falls back to
        position-based decay using ordinal position from the end of the path.
        """
        acc = defaultdict(lambda: [0.0, 0.0])
        for _, row in journeys.iterrows():
            if not row["converted"] or not row["path"]:
                continue
            path = row["path"]
            n = len(path)
            rev = float(row["revenue"])
            j_days = float(row.get("journey_days", 0) or 0)

            if j_days > 0 and n > 1:
                # Approximate each touchpoint's days_before_conversion by
                # distributing journey_days evenly across positions
                days_before = [j_days * (n - 1 - i) / (n - 1) for i in range(n)]
            else:
                # Ordinal fallback: position from the end (0 = last)
                days_before = list(range(n - 1, -1, -1))

            raw_w = [math.exp(-self._decay * d) for d in days_before]
            total = sum(raw_w)
            for ch, w in zip(path, raw_w):
                share = w / total
                acc[ch][0] += share
                acc[ch][1] += share * rev
        return _acc_to_df(acc, channels)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — ADVANCED MODELS (Shapley, Markov, Data-Driven, Ensemble)
# ═══════════════════════════════════════════════════════════════════════════════


class ShapleyModel(_BaseAttributionModel):
    """
    Shapley value attribution based on cooperative game theory.

    Each channel's credit = its average marginal contribution to conversion
    probability across all possible subsets (coalitions) of channels.

    Algorithm
    ---------
    For ≤ ``exact_limit`` unique channels: exact computation over all 2^n
    coalitions.
    For > ``exact_limit`` channels: Monte Carlo sampling with
    ``n_iterations`` random permutations (equivalent to permutation Shapley).

    Parameters
    ----------
    exact_limit   : int   — channel count threshold for switching to Monte Carlo.
                            Default 12.
    n_iterations  : int   — permutation samples for Monte Carlo.  Default 5000.
    random_seed   : int   — reproducibility seed.  Default 42.
    """

    name = "shapley"

    def __init__(
        self,
        exact_limit: int = _EXACT_SHAPLEY_LIMIT,
        n_iterations: int = _MC_SHAPLEY_ITERATIONS,
        random_seed: int = 42,
    ):
        """Store computation parameters and log which mode will be used."""
        self.exact_limit = exact_limit
        self.n_iterations = n_iterations
        self.random_seed = random_seed

    def _compute(self, journeys, channels):
        """
        Compute Shapley values for each channel.

        Step 1: Build a conversion-rate lookup keyed by frozenset of channels.
        Step 2: Compute Shapley values (exact or Monte Carlo).
        Step 3: Scale values to total conversions and revenue.
        """
        # ── Step 1: coalition → conversion rate ──────────────────────────────
        coalition_conv = defaultdict(lambda: [0, 0])  # {frozenset: [conversions, total]}

        for _, row in journeys.iterrows():
            key = frozenset(row["path"])
            coalition_conv[key][1] += 1
            if row["converted"]:
                coalition_conv[key][0] += 1

        conv_rate: dict[frozenset, float] = {
            k: v[0] / v[1] for k, v in coalition_conv.items() if v[1] > 0
        }

        n_ch = len(channels)
        if n_ch == 0:
            return _acc_to_df({}, channels)

        # ── Step 2: Shapley values ────────────────────────────────────────────
        if n_ch <= self.exact_limit:
            sv = self._exact_shapley(channels, conv_rate)
        else:
            warnings.warn(
                f"[ShapleyModel] {n_ch} channels exceeds exact_limit "
                f"({self.exact_limit}). Using Monte Carlo Shapley with "
                f"{self.n_iterations} permutations.",
                UserWarning,
            )
            sv = self._mc_shapley(channels, conv_rate)

        # Normalise so values sum to 1 (handle all-zero edge case)
        total_sv = sum(max(v, 0) for v in sv.values())

        # ── Step 3: scale to actual totals ────────────────────────────────────
        total_conv = int(journeys["converted"].sum())
        total_rev = float(journeys["revenue"].sum())

        acc = {}
        for ch in channels:
            raw = max(sv.get(ch, 0.0), 0.0)
            share = raw / total_sv if total_sv > 0 else 1.0 / n_ch
            acc[ch] = [share * total_conv, share * total_rev]

        return _acc_to_df(acc, channels)

    def _exact_shapley(
        self,
        channels: list[str],
        conv_rate: dict[frozenset, float],
    ) -> dict[str, float]:
        """
        Compute exact Shapley values by iterating over all 2^n channel subsets.

        For each channel, the Shapley value = weighted average marginal
        contribution over all coalitions that do not already contain that channel.
        """
        n = len(channels)
        sv = {ch: 0.0 for ch in channels}

        for ch in channels:
            others = [c for c in channels if c != ch]
            for size in range(len(others) + 1):
                for coalition in itertools.combinations(others, size):
                    s_without = frozenset(coalition)
                    s_with = s_without | {ch}
                    v_without = conv_rate.get(s_without, 0.0)
                    v_with = conv_rate.get(s_with, 0.0)
                    marginal = v_with - v_without
                    # Shapley weight = |S|! × (n-|S|-1)! / n!
                    weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
                    sv[ch] += weight * marginal

        return sv

    def _mc_shapley(
        self,
        channels: list[str],
        conv_rate: dict[frozenset, float],
    ) -> dict[str, float]:
        """
        Approximate Shapley values via random permutation sampling (Monte Carlo).

        Each iteration draws a random ordering of channels and measures the
        marginal contribution of each channel at its position in that ordering.
        """
        rng = random.Random(self.random_seed)
        len(channels)
        sv = {ch: 0.0 for ch in channels}

        for _ in range(self.n_iterations):
            perm = channels[:]
            rng.shuffle(perm)
            current: frozenset = frozenset()
            for ch in perm:
                next_set = current | {ch}
                marginal = conv_rate.get(next_set, 0.0) - conv_rate.get(current, 0.0)
                sv[ch] += marginal / self.n_iterations
                current = next_set

        return sv


class MarkovChainModel(_BaseAttributionModel):
    """
    Markov chain attribution using the removal effect.

    Constructs a transition probability matrix from observed channel sequences,
    then measures the drop in overall conversion probability when each channel
    is removed — that drop becomes the channel's attribution weight.

    Parameters
    ----------
    order : int
        Markov chain order.
        1 = next state depends only on current state (standard).
        2 = next state depends on current + previous state.
        Higher orders capture more path context but require more data.
        Default 1.
    n_simulations : int
        Number of random walk simulations for computing removal effects.
        Default 50_000.
    random_seed : int
        Reproducibility seed.  Default 42.
    """

    name = "markov"

    def __init__(
        self,
        order: int = 1,
        n_simulations: int = 50_000,
        random_seed: int = 42,
    ):
        """Validate order ≥ 1 and store configuration parameters."""
        if order < 1:
            raise ValueError("Markov chain order must be ≥ 1.")
        self.order = order
        self.n_simulations = n_simulations
        self.random_seed = random_seed

    def _compute(self, journeys, channels):
        """
        Compute removal-effect attribution via Markov chain simulation.

        Steps:
          1. Build (order-n) transition matrix from all journeys.
          2. Estimate baseline conversion rate via random walks.
          3. For each channel, zero out its rows in the transition matrix
             and re-estimate conversion rate.
          4. Removal effect = (baseline − reduced) / baseline.
          5. Normalise effects to sum to 1, scale to totals.
        """
        # ── Step 1: build transition matrix ──────────────────────────────────
        # States: channel names + "start", "conversion", "null" (non-conversion end)
        channels + ["start", "conversion", "null"]
        trans: dict[Any, dict[str, float]] = defaultdict(lambda: defaultdict(float))

        for _, row in journeys.iterrows():
            path = row["path"]
            if not path:
                continue
            converted = bool(row["converted"])

            # Build order-n state sequence
            seq = self._make_state_seq(path, converted)

            for i in range(len(seq) - 1):
                src = seq[i]
                dst = seq[i + 1]
                trans[src][dst] += 1.0

        # Normalise to probabilities
        prob: dict[Any, dict[str, float]] = {}
        for src, dst_counts in trans.items():
            total = sum(dst_counts.values())
            prob[src] = {dst: cnt / total for dst, cnt in dst_counts.items()}

        # ── Step 2: baseline conversion rate ─────────────────────────────────
        rng = np.random.default_rng(self.random_seed)
        baseline = self._simulate(prob, channels, rng)

        # ── Steps 3-4: removal effects ────────────────────────────────────────
        removal: dict[str, float] = {}
        for ch in channels:
            prob_removed = self._remove_channel(prob, ch, channels)
            reduced = self._simulate(prob_removed, channels, rng)
            removal[ch] = max(baseline - reduced, 0.0)

        # ── Step 5: normalise and scale ───────────────────────────────────────
        total_re = sum(removal.values())
        total_conv = int(journeys["converted"].sum())
        total_rev = float(journeys["revenue"].sum())

        acc = {}
        for ch in channels:
            share = removal[ch] / total_re if total_re > 0 else 1.0 / len(channels)
            acc[ch] = [share * total_conv, share * total_rev]

        return _acc_to_df(acc, channels)

    def _make_state_seq(self, path: list[str], converted: bool) -> list[Any]:
        """
        Build the state sequence for one journey at the configured order.

        For order=1: ["start", ch1, ch2, …, "conversion"/"null"]
        For order=2: ["start", (ch1,), (ch1,ch2), …, "conversion"/"null"]
        """
        end = "conversion" if converted else "null"
        if self.order == 1:
            return ["start"] + path + [end]
        # Higher-order: use tuples of the last `order` states
        seq: list[Any] = ["start"]
        buffer: list[str] = []
        for ch in path:
            buffer.append(ch)
            if len(buffer) >= self.order:
                seq.append(tuple(buffer[-self.order :]))
        seq.append(end)
        return seq

    def _simulate(
        self,
        prob: dict[Any, dict[str, float]],
        channels: list[str],
        rng: np.random.Generator,
    ) -> float:
        """
        Estimate the conversion probability via random walk simulation.

        Starts at "start", follows transition probabilities until reaching
        "conversion" or "null", and returns the fraction of runs that
        reached "conversion".
        """
        conversions = 0
        for _ in range(self.n_simulations):
            state: Any = "start"
            for _ in range(200):  # max path length guard
                if state == "conversion":
                    conversions += 1
                    break
                if state == "null" or state not in prob:
                    break
                dsts = list(prob[state].keys())
                ps = [prob[state][d] for d in dsts]
                idx = rng.choice(len(dsts), p=np.array(ps) / sum(ps))
                state = dsts[idx]
        return conversions / self.n_simulations

    def _remove_channel(
        self,
        prob: dict[Any, dict[str, float]],
        channel: str,
        channels: list[str],
    ) -> dict[Any, dict[str, float]]:
        """
        Return a new transition matrix with ``channel`` removed.

        Rows from the removed channel redistribute to "null" (user gives up).
        This is the standard removal-effect approach used by Shapley-Markov models.
        """
        new_prob: dict[Any, dict[str, float]] = {}
        for src, dst_prob in prob.items():
            # Skip source states that contain the removed channel
            if _state_contains(src, channel):
                continue
            new_dst = {}
            mass_removed = 0.0
            for dst, p in dst_prob.items():
                if _state_contains(dst, channel):
                    mass_removed += p
                else:
                    new_dst[dst] = p
            # Redistribute removed mass to "null"
            if mass_removed > 0:
                new_dst["null"] = new_dst.get("null", 0.0) + mass_removed
            total = sum(new_dst.values())
            if total > 0:
                new_prob[src] = {k: v / total for k, v in new_dst.items()}
        return new_prob


class DataDrivenModel(_BaseAttributionModel):
    """
    Data-driven attribution using logistic regression + SHAP values.

    Fits a logistic regression model that predicts conversion from a
    channel one-hot feature vector (channel presence in the journey).
    SHAP values from this model give the marginal contribution of each
    channel to each journey's predicted conversion probability.

    This approach mirrors the algorithm used by Google Analytics 4 for
    data-driven attribution.

    Parameters
    ----------
    regularisation : float
        Logistic regression C parameter (inverse of L2 penalty).
        Smaller = stronger regularisation.  Default 1.0.
    random_seed    : int   — Default 42.
    """

    name = "data_driven"

    def __init__(self, regularisation: float = 1.0, random_seed: int = 42):
        """Store regularisation strength and random seed."""
        self.regularisation = regularisation
        self.random_seed = random_seed

    def _compute(self, journeys, channels):
        """
        Fit logistic regression on journey features and compute SHAP attributions.

        Features: binary channel presence indicators + channel frequency (count).
        SHAP values: use the linear model's coefficients × feature values as an
        exact decomposition (equivalent to SHAP for linear models).

        Falls back gracefully when sklearn is not installed or when there are
        fewer than 10 converting journeys.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            warnings.warn(
                "[DataDrivenModel] scikit-learn is required for data-driven "
                "attribution.  Install with: pip install scikit-learn.  "
                "Falling back to linear attribution.",
                UserWarning,
            )
            return LinearModel()._compute(journeys, channels)

        n_conv = int(journeys["converted"].sum())
        if n_conv < 10:
            warnings.warn(
                f"[DataDrivenModel] Only {n_conv} converting journeys — too few "
                "for a reliable logistic model.  Minimum is 10.  "
                "Falling back to linear attribution.",
                UserWarning,
            )
            return LinearModel()._compute(journeys, channels)

        # ── Build feature matrix ──────────────────────────────────────────────
        # Features: presence + frequency of each channel in the path
        n = len(journeys)
        n_ch = len(channels)
        ch_idx = {ch: i for i, ch in enumerate(channels)}

        feat_mat = np.zeros((n, n_ch * 2))  # presence + frequency
        y = np.zeros(n)

        for row_i, (_, row) in enumerate(journeys.iterrows()):
            y[row_i] = float(row["converted"])
            for ch, w in zip(row["path"], row["weights"]):
                if ch in ch_idx:
                    j = ch_idx[ch]
                    feat_mat[row_i, j] = 1.0  # presence
                    feat_mat[row_i, j + n_ch] += w  # weighted frequency

        # Scale features
        scaler = StandardScaler()
        feat_sc = scaler.fit_transform(feat_mat)

        # ── Fit logistic regression ───────────────────────────────────────────
        lr = LogisticRegression(
            C=self.regularisation,
            max_iter=1000,
            random_state=self.random_seed,
            solver="lbfgs",
        )
        lr.fit(feat_sc, y)

        # ── SHAP values (linear exact decomposition) ──────────────────────────
        # For a linear model: SHAP_i(x) = coef_i × (x_i - E[x_i])
        coefs = lr.coef_[0]  # shape: (n_ch * 2,)
        mean_feat = feat_sc.mean(axis=0)

        # Attribution per channel = presence_shap + frequency_shap
        ch_shap = {}
        for ch, j in ch_idx.items():
            presence_shap = coefs[j] * (feat_sc[:, j] - mean_feat[j])
            freq_shap = coefs[j + n_ch] * (feat_sc[:, j + n_ch] - mean_feat[j + n_ch])
            # Keep only converting journeys for credit assignment
            converting_mask = y == 1
            ch_shap[ch] = float((presence_shap[converting_mask] + freq_shap[converting_mask]).sum())

        # Normalise and scale
        total_pos = sum(max(v, 0) for v in ch_shap.values())
        total_conv = int(journeys["converted"].sum())
        total_rev = float(journeys["revenue"].sum())

        acc = {}
        for ch in channels:
            share = max(ch_shap.get(ch, 0.0), 0.0) / total_pos if total_pos > 0 else 1.0 / n_ch
            acc[ch] = [share * total_conv, share * total_rev]

        return _acc_to_df(acc, channels)


class EnsembleModel(_BaseAttributionModel):
    """
    Ensemble attribution: weighted average of credits from multiple models.

    Combines the outputs of any subset of fitted models into a single
    channel credit table, reducing dependence on any one model's assumptions.

    Parameters
    ----------
    model_weights : dict {model_name: float}
        Weight for each model.  Weights are re-normalised to sum to 1.
        If None, equal weights are used.
    """

    name = "ensemble"

    def __init__(self, model_weights: dict[str, float] | None = None):
        """
        Store per-model weights.

        When model_weights is None the ensemble will compute equal weights
        across all models that were fitted on the same journeys.
        """
        self.model_weights = model_weights

    def combine(
        self,
        credits_by_model: dict[str, pd.DataFrame],
        channels: list[str],
    ) -> pd.DataFrame:
        """
        Blend channel credits from multiple pre-fitted models.

        Parameters
        ----------
        credits_by_model : dict {model_name: DataFrame}
            Each DataFrame must have columns: channel, attributed_conversions,
            attributed_revenue  (output of _BaseAttributionModel.fit()).
        channels : list of str

        Returns
        -------
        pd.DataFrame  — same schema as other model outputs, model="ensemble"
        """
        weights = self.model_weights or {m: 1.0 for m in credits_by_model}
        total_w = sum(weights.get(m, 0.0) for m in credits_by_model)
        if total_w <= 0:
            total_w = 1.0

        acc = defaultdict(lambda: [0.0, 0.0])
        for model_name, cdf in credits_by_model.items():
            w = weights.get(model_name, 0.0) / total_w
            for _, row in cdf.iterrows():
                ch = row["channel"]
                acc[ch][0] += w * float(row["attributed_conversions"])
                acc[ch][1] += w * float(row["attributed_revenue"])

        result = _acc_to_df(dict(acc), channels)
        result["model"] = "ensemble"
        return result

    def _compute(self, journeys, channels):
        """Not used directly — EnsembleModel.combine() is the entry point."""
        raise RuntimeError(
            "EnsembleModel should be used via combine(), not fit(). "
            "Pass credits_by_model from the other fitted models."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ATTRIBUTION ANALYZER (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════


class AttributionAnalyzer:
    """
    Fit one or more attribution models on journey data and produce a unified
    result object with channel credits, path-level breakdown, and model
    comparison metrics.

    Parameters
    ----------
    models : list of str
        Attribution models to fit.  Choose from:
        "last_click", "first_click", "linear", "position_based",
        "time_decay", "shapley", "markov", "data_driven", "ensemble".
        Default: all models.
    position_weights : dict, optional
        Override default position-based weights.
        Keys: "first", "last", "middle".  E.g. {"first": 0.3, "last": 0.5, "middle": 0.2}.
    time_decay_half_life : float
        Half-life in days for the time-decay model.  Default 7.
    markov_order : int
        Order of the Markov chain model.  Default 1.
    shapley_exact_limit : int
        Channel count threshold for switching Shapley to Monte Carlo.  Default 12.
    shapley_n_iterations : int
        Monte Carlo iterations for Shapley.  Default 5000.
    ensemble_weights : dict {model_name: float}, optional
        Weights for the ensemble model.  Default: equal weights.
    cost_col : str, optional
        Column in the journey DataFrame containing total touchpoint cost.
        Required for ROI-based budget allocation.
    random_seed : int
        Global seed passed to stochastic models.  Default 42.
    verbose : bool

    Examples
    --------
    >>> analyzer = AttributionAnalyzer(
    ...     models       = ["last_click", "shapley", "markov", "ensemble"],
    ...     markov_order = 2,
    ... )
    >>> result = analyzer.fit(journeys)
    >>> result.print_summary()
    >>> result.plot()
    """

    def __init__(
        self,
        models: list[str] | None = None,
        position_weights: dict[str, float] | None = None,
        time_decay_half_life: float = 7.0,
        markov_order: int = 1,
        shapley_exact_limit: int = _EXACT_SHAPLEY_LIMIT,
        shapley_n_iterations: int = _MC_SHAPLEY_ITERATIONS,
        ensemble_weights: dict[str, float] | None = None,
        cost_col: str | None = None,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Store configuration.  Model instances are created lazily during fit().
        """
        self.models = models or list(_SUPPORTED_MODELS)
        self.position_weights = position_weights or {}
        self.time_decay_half_life = time_decay_half_life
        self.markov_order = markov_order
        self.shapley_exact_limit = shapley_exact_limit
        self.shapley_n_iterations = shapley_n_iterations
        self.ensemble_weights = ensemble_weights
        self.cost_col = cost_col
        self.random_seed = random_seed
        self.verbose = verbose

        # Validate model names
        unknown = [m for m in self.models if m not in _SUPPORTED_MODELS]
        if unknown:
            raise ValueError(f"Unknown models: {unknown}.  " f"Supported: {_SUPPORTED_MODELS}")

    # ── public API ─────────────────────────────────────────────────────────────

    def fit(self, journeys: pd.DataFrame) -> AttributionResult:
        """
        Fit all configured attribution models on a journey DataFrame.

        Parameters
        ----------
        journeys : pd.DataFrame
            Output of JourneyBuilder.build().  Must contain columns:
            path, weights, converted, revenue, journey_days, n_touchpoints.

        Returns
        -------
        AttributionResult
        """
        if len(journeys) == 0:
            raise ValueError("journeys DataFrame is empty.")

        channels = sorted({ch for path in journeys["path"] for ch in path})
        if not channels:
            raise ValueError("No channels found in journeys['path'].")

        self._log(
            f"Fitting {len(self.models)} model(s) on "
            f"{len(journeys):,d} journeys × {len(channels)} channels."
        )

        # ── Fit all models ─────────────────────────────────────────────────────
        model_instances = self._build_model_instances()
        credits_list: list[pd.DataFrame] = []
        credits_by_model: dict[str, pd.DataFrame] = {}

        non_ensemble = [m for m in self.models if m != "ensemble"]
        for model_name in non_ensemble:
            self._log(f"  → {model_name} …")
            inst = model_instances[model_name]
            cdf = inst.fit(journeys, channels)
            credits_list.append(cdf)
            credits_by_model[model_name] = cdf.copy()

        # Ensemble needs the others already computed
        if "ensemble" in self.models:
            self._log("  → ensemble …")
            ens = EnsembleModel(model_weights=self.ensemble_weights)
            ens_cdf = ens.combine(credits_by_model, channels)
            credits_list.append(ens_cdf)
            credits_by_model["ensemble"] = ens_cdf.copy()

        # ── Assemble channel_credits table ─────────────────────────────────────
        channel_credits = pd.concat(credits_list, ignore_index=True)
        channel_credits = self._enrich_credits(channel_credits, journeys)

        # ── Journey-level credits table ────────────────────────────────────────
        # Use enriched table (has credit_share column) keyed by model
        enriched_by_model = {
            m: channel_credits[channel_credits["model"] == m].copy() for m in credits_by_model
        }
        journey_credits = self._build_journey_credits(journeys, channels, enriched_by_model)

        # ── Model comparison ───────────────────────────────────────────────────
        evaluator = AttributionEvaluator()
        model_comparison = evaluator.compare(
            journeys, channel_credits, list(credits_by_model.keys()), channels
        )

        total_conv = int(journeys["converted"].sum())
        total_rev = float(journeys["revenue"].sum())

        self._log(f"Done.  {total_conv:,d} conversions · £{total_rev:,.2f} revenue attributed.")

        return AttributionResult(
            channel_credits=channel_credits,
            journey_credits=journey_credits,
            model_comparison=model_comparison,
            channels=channels,
            total_conversions=total_conv,
            total_revenue=total_rev,
            models_fitted=list(credits_by_model.keys()),
            metadata={
                "markov_order": self.markov_order,
                "cost_col": self.cost_col,
                "random_seed": self.random_seed,
            },
        )

    # ── internal helpers ───────────────────────────────────────────────────────

    def _build_model_instances(self) -> dict[str, _BaseAttributionModel]:
        """Instantiate each configured model with its parameters."""
        pw = self.position_weights
        return {
            "last_click": LastClickModel(),
            "first_click": FirstClickModel(),
            "linear": LinearModel(),
            "position_based": PositionBasedModel(
                first_weight=pw.get("first", 0.40),
                last_weight=pw.get("last", 0.40),
                middle_weight=pw.get("middle", 0.20),
            ),
            "time_decay": TimeDecayModel(half_life_days=self.time_decay_half_life),
            "shapley": ShapleyModel(
                exact_limit=self.shapley_exact_limit,
                n_iterations=self.shapley_n_iterations,
                random_seed=self.random_seed,
            ),
            "markov": MarkovChainModel(
                order=self.markov_order,
                random_seed=self.random_seed,
            ),
            "data_driven": DataDrivenModel(random_seed=self.random_seed),
            "ensemble": EnsembleModel(model_weights=self.ensemble_weights),
        }

    def _enrich_credits(
        self,
        credits: pd.DataFrame,
        journeys: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add derived columns to the channel_credits table.

        Adds: credit_share, attributed_revenue_share, roi (when cost_col available),
        cost (when cost_col available).
        """
        float(journeys["converted"].sum()) or 1.0
        float(journeys["revenue"].sum()) or 1.0

        credits = credits.copy()
        credits["credit_share"] = credits.groupby("model")["attributed_conversions"].transform(
            lambda x: x / max(float(x.sum()), 1e-9)
        )
        credits["attributed_revenue_share"] = credits.groupby("model")[
            "attributed_revenue"
        ].transform(lambda x: x / max(float(x.sum()), 1e-9))

        # ROI = attributed_revenue / cost (when cost column is available)
        if self.cost_col and self.cost_col in journeys.columns:
            ch_cost = (
                journeys.explode("path")
                .rename(columns={"path": "channel"})
                .groupby("channel")[self.cost_col]
                .sum()
                .rename("cost")
            )
            credits = credits.merge(ch_cost.reset_index(), on="channel", how="left")
            credits["roi"] = credits["attributed_revenue"] / credits["cost"].fillna(0).clip(
                lower=1e-9
            )

        return credits

    def _build_journey_credits(
        self,
        journeys: pd.DataFrame,
        channels: list[str],
        credits_by_model: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Build a per-journey credit breakdown for path-level analysis.

        For each converting journey, computes each model's implied
        attribution to the channels present in that journey.
        """
        rows = []
        for model_name, cdf in credits_by_model.items():
            ch_share = cdf.set_index("channel")["credit_share"].to_dict()
            for _, jrow in journeys[journeys["converted"] == 1].iterrows():
                path = jrow["path"]
                rev = float(jrow["revenue"])
                path_chs = set(path)
                # re-normalise shares to channels actually in this path
                in_path = {ch: ch_share.get(ch, 0.0) for ch in path_chs}
                total = sum(in_path.values()) or 1.0
                for ch in path:
                    share = in_path.get(ch, 0.0) / total
                    rows.append(
                        {
                            "journey_id": jrow.get("journey_id", None),
                            "model": model_name,
                            "channel": ch,
                            "attributed_revenue": share * rev,
                            "path_length": len(path),
                            "channel_position": path.index(ch),
                        }
                    )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _log(self, msg: str) -> None:
        """Print msg to stdout when verbose=True."""
        if self.verbose:
            print(f"[AttributionAnalyzer] {msg}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════════


class AttributionEvaluator:
    """
    Compare attribution models using multiple diagnostic metrics and produce
    a ranked model comparison table.

    Evaluation dimensions
    ----------------------
    1. Normalisation accuracy  — does total attributed revenue equal actual revenue?
    2. Path coverage           — fraction of converting paths where the top-credited
                                 channel appears in the path.
    3. Model agreement (pairwise) — Spearman rank correlation between channel
                                    credit rankings across models.
    4. Stability               — variance of channel credits across random
                                 halves of the data (bootstrap).
    5. Conversion alignment    — Pearson correlation between attributed conversion
                                 share and raw channel conversion rate.
    """

    def compare(
        self,
        journeys: pd.DataFrame,
        channel_credits: pd.DataFrame,
        models: list[str],
        channels: list[str],
        n_bootstrap: int = 20,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Compute evaluation metrics for each model and return a ranked DataFrame.

        Parameters
        ----------
        journeys        : journey DataFrame (output of JourneyBuilder.build()).
        channel_credits : combined channel_credits from AttributionAnalyzer.fit().
        models          : list of model names to evaluate.
        channels        : list of channel names.
        n_bootstrap     : number of bootstrap halves for stability metric.
        random_seed     : reproducibility seed.

        Returns
        -------
        pd.DataFrame with one row per model and columns:
            model, normalisation_error, path_coverage, conversion_alignment,
            stability_score, cross_model_agreement, composite_score, rank
        """
        total_rev = float(journeys["revenue"].sum())
        int(journeys["converted"].sum())

        rows = []
        for model in models:
            mdf = channel_credits[channel_credits["model"] == model]

            norm_err = self._normalisation_error(mdf, total_rev)
            coverage = self._path_coverage(mdf, journeys)
            conv_align = self._conversion_alignment(mdf, journeys, channels)
            stability = self._stability(journeys, channels, model, n_bootstrap, random_seed)

            rows.append(
                {
                    "model": model,
                    "normalisation_error": round(norm_err, 4),
                    "path_coverage": round(coverage, 4),
                    "conversion_alignment": round(conv_align, 4),
                    "stability_score": round(stability, 4),
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Add pairwise agreement (mean Spearman correlation with other models)
        agree_series = self._cross_model_agreement(channel_credits, models, channels)
        df["cross_model_agreement"] = df["model"].map(agree_series).fillna(0.0)

        # Composite score: higher is better
        # Weights: normalisation (inverse error) 25%, coverage 25%,
        #          alignment 20%, stability 20%, agreement 10%
        df["composite_score"] = (
            0.25 * (1.0 - df["normalisation_error"].clip(0, 1))
            + 0.25 * df["path_coverage"]
            + 0.20 * df["conversion_alignment"].clip(0, 1)
            + 0.20 * df["stability_score"]
            + 0.10 * df["cross_model_agreement"].clip(0, 1)
        )
        df["rank"] = df["composite_score"].rank(ascending=False).astype(int)
        df = df.sort_values("rank").reset_index(drop=True)
        return df

    # ── metric helpers ────────────────────────────────────────────────────────

    def _normalisation_error(self, mdf: pd.DataFrame, total_rev: float) -> float:
        """
        Measure how accurately the model preserves total revenue.

        Returns |attributed_total - actual_total| / actual_total.
        A perfect model returns 0.0.
        """
        attributed = float(mdf["attributed_revenue"].sum())
        if total_rev == 0:
            return 0.0
        return abs(attributed - total_rev) / total_rev

    def _path_coverage(self, mdf: pd.DataFrame, journeys: pd.DataFrame) -> float:
        """
        Fraction of converting journeys where the model's top-credited channel
        actually appears in the journey path.

        High coverage indicates the model is crediting channels that users
        actually interacted with, rather than channels absent from the path.
        """
        top_ch = mdf.sort_values("attributed_conversions", ascending=False).iloc[0]["channel"]
        converting = journeys[journeys["converted"] == 1]
        if len(converting) == 0:
            return 0.0
        covered = converting["path"].apply(lambda p: top_ch in p)
        return float(covered.mean())

    def _conversion_alignment(
        self,
        mdf: pd.DataFrame,
        journeys: pd.DataFrame,
        channels: list[str],
    ) -> float:
        """
        Pearson correlation between a model's attributed conversion share
        and each channel's raw (last-click) conversion rate.

        A high correlation means the model broadly agrees with the naive
        signal in the data — useful as a sanity check.
        """
        # Raw last-click conversion count per channel
        raw = defaultdict(float)
        for _, row in journeys[journeys["converted"] == 1].iterrows():
            if row["path"]:
                raw[row["path"][-1]] += 1.0
        total_raw = sum(raw.values()) or 1.0

        model_shares = mdf.set_index("channel")["attributed_conversions"].to_dict()
        total_model = sum(model_shares.values()) or 1.0

        x = np.array([raw.get(ch, 0.0) / total_raw for ch in channels])
        y = np.array([model_shares.get(ch, 0.0) / total_model for ch in channels])

        if x.std() == 0 or y.std() == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    def _stability(
        self,
        journeys: pd.DataFrame,
        channels: list[str],
        model_name: str,
        n_bootstrap: int,
        random_seed: int,
    ) -> float:
        """
        Estimate model stability by comparing credit distributions across
        random 50% subsamples.

        Returns 1 - mean_coefficient_of_variation across channels.
        A value close to 1.0 means channel credits are consistent across
        data subsets.
        """
        rng = np.random.default_rng(random_seed)

        # Use a fast proxy: last_click for stability (expensive models
        # like Shapley/Markov would be re-run n_bootstrap times — too slow)
        # Instead, compute stability on the journey-level distribution
        converting = journeys[journeys["converted"] == 1]
        if len(converting) < 20:
            return 0.5  # not enough data to estimate

        channel_cvs = []
        for ch in channels:
            shares = []
            for _ in range(n_bootstrap):
                sample = converting.sample(
                    frac=0.5, replace=True, random_state=int(rng.integers(0, 99999))
                )
                total = len(sample)
                ch_count = sum(1 for _, r in sample.iterrows() if r["path"] and ch in r["path"])
                shares.append(ch_count / total if total > 0 else 0.0)
            if np.mean(shares) > 0:
                channel_cvs.append(np.std(shares) / np.mean(shares))

        if not channel_cvs:
            return 0.5
        return float(np.clip(1.0 - np.mean(channel_cvs), 0.0, 1.0))

    def _cross_model_agreement(
        self,
        channel_credits: pd.DataFrame,
        models: list[str],
        channels: list[str],
    ) -> pd.Series:
        """
        For each model, compute its mean Spearman rank correlation with all
        other models' channel credit orderings.

        High agreement with peers is a sign the model is not an outlier.
        Returns a Series indexed by model name.
        """
        from scipy.stats import spearmanr

        # Build a matrix: rows=channels, cols=models
        mat = pd.DataFrame(index=channels, columns=models, dtype=float)
        for model in models:
            mdf = channel_credits[channel_credits["model"] == model]
            for _, row in mdf.iterrows():
                mat.loc[row["channel"], model] = float(row["attributed_conversions"])
        mat = mat.fillna(0.0)

        agreement = {}
        for model in models:
            corrs = []
            for other in models:
                if other == model:
                    continue
                r, _ = spearmanr(mat[model].values, mat[other].values)
                corrs.append(float(r) if not np.isnan(r) else 0.0)
            agreement[model] = float(np.mean(corrs)) if corrs else 0.0

        return agreement


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_attribution(
    result: AttributionResult,
    model: str | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (18, 14),
) -> None:
    """
    Render a six-panel attribution analysis figure.

    Panels
    ------
    1. Channel credit comparison across all models (grouped bars).
    2. Credit share sankey-style breakdown for the selected model.
    3. Revenue attribution by channel (stacked bar per model).
    4. Path length distribution (converting vs non-converting).
    5. Top-10 most common converting paths (horizontal bar).
    6. Model evaluation radar chart (composite score dimensions).

    Parameters
    ----------
    result    : AttributionResult  — output of AttributionAnalyzer.fit().
    model     : str, optional      — highlight this model in single-model panels.
                                     Defaults to result.best_model().
    save_path : str, optional      — save figure to this path.
    figsize   : (width, height)
    """
    highlight = model or result.best_model()
    ch_colors = _ch_colour(result.channels)

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        "Attribution Analysis",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1], projection="polar")

    _plot_credit_comparison(ax1, result, ch_colors)
    _plot_credit_share_bar(ax2, result, highlight, ch_colors)
    _plot_revenue_by_model(ax3, result, ch_colors)
    _plot_path_length_dist(ax4, result)
    _plot_top_paths(ax5, result)
    _plot_model_radar(ax6, result)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def _plot_credit_comparison(ax, result, ch_colors):
    """
    Grouped bar chart: attributed conversion share per channel, one group of
    bars per channel, one bar per model.
    """
    df = result.channel_credits
    channels = result.channels
    models = result.models_fitted
    n_ch = len(channels)
    n_m = len(models)
    x = np.arange(n_ch)
    bw = 0.8 / n_m

    model_colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}

    for j, model in enumerate(models):
        mdf = df[df["model"] == model].set_index("channel")
        shares = [float(mdf.loc[ch, "credit_share"]) if ch in mdf.index else 0.0 for ch in channels]
        ax.bar(
            x + j * bw - 0.4 + bw / 2,
            shares,
            width=bw * 0.92,
            label=model,
            color=model_colors[model],
            alpha=0.82,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=20, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pct))
    ax.set_title("Conversion Credit Share by Model", fontsize=10, fontweight="bold")
    ax.set_ylabel("Credit share", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1.0)


def _plot_credit_share_bar(ax, result, model, ch_colors):
    """
    Horizontal stacked bar showing how a single model distributes credit
    across channels.
    """
    mdf = result.get_credits(model).sort_values("credit_share", ascending=False)
    shares = mdf["credit_share"].values
    labels = mdf["channel"].values
    left = 0.0
    for ch, share, color in zip(labels, shares, [ch_colors.get(c, _GREY) for c in labels]):
        ax.barh(0, share, left=left, color=color, label=ch, height=0.5)
        if share > 0.04:
            ax.text(
                left + share / 2,
                0,
                f"{ch}\n{share:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
        left += share
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_pct))
    ax.set_yticks([])
    ax.set_title(
        f"Credit Distribution — {model.replace('_', ' ').title()}", fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Credit share", fontsize=9)


def _plot_revenue_by_model(ax, result, ch_colors):
    """
    Stacked horizontal bar chart: attributed revenue per model, coloured by channel.
    """
    models = result.models_fitted
    df = result.channel_credits
    channels = result.channels

    bottoms = np.zeros(len(models))
    for ch in channels:
        vals = []
        for model in models:
            mdf = df[(df["model"] == model) & (df["channel"] == ch)]
            vals.append(float(mdf["attributed_revenue"].values[0]) if len(mdf) else 0.0)
        ax.barh(models, vals, left=bottoms, color=ch_colors.get(ch, _GREY), label=ch, alpha=0.85)
        bottoms += np.array(vals)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_money))
    ax.set_title("Attributed Revenue by Model", fontsize=10, fontweight="bold")
    ax.set_xlabel("Revenue", fontsize=9)
    handles, labels_ = ax.get_legend_handles_labels()
    # Deduplicate
    seen = {}
    for h, lbl in zip(handles, labels_):
        seen[lbl] = h
    ax.legend(seen.values(), seen.keys(), fontsize=7, loc="lower right")


def _plot_path_length_dist(ax, result):
    """
    Histogram of path lengths for converting journeys, drawn from the
    journey_credits table's path_length column.
    """
    jc = result.journey_credits
    if jc is None or jc.empty or "path_length" not in jc.columns:
        ax.set_title("Path Length Distribution (no data)", fontsize=10)
        ax.axis("off")
        return

    lengths = jc["path_length"].dropna()
    if len(lengths) == 0:
        ax.set_title("Path Length Distribution (no data)", fontsize=10)
        ax.axis("off")
        return

    max_len = max(int(lengths.max()), 2)
    bins = list(range(1, min(max_len + 2, 17)))
    ax.hist(lengths, bins=bins, color=_BLUE, alpha=0.75, density=True, edgecolor="white")
    ax.set_title("Path Length Distribution (Converting)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Touchpoints in path", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    mean_len = float(lengths.mean())
    ax.axvline(mean_len, color=_ORANGE, lw=1.8, linestyle="--", label=f"Mean = {mean_len:.1f}")
    ax.legend(fontsize=8)


def _plot_top_paths(ax, result):
    """
    Horizontal bar chart of the 10 most frequent converting channel sequences.
    """
    jc = result.journey_credits
    if jc.empty:
        ax.set_title("Top Converting Paths (no data)", fontsize=10)
        return

    # Reconstruct path strings from journey_credits (grouped by journey_id + model)
    # Use first model only
    first_model = result.models_fitted[0]
    path_data = (
        jc[jc["model"] == first_model]
        .groupby("journey_id")
        .apply(lambda g: " → ".join(g.sort_values("channel_position")["channel"].tolist()))
        .reset_index(name="path_str")
    )
    counts = path_data["path_str"].value_counts().head(10)

    ax.barh(counts.index[::-1], counts.values[::-1], color=_TEAL, alpha=0.82)
    ax.set_title("Top 10 Converting Paths", fontsize=10, fontweight="bold")
    ax.set_xlabel("Journey count", fontsize=9)
    ax.tick_params(labelsize=7)


def _plot_model_radar(ax, result):
    """
    Radar (spider) chart comparing models across evaluation dimensions.
    """
    mc = result.model_comparison
    if mc.empty:
        ax.set_title("Model Evaluation (no comparison data)", fontsize=10)
        return

    dims = [
        "path_coverage",
        "conversion_alignment",
        "stability_score",
        "cross_model_agreement",
        "composite_score",
    ]
    dims = [d for d in dims if d in mc.columns]
    n_d = len(dims)
    if n_d < 3:
        ax.set_title("Model Evaluation", fontsize=10)
        ax.axis("off")
        return

    angles = [2 * math.pi * i / n_d for i in range(n_d)] + [0]
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [d.replace("_", " ").title() for d in dims],
        fontsize=7,
    )
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=6, color="grey")

    for i, row in mc.iterrows():
        vals = [float(row.get(d, 0.0)) for d in dims]
        vals = [max(0.0, min(1.0, v)) for v in vals]
        vals += [vals[0]]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(angles, vals, color=color, lw=2, label=str(row["model"]))
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.set_title("Model Evaluation Dimensions", fontsize=10, fontweight="bold", pad=18)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.1))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — BUDGET ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AttributionBudgetAllocation:
    """
    Budget allocation recommendation derived from attribution model outputs.

    Attributes
    ----------
    total_budget       : float — budget that was distributed.
    allocations        : pd.DataFrame — one row per channel with columns:
                         channel, current_spend, recommended_spend,
                         spend_change, spend_change_pct,
                         attributed_revenue, credit_share,
                         roi (when cost data is available),
                         roi_weighted_share (allocation b),
                         saturation_aware_share (allocation c, when available).
    method             : str — "revenue_share", "roi_weighted", or "hybrid".
    model_used         : str — attribution model that drove the allocation.
    notes              : str
    """

    total_budget: float
    allocations: pd.DataFrame
    method: str
    model_used: str
    notes: str = ""

    def print_summary(self) -> None:
        """Print a formatted budget allocation summary to stdout."""
        sep = "=" * 72
        print(sep)
        print("  ADSAT — ATTRIBUTION BUDGET ALLOCATION")
        print(sep)
        print(f"  Total budget  : {self.total_budget:>15,.0f}")
        print(f"  Method        : {self.method}")
        print(f"  Attribution   : {self.model_used}")
        if self.notes:
            print(f"  Notes         : {self.notes}")
        print()
        disp_cols = [
            c
            for c in [
                "channel",
                "current_spend",
                "recommended_spend",
                "spend_change_pct",
                "attributed_revenue",
                "credit_share",
                "roi",
            ]
            if c in self.allocations.columns
        ]
        fmt = self.allocations[disp_cols].copy()
        if "credit_share" in fmt.columns:
            fmt["credit_share"] = fmt["credit_share"].map("{:.1%}".format)
        if "spend_change_pct" in fmt.columns:
            fmt["spend_change_pct"] = fmt["spend_change_pct"].map(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A"
            )
        for c in ("current_spend", "recommended_spend", "attributed_revenue"):
            if c in fmt.columns:
                fmt[c] = fmt[c].map(lambda v: f"{v:>12,.0f}" if pd.notna(v) else "N/A")
        if "roi" in fmt.columns:
            fmt["roi"] = fmt["roi"].map(lambda v: f"{v:.2f}x" if pd.notna(v) else "N/A")
        print(fmt.to_string(index=False))
        print(sep)

    def plot(self, save_path: str | None = None) -> None:
        """
        Render a three-panel budget allocation figure.

        Panels: current vs recommended spend, spend change %, channel ROI.
        """
        df = self.allocations
        chans = df["channel"].tolist()
        n = len(chans)
        colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"Budget Allocation — {self.method.replace('_', ' ').title()} " f"({self.model_used})",
            fontsize=12,
            fontweight="bold",
        )

        # Panel 1: current vs recommended spend
        ax = axes[0]
        x = np.arange(n)
        bw = 0.35
        if "current_spend" in df.columns:
            ax.bar(
                x - bw / 2, df["current_spend"], width=bw, color=_GREY, alpha=0.7, label="Current"
            )
        ax.bar(
            x + bw / 2,
            df["recommended_spend"],
            width=bw,
            color=[c for c in colors],
            alpha=0.85,
            label="Recommended",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(chans, rotation=20, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_fmt_money))
        ax.set_title("Current vs Recommended Spend", fontsize=10)
        ax.legend(fontsize=8)

        # Panel 2: spend change %
        ax = axes[1]
        if "spend_change_pct" in df.columns:
            vals = df["spend_change_pct"].fillna(0).values
            bcolors = [_GREEN if v >= 0 else _RED for v in vals]
            ax.bar(chans, vals, color=bcolors, alpha=0.85)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticklabels(chans, rotation=20, ha="right", fontsize=8)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
            ax.set_title("Spend Change %", fontsize=10)
        else:
            ax.axis("off")

        # Panel 3: ROI by channel
        ax = axes[2]
        if "roi" in df.columns:
            roi_vals = df["roi"].fillna(0).values
            ax.bar(chans, roi_vals, color=colors, alpha=0.85)
            ax.set_xticklabels(chans, rotation=20, ha="right", fontsize=8)
            ax.set_title("Channel ROI (attributed revenue / cost)", fontsize=10)
            ax.set_ylabel("ROI", fontsize=9)
        else:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                "ROI not available\n(no cost column provided)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color=_GREY,
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)


class AttributionBudgetAdvisor:
    """
    Translate attribution model outputs into a budget reallocation recommendation.

    Two allocation methods are supported and can be used independently or
    combined into a hybrid:

    Method B — ROI-weighted allocation
        channel_share = (attributed_revenue / cost) / Σ(attributed_revenue / cost)
        Channels with higher return per pound of spend receive more budget.
        Requires a cost column in the journey data.

    Method C — Saturation-aware allocation (hybrid)
        Uses Method B as a starting point, then adjusts for diminishing returns
        by integrating with an adsat.campaign CampaignBatchResult (fitted
        saturation curves).  Channels near saturation receive a budget cap.

    When cost data is unavailable, both methods fall back to revenue-share
    allocation (channel_share = attributed_revenue / total_attributed_revenue).

    Parameters
    ----------
    total_budget     : float   — budget to distribute.
    method           : str     — "roi_weighted" (B), "saturation_aware" (C),
                                 or "revenue_share" (simple baseline).
    min_spend        : float or dict — per-channel spend floor.  Default 0.
    max_spend        : float or dict — per-channel spend cap.  Default unbounded.
    current_spend    : dict {channel: float}, optional — current spend per channel.
                       Used to compute spend_change columns.
    """

    def __init__(
        self,
        total_budget: float,
        method: str = "roi_weighted",
        min_spend: float | dict[str, float] = 0.0,
        max_spend: float | dict[str, float] | None = None,
        current_spend: dict[str, float] | None = None,
        verbose: bool = True,
    ):
        """
        Validate parameters and store configuration.  No allocation occurs here.
        """
        if total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {total_budget}.")
        if method not in ("roi_weighted", "saturation_aware", "revenue_share"):
            raise ValueError(
                f"method must be 'roi_weighted', 'saturation_aware', or "
                f"'revenue_share'.  Got '{method}'."
            )
        self.total_budget = float(total_budget)
        self.method = method
        self.min_spend = min_spend
        self.max_spend = max_spend
        self.current_spend = current_spend or {}
        self.verbose = verbose

    def recommend(
        self,
        result: AttributionResult,
        model: str | None = None,
        batch_result: Any | None = None,  # CampaignBatchResult from adsat.campaign
    ) -> AttributionBudgetAllocation:
        """
        Compute budget recommendations from an AttributionResult.

        Parameters
        ----------
        result       : AttributionResult — output of AttributionAnalyzer.fit().
        model        : str, optional — which attribution model to use.
                       Defaults to result.best_model().
        batch_result : CampaignBatchResult, optional — required for
                       method="saturation_aware".

        Returns
        -------
        AttributionBudgetAllocation
        """
        model = model or result.best_model()
        if model not in result.models_fitted:
            raise KeyError(
                f"Model '{model}' was not fitted.  " f"Available: {result.models_fitted}"
            )

        mdf = result.get_credits(model).copy()
        channels = mdf["channel"].tolist()

        # ── Compute raw shares ────────────────────────────────────────────────
        if self.method == "roi_weighted" and "roi" in mdf.columns:
            raw_shares = self._roi_weighted_shares(mdf)
            method_used = "roi_weighted"
        elif self.method == "saturation_aware" and batch_result is not None:
            raw_shares = self._saturation_aware_shares(mdf, batch_result)
            method_used = "saturation_aware"
        else:
            if self.method != "revenue_share":
                warnings.warn(
                    f"[AttributionBudgetAdvisor] Falling back to revenue_share "
                    f"(method='{self.method}' requires "
                    f"{'cost column' if self.method=='roi_weighted' else 'batch_result'}).",
                    UserWarning,
                )
            raw_shares = mdf.set_index("channel")["attributed_revenue_share"].to_dict()
            method_used = "revenue_share"

        # ── Apply floor/cap and allocate ──────────────────────────────────────
        recommended = self._allocate(raw_shares, channels)

        # ── Build output table ────────────────────────────────────────────────
        rows = []
        for ch in channels:
            rec_sp = float(recommended.get(ch, 0.0))
            curr_sp = float(self.current_spend.get(ch, 0.0))
            ch_mdf = mdf[mdf["channel"] == ch]
            rows.append(
                {
                    "channel": ch,
                    "current_spend": curr_sp if curr_sp > 0 else None,
                    "recommended_spend": round(rec_sp, 2),
                    "spend_change": round(rec_sp - curr_sp, 2) if curr_sp > 0 else None,
                    "spend_change_pct": (
                        round((rec_sp - curr_sp) / max(curr_sp, 1.0) * 100, 1)
                        if curr_sp > 0
                        else None
                    ),
                    "attributed_revenue": (
                        float(ch_mdf["attributed_revenue"].values[0]) if len(ch_mdf) else 0.0
                    ),
                    "credit_share": float(ch_mdf["credit_share"].values[0]) if len(ch_mdf) else 0.0,
                    "roi": (
                        float(ch_mdf["roi"].values[0])
                        if "roi" in ch_mdf.columns and len(ch_mdf)
                        else None
                    ),
                }
            )

        alloc_df = pd.DataFrame(rows)

        notes = ""
        if method_used == "saturation_aware":
            notes = "Saturation curves applied: channels near saturation capped."

        return AttributionBudgetAllocation(
            total_budget=self.total_budget,
            allocations=alloc_df,
            method=method_used,
            model_used=model,
            notes=notes,
        )

    # ── allocation helpers ────────────────────────────────────────────────────

    def _roi_weighted_shares(self, mdf: pd.DataFrame) -> dict[str, float]:
        """
        Compute budget shares proportional to each channel's ROI.

        ROI = attributed_revenue / cost.  Channels with missing or zero
        cost fall back to revenue-share weighting.
        """
        roi_vals = mdf.set_index("channel")["roi"].fillna(0.0).to_dict()
        total = sum(max(v, 0.0) for v in roi_vals.values())
        if total == 0:
            return mdf.set_index("channel")["attributed_revenue_share"].to_dict()
        return {ch: max(v, 0.0) / total for ch, v in roi_vals.items()}

    def _saturation_aware_shares(
        self,
        mdf: pd.DataFrame,
        batch_result: Any,
    ) -> dict[str, float]:
        """
        Adjust ROI-weighted shares downward for channels that are near or
        beyond their saturation point, using fitted saturation curves from
        adsat.campaign.CampaignBatchResult.

        Channels beyond 90% of their saturation point are capped at their
        pct_of_saturation ratio relative to the remaining budget.
        """
        roi_shares = self._roi_weighted_shares(mdf)
        adjusted = roi_shares.copy()

        for ch in mdf["channel"].tolist():
            try:
                cr = batch_result.get(ch)
                if cr.pct_of_saturation and cr.pct_of_saturation > 90:
                    # Discount proportional to how far past saturation
                    discount = 1.0 - (cr.pct_of_saturation - 90) / 100.0
                    adjusted[ch] = roi_shares.get(ch, 0.0) * max(discount, 0.1)
            except (KeyError, AttributeError):
                pass  # channel not in saturation results — keep as is

        total = sum(max(v, 0.0) for v in adjusted.values()) or 1.0
        return {ch: v / total for ch, v in adjusted.items()}

    def _allocate(
        self,
        shares: dict[str, float],
        channels: list[str],
    ) -> dict[str, float]:
        """
        Apply floor / cap constraints and allocate the total budget.

        Uses a clamp-and-renormalise procedure: clip each channel's raw
        allocation to [min_spend, max_spend], then redistribute the
        residual proportionally until all constraints are satisfied.
        """
        alloc = {ch: shares.get(ch, 0.0) * self.total_budget for ch in channels}

        for _ in range(50):  # iterative clamping
            clamped_low = {}
            clamped_high = {}
            free_channels = []
            free_total = 0.0

            for ch in channels:
                lo = _resolve_val(self.min_spend, ch, 0.0)
                hi = _resolve_val(self.max_spend, ch, float("inf"))
                if alloc.get(ch, 0.0) < lo:
                    clamped_low[ch] = lo
                elif alloc.get(ch, 0.0) > hi:
                    clamped_high[ch] = hi
                else:
                    free_channels.append(ch)
                    free_total += shares.get(ch, 0.0)

            if not clamped_low and not clamped_high:
                break

            fixed_spend = sum(clamped_low.values()) + sum(clamped_high.values())
            remaining = self.total_budget - fixed_spend

            for ch in clamped_low:
                alloc[ch] = clamped_low[ch]
            for ch in clamped_high:
                alloc[ch] = clamped_high[ch]

            if free_total > 0 and remaining > 0:
                for ch in free_channels:
                    alloc[ch] = (shares.get(ch, 0.0) / free_total) * remaining
            else:
                for ch in free_channels:
                    alloc[ch] = remaining / len(free_channels) if free_channels else 0.0

        return alloc


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def _acc_to_df(
    acc: dict[str, list[float]],
    channels: list[str],
) -> pd.DataFrame:
    """
    Convert a channel → [conversions, revenue] accumulator dict to a DataFrame.

    Fills in zero rows for channels not present in the accumulator, ensuring
    every model's output has the same set of channels.
    """
    rows = []
    for ch in channels:
        v = acc.get(ch, [0.0, 0.0])
        rows.append(
            {
                "channel": ch,
                "attributed_conversions": round(float(v[0]), 4),
                "attributed_revenue": round(float(v[1]), 4),
            }
        )
    return pd.DataFrame(rows)


def _state_contains(state: Any, channel: str) -> bool:
    """
    Return True if the Markov state (str or tuple) includes ``channel``.

    Used by MarkovChainModel to remove a channel from the transition matrix.
    """
    if isinstance(state, str):
        return state == channel
    if isinstance(state, tuple):
        return channel in state
    return False


def _resolve_val(
    param: float | dict | None,
    channel: str,
    default: float,
) -> float:
    """
    Resolve a scalar-or-dict parameter to a per-channel float value.

    Used by AttributionBudgetAdvisor for min_spend / max_spend.
    """
    if param is None:
        return default
    if isinstance(param, dict):
        return float(param.get(channel, default))
    return float(param)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CONVENIENCE FUNCTION + SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════


def attribute_campaigns(
    events: pd.DataFrame,
    models: list[str] | None = None,
    user_col: str = "user_id",
    time_col: str = "timestamp",
    channel_col: str = "channel",
    interaction_col: str = "interaction_type",
    converted_col: str = "converted",
    revenue_col: str = "revenue",
    cost_col: str | None = None,
    lookback_days: int | None = 30,
    interaction_weight: dict[str, float] | None = None,
    multi_conversion: str = "reset",
    markov_order: int = 1,
    time_decay_half_life: float = 7.0,
    position_weights: dict[str, float] | None = None,
    verbose: bool = False,
) -> AttributionResult:
    """
    One-function attribution shortcut.

    Wraps JourneyBuilder → AttributionAnalyzer in a single call.

    Parameters
    ----------
    events              : Raw touchpoint event log DataFrame.
    models              : Attribution models to fit.  Default: all models.
    user_col            : User identifier column name.
    time_col            : Timestamp column name.
    channel_col         : Channel column name.
    interaction_col     : Interaction type column name.
    converted_col       : Conversion flag column name (0/1).
    revenue_col         : Revenue column name.
    cost_col            : Optional cost column name for ROI calculation.
    lookback_days       : Journey window in days.  None = auto-detect.
    interaction_weight  : Dict mapping interaction types to weights.
                          Default {"click": 1.0, "impression": 0.3}.
    multi_conversion    : "reset" or "rolling" — how to handle repeat converters.
    markov_order        : Markov chain order (default 1).
    time_decay_half_life: Half-life in days for time-decay model (default 7).
    position_weights    : Dict with "first", "last", "middle" keys.
    verbose             : Print progress.

    Returns
    -------
    AttributionResult

    Examples
    --------
    >>> from adsat.attribution import attribute_campaigns
    >>> result = attribute_campaigns(
    ...     events,
    ...     models       = ["shapley", "markov", "ensemble"],
    ...     lookback_days = 30,
    ... )
    >>> result.print_summary()
    >>> plot_attribution(result)
    """
    iw = interaction_weight or {"click": 1.0, "impression": 0.3}

    config = JourneyConfig(
        user_col=user_col,
        time_col=time_col,
        channel_col=channel_col,
        interaction_col=interaction_col,
        converted_col=converted_col,
        revenue_col=revenue_col,
        cost_col=cost_col,
        lookback_days=lookback_days,
        multi_conversion=multi_conversion,
        interaction_weight=iw,
    )

    builder = JourneyBuilder(config=config)
    journeys = builder.build(events)

    analyzer = AttributionAnalyzer(
        models=models or list(_SUPPORTED_MODELS),
        position_weights=position_weights or {},
        time_decay_half_life=time_decay_half_life,
        markov_order=markov_order,
        cost_col=cost_col,
        verbose=verbose,
    )
    return analyzer.fit(journeys)


def make_sample_events(
    n_users: int = 2_000,
    channels: list[str] | None = None,
    conv_rate: float = 0.20,
    avg_revenue: float = 85.0,
    lookback_days: int = 30,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic user-level touchpoint event log for testing.

    The generated dataset mirrors real-world advertising data:
    - Users follow a multi-touch journey (1–8 touchpoints per user).
    - Conversion probability increases with number of clicks (sigmoid).
    - Revenue is drawn from a log-normal distribution.
    - Some users convert twice (multi-conversion testing).
    - Impression and click touchpoints are included.
    - A ``cost`` column is included (random per channel × touchpoint).

    Parameters
    ----------
    n_users      : Number of unique users.  Default 2 000.
    channels     : Channel names.  Default: 6 standard digital channels.
    conv_rate    : Overall conversion rate.  Default 0.20.
    avg_revenue  : Mean revenue per conversion.  Default 85.0.
    lookback_days: Maximum journey window in days.  Default 30.
    random_seed  : Reproducibility seed.  Default 42.

    Returns
    -------
    pd.DataFrame  — raw event log ready for JourneyBuilder.build().
    """
    if channels is None:
        channels = [
            "paid_search",
            "display",
            "social_paid",
            "email",
            "organic_search",
            "direct",
        ]

    rng = np.random.default_rng(random_seed)
    start_date = pd.Timestamp("2024-01-01")
    ch_cost = {ch: round(float(rng.uniform(0.5, 5.0)), 2) for ch in channels}

    rows: list[dict] = []
    user_id = 0

    for _ in range(n_users):
        user_id += 1
        n_touches = int(rng.integers(1, 9))
        base_time = start_date + pd.Timedelta(days=float(rng.uniform(0, 90)))

        path_channels = rng.choice(channels, size=n_touches, replace=True).tolist()
        path_times = sorted(
            [
                base_time + pd.Timedelta(hours=float(rng.uniform(0, lookback_days * 24)))
                for _ in range(n_touches)
            ]
        )
        path_types = ["click" if rng.random() > 0.25 else "impression" for _ in range(n_touches)]
        n_clicks = sum(1 for t in path_types if t == "click")

        # Sigmoid conversion probability
        conv_prob = conv_rate / (1 + math.exp(-(n_clicks - 2) * 0.8))
        converted = bool(rng.random() < conv_prob)
        revenue = float(rng.lognormal(math.log(avg_revenue), 0.5)) if converted else 0.0

        for i, (ch, ts, tp) in enumerate(zip(path_channels, path_times, path_types)):
            rows.append(
                {
                    "user_id": f"u_{user_id:05d}",
                    "timestamp": ts,
                    "channel": ch,
                    "interaction_type": tp,
                    "converted": int(converted) if i == n_touches - 1 else 0,
                    "revenue": round(revenue, 2) if i == n_touches - 1 and converted else 0.0,
                    "cost": round(ch_cost[ch] * (1.0 if tp == "click" else 0.1), 4),
                    "session_id": f"s_{user_id:05d}_{i // 3}",
                    "device": rng.choice(["desktop", "mobile", "tablet"], p=[0.5, 0.4, 0.1]),
                }
            )

        # ~10% of converting users convert again
        if converted and rng.random() < 0.10:
            second_conv_time = path_times[-1] + pd.Timedelta(days=float(rng.uniform(1, 15)))
            second_ch = str(rng.choice(channels))
            rows.append(
                {
                    "user_id": f"u_{user_id:05d}",
                    "timestamp": second_conv_time,
                    "channel": second_ch,
                    "interaction_type": "click",
                    "converted": 1,
                    "revenue": round(float(rng.lognormal(math.log(avg_revenue), 0.5)), 2),
                    "cost": round(ch_cost[second_ch], 4),
                    "session_id": f"s_{user_id:05d}_second",
                    "device": "desktop",
                }
            )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
