"""
adsat.campaign
==============
Per-campaign saturation point prediction.

This module answers the question: "For each campaign in my dataset,
at what impression / spend level does performance saturate?"

Key class  : CampaignSaturationAnalyzer
Key function: predict_saturation_per_campaign()

Data contract
-------------
Your DataFrame must contain:
  - a *campaign identifier* column  (e.g. 'campaign_id', 'campaign_name')
  - an *x* column                   (e.g. 'impressions', 'ad_spend')
  - a *y* column                    (e.g. 'conversions', 'revenue')
  - optionally a *date/week* column (used only for sorting)

Each row represents one time period (day / week) for one campaign.

Example
-------
>>> import pandas as pd
>>> from adsat.campaign import CampaignSaturationAnalyzer
>>>
>>> df = pd.read_csv("all_campaigns_weekly.csv")
>>>
>>> analyzer = CampaignSaturationAnalyzer(
...     campaign_col="campaign_id",
...     x_col="impressions",
...     y_col="conversions",
...     min_observations=12,          # skip campaigns with < 12 data points
...     saturation_threshold=0.90,
...     primary_metric="aic",
... )
>>>
>>> results = analyzer.run(df)
>>> print(results.summary_table)
>>> results.plot_all()
>>> results.plot_saturation_comparison()
>>>
>>> # Single-function convenience wrapper
>>> from adsat.campaign import predict_saturation_per_campaign
>>> summary = predict_saturation_per_campaign(df, "campaign_id", "impressions", "conversions")
>>> print(summary)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from adsat.modeling import MODEL_REGISTRY
from adsat.pipeline import PipelineResult, SaturationPipeline

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CampaignResult:
    """Saturation analysis result for a single campaign."""

    campaign_id: Any
    n_observations: int
    x_col: str
    y_col: str
    best_model: str | None
    saturation_point: float | None  # in original x units
    saturation_y: float | None  # predicted y at saturation
    saturation_threshold: float
    r2: float | None
    rmse: float | None
    aic: float | None
    best_model_params: dict[str, float] | None
    # where the campaign currently sits relative to its saturation point
    current_x_median: float | None  # median x observed
    pct_of_saturation: float | None  # current_x_median / saturation_point * 100
    saturation_status: str = "unknown"  # 'below', 'approaching', 'at', 'beyond', 'unknown'
    pipeline_result: PipelineResult | None = field(default=None, repr=False)
    error: str | None = None
    succeeded: bool = True

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    def __post_init__(self):
        """
        Compute pct_of_saturation and saturation_status from the fitted values.
        Called automatically by Python after the dataclass __init__.
        """
        # Use explicit None checks — saturation_point or current_x_median could
        # legitimately be 0.0, which is falsy but valid.
        if (
            self.saturation_point is not None
            and self.current_x_median is not None
            and self.saturation_point != 0
        ):
            pct = self.current_x_median / self.saturation_point * 100
            self.pct_of_saturation = round(pct, 1)
            if pct < 50:
                self.saturation_status = "below"
            elif pct < 80:
                self.saturation_status = "approaching"
            elif pct <= 110:
                self.saturation_status = "at"
            else:
                self.saturation_status = "beyond"

    def as_dict(self) -> dict[str, Any]:
        """
        Return a flat dictionary of key result fields suitable for building a summary DataFrame.
        """
        return {
            "campaign_id": self.campaign_id,
            "n_observations": self.n_observations,
            "best_model": self.best_model,
            "saturation_point": round(self.saturation_point, 2) if self.saturation_point else None,
            "saturation_y": round(self.saturation_y, 2) if self.saturation_y else None,
            "current_x_median": round(self.current_x_median, 2) if self.current_x_median else None,
            "pct_of_saturation": self.pct_of_saturation,
            "saturation_status": self.saturation_status,
            "r2": round(self.r2, 4) if self.r2 is not None else None,
            "rmse": round(self.rmse, 4) if self.rmse is not None else None,
            "aic": round(self.aic, 2) if self.aic is not None else None,
            "succeeded": self.succeeded,
            "error": self.error,
        }

    def print_summary(self) -> None:
        """
        Print a single-campaign formatted summary to stdout.
        Includes model name, fit quality metrics, saturation point, and status.
        """
        sep = "-" * 55
        print(sep)
        print(f"  Campaign : {self.campaign_id}")
        print(f"  Obs      : {self.n_observations}")
        if self.succeeded:
            print(f"  Model    : {self.best_model}  (R²={self.r2:.3f}, AIC={self.aic:.1f})")
            print(
                f"  Sat. pt  : {self.saturation_point:,.2f} {self.x_col}"
                if self.saturation_point
                else "  Sat. pt  : N/A"
            )
            print(
                f"  Sat. y   : {self.saturation_y:,.2f} {self.y_col}"
                if self.saturation_y
                else "  Sat. y   : N/A"
            )
            print(
                f"  Current  : {self.current_x_median:,.2f} ({self.pct_of_saturation}% of saturation) → {self.saturation_status.upper()}"
            )
        else:
            print(f"  FAILED   : {self.error}")
        print(sep)


@dataclass
class CampaignBatchResult:
    """Results for all campaigns in a batch run."""

    campaign_col: str
    x_col: str
    y_col: str
    campaign_results: dict[Any, CampaignResult]  # campaign_id -> CampaignResult
    summary_table: pd.DataFrame  # one row per campaign
    n_total: int
    n_succeeded: int
    n_failed: int

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get(self, campaign_id: Any) -> CampaignResult:
        """
        Retrieve a CampaignResult by campaign ID.

        Raises KeyError if the campaign_id is not in this batch.
        """
        if campaign_id not in self.campaign_results:
            raise KeyError(f"Campaign '{campaign_id}' not found.")
        return self.campaign_results[campaign_id]

    def failed_campaigns(self) -> list[Any]:
        """
        Return a list of campaign IDs whose analysis raised an error or had too few observations.
        """
        return [cid for cid, r in self.campaign_results.items() if not r.succeeded]

    def succeeded_campaigns(self) -> list[Any]:
        """
        Return a list of campaign IDs whose analysis completed successfully.
        """
        return [cid for cid, r in self.campaign_results.items() if r.succeeded]

    def campaigns_by_status(self, status: str) -> pd.DataFrame:
        """Filter summary table by saturation_status: 'below','approaching','at','beyond'."""
        return self.summary_table[self.summary_table["saturation_status"] == status]

    def print_summary(self) -> None:
        """Print a batch-level summary to stdout: total/succeeded/failed counts and a per-campaign table."""
        sep = "=" * 65
        print(sep)
        print("  ADSAT – PER-CAMPAIGN SATURATION RESULTS")
        print(sep)
        print(f"  Campaigns analysed : {self.n_total}")
        print(f"  Succeeded          : {self.n_succeeded}")
        print(f"  Failed             : {self.n_failed}")
        if self.n_failed:
            print(f"  Failed IDs         : {self.failed_campaigns()}")
        print()
        cols = [
            "campaign_id",
            "n_observations",
            "best_model",
            "r2",
            "saturation_point",
            "current_x_median",
            "pct_of_saturation",
            "saturation_status",
        ]
        available = [c for c in cols if c in self.summary_table.columns]
        print(self.summary_table[available].to_string(index=False))
        print(sep)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_all(
        self,
        cols_per_row: int = 3,
        figsize_per_campaign: tuple[int, int] = (5, 4),
        save_path: str | None = None,
    ) -> None:
        """
        Plot the fitted saturation curve for every campaign on a grid.

        Parameters
        ----------
        cols_per_row : int
        figsize_per_campaign : (width, height) per sub-plot
        save_path : str, optional — save figure to this path
        """
        succeeded = self.succeeded_campaigns()
        if not succeeded:
            print("No succeeded campaigns to plot.")
            return

        n = len(succeeded)
        ncols = min(cols_per_row, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(figsize_per_campaign[0] * ncols, figsize_per_campaign[1] * nrows),
        )
        axes = np.array(axes).flatten()

        for ax, cid in zip(axes, succeeded):
            cr = self.campaign_results[cid]
            self._plot_single_campaign(ax, cr)

        # Hide unused axes
        for ax in axes[n:]:
            ax.set_visible(False)

        fig.suptitle(
            f"Saturation Curves – {self.x_col} vs {self.y_col}", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_saturation_comparison(
        self,
        figsize: tuple[int, int] = (12, 6),
        save_path: str | None = None,
    ) -> None:
        """
        Bar chart comparing saturation points across all campaigns,
        with the current median x overlaid as a dot.
        """
        df = self.summary_table[self.summary_table["succeeded"]].copy()
        df = df.dropna(subset=["saturation_point"]).sort_values("saturation_point")

        if df.empty:
            print("No saturation points available to plot.")
            return

        status_colors = {
            "below": "#2ecc71",
            "approaching": "#f39c12",
            "at": "#e74c3c",
            "beyond": "#8e44ad",
            "unknown": "#95a5a6",
        }

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle("Campaign Saturation Point Comparison", fontsize=13, fontweight="bold")

        # --- Left: saturation point bar chart ---
        ax = axes[0]
        bar_colors = [status_colors.get(s, "#95a5a6") for s in df["saturation_status"]]
        ax.barh(df["campaign_id"].astype(str), df["saturation_point"], color=bar_colors, alpha=0.85)

        # Overlay current median x as scatter
        if "current_x_median" in df.columns:
            valid = df.dropna(subset=["current_x_median"])
            y_positions = [
                list(df["campaign_id"].astype(str)).index(str(cid)) for cid in valid["campaign_id"]
            ]
            ax.scatter(
                valid["current_x_median"],
                y_positions,
                color="black",
                zorder=5,
                s=50,
                marker="D",
                label="Current median",
            )
            ax.legend(fontsize=9)

        ax.set_xlabel(f"Saturation Point ({self.x_col})")
        ax.set_title("Saturation Point per Campaign")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))

        # Legend for status colours
        legend_elements = [
            Line2D([0], [0], color=c, lw=6, label=s.capitalize())
            for s, c in status_colors.items()
            if s != "unknown"
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

        # --- Right: % of saturation reached ---
        ax2 = axes[1]
        if "pct_of_saturation" in df.columns:
            df2 = df.dropna(subset=["pct_of_saturation"]).sort_values("pct_of_saturation")
            pct_colors = [status_colors.get(s, "#95a5a6") for s in df2["saturation_status"]]
            ax2.barh(
                df2["campaign_id"].astype(str),
                df2["pct_of_saturation"],
                color=pct_colors,
                alpha=0.85,
            )
            ax2.axvline(90, color="red", lw=1.5, ls="--", label="90% threshold")
            ax2.axvline(100, color="darkred", lw=1.5, ls=":", label="100% (at saturation)")
            ax2.set_xlabel("Current impressions as % of saturation point")
            ax2.set_title("% of Saturation Reached")
            ax2.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def plot_status_breakdown(
        self,
        figsize: tuple[int, int] = (7, 5),
        save_path: str | None = None,
    ) -> None:
        """Pie / donut chart showing how many campaigns are below / approaching / at / beyond saturation."""
        counts = self.summary_table["saturation_status"].value_counts()
        status_colors = {
            "below": "#2ecc71",
            "approaching": "#f39c12",
            "at": "#e74c3c",
            "beyond": "#8e44ad",
            "unknown": "#95a5a6",
        }
        colors = [status_colors.get(s, "#95a5a6") for s in counts.index]

        fig, ax = plt.subplots(figsize=figsize)
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            colors=colors,
            autopct="%1.0f%%",
            startangle=140,
            wedgeprops={"width": 0.55},
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax.set_title("Campaign Saturation Status Breakdown", fontsize=12, fontweight="bold")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------
    # Internal plotting helper
    # ------------------------------------------------------------------

    def _plot_single_campaign(self, ax, cr: CampaignResult) -> None:
        """Draw a single campaign's saturation curve on a given Axes."""
        if not cr.succeeded or cr.pipeline_result is None:
            ax.text(
                0.5,
                0.5,
                f"{cr.campaign_id}\nFAILED\n{cr.error}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="red",
                fontsize=8,
            )
            ax.set_title(str(cr.campaign_id), fontsize=9)
            return

        pr = cr.pipeline_result
        best_res = pr.model_results.get(cr.best_model)
        if best_res is None:
            ax.set_title(str(cr.campaign_id), fontsize=9)
            return

        x_obs = best_res.x_values
        y_obs = best_res.y_true

        func = MODEL_REGISTRY.get(best_res.model_name, {}).get("func")
        x_fine = np.linspace(x_obs.min(), x_obs.max() * 1.5, 400)

        ax.scatter(x_obs, y_obs, s=15, color="steelblue", alpha=0.6, zorder=4)
        if func:
            y_fine = func(x_fine, *list(best_res.params.values()))
            ax.plot(x_fine, y_fine, color="tab:blue", lw=2)

        if cr.saturation_point:
            ax.axvline(cr.saturation_point, color="red", lw=1.5, ls="--")
            ax.text(
                cr.saturation_point,
                ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1,
                "  sat",
                color="red",
                fontsize=7,
                va="top",
            )

        if cr.current_x_median:
            ax.axvline(cr.current_x_median, color="green", lw=1, ls=":")

        title = f"{cr.campaign_id}\n{cr.best_model} | R²={cr.r2:.2f}"
        if cr.saturation_status != "unknown":
            title += f" | {cr.saturation_status.upper()}"
        ax.set_title(title, fontsize=8)
        ax.set_xlabel(self.x_col, fontsize=7)
        ax.set_ylabel(self.y_col, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}")
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class CampaignSaturationAnalyzer:
    """
    Run per-campaign saturation analysis across a multi-campaign DataFrame.

    For each campaign identified in ``campaign_col``, this class:
      1. Slices the data for that campaign.
      2. Runs the full SaturationPipeline (distribution → transform → model → evaluate).
      3. Reports the saturation point in the original x-scale.
      4. Classifies the campaign as below / approaching / at / beyond saturation.

    Parameters
    ----------
    campaign_col : str
        Column that uniquely identifies each campaign (e.g. 'campaign_id').
    x_col : str
        Independent variable: impressions, ad_spend, etc.
    y_col : str
        Dependent variable: conversions, revenue, ROI, etc.
    date_col : str, optional
        If provided, data is sorted by this column before analysis.
    min_observations : int
        Campaigns with fewer rows are skipped. Default 10.
    models : list of str, optional
        Saturation models to try. Defaults to all available.
    transform_strategy : str
        'auto' or a fixed method like 'log'.
    saturation_threshold : float
        Fraction of asymptote defining saturation. Default 0.90.
    primary_metric : str
        Model selection metric: 'aic', 'r2', 'rmse', etc.
    verbose : bool
        If True, prints per-campaign progress.

    Examples
    --------
    >>> analyzer = CampaignSaturationAnalyzer(
    ...     campaign_col="campaign_id",
    ...     x_col="impressions",
    ...     y_col="conversions",
    ...     min_observations=12,
    ... )
    >>> batch = analyzer.run(df)
    >>> batch.print_summary()
    >>> batch.plot_all()
    >>> batch.plot_saturation_comparison()
    """

    def __init__(
        self,
        campaign_col: str,
        x_col: str,
        y_col: str,
        date_col: str | None = None,
        min_observations: int = 10,
        models: list[str] | None = None,
        transform_strategy: str = "auto",
        saturation_threshold: float = 0.90,
        primary_metric: str = "aic",
        verbose: bool = True,
    ):
        """
        Store column names, quality thresholds, and model configuration.

        Parameters
        ----------
        campaign_col : str       — column uniquely identifying each campaign.
        x_col : str              — independent variable (impressions, spend, …).
        y_col : str              — dependent variable (conversions, revenue, …).
        date_col : str, optional — if provided, data is sorted chronologically.
        min_observations : int   — campaigns with fewer rows are skipped.
        models : list, optional  — saturation models to try; defaults to all.
        transform_strategy : str — "auto" or a fixed method name.
        saturation_threshold : float — fraction of asymptote defining saturation.
        primary_metric : str    — model selection criterion ('aic', 'r2', …).
        verbose : bool           — print per-campaign progress.
        """
        self.campaign_col = campaign_col
        self.x_col = x_col
        self.y_col = y_col
        self.date_col = date_col
        self.min_observations = min_observations
        self.models = models
        self.transform_strategy = transform_strategy
        self.saturation_threshold = saturation_threshold
        self.primary_metric = primary_metric
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> CampaignBatchResult:
        """
        Analyse every campaign in ``df`` and return a CampaignBatchResult.

        Parameters
        ----------
        df : pd.DataFrame
            Multi-campaign data. Must contain campaign_col, x_col, y_col.

        Returns
        -------
        CampaignBatchResult
        """
        self._validate_columns(df)
        campaigns = df[self.campaign_col].unique()
        n_total = len(campaigns)

        self._log(f"\n{'='*60}")
        self._log("  ADSAT – Per-Campaign Saturation Analysis")
        self._log(f"  Campaigns : {n_total}  |  x={self.x_col}  |  y={self.y_col}")
        self._log(f"{'='*60}\n")

        campaign_results: dict[Any, CampaignResult] = {}

        for i, cid in enumerate(campaigns, 1):
            self._log(f"[{i}/{n_total}] Campaign: {cid}")
            campaign_df = df[df[self.campaign_col] == cid].copy()

            if self.date_col and self.date_col in campaign_df.columns:
                campaign_df = campaign_df.sort_values(self.date_col)

            campaign_df = campaign_df[[self.x_col, self.y_col]].dropna()
            n_obs = len(campaign_df)

            if n_obs < self.min_observations:
                msg = f"Only {n_obs} observations (min={self.min_observations}) – skipped."
                self._log(f"  ⚠ {msg}")
                campaign_results[cid] = CampaignResult(
                    campaign_id=cid,
                    n_observations=n_obs,
                    x_col=self.x_col,
                    y_col=self.y_col,
                    best_model=None,
                    saturation_point=None,
                    saturation_y=None,
                    saturation_threshold=self.saturation_threshold,
                    r2=None,
                    rmse=None,
                    aic=None,
                    best_model_params=None,
                    current_x_median=float(campaign_df[self.x_col].median()) if n_obs > 0 else None,
                    pct_of_saturation=None,
                    succeeded=False,
                    error=msg,
                )
                continue

            cr = self._analyse_single_campaign(cid, campaign_df, n_obs)
            campaign_results[cid] = cr

            if cr.succeeded:
                self._log(
                    f"  ✓ best_model={cr.best_model} | R²={cr.r2:.3f} | "
                    f"sat_point={cr.saturation_point:,.0f} | status={cr.saturation_status}"
                    if cr.saturation_point
                    else f"  ✓ best_model={cr.best_model} | R²={cr.r2:.3f} | sat_point=N/A"
                )
            else:
                self._log(f"  ✗ FAILED: {cr.error}")

        n_succeeded = sum(1 for r in campaign_results.values() if r.succeeded)
        n_failed = n_total - n_succeeded

        summary_table = pd.DataFrame([r.as_dict() for r in campaign_results.values()])

        self._log(f"\n{'='*60}")
        self._log(f"  Done. {n_succeeded}/{n_total} campaigns succeeded.")
        self._log(f"{'='*60}\n")

        return CampaignBatchResult(
            campaign_col=self.campaign_col,
            x_col=self.x_col,
            y_col=self.y_col,
            campaign_results=campaign_results,
            summary_table=summary_table,
            n_total=n_total,
            n_succeeded=n_succeeded,
            n_failed=n_failed,
        )

    def run_single(self, df: pd.DataFrame, campaign_id: Any) -> CampaignResult:
        """
        Analyse a single campaign by ID. Useful for targeted re-runs.

        Parameters
        ----------
        df : pd.DataFrame  — full dataset (will be filtered to campaign_id)
        campaign_id        — value to match in campaign_col

        Returns
        -------
        CampaignResult
        """
        if campaign_id not in df[self.campaign_col].values:
            raise KeyError(
                f"Campaign '{campaign_id}' not found in column '{self.campaign_col}'. "
                f"Available campaigns: {list(df[self.campaign_col].unique())}"
            )
        campaign_df = df[df[self.campaign_col] == campaign_id].copy()
        if self.date_col and self.date_col in campaign_df.columns:
            campaign_df = campaign_df.sort_values(self.date_col)
        campaign_df = campaign_df[[self.x_col, self.y_col]].dropna()
        n_obs = len(campaign_df)
        return self._analyse_single_campaign(campaign_id, campaign_df, n_obs)

    # ------------------------------------------------------------------
    # Internal: single campaign pipeline
    # ------------------------------------------------------------------

    def _analyse_single_campaign(
        self,
        cid: Any,
        campaign_df: pd.DataFrame,
        n_obs: int,
    ) -> CampaignResult:
        """
        Run the full SaturationPipeline for one campaign slice and wrap the result
        in a CampaignResult dataclass.  Catches all exceptions so a single bad campaign
        does not abort the batch run.
        """
        current_x_median = float(campaign_df[self.x_col].median())

        try:
            pipeline = SaturationPipeline(
                x_col=self.x_col,
                y_col=self.y_col,
                models=self.models,
                transform_strategy=self.transform_strategy,
                saturation_threshold=self.saturation_threshold,
                primary_metric=self.primary_metric,
                verbose=False,
            )
            pr = pipeline.run(campaign_df)

            best_res = pr.model_results.get(pr.best_model)
            r2 = best_res.r2 if best_res else None
            rmse = best_res.rmse if best_res else None
            aic = best_res.aic if best_res else None
            params = best_res.params if best_res else None

            return CampaignResult(
                campaign_id=cid,
                n_observations=n_obs,
                x_col=self.x_col,
                y_col=self.y_col,
                best_model=pr.best_model,
                saturation_point=pr.saturation_point,
                saturation_y=pr.saturation_y,
                saturation_threshold=self.saturation_threshold,
                r2=r2,
                rmse=rmse,
                aic=aic,
                best_model_params=params,
                current_x_median=current_x_median,
                pct_of_saturation=None,  # computed in __post_init__
                pipeline_result=pr,
                succeeded=True,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            if self.verbose:
                traceback.print_exc()
            return CampaignResult(
                campaign_id=cid,
                n_observations=n_obs,
                x_col=self.x_col,
                y_col=self.y_col,
                best_model=None,
                saturation_point=None,
                saturation_y=None,
                saturation_threshold=self.saturation_threshold,
                r2=None,
                rmse=None,
                aic=None,
                best_model_params=None,
                current_x_median=current_x_median,
                pct_of_saturation=None,
                succeeded=False,
                error=error_msg,
            )

    # ------------------------------------------------------------------
    # Validation & logging
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        """
        Raise ValueError listing any required columns that are absent from the DataFrame.
        """
        required = [self.campaign_col, self.x_col, self.y_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

    def _log(self, msg: str) -> None:
        """
        Print msg to stdout when verbose=True. Used throughout the class for progress reporting.
        """
        if self.verbose:
            print(msg)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def predict_saturation_per_campaign(
    df: pd.DataFrame,
    campaign_col: str,
    x_col: str,
    y_col: str,
    date_col: str | None = None,
    min_observations: int = 10,
    models: list[str] | None = None,
    saturation_threshold: float = 0.90,
    primary_metric: str = "aic",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    One-function shortcut: return a summary DataFrame of saturation points
    for every campaign in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Multi-campaign data with campaign_col, x_col, y_col columns.
    campaign_col : str
        Column identifying campaigns.
    x_col : str
        Impression / spend column.
    y_col : str
        Conversion / revenue column.
    date_col : str, optional
        Used to sort data chronologically within each campaign.
    min_observations : int
        Minimum rows per campaign to attempt analysis. Default 10.
    models : list of str, optional
        Model types to try. Default: all (hill, negative_exponential, power,
        michaelis_menten, logistic).
    saturation_threshold : float
        Fraction of asymptote defining saturation. Default 0.90.
    primary_metric : str
        Model selection criterion. Default 'aic'.
    verbose : bool
        Print progress. Default False.

    Returns
    -------
    pd.DataFrame with one row per campaign containing:
        campaign_id, n_observations, best_model, saturation_point,
        saturation_y, current_x_median, pct_of_saturation,
        saturation_status, r2, rmse, aic, succeeded, error

    Examples
    --------
    >>> summary = predict_saturation_per_campaign(
    ...     df, "campaign_id", "impressions", "conversions"
    ... )
    >>> print(summary[["campaign_id","saturation_point","saturation_status"]])
    """
    analyzer = CampaignSaturationAnalyzer(
        campaign_col=campaign_col,
        x_col=x_col,
        y_col=y_col,
        date_col=date_col,
        min_observations=min_observations,
        models=models,
        saturation_threshold=saturation_threshold,
        primary_metric=primary_metric,
        verbose=verbose,
    )
    batch = analyzer.run(df)
    return batch.summary_table
