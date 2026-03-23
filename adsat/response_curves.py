"""
adsat.response_curves
=====================
Marginal return analysis and response curve visualisation.

Knowing *where* saturation is (the saturation point) is the first insight.
This module delivers the second: *how efficiently* is each additional unit
of spend converting?  It characterises the full shape of the response curve
and draws the key business zones that matter for media planning decisions.

Key concepts
------------
- **Response curve**   : the full outcome–vs–spend relationship f(x)
- **Marginal return**  : Δoutcome / Δspend — how much each extra pound buys
- **Elasticity**       : % change in outcome / % change in spend (dimensionless)
- **Efficiency zones** :
    * High efficiency   — marginal return > 80th percentile across the curve
    * Medium efficiency — 20th–80th percentile
    * Low efficiency    — marginal return < 20th percentile (approaching saturation)

Key classes & functions
-----------------------
ResponseCurveAnalyzer   – main class
ResponseCurveResult     – per-campaign result dataclass
analyse_response_curves()  – one-liner shortcut

Typical workflow
----------------
>>> from adsat.response_curves import ResponseCurveAnalyzer
>>> analyzer = ResponseCurveAnalyzer()
>>> results  = analyzer.analyse(batch)          # CampaignBatchResult
>>> analyzer.plot_curves(results)
>>> analyzer.plot_marginal_returns(results)
>>> analyzer.plot_efficiency_zones(results)
>>> print(analyzer.summary_table(results))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adsat.campaign import CampaignBatchResult, CampaignResult
from adsat.modeling import MODEL_REGISTRY

# ── colour palette ────────────────────────────────────────────────────────────
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_GREEN = "#3BB273"
_RED = "#E84855"
_GREY = "#6C757D"
_PURPLE = "#7B2D8B"
PALETTE = [_BLUE, _ORANGE, _GREEN, _RED, _PURPLE, _GREY, "#F4A261", "#264653", "#A8DADC", "#E9C46A"]


def _fmt_large(x, _=None) -> str:
    """
    Format large numbers with k / M suffix for axis tick labels.
    E.g. 2_000_000 -> "2.0M", 50_000 -> "50k".
    """
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.1f}"


def _build_response_fn(cr: CampaignResult) -> Callable[[float], float] | None:
    """Build a scalar callable from a CampaignResult's fitted params."""
    if not cr.succeeded or not cr.best_model or not cr.best_model_params:
        return None
    registry_name = "hill" if cr.best_model == "hill_bayesian" else cr.best_model
    spec = MODEL_REGISTRY.get(registry_name)
    if spec is None:
        return None
    func = spec["func"]
    param_vals = list(cr.best_model_params.values())

    def response(x: float) -> float:
        """
        Evaluate the fitted saturation curve at spend x.  Negative inputs are clipped to 0.
        """
        return float(func(max(float(x), 0.0), *param_vals))

    return response


# ─────────────────────────────────────────────────────────────────────────────
# Per-campaign result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ResponseCurveResult:
    """
    Full response-curve characterisation for a single campaign.

    Attributes
    ----------
    campaign_id : any
    x_values : np.ndarray
        Spend grid used for calculations.
    y_values : np.ndarray
        Predicted outcome at each spend level.
    marginal_returns : np.ndarray
        Δoutcome / Δspend at each spend level.
    elasticities : np.ndarray
        (dy/dx) * (x/y) — % outcome change per 1% spend change.
    inflection_point_x : float or None
        Spend level at which the marginal return starts declining fastest
        (2nd derivative minimum — only meaningful for S-curves).
    efficiency_zone : np.ndarray of str
        'high', 'medium', or 'low' at each x.
    current_x : float or None
        Campaign's current median spend (from CampaignResult).
    current_y : float or None
        Predicted outcome at current spend.
    current_marginal_return : float or None
    current_elasticity : float or None
    saturation_point : float or None
    asymptote : float or None
        Maximum theoretical outcome (model parameter).
    roi_curve : np.ndarray
        outcome / spend at each x (returns per unit spend).
    """

    campaign_id: Any
    x_values: np.ndarray
    y_values: np.ndarray
    marginal_returns: np.ndarray
    elasticities: np.ndarray
    inflection_point_x: float | None
    efficiency_zone: np.ndarray
    current_x: float | None
    current_y: float | None
    current_marginal_return: float | None
    current_elasticity: float | None
    saturation_point: float | None
    asymptote: float | None
    roi_curve: np.ndarray

    def summary_row(self) -> dict[str, Any]:
        """Return a one-row dict for the summary table."""
        return {
            "campaign_id": self.campaign_id,
            "asymptote": round(self.asymptote, 2) if self.asymptote else None,
            "saturation_point": round(self.saturation_point, 0) if self.saturation_point else None,
            "current_spend": round(self.current_x, 0) if self.current_x else None,
            "current_outcome": round(self.current_y, 2) if self.current_y else None,
            "current_marginal_return": (
                round(self.current_marginal_return, 6)
                if self.current_marginal_return is not None
                else None
            ),
            "current_elasticity": (
                round(self.current_elasticity, 4) if self.current_elasticity is not None else None
            ),
            "current_roi": (
                round(self.current_y / self.current_x, 6)
                if self.current_x and self.current_x > 0 and self.current_y
                else None
            ),
            "inflection_point_x": (
                round(self.inflection_point_x, 0) if self.inflection_point_x else None
            ),
            "pct_saturation_reached": (
                round(self.current_x / self.saturation_point * 100, 1)
                if self.current_x and self.saturation_point and self.saturation_point > 0
                else None
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class ResponseCurveAnalyzer:
    """
    Analyse and visualise response curves for advertising campaigns.

    Computes marginal returns, ROI curves, elasticity, and efficiency zones
    for every campaign in a CampaignBatchResult.

    Parameters
    ----------
    n_points : int
        Number of spend grid points per campaign. Default 500.
    x_max_multiplier : float
        Grid extends to this multiple of the saturation point. Default 1.5.
        (Falls back to 2 × current_x_median when no saturation point exists.)
    efficiency_thresholds : tuple (low_pct, high_pct)
        Percentile cutoffs for efficiency zone labelling.
        Default (20, 80) — bottom 20% = low, top 20% = high.
    verbose : bool

    Examples
    --------
    >>> from adsat.response_curves import ResponseCurveAnalyzer
    >>> analyzer = ResponseCurveAnalyzer()
    >>> results  = analyzer.analyse(batch)
    >>> analyzer.plot_curves(results)
    >>> analyzer.plot_marginal_returns(results)
    >>> print(analyzer.summary_table(results))
    """

    def __init__(
        self,
        n_points: int = 500,
        x_max_multiplier: float = 1.5,
        efficiency_thresholds: tuple[float, float] = (20.0, 80.0),
        verbose: bool = True,
    ):
        """
        Configure the analyser with grid resolution, spend range multiplier,
        and efficiency zone thresholds.
        No analysis occurs here; call analyse() to run.
        """
        self.n_points = n_points
        self.x_max_multiplier = x_max_multiplier
        self.efficiency_thresholds = efficiency_thresholds
        self.verbose = verbose

    # ── public API ─────────────────────────────────────────────────────────────

    def analyse(
        self,
        batch: CampaignBatchResult,
    ) -> dict[Any, ResponseCurveResult]:
        """
        Compute full response-curve metrics for every succeeded campaign.

        Parameters
        ----------
        batch : CampaignBatchResult

        Returns
        -------
        dict  {campaign_id: ResponseCurveResult}
        """
        results: dict[Any, ResponseCurveResult] = {}

        for cid in batch.succeeded_campaigns():
            cr = batch.get(cid)
            fn = _build_response_fn(cr)
            if fn is None:
                self._log(f"Skipping '{cid}' — no response function available.")
                continue
            self._log(f"Analysing '{cid}'…")
            results[cid] = self._analyse_single(cid, cr, fn)

        return results

    def summary_table(
        self,
        results: dict[Any, ResponseCurveResult],
    ) -> pd.DataFrame:
        """
        Return a tidy summary DataFrame — one row per campaign.

        Columns: campaign_id, asymptote, saturation_point, current_spend,
        current_outcome, current_marginal_return, current_elasticity,
        current_roi, inflection_point_x, pct_saturation_reached.
        """
        rows = [r.summary_row() for r in results.values()]
        return pd.DataFrame(rows)

    # ── plots ──────────────────────────────────────────────────────────────────

    def plot_curves(
        self,
        results: dict[Any, ResponseCurveResult],
        save_path: str | None = None,
    ) -> None:
        """
        Plot response curves for all campaigns on a single figure.

        Shows outcome vs spend with:
        - Efficiency zone shading (green / amber / red)
        - Current spend marker (●)
        - Saturation point marker (dashed vertical line)
        - Asymptote reference (dotted horizontal line)

        Parameters
        ----------
        results : dict from analyse()
        save_path : str, optional
        """
        n = len(results)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle("Response Curves — Outcome vs Spend", fontsize=13, fontweight="bold")

        for ax, (cid, res), color in zip(
            axes, results.items(), PALETTE * ((n // len(PALETTE)) + 1)
        ):
            xs = res.x_values
            ys = res.y_values
            mr = res.marginal_returns

            # Efficiency zone background shading
            lo_t, hi_t = self.efficiency_thresholds
            lo_mr = np.percentile(mr[mr > 0], lo_t) if (mr > 0).any() else 0
            hi_mr = np.percentile(mr[mr > 0], hi_t) if (mr > 0).any() else 1

            high_mask = mr >= hi_mr
            low_mask = mr <= lo_mr
            mid_mask = ~high_mask & ~low_mask

            def _shade_zone(mask, fc):
                """
                Shade a contiguous zone on the marginal-return chart between y=0 and the curve.

                mask : boolean array indicating which x positions fall in this zone.
                fc   : fill colour for the shading.
                """
                if not mask.any():
                    return
                # Find contiguous segments
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    return
                ax.fill_between(xs, 0, ys, where=mask, alpha=0.10, color=fc, interpolate=True)

            _shade_zone(high_mask, _GREEN)
            _shade_zone(mid_mask, _ORANGE)
            _shade_zone(low_mask, _RED)

            # Main curve
            ax.plot(xs, ys, color=color, lw=2.2)

            # Asymptote
            if res.asymptote:
                ax.axhline(
                    res.asymptote,
                    color=_GREY,
                    lw=1,
                    ls=":",
                    label=f"Asymptote={res.asymptote:,.0f}",
                )

            # Saturation point
            if res.saturation_point:
                ax.axvline(
                    res.saturation_point,
                    color=_RED,
                    lw=1.5,
                    ls="--",
                    label=f"Sat. pt={res.saturation_point:,.0f}",
                )

            # Current spend
            if res.current_x is not None and res.current_y is not None:
                ax.scatter(
                    [res.current_x],
                    [res.current_y],
                    s=70,
                    color="black",
                    marker="o",
                    zorder=6,
                    label=f"Current={res.current_x:,.0f}",
                )

            ax.set_title(str(cid), fontsize=9, fontweight="bold")
            ax.set_xlabel("Spend", fontsize=8)
            ax.set_ylabel("Outcome", fontsize=8)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.legend(fontsize=7)
            ax.set_ylim(bottom=0)

        # Zone legend
        patches = [
            mpatches.Patch(color=_GREEN, alpha=0.4, label="High efficiency"),
            mpatches.Patch(color=_ORANGE, alpha=0.4, label="Medium efficiency"),
            mpatches.Patch(color=_RED, alpha=0.4, label="Low efficiency"),
        ]
        fig.legend(
            handles=patches,
            loc="lower center",
            ncol=3,
            fontsize=9,
            frameon=True,
            bbox_to_anchor=(0.5, -0.02),
        )

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_marginal_returns(
        self,
        results: dict[Any, ResponseCurveResult],
        save_path: str | None = None,
    ) -> None:
        """
        Plot marginal return curves for all campaigns.

        Each panel shows how much additional outcome one extra unit of spend
        buys at each point on the curve.  Campaigns with steep drop-offs are
        approaching saturation quickly.

        Parameters
        ----------
        results : dict from analyse()
        save_path : str, optional
        """
        n = len(results)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle("Marginal Returns — Δoutcome per unit Δspend", fontsize=13, fontweight="bold")

        for ax, (cid, res), color in zip(
            axes, results.items(), PALETTE * ((n // len(PALETTE)) + 1)
        ):
            xs = res.x_values
            mr = res.marginal_returns

            ax.plot(xs, mr, color=color, lw=2)
            ax.fill_between(xs, 0, mr, alpha=0.15, color=color)

            if res.current_x is not None and res.current_marginal_return is not None:
                ax.scatter(
                    [res.current_x],
                    [res.current_marginal_return],
                    s=70,
                    color="black",
                    marker="o",
                    zorder=5,
                    label=f"Current MR={res.current_marginal_return:.4f}",
                )
                ax.axvline(res.current_x, color=_GREY, lw=1, ls=":", alpha=0.6)

            if res.saturation_point:
                ax.axvline(
                    res.saturation_point,
                    color=_RED,
                    lw=1.5,
                    ls="--",
                    alpha=0.7,
                    label="Saturation pt",
                )

            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(str(cid), fontsize=9, fontweight="bold")
            ax.set_xlabel("Spend", fontsize=8)
            ax.set_ylabel("Marginal return", fontsize=8)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.legend(fontsize=7)
            ax.set_ylim(bottom=0)

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_roi_curves(
        self,
        results: dict[Any, ResponseCurveResult],
        save_path: str | None = None,
    ) -> None:
        """
        Plot ROI curves (outcome / spend) for all campaigns.

        ROI typically starts high at low spend (every pound counts) and
        declines as you approach saturation.  The current spend marker
        shows where each campaign currently sits on its ROI curve.

        Parameters
        ----------
        results : dict from analyse()
        save_path : str, optional
        """
        n = len(results)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle("ROI Curves — Outcome per Unit Spend", fontsize=13, fontweight="bold")

        for ax, (cid, res), color in zip(
            axes, results.items(), PALETTE * ((n // len(PALETTE)) + 1)
        ):
            xs = res.x_values
            roi = res.roi_curve
            # Only plot where x > 0
            mask = xs > 0
            ax.plot(xs[mask], roi[mask], color=color, lw=2)
            ax.fill_between(xs[mask], 0, roi[mask], alpha=0.12, color=color)

            if res.current_x and res.current_x > 0:
                curr_roi = res.current_y / res.current_x if res.current_y else None
                if curr_roi is not None:
                    ax.scatter(
                        [res.current_x],
                        [curr_roi],
                        s=70,
                        color="black",
                        marker="o",
                        zorder=5,
                        label=f"Current ROI={curr_roi:.4f}",
                    )
                    ax.axvline(res.current_x, color=_GREY, lw=1, ls=":", alpha=0.6)

            ax.set_title(str(cid), fontsize=9, fontweight="bold")
            ax.set_xlabel("Spend", fontsize=8)
            ax.set_ylabel("ROI (outcome / spend)", fontsize=8)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.legend(fontsize=7)
            ax.set_ylim(bottom=0)

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_efficiency_comparison(
        self,
        results: dict[Any, ResponseCurveResult],
        save_path: str | None = None,
    ) -> None:
        """
        Side-by-side comparison of current marginal return and elasticity
        across all campaigns — a quick cross-campaign efficiency view.

        Parameters
        ----------
        results : dict from analyse()
        save_path : str, optional
        """
        rows = [r.summary_row() for r in results.values()]
        df = pd.DataFrame(rows).set_index("campaign_id")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle("Cross-Campaign Efficiency Comparison", fontsize=13, fontweight="bold")

        n = len(df)
        colors = (PALETTE * ((n // len(PALETTE)) + 1))[:n]

        # ── Marginal return bar chart ─────────────────────────────────────
        ax = axes[0]
        vals = df["current_marginal_return"].fillna(0)
        ax.barh(df.index.astype(str), vals, color=colors, alpha=0.85)
        ax.set_xlabel("Marginal return at current spend", fontsize=9)
        ax.set_title("Marginal Return (higher = more room to grow)", fontsize=10)
        ax.axvline(0, color="black", lw=0.6)
        for i, v in enumerate(vals):
            ax.text(v + vals.abs().max() * 0.01, i, f"{v:.5f}", va="center", fontsize=8)

        # ── Elasticity bar chart ──────────────────────────────────────────
        ax2 = axes[1]
        vals2 = df["current_elasticity"].fillna(0)
        ax2.barh(df.index.astype(str), vals2, color=colors, alpha=0.85)
        ax2.set_xlabel("Elasticity at current spend", fontsize=9)
        ax2.set_title("Elasticity  (1.0 = proportional, <1 = diminishing)", fontsize=10)
        ax2.axvline(1.0, color=_RED, lw=1.2, ls="--", label="Elasticity=1")
        ax2.axvline(0, color="black", lw=0.6)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    # ── internal: single-campaign analysis ────────────────────────────────────

    def _analyse_single(
        self,
        cid: Any,
        cr: CampaignResult,
        fn: Callable,
    ) -> ResponseCurveResult:
        """
        Compute the full response curve for one campaign: build spend grid, evaluate
        outcome and marginal return at each point, find inflection and asymptote,
        and package everything into a ResponseCurveResult.
        """
        # Build spend grid
        if cr.saturation_point and cr.saturation_point > 0:
            x_max = cr.saturation_point * self.x_max_multiplier
        elif cr.current_x_median and cr.current_x_median > 0:
            x_max = cr.current_x_median * 3.0
        else:
            x_max = 1_000_000.0  # safe fallback

        xs = np.linspace(0, x_max, self.n_points)

        # Response values
        ys = np.array([fn(x) for x in xs])

        # Marginal returns — central differences; forward diff at endpoints
        mr = self._marginal_returns(fn, xs)

        # Elasticity: (dy/dx) * (x/y)
        with np.errstate(divide="ignore", invalid="ignore"):
            elasticities = np.where(
                (ys > 0) & (xs > 0),
                mr * xs / ys,
                0.0,
            )

        # ROI curve: y / x
        with np.errstate(divide="ignore", invalid="ignore"):
            roi = np.where(xs > 0, ys / xs, 0.0)

        # Efficiency zones
        lo_t, hi_t = self.efficiency_thresholds
        positive_mr = mr[mr > 0]
        lo_thr = np.percentile(positive_mr, lo_t) if positive_mr.size > 0 else 0.0
        hi_thr = np.percentile(positive_mr, hi_t) if positive_mr.size > 0 else 1.0
        zones = np.where(mr >= hi_thr, "high", np.where(mr <= lo_thr, "low", "medium"))

        # Inflection point: where 2nd derivative of mr is most negative
        inflection_x = self._find_inflection(fn, xs)

        # Current spend metrics
        current_x = float(cr.current_x_median) if cr.current_x_median is not None else None
        current_y = fn(current_x) if current_x is not None else None
        current_mr = None
        current_el = None
        if current_x is not None and current_x > 0:
            current_mr_arr = self._marginal_returns(fn, np.array([current_x]))
            current_mr = float(current_mr_arr[0])
            current_el = current_mr * current_x / current_y if current_y and current_y > 0 else 0.0

        # Asymptote from model parameter
        asymptote = self._get_asymptote(cr)

        return ResponseCurveResult(
            campaign_id=cid,
            x_values=xs,
            y_values=ys,
            marginal_returns=mr,
            elasticities=elasticities,
            inflection_point_x=inflection_x,
            efficiency_zone=zones,
            current_x=current_x,
            current_y=current_y,
            current_marginal_return=current_mr,
            current_elasticity=current_el,
            saturation_point=cr.saturation_point,
            asymptote=asymptote,
            roi_curve=roi,
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _marginal_returns(fn: Callable, xs: np.ndarray) -> np.ndarray:
        """
        Compute marginal return at each x using central differences.

        Works for arrays of any length including a single element.
        Step size is 0.1% of x, with a floor of 1 to avoid division by zero.
        """
        n = len(xs)
        mr = np.zeros(n)
        if n == 0:
            return mr
        for i, x in enumerate(xs):
            step = max(float(x) * 0.001, 1.0)
            mr[i] = (fn(x + step) - fn(max(x - step, 0.0))) / (2.0 * step)
        # Responses are non-decreasing; clip tiny negatives caused by float noise
        return np.maximum(mr, 0.0)

    @staticmethod
    def _find_inflection(fn: Callable, xs: np.ndarray) -> float | None:
        """
        Estimate the inflection point: where the rate of marginal-return
        decline is steepest (minimum of the 2nd derivative of the curve).
        Only meaningful for S-shaped (logistic) curves.
        """
        if len(xs) < 10:
            return None
        try:
            step = xs[1] - xs[0]
            ys = np.array([fn(x) for x in xs])
            d2y = np.gradient(np.gradient(ys, step), step)
            idx = int(np.argmin(d2y))
            # Only report if the 2nd derivative is meaningfully negative
            if d2y[idx] < -1e-15:
                return float(xs[idx])
        except Exception:
            pass
        return None

    @staticmethod
    def _get_asymptote(cr: CampaignResult) -> float | None:
        """Extract the theoretical maximum outcome from the model parameters."""
        if not cr.best_model or not cr.best_model_params:
            return None
        params = list(cr.best_model_params.values())
        if cr.best_model in (
            "hill",
            "hill_bayesian",
            "negative_exponential",
            "michaelis_menten",
            "logistic",
        ):
            return float(params[0])
        return None  # power has no finite asymptote

    def _log(self, msg: str) -> None:
        """
        Print msg to stdout when verbose=True.
        """
        if self.verbose:
            print(f"[ResponseCurveAnalyzer] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def analyse_response_curves(
    batch: CampaignBatchResult,
    n_points: int = 500,
    x_max_multiplier: float = 1.5,
    verbose: bool = False,
) -> dict[Any, ResponseCurveResult]:
    """
    One-function shortcut: analyse response curves for all campaigns in a batch.

    Parameters
    ----------
    batch : CampaignBatchResult
    n_points : int
        Grid resolution per campaign.
    x_max_multiplier : float
        Grid upper bound as multiple of saturation point.
    verbose : bool

    Returns
    -------
    dict {campaign_id: ResponseCurveResult}

    Examples
    --------
    >>> from adsat.response_curves import analyse_response_curves
    >>> results = analyse_response_curves(batch)
    >>> print(ResponseCurveAnalyzer().summary_table(results))
    """
    analyzer = ResponseCurveAnalyzer(
        n_points=n_points,
        x_max_multiplier=x_max_multiplier,
        verbose=verbose,
    )
    return analyzer.analyse(batch)
