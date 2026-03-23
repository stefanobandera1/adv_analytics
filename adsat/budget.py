"""
adsat.budget
============
Budget optimisation across multiple advertising campaigns.

Given fitted saturation curves and a total budget, this module finds the
spend allocation that **maximises total predicted outcome** (conversions,
revenue, ROI) subject to a budget constraint and optional per-campaign
spend floors and caps.

The core insight: once you know each campaign's saturation curve you can
compare the *marginal return* of the next pound/dollar on every campaign
and keep reallocating toward the highest-return option.  This module
automates that process.

Key classes & functions
-----------------------
BudgetOptimizer      – main class; works with CampaignBatchResult
BudgetAllocation     – result dataclass with allocations table and plots
optimise_budget()    – one-liner convenience function

Typical workflow
----------------
>>> from adsat import CampaignSaturationAnalyzer
>>> from adsat.budget import BudgetOptimizer
>>>
>>> batch    = CampaignSaturationAnalyzer(...).run(df)
>>> opt      = BudgetOptimizer(total_budget=5_000_000)
>>> result   = opt.optimise(batch)
>>> result.print_summary()
>>> result.plot()
>>>
>>> # One-liner shortcut
>>> from adsat.budget import optimise_budget
>>> result = optimise_budget(batch, total_budget=5_000_000)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from adsat.campaign import CampaignBatchResult, CampaignResult
from adsat.modeling import MODEL_REGISTRY

# ── colour palette (consistent with rest of adsat) ────────────────────────────
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_GREEN = "#3BB273"
_RED = "#E84855"
_GREY = "#6C757D"
_PURPLE = "#7B2D8B"
PALETTE = [_BLUE, _ORANGE, _GREEN, _RED, _PURPLE, _GREY, "#F4A261", "#264653", "#A8DADC", "#E9C46A"]


def _fmt_large(x, _=None) -> str:
    """Format large numbers with k/M suffix for axis tick labels."""
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.1f}"


# ── helpers ───────────────────────────────────────────────────────────────────


def _build_response_fn(cr: CampaignResult) -> Callable[[float], float] | None:
    """
    Build a scalar callable  y = f(x)  from a CampaignResult.

    Returns None when the campaign failed or has no fitted model.
    """
    if not cr.succeeded or cr.best_model is None or not cr.best_model_params:
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


def _resolve_per_campaign(
    value: float | dict,
    campaign_id: Any,
    default: float,
) -> float:
    """Return a per-campaign value from either a scalar or a dict."""
    if isinstance(value, dict):
        return float(value.get(campaign_id, default))
    if value is None:
        return default
    return float(value)


def _marginal_return(fn: Callable, x: float, budget: float) -> float:
    """
    Numerical marginal return: Δoutcome / Δspend at point x.
    Uses 0.1% of the total budget as the perturbation step.
    """
    delta = max(budget * 0.001, 1.0)
    return (fn(x + delta) - fn(max(x - delta, 0.0))) / (2 * delta)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class BudgetAllocation:
    """
    Result of a budget optimisation run.

    Attributes
    ----------
    total_budget : float
        Budget that was distributed across campaigns.
    allocations : pd.DataFrame
        One row per campaign.  Key columns:

        campaign_id, current_spend, optimal_spend, spend_change,
        spend_change_pct, current_outcome, optimal_outcome,
        outcome_lift, outcome_lift_pct,
        pct_of_saturation_before, pct_of_saturation_after,
        marginal_return_at_optimum
    total_current_outcome : float
    total_optimal_outcome : float
    total_outcome_lift : float
    total_outcome_lift_pct : float
    converged : bool
    notes : str
    """

    total_budget: float
    allocations: pd.DataFrame
    total_current_outcome: float
    total_optimal_outcome: float
    total_outcome_lift: float
    total_outcome_lift_pct: float
    converged: bool
    notes: str = ""

    # ── human-readable output ─────────────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a concise, formatted allocation summary."""
        sep = "=" * 72
        print(sep)
        print("  ADSAT – BUDGET OPTIMISATION RESULTS")
        print(sep)
        print(f"  Total budget       : {self.total_budget:>15,.0f}")
        print(f"  Current outcome    : {self.total_current_outcome:>15,.2f}")
        print(f"  Optimal outcome    : {self.total_optimal_outcome:>15,.2f}")
        print(
            f"  Outcome lift       : {self.total_outcome_lift:>15,.2f}"
            f"  ({self.total_outcome_lift_pct:+.1f}%)"
        )
        print(f"  Converged          : {self.converged}")
        if self.notes:
            print(f"  Notes              : {self.notes}")
        print()

        display_cols = [
            "campaign_id",
            "current_spend",
            "optimal_spend",
            "spend_change_pct",
            "current_outcome",
            "optimal_outcome",
            "outcome_lift_pct",
            "pct_of_saturation_after",
        ]
        df = self.allocations[[c for c in display_cols if c in self.allocations.columns]].copy()

        # Format for readability
        for col in ("current_spend", "optimal_spend"):
            if col in df:
                df[col] = df[col].apply(lambda v: f"{v:>12,.0f}" if pd.notna(v) else "N/A")
        for col in ("spend_change_pct", "outcome_lift_pct", "pct_of_saturation_after"):
            if col in df:
                df[col] = df[col].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A")
        for col in ("current_outcome", "optimal_outcome"):
            if col in df:
                df[col] = df[col].apply(lambda v: f"{v:>10,.1f}" if pd.notna(v) else "N/A")

        print(df.to_string(index=False))
        print(sep)

    # ── visualisation ─────────────────────────────────────────────────────────

    def plot(
        self,
        response_fns: dict[Any, Callable] | None = None,
        save_path: str | None = None,
        figsize: tuple[int, int] = (16, 10),
    ) -> None:
        """
        Four-panel optimisation summary plot.

        Panels
        ------
        Top-left  : Current vs optimal spend per campaign (grouped bars)
        Top-right : Outcome lift per campaign (absolute, annotated with %)
        Bot-left  : % of saturation reached before and after reallocation
        Bot-right : Response curves with current (●) and optimal (★) markers
                    — only rendered when response_fns are provided

        Parameters
        ----------
        response_fns : dict {campaign_id: callable}, optional
            Pass the response_fns dict from the BudgetOptimizer to render
            the response-curve panel.
        save_path : str, optional
        figsize : (width, height)
        """
        df = self.allocations.copy()
        ids = df["campaign_id"].astype(str).tolist()
        n_camps = len(ids)
        colors = (PALETTE * ((n_camps // len(PALETTE)) + 1))[:n_camps]

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"Budget Optimisation  |  Budget: {self.total_budget:,.0f}  |  "
            f"Outcome lift: {self.total_outcome_lift_pct:+.1f}%",
            fontsize=13,
            fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)
        ax_spend = fig.add_subplot(gs[0, 0])
        ax_lift = fig.add_subplot(gs[0, 1])
        ax_sat = fig.add_subplot(gs[1, 0])
        ax_curves = fig.add_subplot(gs[1, 1])

        x_pos = np.arange(n_camps)
        bw = 0.35

        # ── Top-left: current vs optimal spend ───────────────────────────
        ax_spend.bar(
            x_pos - bw / 2,
            df["current_spend"].fillna(0),
            width=bw,
            color=_GREY,
            alpha=0.75,
            label="Current",
        )
        ax_spend.bar(
            x_pos + bw / 2,
            df["optimal_spend"].fillna(0),
            width=bw,
            color=_BLUE,
            alpha=0.85,
            label="Optimal",
        )
        ax_spend.set_xticks(x_pos)
        ax_spend.set_xticklabels(ids, rotation=25, ha="right", fontsize=8)
        ax_spend.set_ylabel("Spend", fontsize=9)
        ax_spend.set_title("Current vs Optimal Spend", fontsize=10)
        ax_spend.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
        ax_spend.legend(fontsize=8)

        # ── Top-right: outcome lift ───────────────────────────────────────
        lift_vals = df["outcome_lift"].fillna(0).values
        lift_pcts = df["outcome_lift_pct"].fillna(0).values
        bar_colors = [_GREEN if v >= 0 else _RED for v in lift_vals]
        ax_lift.bar(ids, lift_vals, color=bar_colors, alpha=0.85)
        ax_lift.axhline(0, color="black", lw=0.8)
        ax_lift.set_xticklabels(ids, rotation=25, ha="right", fontsize=8)
        ax_lift.set_ylabel("Outcome lift (absolute)", fontsize=9)
        ax_lift.set_title("Outcome Lift per Campaign", fontsize=10)
        ax_lift.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
        y_pad = max(abs(lift_vals).max() * 0.04, 1.0)
        for i, (v, pct) in enumerate(zip(lift_vals, lift_pcts)):
            ax_lift.text(
                i,
                v + (y_pad if v >= 0 else -y_pad * 3),
                f"{pct:+.1f}%",
                ha="center",
                fontsize=7,
                fontweight="bold",
                color="black",
            )

        # ── Bot-left: saturation % before / after ────────────────────────
        sat_before = df["pct_of_saturation_before"].fillna(0).values
        sat_after = df["pct_of_saturation_after"].fillna(0).values
        ax_sat.bar(x_pos - bw / 2, sat_before, width=bw, color=_ORANGE, alpha=0.75, label="Before")
        ax_sat.bar(x_pos + bw / 2, sat_after, width=bw, color=_BLUE, alpha=0.85, label="After")
        ax_sat.axhline(90, color=_RED, lw=1.2, ls="--", label="90% sat.")
        ax_sat.axhline(100, color="darkred", lw=1.0, ls=":", alpha=0.7)
        ax_sat.set_xticks(x_pos)
        ax_sat.set_xticklabels(ids, rotation=25, ha="right", fontsize=8)
        ax_sat.set_ylabel("% of Saturation Point", fontsize=9)
        ax_sat.set_title("Saturation Level Before / After", fontsize=10)
        ax_sat.legend(fontsize=8)
        max_sat = max(sat_after.max(), sat_before.max(), 110)
        ax_sat.set_ylim(0, max_sat * 1.12)

        # ── Bot-right: response curves ────────────────────────────────────
        if response_fns:
            for cid, color in zip(response_fns.keys(), colors):
                fn = response_fns[cid]
                row = df[df["campaign_id"] == cid]
                if row.empty:
                    continue
                c_spend = float(row["current_spend"].iloc[0])
                o_spend = float(row["optimal_spend"].iloc[0])
                x_max = max(c_spend, o_spend) * 2.2
                xs = np.linspace(0, x_max, 400)
                ys = np.array([fn(xi) for xi in xs])
                ax_curves.plot(xs, ys, lw=2, color=color, label=str(cid), alpha=0.85)
                ax_curves.scatter([c_spend], [fn(c_spend)], s=55, color=color, marker="o", zorder=5)
                ax_curves.scatter([o_spend], [fn(o_spend)], s=80, color=color, marker="*", zorder=6)
                ax_curves.axvline(o_spend, color=color, lw=1, ls="--", alpha=0.45)
            ax_curves.set_xlabel("Spend", fontsize=9)
            ax_curves.set_ylabel("Outcome", fontsize=9)
            ax_curves.set_title("Response Curves  (● current  ★ optimal)", fontsize=9)
            ax_curves.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax_curves.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax_curves.legend(fontsize=7)
        else:
            ax_curves.axis("off")
            ax_curves.text(
                0.5,
                0.5,
                "Response curves not available.\nPass response_fns=optimizer.response_fns\nto plot() to enable this panel.",
                ha="center",
                va="center",
                transform=ax_curves.transAxes,
                fontsize=9,
                color=_GREY,
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class BudgetOptimizer:
    """
    Optimise budget allocation across campaigns to maximise total outcome.

    Solves the constrained optimisation problem:

        maximise   Σ f_i(x_i)
        subject to Σ x_i  = total_budget
                   lb_i   ≤ x_i ≤ ub_i    for all campaigns i

    where f_i is the fitted saturation curve for campaign i and
    x_i is the spend allocated to that campaign.

    Parameters
    ----------
    total_budget : float
        Total spend budget to distribute across all campaigns.
    min_spend : float or dict {campaign_id: float}
        Minimum spend floor.  Use a dict for campaign-specific floors.
        Default 0.
    max_spend : float or dict {campaign_id: float}
        Maximum spend cap.  Default: 3 × each campaign's saturation point
        (or the full budget when no saturation point is available).
    verbose : bool

    Examples
    --------
    >>> from adsat.budget import BudgetOptimizer
    >>> opt    = BudgetOptimizer(total_budget=5_000_000)
    >>> result = opt.optimise(batch)
    >>> result.print_summary()
    >>> result.plot(response_fns=opt.response_fns)
    """

    def __init__(
        self,
        total_budget: float,
        min_spend: float | dict[Any, float] = 0.0,
        max_spend: float | dict[Any, float] | None = None,
        n_restarts: int = 10,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        total_budget : float
            Total spend budget to distribute across all campaigns.
        min_spend : float or dict {campaign_id: float}
            Minimum spend floor.  Default 0.
        max_spend : float or dict {campaign_id: float}
            Maximum spend cap.  Default 3x each campaign's saturation point.
        n_restarts : int
            Number of random starting points for the SLSQP solver.
            More restarts reduce the risk of settling in a local optimum when
            bound constraints are tight or campaign scales differ widely.
            Default 10.  Set to 1 for a quick single-start run.
        random_seed : int
            Seed for reproducible random starting points.  Default 42.
        verbose : bool
        """
        if total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {total_budget}.")
        if n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {n_restarts}.")
        self.total_budget = float(total_budget)
        self.min_spend = min_spend
        self.max_spend = max_spend
        self.n_restarts = int(n_restarts)
        self.random_seed = int(random_seed)
        self.verbose = verbose

        # Populated during optimise() — exposed so users can pass to plot()
        self.response_fns: dict[Any, Callable] = {}

    # ── public API ─────────────────────────────────────────────────────────────

    def optimise(
        self,
        batch: CampaignBatchResult,
        current_spend: dict[Any, float] | None = None,
    ) -> BudgetAllocation:
        """
        Optimise budget allocation across all succeeded campaigns in a batch.

        Parameters
        ----------
        batch : CampaignBatchResult
            Output of ``CampaignSaturationAnalyzer.run()``.
        current_spend : dict {campaign_id: float}, optional
            Current spend per campaign used for comparison.
            Defaults to each campaign's ``current_x_median``.

        Returns
        -------
        BudgetAllocation
        """
        # collect campaigns that have usable response functions
        campaigns: list[Any] = []
        self.response_fns = {}

        for cid in batch.succeeded_campaigns():
            cr = batch.get(cid)
            fn = _build_response_fn(cr)
            if fn is None:
                self._log(f"Skipping '{cid}' — no fitted response function.")
                continue
            campaigns.append(cid)
            self.response_fns[cid] = fn

        if len(campaigns) < 2:
            raise ValueError(
                f"Need at least 2 campaigns with fitted models. "
                f"Got {len(campaigns)} usable campaign(s)."
            )

        self._log(f"Optimising {len(campaigns)} campaigns | " f"budget = {self.total_budget:,.0f}")

        bounds = self._build_bounds(campaigns, batch)
        curr_spend = self._resolve_current_spend(campaigns, batch, current_spend)
        opt_spend, converged, notes = self._run_optimisation(campaigns, bounds)

        return self._build_result(campaigns, batch, curr_spend, opt_spend, converged, notes)

    def marginal_returns_table(
        self,
        batch: CampaignBatchResult,
        spend_levels: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Return a table of marginal returns at a range of spend levels per campaign.

        Useful for understanding where the curve is steep vs flat before
        committing to an optimisation.

        Parameters
        ----------
        batch : CampaignBatchResult
        spend_levels : np.ndarray, optional
            Spend values at which to evaluate marginal return.
            Defaults to 20 evenly-spaced points from 0 to 1.5× saturation point.

        Returns
        -------
        pd.DataFrame  columns: campaign_id, spend, outcome, marginal_return
        """
        rows = []
        for cid in batch.succeeded_campaigns():
            cr = batch.get(cid)
            fn = _build_response_fn(cr)
            if fn is None:
                continue
            x_max = (
                cr.saturation_point * 1.5
                if cr.saturation_point and cr.saturation_point > 0
                else self.total_budget
            )
            xs = spend_levels if spend_levels is not None else np.linspace(0, x_max, 20)
            for x in xs:
                rows.append(
                    {
                        "campaign_id": cid,
                        "spend": round(float(x), 2),
                        "outcome": round(fn(x), 4),
                        "marginal_return": round(_marginal_return(fn, x, self.total_budget), 6),
                    }
                )
        return pd.DataFrame(rows)

    # ── optimisation engine ───────────────────────────────────────────────────

    def _run_optimisation(
        self,
        campaigns: list[Any],
        bounds: list[tuple[float, float]],
    ) -> tuple[np.ndarray, bool, str]:
        """
        Solve:  min  -Σ f_i(x_i)   s.t.  Σ x_i = budget,  lb_i ≤ x_i ≤ ub_i

        Uses multiple random restarts to avoid local optima, keeps the best.
        Convergence is verified by budget-constraint satisfaction, not just the
        solver's success flag (SLSQP can falsely report failure at corner/boundary
        solutions even when the optimum is correct).

        Returns (optimal_spend_array, converged, notes).
        """
        n = len(campaigns)
        fns = self.response_fns
        budget = self.total_budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Guard: if min-spend sum exceeds budget, scale down proportionally
        if lo.sum() > budget:
            warnings.warn(
                f"Sum of min-spend floors ({lo.sum():,.0f}) exceeds "
                f"total_budget ({budget:,.0f}). Scaling floors down.",
                UserWarning,
            )
            lo = lo * (budget / lo.sum()) * 0.99
            for i in range(n):
                bounds[i] = (lo[i], max(hi[i], lo[i] + 1.0))

        # Per-campaign scalar functions (captured once)
        fn_list = [fns[cid] for cid in campaigns]

        def neg_total(x: np.ndarray) -> float:
            """
            Objective: negative total outcome (minimised by scipy, equivalent to maximising outcome).
            """
            return -sum(f(xi) for f, xi in zip(fn_list, x))

        def neg_total_grad(x: np.ndarray) -> np.ndarray:
            """
            Per-campaign central-difference gradient of the negative total outcome.
            More numerically stable than a full-vector perturbation for independent curves.
            """
            # Independent central-difference per campaign — more stable
            # than perturbing the full vector.
            g = np.zeros(n)
            for j, (f, xj) in enumerate(zip(fn_list, x)):
                eps = max(float(xj) * 1e-4, budget * 1e-6, 1.0)
                g[j] = -(f(xj + eps) - f(max(xj - eps, 0.0))) / (2.0 * eps)
            return g

        constraints = {"type": "eq", "fun": lambda x: x.sum() - budget, "jac": lambda x: np.ones(n)}

        def _clip_renorm(x: np.ndarray) -> np.ndarray:
            """Clip to bounds then rescale to satisfy budget exactly."""
            x = np.clip(x, lo, hi)
            s = x.sum()
            return x / s * budget if s > 0 else x

        def _make_x0(seed: int) -> np.ndarray:
            """Random feasible starting point."""
            rng = np.random.default_rng(seed)
            w = rng.dirichlet(np.ones(n))
            x0 = np.clip(w * budget, lo, hi)
            s = x0.sum()
            return x0 / s * budget if s > 0 else x0

        def _is_feasible(x: np.ndarray, tol: float = 1.0) -> bool:
            """Budget constraint satisfied within tol units."""
            return abs(x.sum() - budget) <= tol

        # --- multi-start loop ---
        # n_restarts is user-configurable via self.n_restarts.
        # We always include 2 deterministic warm starts (equal split + midpoint),
        # then fill the remainder with Dirichlet random starts seeded reproducibly.
        best_x = None
        best_obj = np.inf
        all_msgs: list[str] = []

        n_random = max(0, self.n_restarts - 2)
        starting_points = [
            # 1. Equal split: works well when campaigns are similar scale
            _clip_renorm(np.full(n, budget / n)),
            # 2. Midpoint of each campaign's bounds: covers corner-of-feasible-region cases
            _clip_renorm((lo + hi) / 2.0),
        ] + [_make_x0(seed=self.random_seed + r) for r in range(n_random)]

        for x0 in starting_points:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = minimize(
                    neg_total,
                    x0,
                    jac=neg_total_grad,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 3000, "ftol": 1e-12},
                )
            all_msgs.append(sol.message)
            if sol.fun < best_obj:
                best_obj = sol.fun
                best_x = sol.x.copy()

        opt_x = _clip_renorm(best_x)

        # Convergence: budget constraint met within 1 currency unit
        converged = _is_feasible(opt_x, tol=1.0)
        notes = "" if converged else f"Best solver message: {all_msgs[0]}"

        if not converged:
            self._log(f"Warning: budget constraint not fully met — {notes}")

        return opt_x, converged, notes

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_bounds(
        self,
        campaigns: list[Any],
        batch: CampaignBatchResult,
    ) -> list[tuple[float, float]]:
        """
        Construct the lower/upper bound pairs for each campaign.

        Falls back to 3× the campaign saturation point as the upper bound when no
        explicit max_spend is provided.
        """
        bounds = []
        for cid in campaigns:
            lo = _resolve_per_campaign(self.min_spend, cid, default=0.0)

            if self.max_spend is not None:
                hi = _resolve_per_campaign(self.max_spend, cid, default=self.total_budget)
            else:
                # Default cap: 3 × saturation point; fall back to full budget
                cr = batch.get(cid)
                if cr.saturation_point and cr.saturation_point > 0:
                    hi = cr.saturation_point * 3.0
                else:
                    hi = self.total_budget

            bounds.append((max(lo, 0.0), max(hi, lo + 1.0)))
        return bounds

    def _resolve_current_spend(
        self,
        campaigns: list[Any],
        batch: CampaignBatchResult,
        current_spend: dict | None,
    ) -> np.ndarray:
        """
        Return an array of current spend values, one per campaign.

        Uses the current_spend dict when provided; otherwise falls back to
        each campaign's current_x_median from the batch result.
        """
        arr = []
        for cid in campaigns:
            if current_spend and cid in current_spend:
                arr.append(float(current_spend[cid]))
            else:
                cr = batch.get(cid)
                arr.append(float(cr.current_x_median) if cr.current_x_median is not None else 0.0)
        return np.array(arr)

    def _build_result(
        self,
        campaigns: list[Any],
        batch: CampaignBatchResult,
        curr_spend: np.ndarray,
        opt_spend: np.ndarray,
        converged: bool,
        notes: str,
    ) -> BudgetAllocation:
        """
        Construct a BudgetAllocation from the optimised spend array.

        Computes outcome lift, % of saturation reached, and marginal return
        at the optimum for each campaign.
        """
        rows = []
        for i, cid in enumerate(campaigns):
            cr = batch.get(cid)
            fn = self.response_fns[cid]
            curr = curr_spend[i]
            opt = opt_spend[i]
            curr_y = fn(curr)
            opt_y = fn(opt)
            lift = opt_y - curr_y
            sat = cr.saturation_point

            rows.append(
                {
                    "campaign_id": cid,
                    "current_spend": round(curr, 2),
                    "optimal_spend": round(opt, 2),
                    "spend_change": round(opt - curr, 2),
                    "spend_change_pct": round((opt - curr) / max(curr, 1.0) * 100, 1),
                    "current_outcome": round(curr_y, 4),
                    "optimal_outcome": round(opt_y, 4),
                    "outcome_lift": round(lift, 4),
                    "outcome_lift_pct": round(lift / max(curr_y, 1e-9) * 100, 1),
                    "pct_of_saturation_before": (
                        round(curr / sat * 100, 1) if sat and sat > 0 else None
                    ),
                    "pct_of_saturation_after": (
                        round(opt / sat * 100, 1) if sat and sat > 0 else None
                    ),
                    "marginal_return_at_optimum": round(
                        _marginal_return(fn, opt, self.total_budget), 6
                    ),
                }
            )

        df = pd.DataFrame(rows)
        total_curr = df["current_outcome"].sum()
        total_opt = df["optimal_outcome"].sum()
        total_lift = total_opt - total_curr
        total_lift_pct = total_lift / max(total_curr, 1e-9) * 100

        return BudgetAllocation(
            total_budget=self.total_budget,
            allocations=df,
            total_current_outcome=round(total_curr, 4),
            total_optimal_outcome=round(total_opt, 4),
            total_outcome_lift=round(total_lift, 4),
            total_outcome_lift_pct=round(total_lift_pct, 2),
            converged=converged,
            notes=notes,
        )

    def _log(self, msg: str) -> None:
        """
        Print msg to stdout when verbose=True.
        """
        if self.verbose:
            print(f"[BudgetOptimizer] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def optimise_budget(
    batch: CampaignBatchResult,
    total_budget: float,
    current_spend: dict[Any, float] | None = None,
    min_spend: float | dict[Any, float] = 0.0,
    max_spend: float | dict[Any, float] | None = None,
    n_restarts: int = 10,
    random_seed: int = 42,
    verbose: bool = False,
) -> BudgetAllocation:
    """
    One-function budget optimisation shortcut.

    Parameters
    ----------
    batch : CampaignBatchResult
        Output of ``CampaignSaturationAnalyzer.run()``.
    total_budget : float
        Total budget to distribute.
    current_spend : dict {campaign_id: float}, optional
        Current spend per campaign.  Defaults to ``current_x_median``.
    min_spend : float or dict
        Per-campaign spend floor.  Default 0.
    max_spend : float or dict
        Per-campaign spend cap.  Default 3 x saturation point.
    n_restarts : int
        Number of random starting points for the solver.  Default 10.
        Increase to 20-50 for tighter bound constraints or many campaigns.
    random_seed : int
        Seed for reproducibility.  Default 42.
    verbose : bool

    Returns
    -------
    BudgetAllocation

    Examples
    --------
    >>> from adsat.budget import optimise_budget
    >>> result = optimise_budget(batch, total_budget=5_000_000)
    >>> result.print_summary()
    """
    opt = BudgetOptimizer(
        total_budget=total_budget,
        min_spend=min_spend,
        max_spend=max_spend,
        n_restarts=n_restarts,
        random_seed=random_seed,
        verbose=verbose,
    )
    return opt.optimise(batch, current_spend=current_spend)
