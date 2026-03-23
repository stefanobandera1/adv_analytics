"""
adsat.simulation
================
Scenario simulation for advertising campaign planning.

Once you have fitted saturation curves, this module answers "what if" questions
without running a new optimisation:

  * What happens to total conversions if I increase Brand campaign spend by 20%?
  * What if I shift £500k from Display to Video?
  * How do my top 3 budget scenarios compare against each other?
  * What is the predicted outcome at every spend level from 0 to 5M?

Key classes & functions
-----------------------
ScenarioSimulator      – main class; define and compare named scenarios
Scenario               – dataclass describing a single budget scenario
SimulationResult       – dataclass holding all scenario outcomes and charts
simulate()             – one-liner convenience function

Typical workflow
----------------
>>> from adsat.simulation import ScenarioSimulator, Scenario
>>>
>>> sim = ScenarioSimulator(batch)   # CampaignBatchResult
>>>
>>> # Add "what if" scenarios
>>> sim.add_scenario("Status quo",    spends={"Alpha": 2_000_000, "Beta": 800_000})
>>> sim.add_scenario("+20% Brand",    spends={"Alpha": 2_400_000, "Beta": 800_000})
>>> sim.add_scenario("Shift to Beta", spends={"Alpha": 1_500_000, "Beta": 1_300_000})
>>>
>>> result = sim.run()
>>> result.print_summary()
>>> sim.plot(result)
>>>
>>> # One-liner: auto-generate scenarios from current spend
>>> from adsat.simulation import simulate
>>> result = simulate(batch, budgets=[2_000_000, 2_500_000, 3_000_000])
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.gridspec as gridspec
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
    Format large numbers with k / M suffix for cleaner axis tick labels.
    E.g. 1_500_000 -> "1.5M", 35_000 -> "35k".
    """
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.1f}"


def _build_response_fn(cr: CampaignResult) -> Callable[[float], float] | None:
    """Build a scalar callable  y = f(x)  from a CampaignResult."""
    if not cr.succeeded or not cr.best_model or not cr.best_model_params:
        return None
    registry_name = "hill" if cr.best_model == "hill_bayesian" else cr.best_model
    spec = MODEL_REGISTRY.get(registry_name)
    if spec is None:
        return None
    func = spec["func"]
    params = list(cr.best_model_params.values())

    def response(x: float) -> float:
        """
        Evaluate the fitted saturation curve at spend x, clipping negative inputs to 0.
        """
        return float(func(max(float(x), 0.0), *params))

    return response


# ─────────────────────────────────────────────────────────────────────────────
# Scenario dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Scenario:
    """
    Definition of a single budget scenario.

    Parameters
    ----------
    name : str
        Human-readable name shown in charts and tables.
    spends : dict {campaign_id: float}
        Spend allocated to each campaign in this scenario.
        Campaigns not listed default to 0 spend.
    description : str, optional
        Optional free-text description of the scenario.
    colour : str, optional
        Hex colour for chart rendering.  Auto-assigned if None.

    Examples
    --------
    >>> s = Scenario(
    ...     name="Shift to Brand",
    ...     spends={"Brand": 2_500_000, "Display": 400_000},
    ...     description="Reduce display, invest more in brand search",
    ... )
    """

    name: str
    spends: dict[Any, float]
    description: str = ""
    colour: str | None = None

    def total_spend(self) -> float:
        """
        Return the sum of all campaign spend values in this scenario.
        """
        return float(sum(self.spends.values()))


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SimulationResult:
    """
    Full results of a scenario simulation run.

    Attributes
    ----------
    scenarios : list of Scenario
        The scenario definitions used.
    campaign_ids : list
        All campaign IDs included in the simulation.
    summary_table : pd.DataFrame
        One row per scenario.  Columns:
          scenario_name, total_spend, total_outcome,
          outcome_vs_baseline, outcome_vs_baseline_pct,
          per_campaign outcome columns
    baseline_name : str or None
        Name of the scenario designated as baseline for comparison.
    best_scenario_name : str
        Name of the scenario with the highest total outcome.
    response_fns : dict {campaign_id: callable}
        The response functions used (for external inspection).
    """

    scenarios: list[Scenario]
    campaign_ids: list[Any]
    summary_table: pd.DataFrame
    baseline_name: str | None
    best_scenario_name: str
    response_fns: dict[Any, Callable]

    def print_summary(self) -> None:
        """Print a readable scenario comparison table."""
        sep = "=" * 72
        print(sep)
        print("  ADSAT – SCENARIO SIMULATION RESULTS")
        print(sep)
        if self.baseline_name:
            print(f"  Baseline scenario  : {self.baseline_name}")
        print(f"  Best scenario      : {self.best_scenario_name}  (highest outcome)")
        print()
        cols = [
            "scenario_name",
            "total_spend",
            "total_outcome",
            "outcome_vs_baseline",
            "outcome_vs_baseline_pct",
        ]
        available = [c for c in cols if c in self.summary_table.columns]
        print(self.summary_table[available].to_string(index=False))
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class ScenarioSimulator:
    """
    Define and compare named budget scenarios using fitted saturation curves.

    Parameters
    ----------
    batch : CampaignBatchResult
        Output of CampaignSaturationAnalyzer.run().
    verbose : bool

    Examples
    --------
    >>> sim = ScenarioSimulator(batch)
    >>> sim.add_scenario("Current",    {"Alpha": 2_000_000, "Beta": 800_000})
    >>> sim.add_scenario("+20% Alpha", {"Alpha": 2_400_000, "Beta": 800_000})
    >>> result = sim.run()
    >>> result.print_summary()
    >>> sim.plot(result)
    """

    def __init__(self, batch: CampaignBatchResult, verbose: bool = True):
        """
        Build response functions for all succeeded campaigns in the batch and store them
        as self._response_fns.  Campaigns without fitted models are silently skipped.
        Scenarios are added separately via add_scenario() or add_budget_sweep().
        """
        self.batch = batch
        self.verbose = verbose
        self._scenarios: list[Scenario] = []
        self._response_fns: dict[Any, Callable] = {}

        # Build response functions for all succeeded campaigns
        for cid in batch.succeeded_campaigns():
            fn = _build_response_fn(batch.get(cid))
            if fn is not None:
                self._response_fns[cid] = fn

        if not self._response_fns:
            raise ValueError(
                "No fitted campaigns found in batch. " "Run CampaignSaturationAnalyzer.run() first."
            )

    # ── public API ─────────────────────────────────────────────────────────────

    def add_scenario(
        self,
        name: str,
        spends: dict[Any, float],
        description: str = "",
        colour: str | None = None,
    ) -> ScenarioSimulator:
        """
        Add a named scenario.

        Parameters
        ----------
        name : str
            Unique name for this scenario.
        spends : dict {campaign_id: float}
            Spend per campaign.  Any campaign not included gets 0 spend.
        description : str, optional
            Free-text description.
        colour : str, optional
            Hex colour string for charts.  Auto-assigned if None.

        Returns
        -------
        self  (for method chaining)
        """
        if not spends:
            raise ValueError(f"Scenario '{name}' has an empty spends dict.")

        unknown = set(spends) - set(self._response_fns)
        if unknown:
            warnings.warn(
                f"Scenario '{name}': campaign IDs {unknown} are not in the fitted "
                "batch and will contribute 0 outcome.",
                UserWarning,
            )

        colour = colour or PALETTE[len(self._scenarios) % len(PALETTE)]
        self._scenarios.append(
            Scenario(name=name, spends=spends, description=description, colour=colour)
        )
        return self

    def add_budget_sweep(
        self,
        campaign_id: Any,
        spend_values: list[float] | np.ndarray,
        fixed_spends: dict[Any, float] | None = None,
        prefix: str = "",
    ) -> ScenarioSimulator:
        """
        Add one scenario per spend level for a single campaign, holding all
        other campaigns at fixed_spends (or 0 if not provided).

        Useful for quickly generating a sweep of scenarios to see how
        changing one campaign's budget affects total outcome.

        Parameters
        ----------
        campaign_id : any
            The campaign to sweep.
        spend_values : list or ndarray
            Spend levels to test for campaign_id.
        fixed_spends : dict, optional
            Fixed spend for all other campaigns.
        prefix : str
            Prefix for the auto-generated scenario names.

        Returns
        -------
        self
        """
        if campaign_id not in self._response_fns:
            raise ValueError(
                f"Campaign '{campaign_id}' not in fitted batch. "
                f"Valid: {list(self._response_fns)}"
            )
        base = fixed_spends or {}
        for x in spend_values:
            spends = {**base, campaign_id: float(x)}
            label = f"{prefix}{campaign_id}={x:,.0f}" if prefix else f"{campaign_id}={x:,.0f}"
            self.add_scenario(label, spends)
        return self

    def run(
        self,
        baseline: str | None = None,
    ) -> SimulationResult:
        """
        Evaluate all added scenarios and return a SimulationResult.

        Parameters
        ----------
        baseline : str, optional
            Name of the scenario to use as the comparison baseline.
            Defaults to the first scenario added.

        Returns
        -------
        SimulationResult
        """
        if not self._scenarios:
            raise ValueError("No scenarios added. Call add_scenario() before run().")

        campaign_ids = list(self._response_fns.keys())
        baseline_name = baseline or self._scenarios[0].name

        # Find baseline scenario
        baseline_scenario = None
        for s in self._scenarios:
            if s.name == baseline_name:
                baseline_scenario = s
                break

        if baseline_scenario is None:
            warnings.warn(
                f"Baseline scenario '{baseline_name}' not found. " f"Using first scenario.",
                UserWarning,
            )
            baseline_scenario = self._scenarios[0]
            baseline_name = baseline_scenario.name

        # Evaluate baseline
        baseline_outcome = self._evaluate_total(baseline_scenario)

        rows = []
        for sc in self._scenarios:
            per_campaign = {}
            for cid in campaign_ids:
                spend = float(sc.spends.get(cid, 0.0))
                per_campaign[cid] = self._response_fns[cid](spend)

            total_out = float(sum(per_campaign.values()))
            delta = total_out - baseline_outcome
            delta_pct = (delta / baseline_outcome * 100) if baseline_outcome > 0 else 0.0

            row = {
                "scenario_name": sc.name,
                "description": sc.description,
                "total_spend": round(sc.total_spend(), 0),
                "total_outcome": round(total_out, 2),
                "outcome_vs_baseline": round(delta, 2),
                "outcome_vs_baseline_pct": round(delta_pct, 1),
            }
            for cid in campaign_ids:
                row[f"spend_{cid}"] = round(float(sc.spends.get(cid, 0.0)), 0)
                row[f"outcome_{cid}"] = round(per_campaign[cid], 2)
            rows.append(row)

        summary = pd.DataFrame(rows)
        best_name = summary.loc[summary["total_outcome"].idxmax(), "scenario_name"]

        self._log(f"Evaluated {len(self._scenarios)} scenarios. " f"Best: '{best_name}'.")

        return SimulationResult(
            scenarios=self._scenarios,
            campaign_ids=campaign_ids,
            summary_table=summary,
            baseline_name=baseline_name,
            best_scenario_name=str(best_name),
            response_fns=self._response_fns,
        )

    # ── plots ──────────────────────────────────────────────────────────────────

    def plot(
        self,
        result: SimulationResult,
        figsize: tuple[int, int] = (15, 9),
        save_path: str | None = None,
    ) -> None:
        """
        Four-panel scenario comparison dashboard.

        Panels
        ------
        Top-left  : Total outcome per scenario (horizontal bar, best highlighted)
        Top-right : Outcome vs baseline % change (waterfall-style bar)
        Bot-left  : Per-campaign spend breakdown (stacked bar per scenario)
        Bot-right : Per-campaign outcome breakdown (stacked bar per scenario)

        Parameters
        ----------
        result : SimulationResult — from run()
        figsize : tuple
        save_path : str, optional
        """
        df = result.summary_table
        sc_names = df["scenario_name"].tolist()
        n_sc = len(sc_names)
        n_camps = len(result.campaign_ids)
        cids = result.campaign_ids
        colors = [s.colour or PALETTE[i % len(PALETTE)] for i, s in enumerate(result.scenarios)]

        fig = plt.figure(figsize=figsize)
        fig.suptitle("Scenario Simulation — Budget Comparison", fontsize=13, fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # ── Panel 1: total outcome horizontal bar ─────────────────────────
        totals = df["total_outcome"].values
        best_i = int(np.argmax(totals))
        bar_clr = [_GREEN if i == best_i else c for i, c in enumerate(colors)]
        bars = ax1.barh(sc_names, totals, color=bar_clr, alpha=0.85, edgecolor="white")
        ax1.set_xlabel("Total Predicted Outcome", fontsize=9)
        ax1.set_title("Total Outcome per Scenario", fontsize=10, fontweight="bold")
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
        x_pad = totals.max() * 0.01
        for bar, v in zip(bars, totals):
            ax1.text(
                v + x_pad, bar.get_y() + bar.get_height() / 2, f"{v:,.0f}", va="center", fontsize=8
            )
        ax1.set_xlim(right=totals.max() * 1.15)

        # ── Panel 2: outcome vs baseline % ────────────────────────────────
        pcts = df["outcome_vs_baseline_pct"].values
        clrs = [_GREEN if v >= 0 else _RED for v in pcts]
        ax2.bar(sc_names, pcts, color=clrs, alpha=0.85, edgecolor="white")
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_ylabel("Outcome vs Baseline (%)", fontsize=9)
        ax2.set_title(f"vs Baseline: '{result.baseline_name}'", fontsize=10, fontweight="bold")
        ax2.tick_params(axis="x", rotation=25, labelsize=8)
        y_pad2 = max(np.abs(pcts).max() * 0.04, 0.01)
        for i, (v, sc) in enumerate(zip(pcts, sc_names)):
            if sc != result.baseline_name:
                ax2.text(
                    i,
                    v + (y_pad2 if v >= 0 else -y_pad2 * 4),
                    f"{v:+.1f}%",
                    ha="center",
                    fontsize=8,
                    fontweight="bold",
                )

        # ── Panel 3: per-campaign spend stacked bar ────────────────────────
        if n_camps > 0:
            camp_colors = (PALETTE * ((n_camps // len(PALETTE)) + 1))[:n_camps]
            x_pos = np.arange(n_sc)
            bottom = np.zeros(n_sc)
            for ci, cid in enumerate(cids):
                col = f"spend_{cid}"
                if col in df.columns:
                    vals = df[col].values.astype(float)
                    ax3.bar(
                        x_pos,
                        vals,
                        bottom=bottom,
                        color=camp_colors[ci],
                        alpha=0.85,
                        edgecolor="white",
                        label=str(cid),
                    )
                    bottom += vals
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(sc_names, rotation=25, ha="right", fontsize=8)
            ax3.set_ylabel("Spend", fontsize=9)
            ax3.set_title("Spend Breakdown per Campaign", fontsize=10, fontweight="bold")
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax3.legend(fontsize=7, loc="upper right")

        # ── Panel 4: per-campaign outcome stacked bar ─────────────────────
        if n_camps > 0:
            bottom2 = np.zeros(n_sc)
            for ci, cid in enumerate(cids):
                col = f"outcome_{cid}"
                if col in df.columns:
                    vals = df[col].values.astype(float)
                    ax4.bar(
                        x_pos,
                        vals,
                        bottom=bottom2,
                        color=camp_colors[ci],
                        alpha=0.85,
                        edgecolor="white",
                        label=str(cid),
                    )
                    bottom2 += vals
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(sc_names, rotation=25, ha="right", fontsize=8)
            ax4.set_ylabel("Outcome", fontsize=9)
            ax4.set_title("Outcome Breakdown per Campaign", fontsize=10, fontweight="bold")
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax4.legend(fontsize=7, loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_spend_sweep(
        self,
        campaign_id: Any,
        fixed_spends: dict[Any, float] | None = None,
        x_min: float = 0,
        x_max: float | None = None,
        n_points: int = 200,
        figsize: tuple[int, int] = (10, 4),
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Plot total predicted outcome as the spend for one campaign sweeps
        from x_min to x_max, holding all other campaigns fixed.

        Parameters
        ----------
        campaign_id : any
            Campaign whose spend is varied.
        fixed_spends : dict, optional
            Spend for all other campaigns.  Defaults to 0.
        x_min : float
        x_max : float, optional
            Defaults to 2× the campaign's saturation point.
        n_points : int
        figsize : tuple
        save_path : str, optional

        Returns
        -------
        pd.DataFrame  columns: spend, campaign_outcome, total_outcome
        """
        if campaign_id not in self._response_fns:
            raise ValueError(
                f"Campaign '{campaign_id}' not in fitted batch. "
                f"Valid: {list(self._response_fns)}"
            )

        fn = self._response_fns[campaign_id]
        base = fixed_spends or {}

        # Determine x_max
        if x_max is None:
            cr = self.batch.get(campaign_id)
            if cr.saturation_point and cr.saturation_point > 0:
                x_max = cr.saturation_point * 2.0
            else:
                x_max = float(max(base.values())) * 3 if base else 5_000_000.0

        xs = np.linspace(x_min, x_max, n_points)
        camp_y = np.array([fn(x) for x in xs])

        # Fixed-campaign contribution
        fixed_y = sum(
            self._response_fns[cid](float(base.get(cid, 0.0)))
            for cid in self._response_fns
            if cid != campaign_id
        )
        total_y = camp_y + fixed_y

        df_sweep = pd.DataFrame(
            {
                "spend": xs,
                "campaign_outcome": camp_y,
                "total_outcome": total_y,
            }
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f"Spend Sweep — {campaign_id}", fontsize=12, fontweight="bold")

        ax1.plot(xs, camp_y, color=_BLUE, lw=2.5)
        ax1.fill_between(xs, 0, camp_y, alpha=0.12, color=_BLUE)
        ax1.set_xlabel(f"{campaign_id} spend", fontsize=9)
        ax1.set_ylabel("Campaign outcome", fontsize=9)
        ax1.set_title(f"{campaign_id} response curve", fontsize=10)
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))

        # Saturation point marker
        cr = self.batch.get(campaign_id)
        if cr.saturation_point:
            ax1.axvline(
                cr.saturation_point,
                color=_RED,
                lw=1.5,
                ls="--",
                label=f"Sat. pt={cr.saturation_point:,.0f}",
            )
            ax1.legend(fontsize=8)

        ax2.plot(xs, total_y, color=_GREEN, lw=2.5)
        ax2.fill_between(
            xs, fixed_y, total_y, alpha=0.12, color=_GREEN, label="Incremental from sweep"
        )
        ax2.axhline(fixed_y, color=_GREY, lw=1.2, ls="--", label=f"Fixed base ({fixed_y:,.0f})")
        ax2.set_xlabel(f"{campaign_id} spend", fontsize=9)
        ax2.set_ylabel("Total outcome", fontsize=9)
        ax2.set_title("Total Outcome (all campaigns)", fontsize=10)
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
        ax2.legend(fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

        return df_sweep

    def sensitivity_table(
        self,
        base_spends: dict[Any, float],
        pct_changes: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Show how the total outcome changes as each campaign's spend is
        varied by ±pct_changes% while all others are held fixed.

        Parameters
        ----------
        base_spends : dict {campaign_id: float}
        pct_changes : list of float
            Percentage changes to apply. Default [-20, -10, 0, +10, +20].

        Returns
        -------
        pd.DataFrame  index=campaign_id, columns=pct_changes
        """
        pct_changes = pct_changes or [-20.0, -10.0, 0.0, 10.0, 20.0]

        # Baseline total
        base_total = sum(
            self._response_fns[cid](float(base_spends.get(cid, 0.0))) for cid in self._response_fns
        )

        rows = {}
        for sweep_cid in self._response_fns:
            row = {}
            for pct in pct_changes:
                multiplier = 1.0 + pct / 100.0
                new_spends = {
                    cid: (
                        float(base_spends.get(cid, 0.0)) * multiplier
                        if cid == sweep_cid
                        else float(base_spends.get(cid, 0.0))
                    )
                    for cid in self._response_fns
                }
                new_total = sum(
                    self._response_fns[cid](new_spends[cid]) for cid in self._response_fns
                )
                row[f"{pct:+.0f}%"] = round(new_total - base_total, 2)
            rows[sweep_cid] = row

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "campaign_id"
        return df

    # ── internal helpers ──────────────────────────────────────────────────────

    def _evaluate_total(self, scenario: Scenario) -> float:
        """Compute total predicted outcome for a scenario."""
        total = 0.0
        for cid, fn in self._response_fns.items():
            spend = float(scenario.spends.get(cid, 0.0))
            total += fn(spend)
        return total

    def _log(self, msg: str) -> None:
        """
        Print msg to stdout when verbose=True.
        """
        if self.verbose:
            print(f"[ScenarioSimulator] {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def simulate(
    batch: CampaignBatchResult,
    budgets: list[float],
    campaign_weights: dict[Any, float] | None = None,
    baseline_budget: float | None = None,
    verbose: bool = False,
) -> SimulationResult:
    """
    One-liner scenario simulation: compare a set of total budgets, distributing
    spend across campaigns according to campaign_weights.

    Parameters
    ----------
    batch : CampaignBatchResult
    budgets : list of float
        Total budgets to compare.  Each becomes one scenario.
    campaign_weights : dict {campaign_id: weight}, optional
        How to split each total budget.  Weights are normalised automatically.
        Default: equal weight across all campaigns.
    baseline_budget : float, optional
        Budget to use as the comparison baseline.
        Default: the smallest budget in the list.
    verbose : bool

    Returns
    -------
    SimulationResult

    Examples
    --------
    >>> from adsat.simulation import simulate
    >>> result = simulate(batch, budgets=[2_000_000, 2_500_000, 3_000_000])
    >>> result.print_summary()
    """
    sim = ScenarioSimulator(batch, verbose=verbose)
    campaign_ids = list(sim._response_fns.keys())

    # Normalise weights
    if campaign_weights is None:
        weights = {cid: 1.0 for cid in campaign_ids}
    else:
        weights = {cid: float(campaign_weights.get(cid, 0.0)) for cid in campaign_ids}

    total_w = sum(weights.values())
    if total_w <= 0:
        raise ValueError("Sum of campaign_weights must be > 0.")
    weights = {cid: w / total_w for cid, w in weights.items()}

    baseline_name = None
    for budget in budgets:
        spends = {cid: budget * weights[cid] for cid in campaign_ids}
        name = f"Budget {budget:,.0f}"
        sim.add_scenario(name, spends)
        if baseline_budget is not None and abs(budget - baseline_budget) < 1:
            baseline_name = name

    if baseline_name is None:
        baseline_name = f"Budget {min(budgets):,.0f}"

    return sim.run(baseline=baseline_name)
