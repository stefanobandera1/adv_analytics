"""
adsat.evaluation
================
Compare fitted saturation models and select the best one.

Key class: ModelEvaluator
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EvaluationReport:
    """Outcome of model comparison."""

    best_model: str
    scoring_metric: str
    ranked_models: pd.DataFrame  # full comparison table, best first
    saturation_point: float | None  # from best model
    saturation_y: float | None
    saturation_threshold: float

    def __repr__(self) -> str:
        """
        Short string showing the best model name, saturation point, and scoring metric.
        """
        return (
            f"EvaluationReport(best_model={self.best_model!r}, "
            f"saturation_point={self.saturation_point}, "
            f"metric={self.scoring_metric!r})"
        )


# Score weights (higher = better model)
_METRIC_DIRECTION = {
    "r2": "max",
    "rmse": "min",
    "mae": "min",
    "mape": "min",
    "aic": "min",
    "bic": "min",
}


class ModelEvaluator:
    """
    Compare multiple fitted saturation models and select the best one.

    Parameters
    ----------
    primary_metric : str
        The metric used to rank models. One of: r2, rmse, mae, mape, aic, bic.
        Default is 'aic' (penalises complexity, good for small advertising datasets).
    require_convergence : bool
        If True, models that did not converge are excluded from evaluation.
    min_r2 : float
        Models with R² below this threshold are flagged as poor fits.

    Examples
    --------
    >>> from adsat import ModelEvaluator
    >>> evaluator = ModelEvaluator(primary_metric='aic')
    >>> report = evaluator.evaluate(results)   # results from SaturationModeler.fit()
    >>> print(report.best_model, report.saturation_point)
    """

    def __init__(
        self,
        primary_metric: str = "aic",
        require_convergence: bool = True,
        min_r2: float = 0.5,
    ):
        """
        Store primary metric for model ranking, convergence requirement, and minimum R².
        Raises ValueError for unrecognised metric names at construction time.
        """
        if primary_metric not in _METRIC_DIRECTION:
            raise ValueError(
                f"Unknown metric '{primary_metric}'. " f"Choose from: {list(_METRIC_DIRECTION)}"
            )
        self.primary_metric = primary_metric
        self.require_convergence = require_convergence
        self.min_r2 = min_r2
        self._report: EvaluationReport | None = None
        self._results = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, results: dict) -> EvaluationReport:
        """
        Rank models and return an EvaluationReport.

        Parameters
        ----------
        results : dict
            Output of SaturationModeler.fit() – maps model name -> ModelFitResult.

        Returns
        -------
        EvaluationReport
        """
        self._results = results
        rows = []

        for name, res in results.items():
            row = res.summary()
            row["converged"] = res.converged
            row["poor_fit"] = res.r2 < self.min_r2
            rows.append(row)

        df = pd.DataFrame(rows)

        # Filter
        if self.require_convergence:
            df_eligible = df[df["converged"]]
        else:
            df_eligible = df.copy()

        if df_eligible.empty:
            raise RuntimeError(
                "No models converged. Try different initial parameters, "
                "more data, or set require_convergence=False."
            )

        # Sort by primary metric
        ascending = _METRIC_DIRECTION[self.primary_metric] == "min"
        df_sorted = df_eligible.sort_values(self.primary_metric, ascending=ascending).reset_index(
            drop=True
        )

        # Add rank-based composite score (lower rank number = better)
        for metric, direction in _METRIC_DIRECTION.items():
            if metric in df_sorted.columns and df_sorted[metric].notna().all():
                asc = direction == "min"
                df_sorted[f"{metric}_rank"] = df_sorted[metric].rank(ascending=asc)

        rank_cols = [c for c in df_sorted.columns if c.endswith("_rank")]
        if rank_cols:
            df_sorted["composite_score"] = df_sorted[rank_cols].mean(axis=1)
        else:
            df_sorted["composite_score"] = np.nan

        best_name = df_sorted.iloc[0]["model"]
        best_result = results[best_name]

        self._report = EvaluationReport(
            best_model=best_name,
            scoring_metric=self.primary_metric,
            ranked_models=df_sorted,
            saturation_point=best_result.saturation_point,
            saturation_y=best_result.saturation_y,
            saturation_threshold=best_result.saturation_threshold,
        )

        return self._report

    def plot_model_comparison(
        self,
        results: dict | None = None,
        report: EvaluationReport | None = None,
        x_label: str = "Impressions",
        y_label: str = "Conversions / Revenue",
        save_path: str | None = None,
        figsize: tuple[int, int] = (16, 10),
    ) -> None:
        """
        Plot all model fits alongside metrics and the saturation point.

        Parameters
        ----------
        results : dict, optional  — uses internal cache if None.
        report  : EvaluationReport, optional — uses internal cache if None.
        """
        results = results or self._results
        report = report or self._report

        if results is None or report is None:
            raise RuntimeError("No results available. Run evaluate() first.")

        fig = plt.figure(figsize=figsize)
        fig.suptitle("Saturation Model Comparison", fontsize=15, fontweight="bold")
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        ax_main = fig.add_subplot(gs[0, :2])
        ax_best = fig.add_subplot(gs[0, 2])
        ax_r2 = fig.add_subplot(gs[1, 0])
        ax_aic = fig.add_subplot(gs[1, 1])
        ax_sat = fig.add_subplot(gs[1, 2])

        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        any_result = next(iter(results.values()))
        x_obs = any_result.x_values
        y_obs = any_result.y_true

        x_fine = np.linspace(x_obs.min(), x_obs.max() * 1.5, 500)

        # --- Main: all model fits ---
        ax_main.scatter(x_obs, y_obs, color="black", s=30, zorder=5, label="Observed", alpha=0.7)
        for (name, res), color in zip(results.items(), colors):
            lw = 3 if name == report.best_model else 1.5
            ls = "-" if name == report.best_model else "--"
            label = f"{name} (R²={res.r2:.2f})"
            if name == report.best_model:
                label += " ★ BEST"
            from adsat.modeling import MODEL_REGISTRY

            registry_name = "hill" if name == "hill_bayesian" else name
            func = MODEL_REGISTRY.get(registry_name, {}).get("func")
            if func:
                y_fine = func(x_fine, *list(res.params.values()))
                ax_main.plot(x_fine, y_fine, color=color, lw=lw, ls=ls, label=label)

        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)
        ax_main.set_title("All Model Fits")
        ax_main.legend(fontsize=8)

        # --- Best model zoom with saturation marker ---
        best_res = results[report.best_model]
        from adsat.modeling import MODEL_REGISTRY

        registry_name = "hill" if best_res.model_name == "hill_bayesian" else best_res.model_name
        func = MODEL_REGISTRY.get(registry_name, {}).get("func")
        if func:
            y_fine_best = func(x_fine, *list(best_res.params.values()))
            ax_best.scatter(x_obs, y_obs, color="black", s=25, alpha=0.7, zorder=5)
            ax_best.plot(x_fine, y_fine_best, color="tab:blue", lw=2.5)
            if report.saturation_point:
                ax_best.axvline(
                    report.saturation_point,
                    color="red",
                    lw=2,
                    ls="--",
                    label=f"Saturation ≈ {report.saturation_point:,.0f}",
                )
                ax_best.axhline(report.saturation_y, color="red", lw=1, ls=":", alpha=0.7)
                ax_best.legend(fontsize=8)
        ax_best.set_title(f"Best Model: {report.best_model}")
        ax_best.set_xlabel(x_label)
        ax_best.set_ylabel(y_label)

        # --- R² bar chart ---
        names = list(results.keys())
        r2s = [results[n].r2 for n in names]
        bar_colors = ["gold" if n == report.best_model else "steelblue" for n in names]
        ax_r2.barh(names, r2s, color=bar_colors)
        ax_r2.set_xlabel("R²")
        ax_r2.set_title("R² (higher = better)")
        ax_r2.axvline(0.5, color="red", lw=1, ls="--", alpha=0.5)

        # --- AIC bar chart ---
        aics = [results[n].aic for n in names]
        ax_aic.barh(names, aics, color=bar_colors)
        ax_aic.set_xlabel("AIC")
        ax_aic.set_title("AIC (lower = better)")

        # --- Saturation point summary table ---
        ax_sat.axis("off")
        sat_data = []
        for name in names:
            sp = results[name].saturation_point
            sy = results[name].saturation_y
            sat_data.append(
                [
                    name,
                    f"{sp:,.0f}" if sp else "N/A",
                    f"{sy:,.2f}" if sy else "N/A",
                ]
            )
        table = ax_sat.table(
            cellText=sat_data,
            colLabels=["Model", "Sat. X", "Sat. Y"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        ax_sat.set_title("Saturation Points")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    def print_report(self, report: EvaluationReport | None = None) -> None:
        """Print a human-readable summary of the evaluation."""
        report = report or self._report
        if report is None:
            raise RuntimeError("No report available. Run evaluate() first.")

        sep = "=" * 60
        print(sep)
        print("  SATURATION MODEL EVALUATION REPORT")
        print(sep)
        print(f"  Primary metric : {report.scoring_metric}")
        print(f"  Best model     : {report.best_model}")
        print(
            f"  Saturation pt  : {report.saturation_point:,.2f}"
            if report.saturation_point
            else "  Saturation pt  : N/A"
        )
        print(
            f"  Saturation y   : {report.saturation_y:,.2f}"
            if report.saturation_y
            else "  Saturation y   : N/A"
        )
        print(f"  Threshold      : {report.saturation_threshold * 100:.0f}% of asymptote")
        print(sep)
        print(
            report.ranked_models[
                ["model", "r2", "rmse", "aic", "saturation_point", "converged"]
            ].to_string(index=False)
        )
        print(sep)
