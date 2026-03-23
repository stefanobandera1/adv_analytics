"""
adsat.pipeline
==============
End-to-end saturation analysis pipeline.

Chains: DistributionAnalyzer → DataTransformer → SaturationModeler → ModelEvaluator

Key class: SaturationPipeline
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from adsat.distribution import ColumnDistributionReport, DistributionAnalyzer
from adsat.evaluation import EvaluationReport, ModelEvaluator
from adsat.modeling import ModelFitResult, SaturationModeler
from adsat.transformation import DataTransformer


@dataclass
class PipelineResult:
    """Complete output of SaturationPipeline.run()."""

    x_col: str
    y_col: str
    distribution_reports: dict[str, ColumnDistributionReport]
    transform_summary: pd.DataFrame
    model_results: dict[str, ModelFitResult]
    evaluation_report: EvaluationReport
    best_model: str
    saturation_point: float | None
    saturation_y: float | None

    def print_summary(self) -> None:
        """Print a concise human-readable pipeline summary."""
        sep = "=" * 65
        print(sep)
        print("  ADSAT – SATURATION ANALYSIS PIPELINE RESULTS")
        print(sep)

        # Distribution findings
        print("\n[1] DISTRIBUTION ANALYSIS")
        for col, rep in self.distribution_reports.items():
            best = rep.best_fit.distribution if rep.best_fit else "unknown"
            print(
                f"   {col:20s}  best_dist={best:15s}  "
                f"skew={rep.skewness:+.2f}  transform={rep.recommended_transform}"
            )

        # Transform
        print("\n[2] DATA TRANSFORMATIONS")
        print(self.transform_summary.to_string(index=False))

        # Models
        print("\n[3] MODEL PERFORMANCE")
        print(
            self.evaluation_report.ranked_models[
                ["model", "r2", "rmse", "aic", "saturation_point", "converged"]
            ].to_string(index=False)
        )

        # Conclusion
        print("\n[4] SATURATION POINT PREDICTION")
        print(f"   Best model       : {self.best_model}")
        if self.saturation_point:
            print(f"   Saturation X ({self.x_col:10s}) : {self.saturation_point:,.2f}")
        if self.saturation_y:
            print(f"   Saturation Y ({self.y_col:10s}) : {self.saturation_y:,.2f}")
        print(sep)


class SaturationPipeline:
    """
    Automated end-to-end pipeline for advertising saturation analysis.

    Steps
    -----
    1. **Explore distributions** – fit distributions to x and y columns to
       understand data shape (DistributionAnalyzer).
    2. **Transform** – apply recommended transformation to stabilise variance
       and prepare data for modeling (DataTransformer).
    3. **Fit models** – try Hill, Negative Exponential, Power, Michaelis-Menten,
       Logistic saturation curves (SaturationModeler).
    4. **Evaluate** – rank models by AIC (or chosen metric), select the best,
       predict saturation point (ModelEvaluator).

    Parameters
    ----------
    x_col : str
        Column representing ad exposure (e.g. 'impressions', 'ad_spend').
    y_col : str
        Column representing outcome (e.g. 'conversions', 'revenue', 'roi').
    models : list of str, optional
        Saturation models to try. Defaults to all available.
    transform_strategy : str
        'auto' (recommended), or any specific method like 'log', 'none'.
    saturation_threshold : float
        Fraction of the asymptote defining saturation. Default 0.90.
    primary_metric : str
        Metric for model selection: 'aic', 'r2', 'rmse', etc.
    use_bayesian_hill : bool
        Use Bayesian regression for Hill model (requires PyMC).
    verbose : bool

    Examples
    --------
    >>> import pandas as pd
    >>> from adsat import SaturationPipeline
    >>>
    >>> df = pd.read_csv("campaign_weekly.csv")
    >>>
    >>> pipeline = SaturationPipeline(
    ...     x_col="impressions",
    ...     y_col="conversions",
    ...     models=["hill", "negative_exponential", "power"],
    ...     saturation_threshold=0.90,
    ...     primary_metric="aic",
    ... )
    >>>
    >>> result = pipeline.run(df)
    >>> result.print_summary()
    >>> pipeline.plot(result)
    """

    def __init__(
        self,
        x_col: str,
        y_col: str,
        models: list[str] | None = None,
        transform_strategy: str = "auto",
        saturation_threshold: float = 0.90,
        primary_metric: str = "aic",
        use_bayesian_hill: bool = False,
        verbose: bool = True,
    ):
        """
        Configure the pipeline with column names, model list, transformation strategy,
        saturation threshold, and primary selection metric.
        No computation occurs here; call run() to execute the full pipeline.
        """
        self.x_col = x_col
        self.y_col = y_col
        self.models = models
        self.transform_strategy = transform_strategy
        self.saturation_threshold = saturation_threshold
        self.primary_metric = primary_metric
        self.use_bayesian_hill = use_bayesian_hill
        self.verbose = verbose

        # Sub-components (initialised in run())
        self._dist_analyzer: DistributionAnalyzer | None = None
        self._transformer: DataTransformer | None = None
        self._modeler: SaturationModeler | None = None
        self._evaluator: ModelEvaluator | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> PipelineResult:
        """
        Execute the full pipeline on campaign data.

        Parameters
        ----------
        df : pd.DataFrame
            Weekly (or daily) campaign data. Must contain x_col and y_col.

        Returns
        -------
        PipelineResult
        """
        self._log("=" * 55)
        self._log("  ADSAT SATURATION PIPELINE STARTING")
        self._log("=" * 55)

        # --- Step 1: Distribution analysis ---
        self._log("\n[Step 1] Distribution Analysis…")
        self._dist_analyzer = DistributionAnalyzer(verbose=self.verbose)
        dist_reports = self._dist_analyzer.analyze(df, columns=[self.x_col, self.y_col])

        # --- Step 2: Transformation ---
        self._log("\n[Step 2] Data Transformation…")
        self._transformer = DataTransformer(strategy=self.transform_strategy)
        df_t = self._transformer.fit_transform(
            df,
            columns=[self.x_col, self.y_col],
            distribution_reports=dist_reports,
        )
        transform_summary = self._transformer.get_transform_summary()
        self._log(f"   Transforms applied:\n{transform_summary.to_string(index=False)}")

        # Determine which columns to use for modelling
        x_mod = f"{self.x_col}_t" if f"{self.x_col}_t" in df_t.columns else self.x_col
        y_mod = f"{self.y_col}_t" if f"{self.y_col}_t" in df_t.columns else self.y_col

        # --- Step 3: Model fitting ---
        self._log("\n[Step 3] Saturation Model Fitting…")
        self._modeler = SaturationModeler(
            models=self.models,
            saturation_threshold=self.saturation_threshold,
            use_bayesian_hill=self.use_bayesian_hill,
            verbose=self.verbose,
        )
        model_results = self._modeler.fit(df_t, x_col=x_mod, y_col=y_mod)

        # --- Step 4: Evaluation ---
        self._log("\n[Step 4] Model Evaluation…")
        self._evaluator = ModelEvaluator(
            primary_metric=self.primary_metric,
        )
        eval_report = self._evaluator.evaluate(model_results)

        # Back-transform saturation point to original scale
        sat_x_orig, sat_y_orig = self._inverse_saturation(
            eval_report.saturation_point,
            eval_report.saturation_y,
            df_t,
        )
        eval_report.saturation_point = sat_x_orig
        eval_report.saturation_y = sat_y_orig

        self._log("\n  ✓ Pipeline complete.")
        self._log(f"  Best model    : {eval_report.best_model}")
        self._log(f"  Saturation pt : {sat_x_orig}")

        return PipelineResult(
            x_col=self.x_col,
            y_col=self.y_col,
            distribution_reports=dist_reports,
            transform_summary=transform_summary,
            model_results=model_results,
            evaluation_report=eval_report,
            best_model=eval_report.best_model,
            saturation_point=sat_x_orig,
            saturation_y=sat_y_orig,
        )

    def plot(
        self,
        result: PipelineResult,
        save_path: str | None = None,
    ) -> None:
        """Produce distribution plots + model comparison plots."""
        # Distribution plots
        self._dist_analyzer.plot_distributions(
            result.distribution_reports,
            save_path=f"{save_path}_dist" if save_path else None,
        )
        # Model comparison plot
        self._evaluator.plot_model_comparison(
            result.model_results,
            result.evaluation_report,
            x_label=self.x_col,
            y_label=self.y_col,
            save_path=f"{save_path}_models" if save_path else None,
        )

    def predict(
        self,
        x_new: np.ndarray,
        result: PipelineResult,
    ) -> np.ndarray:
        """
        Generate outcome predictions for new x values using the best model.

        Handles transformation and inverse transformation automatically.

        Parameters
        ----------
        x_new : np.ndarray  — new impression / spend values (original scale).
        result : PipelineResult  — output of run().

        Returns
        -------
        np.ndarray of predicted y values (original scale).
        """
        # Transform x_new
        x_t = self._transformer.transform(
            pd.DataFrame({self.x_col: x_new}),
            columns=[self.x_col],
        )[f"{self.x_col}_t"].values

        # Model predict
        y_t = self._modeler.predict(result.best_model, x_t)

        # Inverse transform y
        df_pred = pd.DataFrame({f"{self.y_col}_t": y_t})
        df_inv = self._transformer.inverse_transform(df_pred, columns=[self.y_col])
        return df_inv[f"{self.y_col}_inv"].values

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inverse_saturation(
        self,
        sat_x: float | None,
        sat_y: float | None,
        df_t: pd.DataFrame,
    ) -> tuple[float | None, float | None]:
        """Back-transform the saturation point to the original data scale."""
        try:
            if sat_x is not None:
                x_df = pd.DataFrame({f"{self.x_col}_t": [sat_x]})
                sat_x_orig = float(
                    self._transformer.inverse_transform(x_df, columns=[self.x_col])[
                        f"{self.x_col}_inv"
                    ].iloc[0]
                )
            else:
                sat_x_orig = None

            if sat_y is not None:
                y_df = pd.DataFrame({f"{self.y_col}_t": [sat_y]})
                sat_y_orig = float(
                    self._transformer.inverse_transform(y_df, columns=[self.y_col])[
                        f"{self.y_col}_inv"
                    ].iloc[0]
                )
            else:
                sat_y_orig = None

            return sat_x_orig, sat_y_orig

        except Exception:
            return sat_x, sat_y

    def _log(self, msg: str) -> None:
        """
        Print msg to stdout when verbose=True.
        """
        if self.verbose:
            print(msg)
