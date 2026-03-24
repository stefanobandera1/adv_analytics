"""
adsat – Unit Tests
==================
Run with: pytest tests/test_adsat.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adsat import (
    CampaignBatchResult,
    CampaignSaturationAnalyzer,
    DataTransformer,
    DistributionAnalyzer,
    ModelEvaluator,
    SaturationModeler,
    SaturationPipeline,
    predict_saturation_per_campaign,
)
from adsat.modeling import hill_function, negative_exponential

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    np.random.seed(0)
    n = 80
    x = np.linspace(100_000, 3_000_000, n)
    y = 30_000 * (x**1.5) / (1_200_000**1.5 + x**1.5) + np.random.normal(0, 300, n)
    return pd.DataFrame({"impressions": x.astype(int), "conversions": np.maximum(y, 0).round(0)})


# ---------------------------------------------------------------------------
# Distribution tests
# ---------------------------------------------------------------------------


class TestDistributionAnalyzer:
    def test_analyze_returns_reports(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        reports = analyzer.analyze(sample_df, columns=["impressions", "conversions"])
        assert "impressions" in reports
        assert "conversions" in reports

    def test_report_has_best_fit(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        reports = analyzer.analyze(sample_df, columns=["impressions"])
        assert reports["impressions"].best_fit is not None

    def test_summary_table_shape(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        analyzer.analyze(sample_df, columns=["impressions", "conversions"])
        tbl = analyzer.summary_table()
        assert len(tbl) == 2
        assert "best_distribution" in tbl.columns

    def test_recommended_transform_type(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        reports = analyzer.analyze(sample_df, columns=["impressions"])
        assert isinstance(reports["impressions"].recommended_transform, str)

    def test_column_report_summary_dataframe(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        reports = analyzer.analyze(sample_df, columns=["impressions"])
        summary = reports["impressions"].summary()
        assert isinstance(summary, pd.DataFrame)
        assert "distribution" in summary.columns
        assert "aic" in summary.columns
        assert len(summary) > 0

    def test_fit_result_repr(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        reports = analyzer.analyze(sample_df, columns=["impressions"])
        best = reports["impressions"].best_fit
        assert best is not None
        r = repr(best)
        assert "DistributionFitResult" in r
        assert "ks_stat" in r

    def test_fit_result_is_acceptable(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        reports = analyzer.analyze(sample_df, columns=["impressions"])
        best = reports["impressions"].best_fit
        assert isinstance(best.is_acceptable, bool)

    def test_get_report(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        analyzer.analyze(sample_df, columns=["impressions"])
        report = analyzer.get_report("impressions")
        assert report.column == "impressions"

    def test_get_report_missing_column_raises(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        analyzer.analyze(sample_df, columns=["impressions"])
        with pytest.raises(KeyError):
            analyzer.get_report("nonexistent")

    def test_analyze_skips_short_column(self):
        df = pd.DataFrame({"short": [1.0, 2.0, 3.0]})
        analyzer = DistributionAnalyzer(verbose=False)
        with pytest.warns(UserWarning, match="fewer than 10"):
            reports = analyzer.analyze(df, columns=["short"])
        assert "short" not in reports

    def test_plot_distributions_runs(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        analyzer.analyze(sample_df, columns=["impressions"])
        # Should complete without raising (Agg backend — no display)
        analyzer.plot_distributions()


# ---------------------------------------------------------------------------
# Transformation tests
# ---------------------------------------------------------------------------


class TestDataTransformer:
    @pytest.mark.parametrize(
        "method", ["none", "log", "log1p", "sqrt", "cbrt", "yeo_johnson", "boxcox", "standard"]
    )
    def test_fit_transform_roundtrip(self, sample_df, method):
        transformer = DataTransformer(strategy=method)
        df_t = transformer.fit_transform(sample_df, columns=["conversions"])
        df_inv = transformer.inverse_transform(df_t, columns=["conversions"])
        original = sample_df["conversions"].values
        recovered = df_inv["conversions_inv"].values
        np.testing.assert_allclose(original, recovered, rtol=1e-3, atol=1e-2)

    def test_auto_strategy_requires_reports(self, sample_df):
        analyzer = DistributionAnalyzer(verbose=False)
        reports = analyzer.analyze(sample_df, columns=["impressions"])
        transformer = DataTransformer(strategy="auto")
        df_t = transformer.fit_transform(
            sample_df, columns=["impressions"], distribution_reports=reports
        )
        assert "impressions_t" in df_t.columns

    def test_dict_strategy(self, sample_df):
        transformer = DataTransformer(strategy={"impressions": "log", "conversions": "sqrt"})
        df_t = transformer.fit_transform(sample_df, columns=["impressions", "conversions"])
        assert "impressions_t" in df_t.columns
        assert "conversions_t" in df_t.columns


# ---------------------------------------------------------------------------
# Modeling tests
# ---------------------------------------------------------------------------


class TestSaturationModeler:
    def test_hill_function_shape(self):
        x = np.array([0, 1, 10, 100, 1000], dtype=float)
        y = hill_function(x, a=100, k=10, n=2)
        assert y[0] == pytest.approx(0, abs=1e-6)
        assert y[-1] < 100
        assert all(np.diff(y) >= 0)  # monotonically increasing

    def test_neg_exp_shape(self):
        x = np.linspace(0, 100, 50)
        y = negative_exponential(x, a=50, b=0.05)
        assert y[0] == pytest.approx(0, abs=1e-6)
        assert y[-1] < 50

    def test_fit_returns_results(self, sample_df):
        modeler = SaturationModeler(models=["hill", "negative_exponential"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        assert "hill" in results
        assert "negative_exponential" in results

    def test_hill_r2_acceptable(self, sample_df):
        modeler = SaturationModeler(models=["hill"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        assert results["hill"].r2 > 0.7

    def test_saturation_point_positive(self, sample_df):
        modeler = SaturationModeler(models=["hill"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        sp = results["hill"].saturation_point
        assert sp is None or sp > 0


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestModelEvaluator:
    def test_evaluate_picks_best(self, sample_df):
        modeler = SaturationModeler(models=["hill", "negative_exponential", "power"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        evaluator = ModelEvaluator(primary_metric="aic")
        report = evaluator.evaluate(results)
        assert report.best_model in results
        assert report.ranked_models is not None

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            ModelEvaluator(primary_metric="xyz")

    def test_report_has_saturation_fields(self, sample_df):
        modeler = SaturationModeler(models=["hill", "power"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        evaluator = ModelEvaluator()
        report = evaluator.evaluate(results)
        assert report.saturation_point is None or report.saturation_point > 0
        assert report.saturation_threshold > 0

    def test_report_repr(self, sample_df):
        modeler = SaturationModeler(models=["hill"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        evaluator = ModelEvaluator()
        report = evaluator.evaluate(results)
        r = repr(report)
        assert "EvaluationReport" in r
        assert "best_model" in r

    def test_require_convergence_false(self, sample_df):
        modeler = SaturationModeler(models=["hill", "power"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        evaluator = ModelEvaluator(require_convergence=False)
        report = evaluator.evaluate(results)
        assert report.best_model in results

    def test_ranked_models_dataframe(self, sample_df):
        modeler = SaturationModeler(models=["hill", "negative_exponential", "power"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        evaluator = ModelEvaluator()
        report = evaluator.evaluate(results)
        assert isinstance(report.ranked_models, pd.DataFrame)
        assert "model" in report.ranked_models.columns
        assert "composite_score" in report.ranked_models.columns

    def test_plot_model_comparison_runs(self, sample_df):
        modeler = SaturationModeler(models=["hill", "negative_exponential"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        evaluator = ModelEvaluator()
        evaluator.evaluate(results)
        # Should complete without raising (Agg backend — no display)
        evaluator.plot_model_comparison()

    def test_print_report_runs(self, sample_df, capsys):
        modeler = SaturationModeler(models=["hill", "power"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        evaluator = ModelEvaluator()
        report = evaluator.evaluate(results)
        evaluator.print_report(report)
        captured = capsys.readouterr()
        assert "SATURATION MODEL EVALUATION REPORT" in captured.out
        assert report.best_model in captured.out


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestSaturationPipeline:
    def test_pipeline_runs(self, sample_df):
        pipeline = SaturationPipeline(
            x_col="impressions",
            y_col="conversions",
            models=["hill", "negative_exponential"],
            verbose=False,
        )
        result = pipeline.run(sample_df)
        assert result.best_model is not None

    def test_pipeline_predict(self, sample_df):
        pipeline = SaturationPipeline(
            x_col="impressions",
            y_col="conversions",
            models=["hill"],
            verbose=False,
        )
        result = pipeline.run(sample_df)
        preds = pipeline.predict(np.array([500_000, 1_000_000, 2_000_000]), result)
        assert len(preds) == 3
        assert all(p >= 0 for p in preds)


# ---------------------------------------------------------------------------
# Multi-campaign fixtures
# ---------------------------------------------------------------------------


def _make_campaign(cid, n, a, k, n_hill, noise, start_imp, end_imp, seed=0):
    np.random.seed(seed)
    x = np.linspace(start_imp, end_imp, n)
    y = a * (x**n_hill) / (k**n_hill + x**n_hill) + np.random.normal(0, noise, n)
    return pd.DataFrame(
        {
            "campaign_id": cid,
            "week": pd.date_range("2023-01-01", periods=n, freq="W"),
            "impressions": x.astype(int),
            "conversions": np.maximum(y, 0).round(0).astype(int),
        }
    )


@pytest.fixture
def multi_campaign_df():
    dfa = _make_campaign("A", 60, 40000, 2_000_000, 1.8, 300, 300_000, 4_000_000, seed=1)
    dfb = _make_campaign("B", 52, 20000, 3_000_000, 2.0, 150, 100_000, 1_500_000, seed=2)
    dfc = _make_campaign("C", 3, 5000, 500_000, 1.5, 50, 50_000, 200_000, seed=3)  # too few
    return pd.concat([dfa, dfb, dfc], ignore_index=True)


# ---------------------------------------------------------------------------
# CampaignSaturationAnalyzer tests
# ---------------------------------------------------------------------------


class TestCampaignSaturationAnalyzer:

    def test_run_returns_batch_result(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        assert isinstance(batch, CampaignBatchResult)

    def test_counts_correct(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        assert batch.n_total == 3
        # Campaign C has 3 obs, should fail
        assert batch.n_failed >= 1

    def test_succeeded_campaigns_have_saturation_point(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        for cid in batch.succeeded_campaigns():
            cr = batch.get(cid)
            assert cr.saturation_point is not None or cr.best_model is not None

    def test_skipped_campaign_in_results(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        assert "C" in batch.campaign_results
        assert not batch.campaign_results["C"].succeeded

    def test_summary_table_has_all_campaigns(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        assert len(batch.summary_table) == 3

    def test_saturation_status_populated(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        for cid in batch.succeeded_campaigns():
            cr = batch.get(cid)
            if cr.saturation_point:
                assert cr.saturation_status in ("below", "approaching", "at", "beyond")

    def test_pct_of_saturation_positive(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        for cid in batch.succeeded_campaigns():
            cr = batch.get(cid)
            if cr.pct_of_saturation is not None:
                assert cr.pct_of_saturation > 0

    def test_run_single(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        cr = analyzer.run_single(multi_campaign_df, "A")
        assert cr.campaign_id == "A"

    def test_missing_column_raises(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="spend",  # non-existent
            y_col="conversions",
            verbose=False,
        )
        with pytest.raises(ValueError, match="missing required columns"):
            analyzer.run(multi_campaign_df)

    def test_campaigns_by_status_filter(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        for status in ("below", "approaching", "at", "beyond", "unknown"):
            result = batch.campaigns_by_status(status)
            assert isinstance(result, pd.DataFrame)

    def test_batch_plot_all_runs(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        # Should complete without raising (Agg backend — no display)
        batch.plot_all()

    def test_batch_plot_saturation_comparison_runs(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        batch.plot_saturation_comparison()

    def test_batch_plot_status_breakdown_runs(self, multi_campaign_df):
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            min_observations=10,
            verbose=False,
        )
        batch = analyzer.run(multi_campaign_df)
        batch.plot_status_breakdown()


# ---------------------------------------------------------------------------
# predict_saturation_per_campaign convenience function tests
# ---------------------------------------------------------------------------


class TestPredictSaturationPerCampaign:

    def test_returns_dataframe(self, multi_campaign_df):
        result = predict_saturation_per_campaign(
            multi_campaign_df,
            "campaign_id",
            "impressions",
            "conversions",
            min_observations=10,
            verbose=False,
        )
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_campaign(self, multi_campaign_df):
        result = predict_saturation_per_campaign(
            multi_campaign_df,
            "campaign_id",
            "impressions",
            "conversions",
            min_observations=10,
            verbose=False,
        )
        assert len(result) == multi_campaign_df["campaign_id"].nunique()

    def test_required_columns_present(self, multi_campaign_df):
        result = predict_saturation_per_campaign(
            multi_campaign_df,
            "campaign_id",
            "impressions",
            "conversions",
            min_observations=10,
            verbose=False,
        )
        expected_cols = {
            "campaign_id",
            "n_observations",
            "best_model",
            "saturation_point",
            "saturation_status",
            "succeeded",
        }
        assert expected_cols.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Shared fixtures for extension modules
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_result(multi_campaign_df):
    """CampaignBatchResult from campaigns A and B (C skipped — too few rows)."""
    analyzer = CampaignSaturationAnalyzer(
        campaign_col="campaign_id",
        x_col="impressions",
        y_col="conversions",
        min_observations=10,
        verbose=False,
    )
    return analyzer.run(multi_campaign_df)


# ---------------------------------------------------------------------------
# Exploratory tests
# ---------------------------------------------------------------------------


class TestCampaignExplorer:
    from adsat.exploratory import CampaignExplorer, explore

    def test_init_resolves_numeric_cols(self, sample_df):
        from adsat.exploratory import CampaignExplorer

        explorer = CampaignExplorer(sample_df)
        assert "impressions" in explorer.numeric_cols
        assert "conversions" in explorer.numeric_cols

    def test_init_explicit_numeric_cols(self, sample_df):
        from adsat.exploratory import CampaignExplorer

        explorer = CampaignExplorer(sample_df, numeric_cols=["impressions"])
        assert explorer.numeric_cols == ["impressions"]

    def test_init_excludes_campaign_and_date_cols(self, multi_campaign_df):
        from adsat.exploratory import CampaignExplorer

        explorer = CampaignExplorer(
            multi_campaign_df,
            campaign_col="campaign_id",
            date_col="week",
        )
        assert "campaign_id" not in explorer.numeric_cols
        assert "week" not in explorer.numeric_cols

    def test_plot_descriptive_summary_returns_dataframe(self, sample_df):
        import matplotlib.pyplot as plt

        from adsat.exploratory import CampaignExplorer

        explorer = CampaignExplorer(sample_df)
        result = explorer.plot_descriptive_summary()
        plt.close("all")
        assert isinstance(result, pd.DataFrame)
        assert "column" in result.columns
        assert len(result) == len(explorer.numeric_cols)

    def test_descriptive_summary_has_expected_stats(self, sample_df):
        import matplotlib.pyplot as plt

        from adsat.exploratory import CampaignExplorer

        explorer = CampaignExplorer(sample_df, numeric_cols=["impressions"])
        result = explorer.plot_descriptive_summary()
        plt.close("all")
        for col in ("mean", "std", "min", "max", "skewness"):
            assert col in result.columns

    def test_explore_convenience_function_runs(self, sample_df):
        import matplotlib.pyplot as plt

        from adsat.exploratory import explore

        # should not raise
        explore(
            sample_df,
            numeric_cols=["impressions", "conversions"],
            x_col="impressions",
            y_col="conversions",
        )
        plt.close("all")

    def test_single_numeric_col_does_not_crash(self, sample_df):
        import matplotlib.pyplot as plt

        from adsat.exploratory import CampaignExplorer

        explorer = CampaignExplorer(sample_df, numeric_cols=["impressions"])
        result = explorer.plot_descriptive_summary()
        plt.close("all")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Diagnostics tests
# ---------------------------------------------------------------------------


class TestModelDiagnostics:

    @pytest.fixture
    def hill_result(self, sample_df):
        modeler = SaturationModeler(models=["hill"], verbose=False)
        return modeler.fit(sample_df, x_col="impressions", y_col="conversions")["hill"]

    def test_run_returns_diagnostics_report(self, hill_result):
        from adsat.diagnostics import DiagnosticsReport, ModelDiagnostics

        diag = ModelDiagnostics(verbose=False)
        report = diag.run(hill_result)
        assert isinstance(report, DiagnosticsReport)

    def test_report_has_correct_n(self, sample_df, hill_result):
        from adsat.diagnostics import ModelDiagnostics

        diag = ModelDiagnostics(verbose=False)
        report = diag.run(hill_result)
        assert report.n == len(sample_df)

    def test_report_residuals_shape(self, sample_df, hill_result):
        from adsat.diagnostics import ModelDiagnostics

        diag = ModelDiagnostics(verbose=False)
        report = diag.run(hill_result)
        assert len(report.residuals) == len(sample_df)
        assert len(report.fitted_values) == len(sample_df)

    def test_report_pvalues_in_range(self, hill_result):
        from adsat.diagnostics import ModelDiagnostics

        diag = ModelDiagnostics(verbose=False)
        report = diag.run(hill_result)
        for pval in (
            report.shapiro_pvalue,
            report.ks_pvalue,
            report.jb_pvalue,
            report.levene_pvalue,
        ):
            assert 0.0 <= pval <= 1.0

    def test_durbin_watson_in_range(self, hill_result):
        from adsat.diagnostics import ModelDiagnostics

        diag = ModelDiagnostics(verbose=False)
        report = diag.run(hill_result)
        assert 0.0 <= report.durbin_watson <= 4.0

    def test_cook_distances_non_negative(self, hill_result):
        from adsat.diagnostics import ModelDiagnostics

        diag = ModelDiagnostics(verbose=False)
        report = diag.run(hill_result)
        assert (report.cook_distances >= 0).all()

    def test_run_all_returns_dict(self, sample_df):
        from adsat.diagnostics import ModelDiagnostics

        modeler = SaturationModeler(models=["hill", "negative_exponential"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        diag = ModelDiagnostics(verbose=False)
        reports = diag.run_all(results)
        assert set(reports.keys()) == {"hill", "negative_exponential"}

    def test_summary_table_shape(self, sample_df):
        from adsat.diagnostics import ModelDiagnostics

        modeler = SaturationModeler(models=["hill", "power"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        diag = ModelDiagnostics(verbose=False)
        reports = diag.run_all(results)
        tbl = diag.summary_table(reports)
        assert len(tbl) == 2
        assert "overall_ok" in tbl.columns

    def test_run_diagnostics_convenience_fn(self, hill_result):
        from adsat.diagnostics import DiagnosticsReport, run_diagnostics

        report = run_diagnostics(hill_result, verbose=False)
        assert isinstance(report, DiagnosticsReport)

    def test_invalid_alpha_raises(self):
        from adsat.diagnostics import ModelDiagnostics

        with pytest.raises(ValueError):
            ModelDiagnostics(alpha=1.5, verbose=False)

    def test_plot_single_runs(self, hill_result):
        from adsat.diagnostics import ModelDiagnostics

        diag = ModelDiagnostics(verbose=False)
        report = diag.run(hill_result)
        # Should complete without raising (Agg backend — no display)
        diag.plot(report)

    def test_plot_comparison_runs(self, sample_df):
        from adsat.diagnostics import ModelDiagnostics

        modeler = SaturationModeler(models=["hill", "power"], verbose=False)
        results = modeler.fit(sample_df, x_col="impressions", y_col="conversions")
        diag = ModelDiagnostics(verbose=False)
        reports = diag.run_all(results)
        diag.plot_comparison(reports)

    def test_print_summary_runs(self, hill_result, capsys):
        from adsat.diagnostics import ModelDiagnostics

        diag = ModelDiagnostics(verbose=False)
        report = diag.run(hill_result)
        report.print_summary()
        captured = capsys.readouterr()
        assert "DIAGNOSTICS REPORT" in captured.out
        assert "Shapiro-Wilk" in captured.out


# ---------------------------------------------------------------------------
# Budget optimiser tests
# ---------------------------------------------------------------------------


class TestBudgetOptimizer:

    def test_optimise_returns_budget_allocation(self, batch_result):
        from adsat.budget import BudgetAllocation, BudgetOptimizer

        opt = BudgetOptimizer(total_budget=5_000_000, n_restarts=2, verbose=False)
        result = opt.optimise(batch_result)
        assert isinstance(result, BudgetAllocation)

    def test_allocation_columns_present(self, batch_result):
        from adsat.budget import BudgetOptimizer

        opt = BudgetOptimizer(total_budget=5_000_000, n_restarts=2, verbose=False)
        result = opt.optimise(batch_result)
        expected = {
            "campaign_id",
            "current_spend",
            "optimal_spend",
            "spend_change_pct",
            "current_outcome",
            "optimal_outcome",
        }
        assert expected.issubset(set(result.allocations.columns))

    def test_optimal_spend_sums_to_budget(self, batch_result):
        from adsat.budget import BudgetOptimizer

        budget = 5_000_000
        opt = BudgetOptimizer(total_budget=budget, n_restarts=2, verbose=False)
        result = opt.optimise(batch_result)
        assert abs(result.allocations["optimal_spend"].sum() - budget) < 10.0

    def test_total_outcome_positive(self, batch_result):
        from adsat.budget import BudgetOptimizer

        opt = BudgetOptimizer(total_budget=5_000_000, n_restarts=2, verbose=False)
        result = opt.optimise(batch_result)
        assert result.total_optimal_outcome > 0

    def test_invalid_budget_raises(self):
        from adsat.budget import BudgetOptimizer

        with pytest.raises(ValueError):
            BudgetOptimizer(total_budget=-1, verbose=False)

    def test_too_few_campaigns_raises(self, sample_df):
        from adsat import CampaignSaturationAnalyzer
        from adsat.budget import BudgetOptimizer

        # Single-campaign batch has only one succeeded campaign
        single = sample_df.copy()
        single["campaign_id"] = "X"
        analyzer = CampaignSaturationAnalyzer(
            campaign_col="campaign_id",
            x_col="impressions",
            y_col="conversions",
            verbose=False,
        )
        single_batch = analyzer.run(single)
        opt = BudgetOptimizer(total_budget=1_000_000, verbose=False)
        with pytest.raises(ValueError):
            opt.optimise(single_batch)

    def test_optimise_budget_convenience_fn(self, batch_result):
        from adsat.budget import BudgetAllocation, optimise_budget

        result = optimise_budget(batch_result, total_budget=5_000_000, n_restarts=2, verbose=False)
        assert isinstance(result, BudgetAllocation)

    def test_marginal_returns_table_shape(self, batch_result):
        from adsat.budget import BudgetOptimizer

        opt = BudgetOptimizer(total_budget=5_000_000, n_restarts=2, verbose=False)
        tbl = opt.marginal_returns_table(batch_result)
        assert isinstance(tbl, pd.DataFrame)
        assert "campaign_id" in tbl.columns
        assert "marginal_return" in tbl.columns

    def test_budget_allocation_print_summary_runs(self, batch_result, capsys):
        from adsat.budget import BudgetOptimizer

        opt = BudgetOptimizer(total_budget=5_000_000, n_restarts=2, verbose=False)
        result = opt.optimise(batch_result)
        result.print_summary()
        captured = capsys.readouterr()
        assert "BUDGET OPTIMISATION" in captured.out

    def test_budget_allocation_plot_runs(self, batch_result):
        from adsat.budget import BudgetOptimizer

        opt = BudgetOptimizer(total_budget=5_000_000, n_restarts=2, verbose=False)
        result = opt.optimise(batch_result)
        # plot() accepts response_fns from the optimizer
        result.plot(response_fns=opt.response_fns)


# ---------------------------------------------------------------------------
# Response curves tests
# ---------------------------------------------------------------------------


class TestResponseCurveAnalyzer:

    def test_analyse_returns_dict(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        assert isinstance(results, dict)
        assert len(results) >= 1

    def test_result_arrays_same_length(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        for cid, res in results.items():
            n = len(res.x_values)
            assert len(res.y_values) == n
            assert len(res.marginal_returns) == n
            assert len(res.elasticities) == n
            assert len(res.roi_curve) == n

    def test_y_values_non_negative(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        for res in results.values():
            assert (res.y_values >= 0).all()

    def test_marginal_returns_non_negative(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        for res in results.values():
            assert (res.marginal_returns >= 0).all()

    def test_efficiency_zone_valid_labels(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        valid = {"high", "medium", "low"}
        for res in results.values():
            assert set(res.efficiency_zone).issubset(valid)

    def test_summary_table_one_row_per_campaign(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        tbl = analyzer.summary_table(results)
        assert isinstance(tbl, pd.DataFrame)
        assert len(tbl) == len(results)
        assert "campaign_id" in tbl.columns

    def test_analyse_response_curves_convenience_fn(self, batch_result):
        from adsat.response_curves import ResponseCurveResult, analyse_response_curves

        results = analyse_response_curves(batch_result, n_points=50, verbose=False)
        assert isinstance(results, dict)
        for res in results.values():
            assert isinstance(res, ResponseCurveResult)

    def test_plot_curves_runs(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        # Should complete without raising (Agg backend — no display)
        analyzer.plot_curves(results)

    def test_plot_marginal_returns_runs(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        analyzer.plot_marginal_returns(results)

    def test_plot_roi_curves_runs(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        analyzer.plot_roi_curves(results)

    def test_plot_efficiency_comparison_runs(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        analyzer.plot_efficiency_comparison(results)

    def test_summary_row_has_expected_keys(self, batch_result):
        from adsat.response_curves import ResponseCurveAnalyzer

        analyzer = ResponseCurveAnalyzer(n_points=50, verbose=False)
        results = analyzer.analyse(batch_result)
        for res in results.values():
            row = res.summary_row()
            assert "campaign_id" in row
            assert "asymptote" in row
            assert "pct_saturation_reached" in row


# ---------------------------------------------------------------------------
# Seasonality tests
# ---------------------------------------------------------------------------


class TestSeasonalDecomposer:

    @pytest.fixture
    def seasonal_series(self):
        np.random.seed(42)
        n = 104  # 2 years of weekly data
        t = np.arange(n)
        trend = 1000 + 5 * t
        seasonal = 200 * np.sin(2 * np.pi * t / 52)
        noise = np.random.normal(0, 30, n)
        return pd.Series(trend + seasonal + noise)

    def test_fit_returns_decomposition(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer, SeasonalDecomposition

        decomp = SeasonalDecomposer(period=52, model="additive", verbose=False)
        result = decomp.fit(seasonal_series)
        assert isinstance(result, SeasonalDecomposition)

    def test_decomposition_components_same_length(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer

        decomp = SeasonalDecomposer(period=52, model="additive", verbose=False)
        result = decomp.fit(seasonal_series)
        n = len(seasonal_series)
        assert len(result.original) == n
        assert len(result.trend) == n
        assert len(result.seasonal) == n
        assert len(result.residual) == n
        assert len(result.adjusted) == n

    def test_seasonal_factors_length_equals_period(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer

        period = 52
        decomp = SeasonalDecomposer(period=period, model="additive", verbose=False)
        result = decomp.fit(seasonal_series)
        assert len(result.seasonal_factors) == period

    def test_seasonal_strength_in_range(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer

        decomp = SeasonalDecomposer(period=52, model="additive", verbose=False)
        result = decomp.fit(seasonal_series)
        assert 0.0 <= result.strength_of_seasonality <= 1.0

    def test_additive_adjusted_removes_season(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer

        decomp = SeasonalDecomposer(period=52, model="additive", verbose=False)
        result = decomp.fit(seasonal_series)
        # Adjusted series should have less variance in the seasonal component
        var_orig = float(np.var(result.original.values))
        var_adj = float(np.var(result.adjusted.values))
        assert var_adj < var_orig  # seasonal removed → lower amplitude swings

    def test_multiplicative_model(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer, SeasonalDecomposition

        # Ensure positive series for multiplicative
        series = seasonal_series + abs(seasonal_series.min()) + 10
        decomp = SeasonalDecomposer(period=4, model="multiplicative", verbose=False)
        result = decomp.fit(series)
        assert isinstance(result, SeasonalDecomposition)
        assert result.model == "multiplicative"

    def test_inverse_adjust_roundtrip(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer

        decomp = SeasonalDecomposer(period=52, model="additive", verbose=False)
        result = decomp.fit(seasonal_series)
        recovered = decomp.inverse_adjust(result.adjusted.values, result)
        # Recovered should be close to original
        np.testing.assert_allclose(recovered, result.original.values, rtol=0.01, atol=50)

    def test_as_dataframe_has_all_components(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer

        decomp = SeasonalDecomposer(period=52, model="additive", verbose=False)
        result = decomp.fit(seasonal_series)
        df = result.as_dataframe()
        assert set(df.columns) == {"original", "trend", "seasonal", "residual", "adjusted"}

    def test_fit_transform_adds_adj_column(self, sample_df):
        from adsat.seasonality import SeasonalDecomposer

        decomp = SeasonalDecomposer(period=4, model="additive", verbose=False)
        df_out, decomps = decomp.fit_transform(sample_df, columns=["conversions"])
        assert "conversions_adj" in df_out.columns
        assert "conversions" in decomps

    def test_too_short_series_raises(self):
        from adsat.seasonality import SeasonalDecomposer

        decomp = SeasonalDecomposer(period=4, model="additive", verbose=False)
        with pytest.raises(ValueError, match="too short"):
            decomp.fit(pd.Series([1.0, 2.0, 3.0]))

    def test_invalid_model_raises(self):
        from adsat.seasonality import SeasonalDecomposer

        with pytest.raises(ValueError):
            SeasonalDecomposer(model="unknown", verbose=False)

    def test_adjust_for_seasonality_convenience_fn(self, seasonal_series):
        from adsat.seasonality import adjust_for_seasonality

        adjusted, decomp = adjust_for_seasonality(
            seasonal_series, period=52, model="additive", verbose=False
        )
        assert isinstance(adjusted, pd.Series)
        assert len(adjusted) == len(seasonal_series)

    def test_auto_period_detection(self, seasonal_series):
        from adsat.seasonality import SeasonalDecomposer, SeasonalDecomposition

        decomp = SeasonalDecomposer(period="auto", model="additive", verbose=False)
        result = decomp.fit(seasonal_series)
        assert isinstance(result, SeasonalDecomposition)
        assert result.period >= 2


# ---------------------------------------------------------------------------
# Simulation tests
# ---------------------------------------------------------------------------


class TestScenarioSimulator:

    def test_add_scenario_and_run(self, batch_result):
        from adsat.simulation import ScenarioSimulator, SimulationResult

        sim = ScenarioSimulator(batch_result, verbose=False)
        succeeded = batch_result.succeeded_campaigns()
        spends_a = {cid: 1_000_000.0 for cid in succeeded}
        spends_b = {cid: 2_000_000.0 for cid in succeeded}
        sim.add_scenario("Low spend", spends_a)
        sim.add_scenario("High spend", spends_b)
        result = sim.run()
        assert isinstance(result, SimulationResult)

    def test_summary_table_one_row_per_scenario(self, batch_result):
        from adsat.simulation import ScenarioSimulator

        sim = ScenarioSimulator(batch_result, verbose=False)
        succeeded = batch_result.succeeded_campaigns()
        for i in range(3):
            spends = {cid: float((i + 1) * 500_000) for cid in succeeded}
            sim.add_scenario(f"Scenario {i}", spends)
        result = sim.run()
        assert len(result.summary_table) == 3

    def test_best_scenario_has_highest_outcome(self, batch_result):
        from adsat.simulation import ScenarioSimulator

        sim = ScenarioSimulator(batch_result, verbose=False)
        succeeded = batch_result.succeeded_campaigns()
        sim.add_scenario("Low", {cid: 500_000.0 for cid in succeeded})
        sim.add_scenario("High", {cid: 5_000_000.0 for cid in succeeded})
        result = sim.run()
        tbl = result.summary_table
        best_row = tbl[tbl["scenario_name"] == result.best_scenario_name]
        assert best_row["total_outcome"].values[0] == tbl["total_outcome"].max()

    def test_empty_scenarios_raises(self, batch_result):
        from adsat.simulation import ScenarioSimulator

        sim = ScenarioSimulator(batch_result, verbose=False)
        with pytest.raises(ValueError, match="No scenarios"):
            sim.run()

    def test_empty_spends_raises(self, batch_result):
        from adsat.simulation import ScenarioSimulator

        sim = ScenarioSimulator(batch_result, verbose=False)
        with pytest.raises(ValueError):
            sim.add_scenario("Empty", {})

    def test_sensitivity_table_shape(self, batch_result):
        from adsat.simulation import ScenarioSimulator

        sim = ScenarioSimulator(batch_result, verbose=False)
        succeeded = batch_result.succeeded_campaigns()
        base_spends = {cid: 1_000_000.0 for cid in succeeded}
        tbl = sim.sensitivity_table(base_spends, pct_changes=[-10.0, 0.0, 10.0])
        assert isinstance(tbl, pd.DataFrame)
        assert tbl.shape[1] == 3  # three pct-change columns
        assert len(tbl) == len(succeeded)

    def test_add_budget_sweep_adds_scenarios(self, batch_result):
        from adsat.simulation import ScenarioSimulator

        sim = ScenarioSimulator(batch_result, verbose=False)
        cid = list(batch_result.succeeded_campaigns())[0]
        sim.add_budget_sweep(cid, [500_000, 1_000_000, 2_000_000])
        result = sim.run()
        assert len(result.scenarios) == 3

    def test_simulate_convenience_fn(self, batch_result):
        from adsat.simulation import SimulationResult, simulate

        result = simulate(
            batch_result,
            budgets=[3_000_000, 5_000_000, 7_000_000],
            verbose=False,
        )
        assert isinstance(result, SimulationResult)
        assert len(result.summary_table) == 3

    def test_simulate_with_custom_weights(self, batch_result):
        from adsat.simulation import simulate

        succeeded = batch_result.succeeded_campaigns()
        weights = {cid: 1.0 for cid in succeeded}
        result = simulate(
            batch_result, budgets=[4_000_000], campaign_weights=weights, verbose=False
        )
        assert len(result.summary_table) == 1


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestReportBuilder:

    def test_build_and_save_creates_html_file(self, batch_result, tmp_path):
        from adsat.report import ReportBuilder

        builder = ReportBuilder(title="Test Report")
        builder.add_campaign_batch(batch_result)
        out = str(tmp_path / "report.html")
        builder.save(out, open_browser=False)
        assert os.path.exists(out)
        content = open(out).read()
        assert "<html" in content.lower()
        assert "Test Report" in content

    def test_report_contains_campaign_section(self, batch_result, tmp_path):
        from adsat.report import ReportBuilder

        builder = ReportBuilder()
        builder.add_campaign_batch(batch_result)
        out = str(tmp_path / "report2.html")
        builder.save(out, open_browser=False)
        content = open(out).read()
        assert "Saturation" in content

    def test_add_budget_allocation_section(self, batch_result, tmp_path):
        from adsat.budget import optimise_budget
        from adsat.report import ReportBuilder

        budget_result = optimise_budget(
            batch_result, total_budget=5_000_000, n_restarts=2, verbose=False
        )
        builder = ReportBuilder()
        builder.add_campaign_batch(batch_result)
        builder.add_budget_allocation(budget_result)
        out = str(tmp_path / "report3.html")
        builder.save(out, open_browser=False)
        open(out).read()
        assert os.path.getsize(out) > 1000  # non-trivial HTML

    def test_add_custom_section(self, batch_result, tmp_path):
        from adsat.report import ReportBuilder

        builder = ReportBuilder()
        builder.add_custom_section("My Note", "<p>Custom content here</p>")
        out = str(tmp_path / "report4.html")
        builder.save(out, open_browser=False)
        content = open(out).read()
        assert "Custom content here" in content

    def test_generate_report_convenience_fn(self, batch_result, tmp_path):
        from adsat.report import generate_report

        out = str(tmp_path / "gen_report.html")
        generate_report(batch_result, output_path=out, open_browser=False)
        assert os.path.exists(out)

    def test_method_chaining(self, batch_result):
        from adsat.report import ReportBuilder

        builder = ReportBuilder(title="Chain Test")
        # add_campaign_batch should return self
        result = builder.add_campaign_batch(batch_result)
        assert result is builder


# ---------------------------------------------------------------------------
# Attribution tests
# ---------------------------------------------------------------------------


class TestAttribution:

    @pytest.fixture
    def events_df(self):
        from adsat.attribution import make_sample_events

        return make_sample_events(n_users=200, random_seed=42)

    @pytest.fixture
    def journeys_df(self, events_df):
        from adsat.attribution import JourneyBuilder

        builder = JourneyBuilder(lookback_days=30)
        return builder.build(events_df)

    def test_make_sample_events_returns_dataframe(self, events_df):
        assert isinstance(events_df, pd.DataFrame)
        required = {"user_id", "timestamp", "channel", "interaction_type", "converted", "revenue"}
        assert required.issubset(set(events_df.columns))

    def test_make_sample_events_has_positive_conversions(self, events_df):
        assert events_df["converted"].sum() > 0

    def test_journey_builder_returns_dataframe(self, journeys_df):
        assert isinstance(journeys_df, pd.DataFrame)

    def test_journeys_required_columns(self, journeys_df):
        required = {"path", "converted", "revenue", "n_touchpoints"}
        assert required.issubset(set(journeys_df.columns))

    def test_journeys_include_non_converting(self, journeys_df):
        # Markov/Shapley require non-converting paths
        assert (journeys_df["converted"] == 0).any()

    def test_attribution_analyzer_fit_returns_result(self, journeys_df):
        from adsat.attribution import AttributionAnalyzer, AttributionResult

        analyzer = AttributionAnalyzer(
            models=["last_click", "first_click", "linear"],
            verbose=False,
        )
        result = analyzer.fit(journeys_df)
        assert isinstance(result, AttributionResult)

    def test_attribution_credits_non_negative(self, journeys_df):
        from adsat.attribution import AttributionAnalyzer

        analyzer = AttributionAnalyzer(
            models=["last_click", "linear"],
            verbose=False,
        )
        result = analyzer.fit(journeys_df)
        assert (result.channel_credits["attributed_conversions"] >= 0).all()
        assert (result.channel_credits["attributed_revenue"] >= 0).all()

    def test_attribution_credits_sum_to_total_revenue(self, journeys_df):
        from adsat.attribution import AttributionAnalyzer

        analyzer = AttributionAnalyzer(models=["linear"], verbose=False)
        result = analyzer.fit(journeys_df)
        total_revenue = journeys_df[journeys_df["converted"] == 1]["revenue"].sum()
        credit_sum = result.get_credits("linear")["attributed_revenue"].sum()
        assert abs(credit_sum - total_revenue) / max(total_revenue, 1) < 0.05

    def test_all_channels_covered(self, events_df, journeys_df):
        from adsat.attribution import AttributionAnalyzer

        analyzer = AttributionAnalyzer(models=["last_click"], verbose=False)
        result = analyzer.fit(journeys_df)
        channels_in_events = set(events_df["channel"].unique())
        channels_in_credits = set(result.get_credits("last_click")["channel"].tolist())
        assert channels_in_events == channels_in_credits

    def test_shapley_model_runs(self, journeys_df):
        from adsat.attribution import AttributionAnalyzer

        analyzer = AttributionAnalyzer(
            models=["shapley"],
            shapley_n_iterations=500,
            verbose=False,
        )
        result = analyzer.fit(journeys_df)
        assert "shapley" in result.models_fitted

    def test_invalid_model_raises(self):
        from adsat.attribution import AttributionAnalyzer

        with pytest.raises(ValueError, match="Unknown models"):
            AttributionAnalyzer(models=["not_a_model"])

    def test_empty_journeys_raises(self):
        from adsat.attribution import AttributionAnalyzer

        analyzer = AttributionAnalyzer(models=["last_click"], verbose=False)
        with pytest.raises(ValueError):
            analyzer.fit(pd.DataFrame())


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


class TestCampaignBenchmarker:

    @pytest.fixture
    def benchmark_df(self):
        np.random.seed(42)
        n_hist = 40  # historical weeks per segment
        n_curr = 4  # current weeks per segment
        segments = ["UK_paid", "DE_paid"]

        rows = []
        for seg in segments:
            # Historical data
            base_cvr = np.random.uniform(0.03, 0.08)
            for i in range(n_hist):
                rows.append(
                    {
                        "week_start": pd.Timestamp("2023-01-02") + pd.Timedelta(weeks=i),
                        "segment": seg,
                        "impressions": int(np.random.uniform(10_000, 100_000)),
                        "conversions": int(np.random.uniform(300, 800)),
                        "cvr": round(float(np.random.normal(base_cvr, 0.005)), 5),
                        "spend": round(float(np.random.uniform(5_000, 50_000)), 2),
                    }
                )
            # Current period
            for i in range(n_curr):
                rows.append(
                    {
                        "week_start": pd.Timestamp("2023-10-30") + pd.Timedelta(weeks=i),
                        "segment": seg,
                        "impressions": int(np.random.uniform(10_000, 100_000)),
                        "conversions": int(np.random.uniform(300, 800)),
                        "cvr": round(float(np.random.normal(base_cvr, 0.005)), 5),
                        "spend": round(float(np.random.uniform(5_000, 50_000)), 2),
                    }
                )
        return pd.DataFrame(rows)

    def test_fit_returns_benchmark_result(self, benchmark_df):
        from adsat.benchmark import BenchmarkResult, CampaignBenchmarker

        bm = CampaignBenchmarker(
            metric_col="cvr",
            metric_type="proportion",
            date_col="week_start",
            volume_col="impressions",
            segment_cols=["segment"],
            current_period_start="2023-10-30",
            use_seasonality=False,
            verbose=False,
        )
        result = bm.fit(benchmark_df)
        assert isinstance(result, BenchmarkResult)

    def test_enriched_df_same_length_as_input(self, benchmark_df):
        from adsat.benchmark import CampaignBenchmarker

        bm = CampaignBenchmarker(
            metric_col="cvr",
            metric_type="proportion",
            date_col="week_start",
            segment_cols=["segment"],
            current_period_start="2023-10-30",
            use_seasonality=False,
            verbose=False,
        )
        result = bm.fit(benchmark_df)
        assert len(result.enriched_df) == len(benchmark_df)

    def test_m1_columns_present(self, benchmark_df):
        from adsat.benchmark import CampaignBenchmarker

        bm = CampaignBenchmarker(
            metric_col="cvr",
            metric_type="proportion",
            date_col="week_start",
            segment_cols=["segment"],
            current_period_start="2023-10-30",
            use_seasonality=False,
            verbose=False,
        )
        result = bm.fit(benchmark_df)
        for col in ("m1_lower90", "m1_upper90", "m1_class"):
            assert col in result.enriched_df.columns, f"Missing column: {col}"

    def test_m1_class_values_valid(self, benchmark_df):
        from adsat.benchmark import CampaignBenchmarker

        bm = CampaignBenchmarker(
            metric_col="cvr",
            metric_type="proportion",
            date_col="week_start",
            segment_cols=["segment"],
            current_period_start="2023-10-30",
            use_seasonality=False,
            verbose=False,
        )
        result = bm.fit(benchmark_df)
        classes = set(result.enriched_df["m1_class"].dropna().unique())
        # All non-null classes must be recognised strings
        known = {"Above", "Within", "Below", "Insufficient data", "N/A"}
        assert classes.issubset(known), f"Unexpected classes: {classes - known}"

    def test_summary_compact_has_segment_rows(self, benchmark_df):
        from adsat.benchmark import CampaignBenchmarker

        bm = CampaignBenchmarker(
            metric_col="cvr",
            metric_type="proportion",
            date_col="week_start",
            segment_cols=["segment"],
            current_period_start="2023-10-30",
            use_seasonality=False,
            verbose=False,
        )
        result = bm.fit(benchmark_df)
        assert len(result.summary_compact) >= 1

    def test_continuous_metric_type(self, benchmark_df):
        from adsat.benchmark import BenchmarkResult, CampaignBenchmarker

        bm = CampaignBenchmarker(
            metric_col="spend",
            metric_type="continuous",
            date_col="week_start",
            segment_cols=["segment"],
            current_period_start="2023-10-30",
            use_seasonality=False,
            verbose=False,
        )
        result = bm.fit(benchmark_df)
        assert isinstance(result, BenchmarkResult)

    def test_benchmark_campaigns_convenience_fn(self, benchmark_df):
        from adsat.benchmark import BenchmarkResult, benchmark_campaigns

        result = benchmark_campaigns(
            benchmark_df,
            metric_col="cvr",
            metric_type="proportion",
            date_col="week_start",
            segment_cols=["segment"],
            current_period_start="2023-10-30",
            use_seasonality=False,
            verbose=False,
        )
        assert isinstance(result, BenchmarkResult)
