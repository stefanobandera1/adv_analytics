"""
adsat – End-to-End Test
========================
3 campaigns × 10 days of daily data.
Walks through every function in the package:
  1. Build sample dataset
  2. DistributionAnalyzer  – assess distribution shape
  3. DataTransformer       – apply recommended transformation
  4. SaturationModeler     – fit all saturation curves
  5. ModelEvaluator        – rank models, pick the best
  6. SaturationPipeline    – single-campaign automated run
  7. CampaignSaturationAnalyzer / predict_saturation_per_campaign
                           – per-campaign saturation prediction
"""

import sys
import os
import numpy as np
import pandas as pd

# ── make sure local package is on the path ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# ── imports from adsat ──────────────────────────────────────────────────────
from adsat.distribution  import DistributionAnalyzer
from adsat.transformation import DataTransformer
from adsat.modeling      import SaturationModeler
from adsat.evaluation    import ModelEvaluator
from adsat.pipeline      import SaturationPipeline
from adsat.campaign      import CampaignSaturationAnalyzer, predict_saturation_per_campaign

DIVIDER  = "=" * 70
SECTION  = "-" * 70


# ═══════════════════════════════════════════════════════════════════════════
# STEP 0 – Build the sample dataset
# ═══════════════════════════════════════════════════════════════════════════

def build_sample_dataset() -> pd.DataFrame:
    """
    Generate 10 days of daily campaign data for 3 campaigns.

    Campaign dynamics
    -----------------
    Campaign_Alpha  – large spend, S-shaped Hill curve, already nearing
                      saturation by day 10.
    Campaign_Beta   – mid-size, negative-exponential style, well below sat.
    Campaign_Gamma  – small niche brand, saturates very early (< day 4).

    Note: 10 days is deliberately sparse – it exercises the package's
    robustness with minimal data, exactly the real-world daily scenario.
    """
    np.random.seed(42)
    records = []

    campaigns = {
        # name          a_max   k_half   n_hill  noise  imp_start   imp_end
        "Campaign_Alpha": (50_000, 1_500_000, 2.0,  600,  400_000, 2_800_000),
        "Campaign_Beta":  (18_000, 2_800_000, 1.8,  200,  150_000,   900_000),
        "Campaign_Gamma": ( 6_000,   200_000, 1.5,  100,   80_000,   600_000),
    }

    dates = pd.date_range("2025-06-01", periods=10, freq="D")

    for campaign, (a, k, n_hill, noise, imp_start, imp_end) in campaigns.items():
        impressions = np.linspace(imp_start, imp_end, 10)
        impressions += np.random.normal(0, imp_start * 0.04, 10)
        impressions = np.maximum(impressions, 1000).astype(int)

        conv_true = a * (impressions ** n_hill) / (k ** n_hill + impressions ** n_hill)
        conversions = np.maximum(conv_true + np.random.normal(0, noise, 10), 0).round(0).astype(int)

        revenue   = conversions * np.random.uniform(30, 50, 10)
        ad_spend  = impressions * 0.005 + np.random.normal(0, 500, 10)
        ctr       = conversions / impressions * 100

        for i, date in enumerate(dates):
            records.append({
                "date":        date,
                "campaign_id": campaign,
                "impressions": impressions[i],
                "conversions": conversions[i],
                "revenue":     round(revenue[i], 2),
                "ad_spend":    round(max(ad_spend[i], 0), 2),
                "ctr":         round(ctr[i], 4),
            })

    return pd.DataFrame(records).sort_values(["campaign_id", "date"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)

def subsection(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():

    # ── 0. Build dataset ────────────────────────────────────────────────────
    section("STEP 0 – Sample Dataset (3 campaigns × 10 days)")

    df = build_sample_dataset()

    print(f"\n  Shape : {df.shape}  ({df['campaign_id'].nunique()} campaigns, "
          f"{df.groupby('campaign_id').size().iloc[0]} days each)")
    print(f"\n  Columns : {list(df.columns)}")
    print(f"\n  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n  Per-campaign stats:")
    print(
        df.groupby("campaign_id")[["impressions", "conversions", "revenue", "ad_spend"]]
        .agg(["min", "max", "mean"])
        .round(0)
        .to_string()
    )
    print(f"\n  Raw data (first 15 rows):")
    print(df.head(15).to_string(index=False))


    # ── 1. Distribution analysis – ALL campaigns pooled first ───────────────
    section("STEP 1 – Distribution Analysis  [DistributionAnalyzer]")

    print("""
  Function used : DistributionAnalyzer.analyze(df, columns=[...])
  What it does  : Fits 14+ statistical distributions to each column,
                  ranks them by AIC/BIC, tests normality (Shapiro-Wilk),
                  computes skewness & kurtosis, recommends a transform.
""")

    analyzer = DistributionAnalyzer(
        candidates=["norm", "lognorm", "gamma", "expon", "weibull_min",
                    "beta", "powerlaw", "logistic"],
        alpha=0.05,
        verbose=False,         # silence per-distribution chatter
    )

    dist_reports = analyzer.analyze(df, columns=["impressions", "conversions", "revenue"])

    subsection("1a – Summary table (best fit per column)")
    print(analyzer.summary_table().to_string(index=False))

    subsection("1b – Column detail")
    for col, report in dist_reports.items():
        best = report.best_fit
        print(f"\n  ── {col} ──")
        print(f"     Observations   : {report.n_observations}")
        print(f"     Mean / Std     : {report.descriptive_stats['mean']:,.1f} / "
              f"{report.descriptive_stats['std']:,.1f}")
        print(f"     Skewness       : {report.skewness:+.3f}")
        print(f"     Kurtosis       : {report.kurtosis:+.3f}")
        print(f"     Normal?        : {report.is_normal}  (Shapiro-Wilk p={report.shapiro_pvalue:.4f})")
        if best:
            print(f"     Best fit       : {best.distribution}  "
                  f"(KS stat={best.ks_statistic:.4f}, p={best.ks_pvalue:.4f}, "
                  f"AIC={best.aic:.2f})")
        print(f"     Recommended Δ : {report.recommended_transform}")

    subsection("1c – Top-3 distributions for 'impressions'")
    imp_summary = dist_reports["impressions"].summary()
    print(imp_summary.head(3).to_string(index=False))


    # ── 2. Data transformation ───────────────────────────────────────────────
    section("STEP 2 – Data Transformation  [DataTransformer]")

    print("""
  Function used : DataTransformer.fit_transform(df, columns, distribution_reports)
  What it does  : Reads recommended_transform from each DistributionReport and
                  applies the appropriate transform (log, sqrt, Box-Cox,
                  Yeo-Johnson, etc.).  Stores the fit for later inversion.
""")

    transformer = DataTransformer(
        strategy="auto",   # reads recommended_transform from dist_reports
        epsilon=1e-6,
    )

    df_transformed = transformer.fit_transform(
        df,
        columns=["impressions", "conversions", "revenue"],
        distribution_reports=dist_reports,
    )

    print("  Transforms applied:")
    print(transformer.get_transform_summary().to_string(index=False))

    print("\n  Transformed columns added to DataFrame:")
    new_cols = [c for c in df_transformed.columns if c.endswith("_t")]
    print(f"  {new_cols}")

    subsection("2a – Before / after stats for impressions")
    print(f"  Original  : mean={df['impressions'].mean():>12,.1f}  "
          f"std={df['impressions'].std():>12,.1f}  "
          f"skew={df['impressions'].skew():+.3f}")
    if "impressions_t" in df_transformed.columns:
        print(f"  Transformed: mean={df_transformed['impressions_t'].mean():>12.4f}  "
              f"std={df_transformed['impressions_t'].std():>12.4f}  "
              f"skew={df_transformed['impressions_t'].skew():+.3f}")

    subsection("2b – Inverse transform (round-trip check)")
    df_inv = transformer.inverse_transform(df_transformed, columns=["impressions", "conversions"])
    err_imp  = abs(df["impressions"] - df_inv["impressions_inv"]).max()
    err_conv = abs(df["conversions"] - df_inv["conversions_inv"]).max()
    print(f"  Max abs error impressions  : {err_imp:.6f}")
    print(f"  Max abs error conversions  : {err_conv:.6f}")
    print(f"  Round-trip accurate        : {err_imp < 1 and err_conv < 1}")


    # ── 3. Saturation model fitting (single campaign slice) ──────────────────
    section("STEP 3 – Saturation Model Fitting  [SaturationModeler]")

    print("""
  Function used : SaturationModeler.fit(df, x_col, y_col)
  What it does  : Fits Hill, Negative Exponential, Power, Michaelis-Menten
                  and Logistic saturation curves to the (transformed) data.
                  Returns a ModelFitResult per model with params, R², RMSE,
                  AIC/BIC and a raw saturation point estimate.

  We demonstrate on Campaign_Alpha (the richest signal).
""")

    alpha_t = df_transformed[df_transformed["campaign_id"] == "Campaign_Alpha"].copy()

    x_col_t = "impressions_t" if "impressions_t" in alpha_t.columns else "impressions"
    y_col_t = "conversions_t" if "conversions_t" in alpha_t.columns else "conversions"

    modeler = SaturationModeler(
        models=["hill", "negative_exponential", "power", "michaelis_menten", "logistic"],
        saturation_threshold=0.90,
        use_bayesian_hill=False,
        verbose=False,
    )

    model_results = modeler.fit(alpha_t, x_col=x_col_t, y_col=y_col_t)

    subsection("3a – Performance table (all models)")
    print(modeler.summary_table().to_string(index=False))

    subsection("3b – Best-model parameters")
    best_name_raw = modeler.summary_table().iloc[0]["model"]
    best_res = model_results[best_name_raw]
    print(f"\n  Model : {best_res.model_name}")
    print(f"  R²    : {best_res.r2:.4f}")
    print(f"  RMSE  : {best_res.rmse:.4f}")
    print(f"  AIC   : {best_res.aic:.2f}")
    print(f"  Converged : {best_res.converged}")
    print(f"\n  Parameters:")
    for pname, pval in best_res.params.items():
        print(f"    {pname:20s} = {pval:.6f}")
    print(f"\n  Saturation point (transformed scale) : {best_res.saturation_point}")

    subsection("3c – Point predictions at 5 impression levels (transformed)")
    x_test = np.linspace(alpha_t[x_col_t].min(), alpha_t[x_col_t].max(), 5)
    y_test = modeler.predict(best_name_raw, x_test)
    for x, y in zip(x_test, y_test):
        print(f"    x_t={x:.4f}  →  y_t={y:.4f}")


    # ── 4. Model evaluation ──────────────────────────────────────────────────
    section("STEP 4 – Model Evaluation & Selection  [ModelEvaluator]")

    print("""
  Function used : ModelEvaluator.evaluate(model_results)
  What it does  : Ranks all fitted models by the primary metric (AIC by
                  default), adds a composite rank score across all metrics,
                  selects the best model and exposes its saturation point.
""")

    evaluator = ModelEvaluator(
        primary_metric="aic",
        require_convergence=True,
        min_r2=0.0,          # lenient – 10 points is very sparse
    )

    eval_report = evaluator.evaluate(model_results)

    subsection("4a – Full ranked comparison")
    rank_cols = ["model", "r2", "rmse", "aic", "bic", "saturation_point", "converged"]
    available = [c for c in rank_cols if c in eval_report.ranked_models.columns]
    print(eval_report.ranked_models[available].to_string(index=False))

    subsection("4b – Winner")
    print(f"\n  Best model       : {eval_report.best_model}")
    print(f"  Sat. point (Δ)   : {eval_report.saturation_point}")
    print(f"  Sat. y     (Δ)   : {eval_report.saturation_y}")
    print(f"  Threshold        : {eval_report.saturation_threshold * 100:.0f}% of asymptote")

    evaluator.print_report(eval_report)


    # ── 5. Single-campaign automated pipeline ───────────────────────────────
    section("STEP 5 – Automated Single-Campaign Pipeline  [SaturationPipeline]")

    print("""
  Function used : SaturationPipeline(x_col, y_col, ...).run(df)
  What it does  : Chains steps 1-4 automatically for a single campaign
                  slice. Saturation point is back-transformed to the
                  original data scale.
""")

    for campaign_name in ["Campaign_Alpha", "Campaign_Beta", "Campaign_Gamma"]:

        subsection(f"Campaign: {campaign_name}")

        campaign_df = df[df["campaign_id"] == campaign_name][
            ["impressions", "conversions"]
        ].copy()

        pipeline = SaturationPipeline(
            x_col="impressions",
            y_col="conversions",
            models=["hill", "negative_exponential", "power"],
            transform_strategy="auto",
            saturation_threshold=0.90,
            primary_metric="aic",
            verbose=False,
        )

        result = pipeline.run(campaign_df)

        print(f"\n  Distribution of impressions  : {result.distribution_reports['impressions'].best_fit.distribution if result.distribution_reports['impressions'].best_fit else 'N/A'}")
        print(f"  Recommended transform        : {result.distribution_reports['impressions'].recommended_transform}")
        print(f"  Best saturation model        : {result.best_model}")

        best = result.model_results.get(result.best_model)
        if best:
            print(f"  R²                          : {best.r2:.4f}")
            print(f"  AIC                         : {best.aic:.2f}")
            print(f"  Model parameters            :")
            for k, v in best.params.items():
                print(f"      {k:20s} = {v:.4f}")

        if result.saturation_point:
            print(f"\n  ★ SATURATION POINT  = {result.saturation_point:>12,.0f} impressions")
            print(f"  ★ SATURATION Y      = {result.saturation_y:>12,.0f} conversions")
        else:
            print(f"\n  ★ Saturation point  : could not be determined with {len(campaign_df)} obs")

        # Predict at 3 scenarios
        print(f"\n  Scenario predictions (original scale):")
        x_scenarios = np.array([
            campaign_df["impressions"].min(),
            campaign_df["impressions"].median(),
            campaign_df["impressions"].max(),
        ]).astype(float)

        try:
            y_pred = pipeline.predict(x_scenarios, result)
            for x, y in zip(x_scenarios, y_pred):
                print(f"      {x:>12,.0f} impressions  →  {y:>8,.0f} conversions")
        except Exception as e:
            print(f"      (prediction failed: {e})")


    # ── 6. Per-campaign batch analysis ──────────────────────────────────────
    section("STEP 6 – Per-Campaign Saturation Prediction  [CampaignSaturationAnalyzer]")

    print("""
  Function used : CampaignSaturationAnalyzer(campaign_col, x_col, y_col).run(df)
  What it does  : Iterates over every campaign_id in the DataFrame,
                  runs the full SaturationPipeline on each slice,
                  back-transforms the saturation point, and classifies
                  each campaign as below / approaching / at / beyond saturation.

  Also shown    : predict_saturation_per_campaign() — one-liner convenience wrapper.
""")

    batch_analyzer = CampaignSaturationAnalyzer(
        campaign_col="campaign_id",
        x_col="impressions",
        y_col="conversions",
        date_col="date",
        min_observations=5,            # allow sparse 10-day campaigns
        models=["hill", "negative_exponential", "power"],
        transform_strategy="auto",
        saturation_threshold=0.90,
        primary_metric="aic",
        verbose=True,
    )

    batch_result = batch_analyzer.run(df)

    subsection("6a – Batch summary")
    batch_result.print_summary()

    subsection("6b – Individual campaign detail")
    for cid in batch_result.succeeded_campaigns():
        cr = batch_result.get(cid)
        cr.print_summary()

    if batch_result.failed_campaigns():
        subsection("6c – Failed campaigns")
        for cid in batch_result.failed_campaigns():
            cr = batch_result.get(cid)
            print(f"  {cid}: {cr.error}")

    subsection("6d – Saturation status breakdown")
    for status in ("below", "approaching", "at", "beyond", "unknown"):
        subset = batch_result.campaigns_by_status(status)
        if not subset.empty:
            ids = subset["campaign_id"].tolist()
            print(f"  {status.upper():12s} : {ids}")

    subsection("6e – One-liner convenience function  predict_saturation_per_campaign()")

    print("""
  Function used : predict_saturation_per_campaign(df, campaign_col, x_col, y_col)
  Returns       : pd.DataFrame – one row per campaign
""")

    summary_df = predict_saturation_per_campaign(
        df=df,
        campaign_col="campaign_id",
        x_col="impressions",
        y_col="conversions",
        date_col="date",
        min_observations=5,
        models=["hill", "negative_exponential", "power"],
        saturation_threshold=0.90,
        primary_metric="aic",
        verbose=False,
    )

    display_cols = [
        "campaign_id", "n_observations", "best_model",
        "r2", "saturation_point", "saturation_y",
        "current_x_median", "pct_of_saturation", "saturation_status",
    ]
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))


    # ── 7. Final narrative summary ───────────────────────────────────────────
    section("STEP 7 – Final Saturation Narrative")

    print()
    for _, row in summary_df.iterrows():
        cid   = row["campaign_id"]
        sp    = row.get("saturation_point")
        pct   = row.get("pct_of_saturation")
        stat  = row.get("saturation_status", "unknown")
        model = row.get("best_model", "N/A")
        r2    = row.get("r2")

        if not row.get("succeeded", False) or sp is None:
            print(f"  {cid:20s}  ⚠  Insufficient data or model did not converge.")
            continue

        status_icon = {"below": "🟢", "approaching": "🟡", "at": "🔴", "beyond": "🟣"}.get(stat, "⚪")

        print(f"  {cid:20s}  {status_icon}  {stat.upper():12s}  "
              f"sat_pt={sp:>12,.0f} impressions  "
              f"({pct:.1f}% reached)  "
              f"model={model}  R²={r2:.3f}" if r2 is not None else
              f"  {cid:20s}  {status_icon}  {stat.upper():12s}  "
              f"sat_pt={sp:>12,.0f} impressions  "
              f"({pct:.1f}% reached)  model={model}")

    print(f"\n{DIVIDER}")
    print("  END OF END-TO-END TEST – all steps completed successfully.")
    print(DIVIDER)


if __name__ == "__main__":
    main()
