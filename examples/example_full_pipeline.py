"""
adsat – Example: Full Saturation Analysis on Simulated Campaign Data
====================================================================

This script demonstrates the full adsat pipeline with synthetic data
that mimics a real weekly advertising campaign dataset.

Run:
    python examples/example_full_pipeline.py
"""

import numpy as np
import pandas as pd
import sys
import os

# Allow running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adsat import (
    DistributionAnalyzer,
    DataTransformer,
    SaturationModeler,
    ModelEvaluator,
    SaturationPipeline,
)

# ---------------------------------------------------------------------------
# 1. Generate synthetic campaign data with saturation behaviour
# ---------------------------------------------------------------------------

np.random.seed(42)
n_weeks = 104  # 2 years of weekly data

# Simulate impressions growing week-over-week with some noise
impressions = np.linspace(100_000, 5_000_000, n_weeks) + np.random.normal(0, 100_000, n_weeks)
impressions = np.maximum(impressions, 10_000)

# True underlying relationship: Hill function
TRUE_A = 50_000      # max conversions
TRUE_K = 1_800_000   # half-saturation at 1.8M impressions
TRUE_N = 1.8         # Hill exponent

conversions_true = TRUE_A * (impressions ** TRUE_N) / (TRUE_K ** TRUE_N + impressions ** TRUE_N)
conversions = conversions_true + np.random.normal(0, 800, n_weeks)
conversions = np.maximum(conversions, 0)

revenue = conversions * 35 + np.random.normal(0, 5000, n_weeks)  # ~£35 per conversion
ad_spend = impressions * 0.005 + np.random.normal(0, 2000, n_weeks)

campaign_data = pd.DataFrame({
    "week": pd.date_range("2023-01-01", periods=n_weeks, freq="W"),
    "impressions": impressions.astype(int),
    "conversions": conversions.round(0).astype(int),
    "revenue": revenue.round(2),
    "ad_spend": ad_spend.round(2),
})

print("=" * 60)
print("  SYNTHETIC CAMPAIGN DATA SAMPLE")
print("=" * 60)
print(campaign_data.head(8).to_string(index=False))
print(f"\n  Rows: {len(campaign_data)}")
print(f"  True saturation (90% of {TRUE_A:,}) ≈ {TRUE_K * (0.9/0.1) ** (1/TRUE_N):,.0f} impressions")

# ---------------------------------------------------------------------------
# 2. Quick approach: use the Pipeline for everything automatically
# ---------------------------------------------------------------------------

print("\n\n" + "=" * 60)
print("  APPROACH A: Automated Pipeline")
print("=" * 60)

pipeline = SaturationPipeline(
    x_col="impressions",
    y_col="conversions",
    models=["hill", "negative_exponential", "power", "michaelis_menten", "logistic"],
    transform_strategy="auto",
    saturation_threshold=0.90,
    primary_metric="aic",
    verbose=True,
)

result = pipeline.run(campaign_data)
result.print_summary()

# Predict for new impression levels
x_new = np.array([500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000])
y_pred = pipeline.predict(x_new, result)

print("\n  Predictions (best model):")
for x, y in zip(x_new, y_pred):
    print(f"   Impressions={x:>10,.0f} → Predicted conversions={y:>7,.1f}")

# ---------------------------------------------------------------------------
# 3. Manual step-by-step approach (more control)
# ---------------------------------------------------------------------------

print("\n\n" + "=" * 60)
print("  APPROACH B: Step-by-Step (Manual)")
print("=" * 60)

# Step 1: Explore distributions
analyzer = DistributionAnalyzer(verbose=False)
dist_reports = analyzer.analyze(campaign_data, columns=["impressions", "conversions"])

print("\n  Distribution Summary:")
print(analyzer.summary_table().to_string(index=False))

# Step 2: Transform data
transformer = DataTransformer(strategy="auto")
df_t = transformer.fit_transform(
    campaign_data,
    columns=["impressions", "conversions"],
    distribution_reports=dist_reports,
)
print("\n  Transforms applied:")
print(transformer.get_transform_summary().to_string(index=False))

# Step 3: Fit models
modeler = SaturationModeler(
    models=["hill", "negative_exponential", "power"],
    saturation_threshold=0.90,
    verbose=False,
)
model_results = modeler.fit(df_t, x_col="impressions_t", y_col="conversions_t")

print("\n  Model Performance Table:")
print(modeler.summary_table().to_string(index=False))

# Step 4: Evaluate & select best
evaluator = ModelEvaluator(primary_metric="aic")
report = evaluator.evaluate(model_results)
evaluator.print_report(report)

print(f"\n  Best model: {report.best_model}")
print(f"  Saturation point (transformed scale): {report.saturation_point}")
