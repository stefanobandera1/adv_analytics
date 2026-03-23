"""
adsat – Example: Per-Campaign Saturation Analysis
==================================================
Demonstrates CampaignSaturationAnalyzer and predict_saturation_per_campaign()
on a synthetic multi-campaign dataset.

Run:
    python examples/example_per_campaign.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adsat.campaign import CampaignSaturationAnalyzer, predict_saturation_per_campaign

# ---------------------------------------------------------------------------
# 1. Generate synthetic multi-campaign data
#    Each campaign has different saturation behaviour
# ---------------------------------------------------------------------------

np.random.seed(42)

def make_campaign(cid, n_weeks, a, k, n_hill, noise_scale, start_imp, end_imp):
    """Generate weekly data for one campaign following a Hill curve."""
    impressions = np.linspace(start_imp, end_imp, n_weeks)
    impressions += np.random.normal(0, start_imp * 0.05, n_weeks)
    impressions = np.maximum(impressions, 1000).astype(int)
    conv_true = a * (impressions ** n_hill) / (k ** n_hill + impressions ** n_hill)
    conversions = np.maximum(conv_true + np.random.normal(0, noise_scale, n_weeks), 0).round(0).astype(int)
    dates = pd.date_range("2023-01-01", periods=n_weeks, freq="W")
    return pd.DataFrame({
        "campaign_id": cid,
        "week": dates,
        "impressions": impressions,
        "conversions": conversions,
        "revenue": conversions * np.random.uniform(28, 42),
        "ad_spend": impressions * 0.004 + np.random.normal(0, 500, n_weeks),
    })

# Campaign A: Large, well-funded, already approaching saturation
df_a = make_campaign("Campaign_A", 80, a=45000, k=2_000_000, n_hill=1.8,
                     noise_scale=400, start_imp=500_000, end_imp=4_500_000)

# Campaign B: Mid-size, still well below saturation
df_b = make_campaign("Campaign_B", 60, a=20000, k=3_000_000, n_hill=2.0,
                     noise_scale=200, start_imp=200_000, end_imp=1_200_000)

# Campaign C: Small brand, saturates early
df_c = make_campaign("Campaign_C", 52, a=8000,  k=400_000,  n_hill=1.5,
                     noise_scale=100, start_imp=50_000,  end_imp=800_000)

# Campaign D: High spend, far beyond saturation
df_d = make_campaign("Campaign_D", 70, a=30000, k=600_000,  n_hill=2.2,
                     noise_scale=300, start_imp=800_000, end_imp=3_000_000)

# Campaign E: Too few observations – should be gracefully skipped
df_e = make_campaign("Campaign_E", 5, a=15000, k=1_500_000, n_hill=1.6,
                     noise_scale=150, start_imp=100_000, end_imp=300_000)

all_campaigns = pd.concat([df_a, df_b, df_c, df_d, df_e], ignore_index=True)

print(f"Total rows: {len(all_campaigns)}")
print(f"Campaigns : {all_campaigns['campaign_id'].unique().tolist()}")
print(all_campaigns.head(5).to_string(index=False))


# ---------------------------------------------------------------------------
# 2. APPROACH A – convenience one-liner
#    Returns a tidy summary DataFrame, one row per campaign
# ---------------------------------------------------------------------------

print("\n\n" + "=" * 60)
print("  APPROACH A: predict_saturation_per_campaign() one-liner")
print("=" * 60)

summary = predict_saturation_per_campaign(
    df=all_campaigns,
    campaign_col="campaign_id",
    x_col="impressions",
    y_col="conversions",
    date_col="week",
    min_observations=10,
    saturation_threshold=0.90,
    primary_metric="aic",
    verbose=True,
)

print("\nSummary table:")
print(summary[[
    "campaign_id", "n_observations", "best_model",
    "r2", "saturation_point", "current_x_median",
    "pct_of_saturation", "saturation_status"
]].to_string(index=False))


# ---------------------------------------------------------------------------
# 3. APPROACH B – full CampaignSaturationAnalyzer for plots & deep-dive
# ---------------------------------------------------------------------------

print("\n\n" + "=" * 60)
print("  APPROACH B: CampaignSaturationAnalyzer (full control)")
print("=" * 60)

analyzer = CampaignSaturationAnalyzer(
    campaign_col="campaign_id",
    x_col="impressions",
    y_col="conversions",
    date_col="week",
    min_observations=10,
    models=["hill", "negative_exponential", "power"],
    saturation_threshold=0.90,
    primary_metric="aic",
    verbose=True,
)

batch = analyzer.run(all_campaigns)

# High-level print
batch.print_summary()

# Per-campaign detail
for cid in batch.succeeded_campaigns():
    batch.get(cid).print_summary()

# Campaigns that still have budget headroom
print("\nCampaigns BELOW saturation (budget headroom available):")
print(batch.campaigns_by_status("below")[["campaign_id", "saturation_point", "pct_of_saturation"]])

print("\nCampaigns BEYOND saturation (overspending):")
print(batch.campaigns_by_status("beyond")[["campaign_id", "saturation_point", "pct_of_saturation"]])

# Plots (comment out if running in a non-display environment)
# batch.plot_all()
# batch.plot_saturation_comparison()
# batch.plot_status_breakdown()

# ---------------------------------------------------------------------------
# 4. Predict outcomes at new impression levels for a specific campaign
# ---------------------------------------------------------------------------

print("\n\n" + "=" * 60)
print("  Predicting conversions at new impression levels for Campaign_B")
print("=" * 60)

# Re-run single campaign for prediction
cr = analyzer.run_single(all_campaigns, "Campaign_B")
cr.print_summary()

# Manually use pipeline to predict new values
pipeline = cr.pipeline_result
if pipeline:
    best_res = pipeline.model_results[cr.best_model]
    from adsat.modeling import MODEL_REGISTRY
    func = MODEL_REGISTRY[best_res.model_name]["func"]
    x_scenarios = np.array([500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000])
    y_pred = func(x_scenarios, *list(best_res.params.values()))
    print(f"\n  {'Impressions':>15}  {'Pred. Conversions':>18}")
    print(f"  {'-'*35}")
    for x, y in zip(x_scenarios, y_pred):
        marker = " ← SAT" if cr.saturation_point and abs(x - cr.saturation_point) / cr.saturation_point < 0.15 else ""
        print(f"  {x:>15,.0f}  {y:>18,.1f}{marker}")
