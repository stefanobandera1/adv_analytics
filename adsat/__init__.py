"""
adsat - Advertising Saturation Analysis Toolkit
================================================
A framework for identifying impression saturation points in advertising campaigns.

Modules:
    exploratory    : Full EDA suite — histograms, Q-Q, ECDF, scatter, correlation,
                     time-series, outlier detection, distribution fitting
    distribution   : Fit and rank statistical distributions to campaign metrics
    transformation : Apply data transformations based on distribution shape
    modeling       : Fit saturation models (Hill, Exponential, Power, etc.)
    evaluation     : Compare models and select the best fit
    pipeline       : End-to-end saturation analysis pipeline (single dataset)
    campaign       : Per-campaign saturation prediction across a multi-campaign DataFrame
    budget         : Budget optimisation across campaigns (maximise total outcome)
    response_curves: Marginal returns, ROI curves, elasticity and efficiency zones
    diagnostics    : Model residual diagnostics (normality, autocorrelation, Cook's D)
    seasonality    : Seasonal decomposition and adjustment for time-series data
    report         : Automated HTML report generation
    simulation     : Scenario simulation and "what if" spend analysis
"""

from adsat.budget import BudgetAllocation, BudgetOptimizer, optimise_budget
from adsat.campaign import (
    CampaignBatchResult,
    CampaignResult,
    CampaignSaturationAnalyzer,
    predict_saturation_per_campaign,
)
from adsat.diagnostics import DiagnosticsReport, ModelDiagnostics, run_diagnostics
from adsat.distribution import DistributionAnalyzer
from adsat.evaluation import ModelEvaluator
from adsat.exploratory import CampaignExplorer, explore
from adsat.modeling import SaturationModeler
from adsat.pipeline import SaturationPipeline
from adsat.report import ReportBuilder, generate_report
from adsat.response_curves import (
    ResponseCurveAnalyzer,
    ResponseCurveResult,
    analyse_response_curves,
)
from adsat.seasonality import (
    SeasonalDecomposer,
    SeasonalDecomposition,
    adjust_for_seasonality,
)
from adsat.simulation import (
    Scenario,
    ScenarioSimulator,
    SimulationResult,
    simulate,
)
from adsat.transformation import DataTransformer

__version__ = "0.5.1"
__author__ = "adsat contributors"

from .benchmark import (
    BenchmarkResult,
    CampaignBenchmarker,
    benchmark_campaigns,
)

__all__ = [
    # Exploratory Data Analysis
    "CampaignExplorer",
    "explore",
    # Distribution analysis
    "DistributionAnalyzer",
    # Transformation
    "DataTransformer",
    # Modelling
    "SaturationModeler",
    "ModelEvaluator",
    # Single-dataset pipeline
    "SaturationPipeline",
    # Per-campaign analysis
    "CampaignSaturationAnalyzer",
    "CampaignResult",
    "CampaignBatchResult",
    "predict_saturation_per_campaign",
    # Budget optimisation
    "BudgetOptimizer",
    "BudgetAllocation",
    "optimise_budget",
    # Response curves
    "ResponseCurveAnalyzer",
    "ResponseCurveResult",
    "analyse_response_curves",
    # Diagnostics
    "ModelDiagnostics",
    "DiagnosticsReport",
    "run_diagnostics",
    # Seasonality
    "SeasonalDecomposer",
    "SeasonalDecomposition",
    "adjust_for_seasonality",
    # Report
    "ReportBuilder",
    "generate_report",
    # Simulation
    "ScenarioSimulator",
    "Scenario",
    "SimulationResult",
    "simulate",
    # Benchmarking
    "CampaignBenchmarker",
    "BenchmarkResult",
    "benchmark_campaigns",
    # Attribution modelling
    "JourneyConfig",
    "JourneyBuilder",
    "AttributionAnalyzer",
    "AttributionResult",
    "AttributionEvaluator",
    "AttributionBudgetAdvisor",
    "AttributionBudgetAllocation",
    "plot_attribution",
    "attribute_campaigns",
    "make_sample_events",
    "LastClickModel",
    "FirstClickModel",
    "LinearModel",
    "PositionBasedModel",
    "TimeDecayModel",
    "ShapleyModel",
    "MarkovChainModel",
    "DataDrivenModel",
    "EnsembleModel",
]

# Attribution modelling
from adsat.attribution import (
    AttributionAnalyzer,
    AttributionBudgetAdvisor,
    AttributionBudgetAllocation,
    AttributionEvaluator,
    AttributionResult,
    DataDrivenModel,
    EnsembleModel,
    FirstClickModel,
    JourneyBuilder,
    JourneyConfig,
    # Individual models (for direct use or subclassing)
    LastClickModel,
    LinearModel,
    MarkovChainModel,
    PositionBasedModel,
    ShapleyModel,
    TimeDecayModel,
    attribute_campaigns,
    make_sample_events,
    plot_attribution,
)
