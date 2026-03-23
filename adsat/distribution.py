"""
adsat.distribution
==================
Explore and identify the best-fitting statistical distribution for campaign metrics.

Key class: DistributionAnalyzer
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# ---------------------------------------------------------------------------
# Candidate distributions considered during fitting
# ---------------------------------------------------------------------------
CANDIDATE_DISTRIBUTIONS: list[str] = [
    "norm",
    "lognorm",
    "expon",
    "gamma",
    "beta",
    "weibull_min",
    "weibull_max",
    "pareto",
    "powerlaw",
    "uniform",
    "chi2",
    "rayleigh",
    "logistic",
    "cauchy",
]


@dataclass
class DistributionFitResult:
    """Result of fitting a single distribution to a data column."""

    distribution: str
    params: tuple
    ks_statistic: float
    ks_pvalue: float
    aic: float
    bic: float
    ad_statistic: float | None = None
    ad_critical_values: np.ndarray | None = None

    @property
    def is_acceptable(self) -> bool:
        """KS p-value > 0.05 indicates we cannot reject the distribution hypothesis."""
        return self.ks_pvalue > 0.05

    def __repr__(self) -> str:
        """
        Short string representation showing distribution name and AIC score.
        """
        return (
            f"DistributionFitResult(distribution={self.distribution!r}, "
            f"ks_stat={self.ks_statistic:.4f}, ks_pvalue={self.ks_pvalue:.4f}, "
            f"aic={self.aic:.2f}, bic={self.bic:.2f})"
        )


@dataclass
class ColumnDistributionReport:
    """Full distribution report for a single numeric column."""

    column: str
    n_observations: int
    descriptive_stats: dict
    skewness: float
    kurtosis: float
    is_normal: bool  # Shapiro-Wilk result
    shapiro_pvalue: float
    best_fit: DistributionFitResult | None
    all_fits: list[DistributionFitResult] = field(default_factory=list)
    recommended_transform: str = "none"

    def summary(self) -> pd.DataFrame:
        """Return a tidy summary DataFrame of all fitted distributions."""
        rows = []
        for f in self.all_fits:
            rows.append(
                {
                    "distribution": f.distribution,
                    "ks_statistic": round(f.ks_statistic, 4),
                    "ks_pvalue": round(f.ks_pvalue, 4),
                    "aic": round(f.aic, 2),
                    "bic": round(f.bic, 2),
                    "acceptable": f.is_acceptable,
                }
            )
        return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


class DistributionAnalyzer:
    """
    Analyze the distributional properties of advertising campaign metrics.

    Parameters
    ----------
    candidates : list of str, optional
        scipy.stats distribution names to test. Defaults to a broad set
        covering most advertising data shapes.
    alpha : float
        Significance level for goodness-of-fit tests. Default 0.05.
    verbose : bool
        Print progress information during fitting.

    Examples
    --------
    >>> import pandas as pd
    >>> from adsat import DistributionAnalyzer
    >>> df = pd.read_csv("campaign_data.csv")
    >>> analyzer = DistributionAnalyzer()
    >>> reports = analyzer.analyze(df, columns=["impressions", "conversions", "revenue"])
    >>> analyzer.plot_distributions(reports)
    """

    def __init__(
        self,
        candidates: list[str] | None = None,
        alpha: float = 0.05,
        verbose: bool = True,
    ):
        """
        Store candidate distribution names, significance level, and verbosity flag.
        No fitting occurs here; call analyze() to fit distributions to data columns.
        """
        self.candidates = candidates or CANDIDATE_DISTRIBUTIONS
        self.alpha = alpha
        self.verbose = verbose
        self._reports: dict[str, ColumnDistributionReport] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, ColumnDistributionReport]:
        """
        Fit multiple distributions to each numeric column and return reports.

        Parameters
        ----------
        df : pd.DataFrame
            Campaign data (weekly aggregated).
        columns : list of str, optional
            Subset of columns to analyze. Defaults to all numeric columns.

        Returns
        -------
        dict mapping column name -> ColumnDistributionReport
        """
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        self._reports = {}

        for col in cols:
            series = df[col].dropna()
            if len(series) < 10:
                warnings.warn(
                    f"Column '{col}' has fewer than 10 observations – skipping.",
                    UserWarning,
                )
                continue

            if self.verbose:
                print(f"[DistributionAnalyzer] Analyzing '{col}' ({len(series)} obs)…")

            report = self._analyze_column(col, series)
            self._reports[col] = report

        return self._reports

    def get_report(self, column: str) -> ColumnDistributionReport:
        """Retrieve the report for a specific column (after calling analyze)."""
        if column not in self._reports:
            raise KeyError(f"No report found for column '{column}'. Run analyze() first.")
        return self._reports[column]

    def summary_table(self) -> pd.DataFrame:
        """Return a high-level summary of best fits across all analyzed columns."""
        rows = []
        for col, report in self._reports.items():
            best = report.best_fit
            rows.append(
                {
                    "column": col,
                    "n": report.n_observations,
                    "skewness": round(report.skewness, 3),
                    "kurtosis": round(report.kurtosis, 3),
                    "is_normal": report.is_normal,
                    "best_distribution": best.distribution if best else "none",
                    "ks_pvalue": round(best.ks_pvalue, 4) if best else None,
                    "aic": round(best.aic, 2) if best else None,
                    "recommended_transform": report.recommended_transform,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_distributions(
        self,
        reports: dict[str, ColumnDistributionReport] | None = None,
        top_n: int = 4,
        figsize_per_col: tuple[int, int] = (14, 10),
        save_path: str | None = None,
    ) -> None:
        """
        4-panel distribution plot for each analysed column.

        Panels
        ------
        Top-left  : Histogram + KDE + top-N fitted PDFs
        Top-right : ECDF + top-N fitted CDFs
        Bot-left  : Q-Q plot vs best-fitting distribution (shaded deviation)
        Bot-right : AIC / BIC bar chart for all candidate distributions

        Parameters
        ----------
        reports : dict, optional
            Output of analyze(). Uses internal cache if None.
        top_n : int
            Number of best distributions to overlay. Default 4.
        figsize_per_col : tuple
            Figure size (width, height) per column. Default (14, 10).
        save_path : str, optional
            If provided, saves each figure as ``{save_path}_{col}.png``.
        """
        reports = reports or self._reports
        if not reports:
            raise RuntimeError("No reports available. Run analyze() first.")

        for col, report in reports.items():
            raw = report.descriptive_stats.get("_raw_data", np.array([]))
            if len(raw) == 0:
                warnings.warn(f"No raw data stored for '{col}' – skipping plot.")
                continue

            top_fits = report.all_fits[:top_n]
            colors = plt.cm.tab10(np.linspace(0, 0.9, max(top_n, 1)))

            fig = plt.figure(figsize=figsize_per_col)
            best_name = report.best_fit.distribution if report.best_fit else "N/A"
            best_aic = f"{report.best_fit.aic:.1f}" if report.best_fit else "–"
            best_ks_p = f"{report.best_fit.ks_pvalue:.3f}" if report.best_fit else "–"
            fig.suptitle(
                f"Distribution Analysis: {col}  |  "
                f"best fit: {best_name}  AIC={best_aic}  KS-p={best_ks_p}  "
                f"skew={report.skewness:+.2f}  normal={report.is_normal}",
                fontsize=11,
                fontweight="bold",
            )
            gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
            ax_hist = fig.add_subplot(gs[0, 0])
            ax_ecdf = fig.add_subplot(gs[0, 1])
            ax_qq = fig.add_subplot(gs[1, 0])
            ax_aic = fig.add_subplot(gs[1, 1])

            x_range = np.linspace(raw.min(), raw.max(), 400)

            # ── Panel 1: Histogram + KDE + PDFs ──────────────────────────────
            ax_hist.hist(
                raw,
                bins="auto",
                density=True,
                alpha=0.35,
                color="#6C757D",
                edgecolor="white",
                label="Data",
            )
            try:
                from scipy.stats import gaussian_kde

                kde = gaussian_kde(raw)
                ax_hist.plot(x_range, kde(x_range), color="black", lw=1.5, ls=":", label="KDE")
            except Exception:
                pass

            for fit, color in zip(top_fits, colors):
                try:
                    dist_obj = getattr(stats, fit.distribution)
                    shifted_range = self._shift_range(x_range, raw, fit.distribution)
                    pdf = dist_obj.pdf(shifted_range, *fit.params)
                    if fit.distribution == "beta":
                        pdf = pdf / (raw.max() - raw.min() + 2e-6)
                    lw = 3.0 if fit == report.best_fit else 1.5
                    label = f"{fit.distribution}{'★' if fit == report.best_fit else ''} AIC={fit.aic:.0f}"
                    ax_hist.plot(x_range, pdf, lw=lw, color=color, label=label)
                except Exception:
                    pass

            ax_hist.set_title("Histogram + Fitted PDFs", fontsize=9)
            ax_hist.set_xlabel(col, fontsize=8)
            ax_hist.set_ylabel("Density", fontsize=8)
            ax_hist.legend(fontsize=7)

            # Mean / median lines
            ax_hist.axvline(raw.mean(), color="#E84855", lw=1.5, ls="--", alpha=0.8)
            ax_hist.axvline(np.median(raw), color="#2E86AB", lw=1.5, ls=":", alpha=0.8)

            # ── Panel 2: ECDF + CDFs ─────────────────────────────────────────
            sorted_raw = np.sort(raw)
            ecdf_y = np.arange(1, len(sorted_raw) + 1) / len(sorted_raw)
            ax_ecdf.step(
                sorted_raw, ecdf_y, where="post", color="black", lw=2, label="ECDF", zorder=5
            )

            for fit, color in zip(top_fits, colors):
                try:
                    dist_obj = getattr(stats, fit.distribution)
                    shifted_range = self._shift_range(x_range, raw, fit.distribution)
                    cdf = dist_obj.cdf(shifted_range, *fit.params)
                    lw = 2.5 if fit == report.best_fit else 1.2
                    ax_ecdf.plot(
                        x_range,
                        cdf,
                        lw=lw,
                        color=color,
                        label=fit.distribution + ("★" if fit == report.best_fit else ""),
                    )
                except Exception:
                    pass

            ax_ecdf.set_title("ECDF + Fitted CDFs", fontsize=9)
            ax_ecdf.set_xlabel(col, fontsize=8)
            ax_ecdf.set_ylabel("Cumulative Probability", fontsize=8)
            ax_ecdf.set_ylim(0, 1.05)
            ax_ecdf.legend(fontsize=7)

            # ── Panel 3: Q-Q plot ─────────────────────────────────────────────
            if report.best_fit:
                try:
                    dist_obj = getattr(stats, report.best_fit.distribution)
                    shifted = self._get_shifted(raw, report.best_fit.distribution)
                    sparams = (
                        report.best_fit.params[:-2]
                        if len(report.best_fit.params) > 2
                        else report.best_fit.params
                    )
                    (osm, osr), (slope, intercept, r) = stats.probplot(
                        shifted, dist=dist_obj, sparams=sparams, fit=True
                    )
                    ax_qq.scatter(osm, osr, s=18, alpha=0.6, color="#2E86AB", zorder=4)
                    x_line = np.array([osm.min(), osm.max()])
                    y_fit = slope * x_line + intercept
                    ax_qq.plot(x_line, y_fit, color="#E84855", lw=2)
                    ax_qq.fill_between(
                        osm, osr, slope * osm + intercept, alpha=0.12, color="#E07B39"
                    )
                    ax_qq.set_title(f"Q-Q: {report.best_fit.distribution}  (R={r:.3f})", fontsize=9)
                except Exception:
                    ax_qq.text(
                        0.5,
                        0.5,
                        "Q-Q unavailable",
                        ha="center",
                        va="center",
                        transform=ax_qq.transAxes,
                    )
            ax_qq.set_xlabel("Theoretical Quantiles", fontsize=8)
            ax_qq.set_ylabel("Sample Quantiles", fontsize=8)

            # ── Panel 4: AIC / BIC bar chart ──────────────────────────────────
            all_fits = report.all_fits
            names = [f.distribution for f in all_fits]
            aics = [f.aic for f in all_fits]
            bics = [f.bic for f in all_fits]
            bar_colors = ["#3BB273" if f.is_acceptable else "#E84855" for f in all_fits]

            y_pos = np.arange(len(names))
            ax_aic.barh(y_pos - 0.2, aics, height=0.35, color=bar_colors, alpha=0.80, label="AIC")
            ax_aic.barh(
                y_pos + 0.2,
                bics,
                height=0.35,
                color=bar_colors,
                alpha=0.45,
                label="BIC",
                hatch="//",
            )
            ax_aic.set_yticks(y_pos)
            ax_aic.set_yticklabels(names, fontsize=8)
            ax_aic.set_xlabel("Information Criterion (lower = better)", fontsize=8)
            ax_aic.set_title(
                "AIC / BIC Ranking\n(green = KS acceptable, red = rejected)", fontsize=9
            )
            ax_aic.legend(fontsize=8)
            ax_aic.axvline(0, color="black", lw=0.5)

            plt.tight_layout()
            if save_path:
                fig.savefig(f"{save_path}_{col}.png", dpi=150, bbox_inches="tight")
            plt.show()
            plt.close(fig)

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shift_range(x_range: np.ndarray, raw: np.ndarray, dist_name: str) -> np.ndarray:
        """Return x_range adjusted to the support of the given distribution."""
        if dist_name == "beta":
            return (x_range - raw.min() + 1e-6) / (raw.max() - raw.min() + 2e-6)
        if dist_name in (
            "lognorm",
            "expon",
            "gamma",
            "weibull_min",
            "weibull_max",
            "pareto",
            "powerlaw",
            "rayleigh",
            "chi2",
        ):
            shift = max(0, -raw.min() + 1e-6)
            return x_range + shift
        return x_range

    @staticmethod
    def _get_shifted(raw: np.ndarray, dist_name: str) -> np.ndarray:
        """Return raw data shifted to the support of the given distribution."""
        if dist_name == "beta":
            return (raw - raw.min() + 1e-6) / (raw.max() - raw.min() + 2e-6)
        if dist_name in (
            "lognorm",
            "expon",
            "gamma",
            "weibull_min",
            "weibull_max",
            "pareto",
            "powerlaw",
            "rayleigh",
            "chi2",
        ):
            shift = max(0, -raw.min() + 1e-6)
            return raw + shift
        return raw

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _analyze_column(self, col: str, series: pd.Series) -> ColumnDistributionReport:
        """
        Compute descriptive statistics, fit all candidate distributions, run a Shapiro-Wilk
        normality test, and recommend a transformation strategy for one column.
        Returns a ColumnDistributionReport.
        """
        raw = series.values.astype(float)

        # Descriptive stats
        desc = {
            "mean": float(np.mean(raw)),
            "median": float(np.median(raw)),
            "std": float(np.std(raw)),
            "min": float(np.min(raw)),
            "max": float(np.max(raw)),
            "q25": float(np.percentile(raw, 25)),
            "q75": float(np.percentile(raw, 75)),
            "_raw_data": raw,  # stored for plotting
        }

        skewness = float(stats.skew(raw))
        kurt = float(stats.kurtosis(raw))

        # Normality test (Shapiro-Wilk, capped at 5000 obs)
        sw_sample = raw[:5000]
        _, sw_pvalue = stats.shapiro(sw_sample)
        is_normal = sw_pvalue > self.alpha

        # Fit all candidate distributions
        fits = self._fit_distributions(raw)
        fits.sort(key=lambda f: f.aic)

        best_fit = fits[0] if fits else None
        recommended_transform = self._recommend_transform(skewness, kurt, is_normal, best_fit)

        return ColumnDistributionReport(
            column=col,
            n_observations=len(raw),
            descriptive_stats=desc,
            skewness=skewness,
            kurtosis=kurt,
            is_normal=is_normal,
            shapiro_pvalue=float(sw_pvalue),
            best_fit=best_fit,
            all_fits=fits,
            recommended_transform=recommended_transform,
        )

    def _fit_distributions(self, data: np.ndarray) -> list[DistributionFitResult]:
        """
        Fit every candidate distribution to data and return a sorted list of
        DistributionFitResult objects (sorted by AIC, best first).
        Distributions requiring positive support are automatically shifted.
        """
        results = []
        n = len(data)

        for dist_name in self.candidates:
            try:
                dist = getattr(stats, dist_name)

                # Shift data for distributions requiring positive support
                shifted = data
                if dist_name in (
                    "lognorm",
                    "expon",
                    "gamma",
                    "weibull_min",
                    "weibull_max",
                    "pareto",
                    "powerlaw",
                    "rayleigh",
                    "chi2",
                ):
                    shift = abs(data.min()) + 1e-6 if data.min() <= 0 else 0
                    shifted = data + shift

                if dist_name == "beta":
                    # Beta requires (0,1) range
                    shifted = (data - data.min() + 1e-6) / (data.max() - data.min() + 2e-6)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params = dist.fit(shifted)

                ks_stat, ks_pval = stats.kstest(shifted, dist_name, args=params)

                # AIC / BIC via log-likelihood
                log_lik = np.sum(dist.logpdf(shifted, *params))
                k = len(params)
                aic = 2 * k - 2 * log_lik
                bic = k * np.log(n) - 2 * log_lik

                results.append(
                    DistributionFitResult(
                        distribution=dist_name,
                        params=params,
                        ks_statistic=float(ks_stat),
                        ks_pvalue=float(ks_pval),
                        aic=float(aic),
                        bic=float(bic),
                    )
                )
            except Exception:
                pass

        return results

    def _recommend_transform(
        self,
        skewness: float,
        kurtosis: float,
        is_normal: bool,
        best_fit: DistributionFitResult | None,
    ) -> str:
        """
        Heuristic recommendation for data transformation.

        Priority order:
        1. If data is already normal → 'none'
        2. Skewness thresholds (most reliable signal for advertising data)
        3. Best-fit distribution name as a secondary hint
        4. Kurtosis as a fallback
        """
        if is_normal:
            return "none"

        # Positive skew (most common in impression / spend data)
        if skewness > 1.5:
            return "log"
        if skewness > 0.5:
            return "sqrt"

        # Negative skew
        if skewness < -1.5:
            return "reflect_log"
        if skewness < -0.5:
            return "reflect_sqrt"

        # Mild skew — use best-fit distribution as a secondary hint
        if best_fit:
            if best_fit.distribution == "lognorm":
                return "log"
            if best_fit.distribution in ("expon", "gamma", "weibull_min", "pareto", "powerlaw"):
                return "sqrt"

        # High kurtosis (heavy tails) → Box-Cox
        if abs(kurtosis) > 3:
            return "boxcox"

        # Default: Yeo-Johnson handles any remaining shape
        return "yeo_johnson"
