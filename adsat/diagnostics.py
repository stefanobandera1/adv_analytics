"""
adsat.diagnostics
=================
Model residual and fit diagnostics for saturation curves.

After fitting a saturation model, this module answers:
  - Are the residuals random (good) or structured (bad)?
  - Are there observations with outsized influence on the fit?
  - Is the variance constant across the fit range (homoscedastic)?
  - Is the model reliable enough to use for optimisation decisions?

Key classes & functions
-----------------------
ModelDiagnostics      – main class; works on ModelFitResult or dict of results
DiagnosticsReport     – result dataclass with all residual statistics
run_diagnostics()     – one-liner convenience function

Typical workflow
----------------
>>> from adsat import SaturationModeler
>>> from adsat.diagnostics import ModelDiagnostics
>>>
>>> modeler = SaturationModeler()
>>> results = modeler.fit(df, x_col='impressions', y_col='conversions')
>>>
>>> diag   = ModelDiagnostics()
>>> report = diag.run(results['hill'])
>>> report.print_summary()
>>> diag.plot(report)
>>>
>>> # Compare all models
>>> reports = diag.run_all(results)
>>> diag.plot_comparison(reports)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, kstest, levene, shapiro

from adsat.modeling import ModelFitResult

# ── colour palette ────────────────────────────────────────────────────────────
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_GREEN = "#3BB273"
_RED = "#E84855"
_GREY = "#6C757D"


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DiagnosticsReport:
    """
    Full residual diagnostics for a single fitted model.

    Attributes
    ----------
    model_name : str
    n : int
        Number of observations.
    residuals : np.ndarray
        Raw residuals  y_true − y_pred.
    standardised_residuals : np.ndarray
        Residuals divided by their standard deviation.
    fitted_values : np.ndarray
        Model predictions (y_pred).
    x_values : np.ndarray
        Input values (x).

    Normality tests  (p-value; H0 = residuals are normally distributed)
    ----------------
    shapiro_pvalue : float   Shapiro-Wilk (reliable up to ~5 000 obs)
    ks_pvalue      : float   Kolmogorov-Smirnov vs N(0,1)
    jb_pvalue      : float   Jarque-Bera
    normality_ok   : bool    True when ≥2/3 tests pass at alpha

    Autocorrelation
    ---------------
    durbin_watson    : float  ~2 = no autocorrelation, <1 or >3 = concern
    autocorrelation_ok : bool

    Homoscedasticity (constant variance)
    ------------------------------------
    levene_pvalue      : float  Levene test splitting residuals at median fitted
    homoscedasticity_ok : bool

    Outlier / influence
    -------------------
    cook_distances     : np.ndarray  Cook's distance per observation
    high_influence_idx : np.ndarray  Indices with Cook's D > 4/n
    n_high_influence   : int

    Overall verdict
    ---------------
    overall_ok : bool    True when normality + autocorrelation + homo all pass
    warnings   : list of str
    """

    model_name: str
    n: int
    residuals: np.ndarray
    standardised_residuals: np.ndarray
    fitted_values: np.ndarray
    x_values: np.ndarray
    shapiro_pvalue: float
    ks_pvalue: float
    jb_pvalue: float
    normality_ok: bool
    durbin_watson: float
    autocorrelation_ok: bool
    levene_pvalue: float
    homoscedasticity_ok: bool
    cook_distances: np.ndarray
    high_influence_idx: np.ndarray
    n_high_influence: int
    overall_ok: bool
    warnings: list[str] = field(default_factory=list)

    def print_summary(self) -> None:
        """Print a structured diagnostics report to stdout."""
        sep = "=" * 60
        tick = "OK"
        cross = "!!"

        print(sep)
        print(f"  DIAGNOSTICS REPORT — {self.model_name}")
        print(sep)
        print(f"  Observations      : {self.n}")
        print(
            f"  High-influence pts: {self.n_high_influence}  "
            f"(threshold Cook's D > {4/self.n:.4f})"
        )
        print()

        def row(label, ok, detail):
            """
            Helper to print one diagnostic check row with pass/fail icon and detail string.
            """
            icon = tick if ok else cross
            print(f"  [{icon}]  {label:<30s} {detail}")

        print("  [Normality of residuals]")
        row("Shapiro-Wilk", self.shapiro_pvalue > 0.05, f"p = {self.shapiro_pvalue:.4f}")
        row("Kolmogorov-Smirnov", self.ks_pvalue > 0.05, f"p = {self.ks_pvalue:.4f}")
        row("Jarque-Bera", self.jb_pvalue > 0.05, f"p = {self.jb_pvalue:.4f}")
        row("Normality overall", self.normality_ok, ">=2/3 tests pass at alpha=0.05")

        print()
        print("  [Autocorrelation]")
        row(
            "Durbin-Watson",
            self.autocorrelation_ok,
            f"DW = {self.durbin_watson:.3f}  (ideal ~2.0, concern <1 or >3)",
        )

        print()
        print("  [Homoscedasticity]")
        row("Levene test", self.homoscedasticity_ok, f"p = {self.levene_pvalue:.4f}")

        print()
        if self.warnings:
            print("  [Issues detected]")
            for w in self.warnings:
                print(f"    ** {w}")
        else:
            print("  No issues detected.")

        print()
        verdict = (
            "PASS — model diagnostics acceptable"
            if self.overall_ok
            else "FAIL — review warnings above"
        )
        icon = tick if self.overall_ok else cross
        print(f"  [{icon}]  Overall: {verdict}")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class ModelDiagnostics:
    """
    Residual diagnostics for fitted saturation models.

    Parameters
    ----------
    alpha : float
        Significance level for all hypothesis tests. Default 0.05.
    cook_threshold_multiplier : float
        Cook's distance threshold = multiplier / n. Default 4 (standard rule).
    verbose : bool

    Examples
    --------
    >>> from adsat.diagnostics import ModelDiagnostics
    >>> diag   = ModelDiagnostics()
    >>> report = diag.run(results['hill'])
    >>> report.print_summary()
    >>> diag.plot(report)

    >>> reports = diag.run_all(results)
    >>> print(diag.summary_table(reports))
    >>> diag.plot_comparison(reports)
    """

    def __init__(
        self,
        alpha: float = 0.05,
        cook_threshold_multiplier: float = 4.0,
        verbose: bool = True,
    ):
        """
        Store significance level, Cook's distance multiplier, and verbosity flag.
        No computation occurs here; call run() or run_all() to produce results.
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
        self.alpha = alpha
        self.cook_k = cook_threshold_multiplier
        self.verbose = verbose

    # ── public API ─────────────────────────────────────────────────────────────

    def run(self, result: ModelFitResult) -> DiagnosticsReport:
        """
        Run full diagnostics on a single fitted model.

        Parameters
        ----------
        result : ModelFitResult
            One entry from ``SaturationModeler.fit()``.

        Returns
        -------
        DiagnosticsReport
        """
        self._log(f"Running diagnostics for '{result.model_name}'...")
        return self._compute(result)

    def run_all(
        self,
        results: dict[str, ModelFitResult],
    ) -> dict[str, DiagnosticsReport]:
        """
        Run diagnostics for every model in a results dict.

        Parameters
        ----------
        results : dict  {model_name: ModelFitResult}

        Returns
        -------
        dict  {model_name: DiagnosticsReport}
        """
        return {name: self._compute(res) for name, res in results.items()}

    def summary_table(
        self,
        reports: dict[str, DiagnosticsReport],
    ) -> pd.DataFrame:
        """
        Compare all diagnostic statistics across models in one DataFrame.
        """
        rows = []
        for name, rep in reports.items():
            rows.append(
                {
                    "model": name,
                    "n": rep.n,
                    "shapiro_p": round(rep.shapiro_pvalue, 4),
                    "ks_p": round(rep.ks_pvalue, 4),
                    "jb_p": round(rep.jb_pvalue, 4),
                    "normality_ok": rep.normality_ok,
                    "durbin_watson": round(rep.durbin_watson, 3),
                    "autocorr_ok": rep.autocorrelation_ok,
                    "levene_p": round(rep.levene_pvalue, 4),
                    "homo_ok": rep.homoscedasticity_ok,
                    "n_high_influence": rep.n_high_influence,
                    "overall_ok": rep.overall_ok,
                }
            )
        return pd.DataFrame(rows)

    # ── plots ──────────────────────────────────────────────────────────────────

    def plot(
        self,
        report: DiagnosticsReport,
        save_path: str | None = None,
        figsize: tuple[int, int] = (14, 10),
    ) -> None:
        """
        Six-panel residual diagnostic plot for a single model.

        Panels
        ------
        1. Residuals vs fitted        – should scatter randomly around 0
        2. Residuals vs x             – should show no trend
        3. Normal Q-Q plot            – should hug the diagonal
        4. Histogram of residuals     – should look roughly bell-shaped
        5. Cook's distance stem plot  – high-influence points highlighted in red
        6. Scale-location             – tests constant variance (homoscedasticity)

        Parameters
        ----------
        report : DiagnosticsReport
        save_path : str, optional
        figsize : (width, height)
        """
        fig = plt.figure(figsize=figsize)
        verdict = "PASS" if report.overall_ok else "FAIL"
        fig.suptitle(
            f"Residual Diagnostics — {report.model_name}"
            f"  |  n={report.n}  |  Overall: {verdict}",
            fontsize=12,
            fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
        axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

        res = report.residuals
        sres = report.standardised_residuals
        fit = report.fitted_values
        xs = report.x_values
        cook = report.cook_distances
        hi = report.high_influence_idx
        n = report.n

        # ── 1: Residuals vs fitted ────────────────────────────────────────
        ax = axs[0]
        ax.scatter(fit, res, s=25, color=_BLUE, alpha=0.65, zorder=3)
        ax.axhline(0, color=_RED, lw=1.2, ls="--")
        if hi.size:
            ax.scatter(
                fit[hi], res[hi], s=60, color=_RED, zorder=5, label=f"High influence (n={len(hi)})"
            )
            ax.legend(fontsize=7)
        self._smooth_line(ax, fit, res, color=_ORANGE)
        ax.set_xlabel("Fitted values", fontsize=8)
        ax.set_ylabel("Residuals", fontsize=8)
        ax.set_title("Residuals vs Fitted", fontsize=9)

        # ── 2: Residuals vs x ─────────────────────────────────────────────
        ax = axs[1]
        ax.scatter(xs, res, s=25, color=_BLUE, alpha=0.65, zorder=3)
        ax.axhline(0, color=_RED, lw=1.2, ls="--")
        self._smooth_line(ax, xs, res, color=_ORANGE)
        ax.set_xlabel("x (input variable)", fontsize=8)
        ax.set_ylabel("Residuals", fontsize=8)
        ax.set_title("Residuals vs X", fontsize=9)

        # ── 3: Normal Q-Q ─────────────────────────────────────────────────
        ax = axs[2]
        (osm, osr), (slope, intercept, _) = stats.probplot(sres, dist="norm")
        ax.scatter(osm, osr, s=20, color=_BLUE, alpha=0.7)
        line_x = np.array([osm[0], osm[-1]])
        ax.plot(line_x, slope * line_x + intercept, color=_RED, lw=1.5, ls="--", label="Normal ref")
        ax.set_xlabel("Theoretical quantiles", fontsize=8)
        ax.set_ylabel("Standardised residuals", fontsize=8)
        ax.set_title(f"Q-Q Plot  (Shapiro p={report.shapiro_pvalue:.3f})", fontsize=9)
        ax.legend(fontsize=7)

        # ── 4: Histogram ──────────────────────────────────────────────────
        ax = axs[3]
        ax.hist(res, bins="auto", color=_BLUE, alpha=0.65, density=True, edgecolor="white")
        xr = np.linspace(res.min() - res.std(), res.max() + res.std(), 200)
        mu, sigma = res.mean(), res.std()
        if sigma > 0:
            ax.plot(xr, stats.norm.pdf(xr, mu, sigma), color=_RED, lw=2, label="Normal fit")
        ax.axvline(0, color=_GREY, lw=1, ls=":")
        ax.set_xlabel("Residuals", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title("Residual Distribution", fontsize=9)
        ax.legend(fontsize=7)

        # ── 5: Cook's distance ────────────────────────────────────────────
        ax = axs[4]
        idx_arr = np.arange(n)
        threshold = self.cook_k / n
        c_colors = [_RED if i in set(hi.tolist()) else _BLUE for i in idx_arr]
        ax.vlines(idx_arr, 0, cook, colors=c_colors, lw=1.2, alpha=0.7)
        ax.scatter(idx_arr, cook, s=15, c=c_colors, zorder=4)
        ax.axhline(threshold, color=_RED, lw=1.2, ls="--", label=f"Threshold={threshold:.4f}")
        ax.set_xlabel("Observation index", fontsize=8)
        ax.set_ylabel("Cook's distance", fontsize=8)
        ax.set_title(f"Cook's Distance  ({len(hi)} flagged)", fontsize=9)
        ax.legend(fontsize=7)

        # ── 6: Scale-location ─────────────────────────────────────────────
        ax = axs[5]
        sqrt_abs = np.sqrt(np.abs(sres))
        ax.scatter(fit, sqrt_abs, s=25, color=_BLUE, alpha=0.65)
        self._smooth_line(ax, fit, sqrt_abs, color=_ORANGE)
        ax.set_xlabel("Fitted values", fontsize=8)
        ax.set_ylabel("|Standardised residuals|^0.5", fontsize=8)
        ax.set_title(f"Scale-Location  (Levene p={report.levene_pvalue:.3f})", fontsize=9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    def plot_comparison(
        self,
        reports: dict[str, DiagnosticsReport],
        save_path: str | None = None,
    ) -> None:
        """
        Bar-chart comparison of all key diagnostic statistics across models.

        Green bars pass the test, red bars fail.

        Parameters
        ----------
        reports : dict from run_all()
        save_path : str, optional
        """
        df = self.summary_table(reports)
        models = df["model"].tolist()

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Diagnostics Comparison Across Models", fontsize=13, fontweight="bold")

        metrics = [
            ("shapiro_p", "Shapiro-Wilk p", "pval"),
            ("ks_p", "KS test p", "pval"),
            ("jb_p", "Jarque-Bera p", "pval"),
            ("durbin_watson", "Durbin-Watson", "dw"),
            ("levene_p", "Levene p", "pval"),
            ("n_high_influence", "High-influence points", "count"),
        ]

        for ax, (col, title, kind) in zip(axes.flatten(), metrics):
            vals = df[col].values.astype(float)
            if kind == "pval":
                colors = [_GREEN if v > self.alpha else _RED for v in vals]
            elif kind == "dw":
                colors = [_GREEN if 1.5 < v < 2.5 else _RED for v in vals]
            else:  # count
                colors = [_GREEN if v == 0 else (_ORANGE if v <= 2 else _RED) for v in vals]
            ax.bar(models, vals, color=colors, alpha=0.85)
            if kind == "pval":
                ax.axhline(self.alpha, color=_RED, lw=1.2, ls="--", label=f"alpha={self.alpha}")
                ax.legend(fontsize=7)
            elif kind == "dw":
                ax.axhline(2.0, color=_GREY, lw=1, ls=":", label="Ideal=2")
                ax.axhline(1.5, color=_RED, lw=0.8, ls="--", alpha=0.5)
                ax.axhline(2.5, color=_RED, lw=0.8, ls="--", alpha=0.5)
                ax.legend(fontsize=7)
            ax.set_xticklabels(models, rotation=20, ha="right", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.set_ylim(bottom=0)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)

    # ── internal: compute all statistics ──────────────────────────────────────

    def _compute(self, result: ModelFitResult) -> DiagnosticsReport:
        """
        Compute all diagnostic statistics for a single ModelFitResult.

        Runs normality tests (Shapiro-Wilk, KS, Jarque-Bera), Durbin-Watson autocorrelation,
        Levene homoscedasticity test, and Cook's distances for influence detection.
        Returns a fully-populated DiagnosticsReport.
        """
        y_true = result.y_true.astype(float)
        y_pred = result.y_pred.astype(float)
        x_vals = result.x_values.astype(float)
        n = len(y_true)

        res = y_true - y_pred
        std = res.std()
        sres = res / std if std > 0 else res.copy()

        # ── Normality ─────────────────────────────────────────────────────
        sample = res if n <= 5000 else np.random.default_rng(42).choice(res, 5000, replace=False)
        try:
            _, sw_p = shapiro(sample)
        except Exception:
            sw_p = 1.0
        try:
            _, ks_p = kstest(sres, "norm", args=(0, 1))
        except Exception:
            ks_p = 1.0
        try:
            _, jb_p = jarque_bera(res)
        except Exception:
            jb_p = 1.0

        normality_ok = sum(p > self.alpha for p in (sw_p, ks_p, jb_p)) >= 2

        # ── Durbin-Watson ─────────────────────────────────────────────────
        dw = self._durbin_watson(res)
        autocorr_ok = 1.0 < dw < 3.0

        # ── Levene homoscedasticity ───────────────────────────────────────
        try:
            med = np.median(y_pred)
            g1 = res[y_pred <= med]
            g2 = res[y_pred > med]
            if len(g1) >= 2 and len(g2) >= 2:
                _, lev_p = levene(g1, g2)
            else:
                lev_p = 1.0
        except Exception:
            lev_p = 1.0
        homo_ok = lev_p > self.alpha

        # ── Cook's distances ──────────────────────────────────────────────
        k_params = len(result.params)
        cook = self._cook_distances(res, y_pred, n, k_params)
        threshold = self.cook_k / n
        hi_idx = np.where(cook > threshold)[0]

        # ── Warnings ──────────────────────────────────────────────────────
        w = []
        if not normality_ok:
            w.append(
                f"Residuals not normally distributed "
                f"(SW p={sw_p:.3f}, KS p={ks_p:.3f}, JB p={jb_p:.3f}). "
                "Prediction intervals may be unreliable."
            )
        if not autocorr_ok:
            w.append(
                f"Autocorrelation in residuals (DW={dw:.3f}). "
                "Consider time-series adjustments or seasonality decomposition."
            )
        if not homo_ok:
            w.append(
                f"Heteroscedasticity detected (Levene p={lev_p:.3f}). "
                "Variance is not constant — consider weighted fitting."
            )
        if len(hi_idx) > 0:
            w.append(
                f"{len(hi_idx)} high-influence observation(s) detected "
                f"(Cook's D > {threshold:.4f}). Indices: {hi_idx.tolist()}."
            )
        if not result.converged:
            w.append(
                "Model did not fully converge — all statistics should be " "treated with caution."
            )

        return DiagnosticsReport(
            model_name=result.model_name,
            n=n,
            residuals=res,
            standardised_residuals=sres,
            fitted_values=y_pred,
            x_values=x_vals,
            shapiro_pvalue=float(sw_p),
            ks_pvalue=float(ks_p),
            jb_pvalue=float(jb_p),
            normality_ok=normality_ok,
            durbin_watson=float(dw),
            autocorrelation_ok=autocorr_ok,
            levene_pvalue=float(lev_p),
            homoscedasticity_ok=homo_ok,
            cook_distances=cook,
            high_influence_idx=hi_idx,
            n_high_influence=int(len(hi_idx)),
            overall_ok=normality_ok and autocorr_ok and homo_ok,
            warnings=w,
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _durbin_watson(residuals: np.ndarray) -> float:
        """Compute the Durbin-Watson statistic."""
        if len(residuals) < 2:
            return 2.0
        diff = np.diff(residuals)
        denom = np.sum(residuals**2)
        return float(np.sum(diff**2) / denom) if denom > 0 else 2.0

    @staticmethod
    def _cook_distances(
        residuals: np.ndarray,
        fitted: np.ndarray,
        n: int,
        k: int,
    ) -> np.ndarray:
        """
        Approximate Cook's distances using an OLS leverage approximation.

        h_i = 1/n + (f_i - f_mean)^2 / sum((f_j - f_mean)^2)
        Cook_i = (res_i^2 * h_i) / (k * MSE * (1 - h_i)^2)
        """
        f_mean = fitted.mean()
        ss = np.sum((fitted - f_mean) ** 2)
        if ss == 0:
            return np.zeros(n)
        h = 1.0 / n + (fitted - f_mean) ** 2 / ss
        h = np.clip(h, 0.0, 1.0 - 1e-9)
        mse = np.mean(residuals**2)
        if mse == 0:
            return np.zeros(n)
        denom = k * mse * (1.0 - h) ** 2
        denom = np.where(denom > 0, denom, 1e-12)
        return (residuals**2 * h) / denom

    def _log(self, msg: str) -> None:
        """
        Print msg to stdout when verbose=True.
        """
        if self.verbose:
            print(f"[ModelDiagnostics] {msg}")

    @staticmethod
    def _smooth_line(ax, x: np.ndarray, y: np.ndarray, color: str, frac: float = 0.4) -> None:
        """Add a rolling-mean smoothed trend line (lightweight LOWESS)."""
        if len(x) < 6:
            return
        order = np.argsort(x)
        xs, ys = x[order], y[order]
        window = max(3, int(len(xs) * frac))
        kernel = np.ones(window) / window
        try:
            sm = np.convolve(ys, kernel, mode="valid")
            xsm = xs[window // 2 : window // 2 + len(sm)]
            ax.plot(xsm, sm, color=color, lw=1.5, ls="--", alpha=0.8)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def run_diagnostics(
    result: ModelFitResult,
    alpha: float = 0.05,
    verbose: bool = False,
) -> DiagnosticsReport:
    """
    One-function shortcut: run diagnostics on a single ModelFitResult.

    Parameters
    ----------
    result : ModelFitResult
        One model from ``SaturationModeler.fit()``.
    alpha : float
        Significance level. Default 0.05.
    verbose : bool

    Returns
    -------
    DiagnosticsReport

    Examples
    --------
    >>> from adsat.diagnostics import run_diagnostics
    >>> report = run_diagnostics(results['hill'])
    >>> report.print_summary()
    """
    return ModelDiagnostics(alpha=alpha, verbose=verbose).run(result)
