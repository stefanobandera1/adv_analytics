"""
adsat.exploratory
=================
Exploratory Data Analysis (EDA) for advertising campaign metrics.

Provides a suite of visualisations and descriptive statistics designed to:
  - Understand the shape and spread of each metric
  - Identify the best-fitting statistical distribution
  - Reveal relationships between variables (impressions vs conversions, spend vs ROI)
  - Surface outliers, skewness, and non-normality before modelling

Key class  : CampaignExplorer
Key function: explore(df, ...)   ← one-liner that runs everything

Plots produced
--------------
  plot_descriptive_summary()   – table of stats + boxplots per column
  plot_histograms()            – histogram with KDE per column
  plot_qq()                    – Q-Q plots vs Normal and best-fit distribution
  plot_ecdf()                  – Empirical Cumulative Distribution Function
  plot_correlation()           – Pearson/Spearman heatmap + pairplot
  plot_scatter()               – x vs y scatter with regression line + confidence band
  plot_time_series()           – metrics over time, per campaign
  plot_outliers()              – boxplots + z-score / IQR flagging
  plot_distribution_fits()     – histogram + top-N fitted PDFs + AIC ranking
  explore()                    – runs all of the above in one call
"""

from __future__ import annotations

import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

# ── colour palette ────────────────────────────────────────────────────────────
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_GREEN = "#3BB273"
_RED = "#E84855"
_PURPLE = "#7B2D8B"
_GREY = "#6C757D"
_LIGHT = "#F0F4F8"
PALETTE = [_BLUE, _ORANGE, _GREEN, _RED, _PURPLE, _GREY]


def _fmt_large(x, _=None) -> str:
    """Format large numbers with k / M suffix for axis labels."""
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}k"
    return f"{x:.1f}"


def _save_or_show(fig: plt.Figure, save_path: str | None, suffix: str) -> None:
    """
    Save the figure to save_path when provided; otherwise call plt.show().
    Always closes the figure after saving/showing to free memory.
    """
    plt.tight_layout()
    if save_path:
        fig.savefig(f"{save_path}_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════════


class CampaignExplorer:
    """
    Exploratory Data Analysis for advertising campaign datasets.

    Produces a comprehensive set of visualisations covering univariate
    distribution analysis, bivariate relationships, time-series trends,
    outlier detection, and distribution fitting.

    Parameters
    ----------
    df : pd.DataFrame
        Campaign data. Each row is one time period for one campaign.
    numeric_cols : list of str, optional
        Columns to analyse. Defaults to all numeric columns.
    campaign_col : str, optional
        Column identifying campaigns. Used for grouping in time-series plots.
    date_col : str, optional
        Date/period column. Used for time-series plots.
    figsize : tuple
        Default figure size (width, height).
    style : str
        Matplotlib style. Default 'seaborn-v0_8-whitegrid'.

    Examples
    --------
    >>> from adsat.exploratory import CampaignExplorer
    >>> explorer = CampaignExplorer(
    ...     df,
    ...     numeric_cols=["impressions", "conversions", "ad_spend", "revenue"],
    ...     campaign_col="campaign_id",
    ...     date_col="date",
    ... )
    >>> explorer.explore()                          # run everything
    >>> explorer.plot_scatter("impressions", "conversions")
    >>> explorer.plot_distribution_fits("impressions")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str] | None = None,
        campaign_col: str | None = None,
        date_col: str | None = None,
        figsize: tuple[int, int] = (14, 6),
        style: str = "seaborn-v0_8-whitegrid",
    ):
        """
        Store column configuration, display preferences, and output path.
        No computation occurs here; call explore() or individual plot_* methods.
        """
        self.df = df.copy()
        self.campaign_col = campaign_col
        self.date_col = date_col
        self.figsize = figsize

        # resolve numeric columns
        if numeric_cols:
            self.numeric_cols = [c for c in numeric_cols if c in df.columns]
        else:
            exclude = {campaign_col, date_col}
            self.numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
            ]

        # apply style safely
        try:
            plt.style.use(style)
        except Exception:
            try:
                plt.style.use("seaborn-whitegrid")
            except Exception:
                pass  # fall back to matplotlib default

    # ──────────────────────────────────────────────────────────────────────────
    # Master method
    # ──────────────────────────────────────────────────────────────────────────

    def explore(
        self,
        x_col: str | None = None,
        y_col: str | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Run the full EDA suite.

        Produces all available plots in sequence:
          1. Descriptive summary table + boxplots
          2. Histograms with KDE
          3. Q-Q plots
          4. ECDF
          5. Correlation heatmap
          6. Scatter plots (x vs y if supplied, else first two numeric cols)
          7. Time-series (if date_col set)
          8. Outlier detection
          9. Distribution fitting

        Parameters
        ----------
        x_col : str, optional  – x axis for scatter / relationship plots
        y_col : str, optional  – y axis for scatter / relationship plots
        save_path : str, optional – base path; each plot appends a suffix
        """
        print("=" * 60)
        print("  ADSAT – Exploratory Data Analysis")
        print("=" * 60)

        self.plot_descriptive_summary(save_path=save_path)
        self.plot_histograms(save_path=save_path)
        self.plot_qq(save_path=save_path)
        self.plot_ecdf(save_path=save_path)
        self.plot_correlation(save_path=save_path)

        # Scatter: use provided cols or default to first two numeric
        _x = x_col or (self.numeric_cols[0] if len(self.numeric_cols) >= 1 else None)
        _y = y_col or (self.numeric_cols[1] if len(self.numeric_cols) >= 2 else None)
        if _x and _y:
            self.plot_scatter(_x, _y, save_path=save_path)

        if self.date_col and self.date_col in self.df.columns:
            self.plot_time_series(save_path=save_path)

        self.plot_outliers(save_path=save_path)

        for col in self.numeric_cols:
            self.plot_distribution_fits(col, save_path=save_path)

        print("\n  ✓ EDA complete.")

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Descriptive summary
    # ──────────────────────────────────────────────────────────────────────────

    def plot_descriptive_summary(
        self,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Print a rich descriptive statistics table and plot side-by-side boxplots.

        Statistics reported per column:
          count, mean, std, min, 25th/50th/75th percentile, max,
          skewness, kurtosis, % missing, coefficient of variation.

        Returns
        -------
        pd.DataFrame  – the descriptive statistics table
        """
        cols = self.numeric_cols
        df = self.df[cols].copy()

        rows = []
        for col in cols:
            s = df[col].dropna()
            cv = (s.std() / s.mean() * 100) if s.mean() != 0 else np.nan
            rows.append(
                {
                    "column": col,
                    "count": int(s.count()),
                    "missing_%": round(df[col].isna().mean() * 100, 1),
                    "mean": round(s.mean(), 2),
                    "std": round(s.std(), 2),
                    "cv_%": round(cv, 1),
                    "min": round(s.min(), 2),
                    "p25": round(s.quantile(0.25), 2),
                    "median": round(s.median(), 2),
                    "p75": round(s.quantile(0.75), 2),
                    "max": round(s.max(), 2),
                    "skewness": round(stats.skew(s), 3),
                    "kurtosis": round(stats.kurtosis(s), 3),
                }
            )

        tbl = pd.DataFrame(rows)

        print("\n[1] Descriptive Statistics")
        print(tbl.to_string(index=False))

        # Boxplots
        n = len(cols)
        fig, axes = plt.subplots(1, n, figsize=(max(self.figsize[0], 4 * n), self.figsize[1]))
        if n == 1:
            axes = [axes]
        fig.suptitle("Boxplots – Distribution Spread per Metric", fontsize=13, fontweight="bold")

        for ax, col, color in zip(axes, cols, PALETTE * 10):
            data = df[col].dropna()
            bp = ax.boxplot(
                data,
                patch_artist=True,
                notch=False,
                medianprops=dict(color="white", lw=2.5),
                whiskerprops=dict(color=_GREY),
                capprops=dict(color=_GREY),
                flierprops=dict(marker="o", color=color, alpha=0.5, markersize=4),
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.75)

            # Annotate mean
            mean_val = data.mean()
            ax.axhline(mean_val, color=_RED, lw=1.5, ls="--", label=f"mean={mean_val:,.0f}")
            ax.set_title(col, fontsize=10)
            ax.set_xticklabels([])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.legend(fontsize=7)

        _save_or_show(fig, save_path, "01_boxplots")
        return tbl

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Histograms + KDE
    # ──────────────────────────────────────────────────────────────────────────

    def plot_histograms(
        self,
        bins: int | str = "auto",
        save_path: str | None = None,
    ) -> None:
        """
        Plot a histogram with a KDE overlay for each numeric column.

        Annotations include mean, median, and skewness.

        Parameters
        ----------
        bins : int or 'auto'
        save_path : str, optional
        """
        cols = self.numeric_cols
        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle("Histograms + KDE – Metric Distributions", fontsize=13, fontweight="bold")

        for ax, col, color in zip(axes, cols, PALETTE * 10):
            data = self.df[col].dropna().values.astype(float)

            ax.hist(
                data, bins=bins, density=True, alpha=0.45, color=color, edgecolor="white", lw=0.5
            )

            # KDE
            try:
                kde = stats.gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 300)
                ax.plot(x_range, kde(x_range), color=color, lw=2.5)
            except Exception:
                pass

            # Mean & median lines
            ax.axvline(data.mean(), color=_RED, lw=1.5, ls="--", label=f"mean={data.mean():,.0f}")
            ax.axvline(
                np.median(data), color=_GREY, lw=1.5, ls=":", label=f"median={np.median(data):,.0f}"
            )

            skew_val = stats.skew(data)
            ax.set_title(f"{col}\nskew={skew_val:+.2f}", fontsize=9)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.legend(fontsize=7)

        for ax in axes[n:]:
            ax.set_visible(False)

        _save_or_show(fig, save_path, "02_histograms")

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Q-Q plots
    # ──────────────────────────────────────────────────────────────────────────

    def plot_qq(
        self,
        distributions: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Q-Q plots for each numeric column against one or more reference distributions.

        Points lying on the diagonal line indicate a good fit.
        Deviations reveal skewness, heavy tails, or other non-normality.

        Parameters
        ----------
        distributions : list of str
            scipy.stats distribution names to test against.
            Default: ['norm', 'lognorm', 'expon'].
        save_path : str, optional
        """
        dists = distributions or ["norm", "lognorm", "expon"]
        cols = self.numeric_cols
        n_cols = len(cols)
        n_dists = len(dists)

        fig, axes = plt.subplots(
            n_cols,
            n_dists,
            figsize=(5 * n_dists, 4 * n_cols),
            squeeze=False,
        )
        fig.suptitle(
            "Q-Q Plots – Sample vs Theoretical Distributions", fontsize=13, fontweight="bold"
        )

        for row, col in enumerate(cols):
            data = self.df[col].dropna().values.astype(float)

            for c, dist_name in enumerate(dists):
                ax = axes[row][c]
                try:
                    dist_obj = getattr(stats, dist_name)

                    # For lognorm / expon shift to positive
                    shifted = data
                    if dist_name in ("lognorm", "expon", "gamma", "weibull_min"):
                        shift = max(0, -data.min() + 1e-6)
                        shifted = data + shift

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        params = dist_obj.fit(shifted)

                    (osm, osr), (slope, intercept, r) = stats.probplot(
                        shifted,
                        dist=dist_obj,
                        sparams=params[:-2] if len(params) > 2 else params,
                        fit=True,
                    )

                    ax.scatter(osm, osr, s=18, alpha=0.65, color=_BLUE, zorder=4)
                    x_line = np.array([osm.min(), osm.max()])
                    ax.plot(x_line, slope * x_line + intercept, color=_RED, lw=1.5)

                    ax.set_title(f"{col} vs {dist_name}\nR={r:.3f}", fontsize=8)
                    ax.set_xlabel("Theoretical Quantiles", fontsize=7)
                    ax.set_ylabel("Sample Quantiles", fontsize=7)

                    # Shade deviation from line
                    y_fit = slope * osm + intercept
                    ax.fill_between(osm, osr, y_fit, alpha=0.12, color=_ORANGE)

                except Exception:
                    ax.text(
                        0.5,
                        0.5,
                        f"Could not fit\n{dist_name}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"{col} vs {dist_name}", fontsize=8)

        _save_or_show(fig, save_path, "03_qq_plots")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. ECDF
    # ──────────────────────────────────────────────────────────────────────────

    def plot_ecdf(
        self,
        overlay_normal: bool = True,
        save_path: str | None = None,
    ) -> None:
        """
        Empirical Cumulative Distribution Function (ECDF) for each column.

        The ECDF is a non-parametric estimate of the CDF — no binning, no
        smoothing. It shows exactly where data points accumulate and makes
        it easy to read off percentiles directly from the chart.

        A fitted Normal CDF is overlaid for reference.

        Parameters
        ----------
        overlay_normal : bool
            Overlay the fitted Normal CDF for comparison. Default True.
        save_path : str, optional
        """
        cols = self.numeric_cols
        n = len(cols)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle(
            "ECDF – Empirical Cumulative Distribution Functions", fontsize=13, fontweight="bold"
        )

        for ax, col, color in zip(axes, cols, PALETTE * 10):
            data = np.sort(self.df[col].dropna().values.astype(float))
            n_pts = len(data)
            ecdf_y = np.arange(1, n_pts + 1) / n_pts

            ax.step(data, ecdf_y, where="post", color=color, lw=2, label="ECDF")
            ax.scatter(data, ecdf_y, s=10, color=color, alpha=0.4, zorder=3)

            if overlay_normal:
                mu, sigma = data.mean(), data.std()
                if sigma > 0:
                    x_norm = np.linspace(data.min(), data.max(), 300)
                    ax.plot(
                        x_norm,
                        stats.norm.cdf(x_norm, mu, sigma),
                        color=_RED,
                        lw=1.5,
                        ls="--",
                        label="Normal CDF",
                    )

            # Mark median and p90
            for pct, ls, label in [(50, ":", "p50"), (90, "--", "p90")]:
                val = np.percentile(data, pct)
                ax.axvline(val, color=_GREY, lw=1, ls=ls)
                ax.text(
                    val,
                    0.02,
                    f" {label}={val:,.0f}",
                    fontsize=6,
                    color=_GREY,
                    rotation=90,
                    va="bottom",
                )

            ax.set_title(col, fontsize=9)
            ax.set_xlabel(col, fontsize=8)
            ax.set_ylabel("Cumulative Probability", fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.legend(fontsize=7)

        for ax in axes[n:]:
            ax.set_visible(False)

        _save_or_show(fig, save_path, "04_ecdf")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Correlation heatmap + pairplot
    # ──────────────────────────────────────────────────────────────────────────

    def plot_correlation(
        self,
        method: str = "pearson",
        save_path: str | None = None,
    ) -> None:
        """
        Correlation heatmap and scatter pairplot for all numeric columns.

        Parameters
        ----------
        method : str
            'pearson' (linear) or 'spearman' (rank-based, robust to outliers).
        save_path : str, optional
        """
        cols = self.numeric_cols
        df_num = self.df[cols].dropna()

        # ── Heatmap ──────────────────────────────────────────────────────────
        corr = df_num.corr(method=method)

        fig, ax = plt.subplots(figsize=(max(7, len(cols) * 1.2), max(6, len(cols) * 1.0)))
        fig.suptitle(f"Correlation Heatmap ({method.capitalize()})", fontsize=13, fontweight="bold")

        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(cols, fontsize=9)

        for i in range(len(cols)):
            for j in range(len(cols)):
                val = corr.iloc[i, j]
                text_color = "white" if abs(val) > 0.7 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )

        _save_or_show(fig, save_path, "05a_correlation_heatmap")

        # ── Pairplot (scatter matrix) ─────────────────────────────────────────
        n = len(cols)
        if n < 2:
            # Nothing to compare — skip pairplot silently
            return

        fig2, axes2 = plt.subplots(n, n, figsize=(3 * n, 3 * n), squeeze=False)
        fig2.suptitle(
            f"Scatter Matrix ({method.capitalize()} correlations)", fontsize=12, fontweight="bold"
        )

        for i, col_i in enumerate(cols):
            for j, col_j in enumerate(cols):
                ax = axes2[i][j]
                xi = df_num[col_i].values
                xj = df_num[col_j].values

                if i == j:
                    # Diagonal: KDE
                    try:
                        kde = stats.gaussian_kde(xi)
                        xr = np.linspace(xi.min(), xi.max(), 200)
                        ax.plot(xr, kde(xr), color=PALETTE[i % len(PALETTE)], lw=1.5)
                        ax.fill_between(xr, kde(xr), alpha=0.2, color=PALETTE[i % len(PALETTE)])
                    except Exception:
                        ax.hist(
                            xi, bins=15, density=True, color=PALETTE[i % len(PALETTE)], alpha=0.5
                        )
                    ax.set_title(col_i, fontsize=7, pad=2)
                else:
                    r, p = stats.pearsonr(xi, xj)
                    ax.scatter(xj, xi, s=10, alpha=0.45, color=PALETTE[i % len(PALETTE)])
                    # Trend line
                    try:
                        m, b = np.polyfit(xj, xi, 1)
                        xfit = np.linspace(xj.min(), xj.max(), 100)
                        ax.plot(xfit, m * xfit + b, color=_RED, lw=1, alpha=0.8)
                    except Exception:
                        pass
                    ax.text(
                        0.05,
                        0.92,
                        f"r={r:.2f}",
                        transform=ax.transAxes,
                        fontsize=6,
                        color="black",
                        fontweight="bold" if abs(r) > 0.6 else "normal",
                    )

                ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
                ax.tick_params(labelsize=6)

                if i == n - 1:
                    ax.set_xlabel(col_j, fontsize=7)
                if j == 0:
                    ax.set_ylabel(col_i, fontsize=7)

        _save_or_show(fig2, save_path, "05b_scatter_matrix")

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Scatter + regression
    # ──────────────────────────────────────────────────────────────────────────

    def plot_scatter(
        self,
        x_col: str,
        y_col: str,
        color_by: str | None = None,
        log_scale: bool = False,
        save_path: str | None = None,
    ) -> None:
        """
        Scatter plot of x_col vs y_col with regression line and 95% confidence band.

        Optionally colour points by campaign or another categorical column.

        Parameters
        ----------
        x_col : str  – independent variable (e.g. 'impressions')
        y_col : str  – dependent variable   (e.g. 'conversions')
        color_by : str, optional  – categorical column for colour grouping
        log_scale : bool  – apply log scale to both axes
        save_path : str, optional
        """
        df = self.df[[x_col, y_col]].dropna()
        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)

        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        fig.suptitle(f"Relationship: {x_col}  vs  {y_col}", fontsize=13, fontweight="bold")

        for ax_idx, ax in enumerate(axes):
            # Colour grouping
            if color_by and color_by in self.df.columns:
                groups = self.df[color_by].unique()
                for grp, color in zip(groups, PALETTE * 10):
                    mask = self.df[color_by] == grp
                    gdf = self.df[mask][[x_col, y_col]].dropna()
                    ax.scatter(
                        gdf[x_col],
                        gdf[y_col],
                        s=30,
                        alpha=0.65,
                        color=color,
                        label=str(grp),
                        zorder=4,
                    )
                ax.legend(fontsize=8, title=color_by)
            else:
                ax.scatter(x, y, s=30, alpha=0.55, color=_BLUE, zorder=4)

            # OLS regression line + 95% CI
            try:
                sort_idx = np.argsort(x)
                xs = x[sort_idx]
                ys = y[sort_idx]
                m, b, r_val, p_val, se = stats.linregress(xs, ys)
                y_hat = m * xs + b
                ax.plot(xs, y_hat, color=_RED, lw=2, label=f"OLS  r={r_val:.2f}")

                # 95% confidence band
                n = len(xs)
                t_crit = stats.t.ppf(0.975, df=n - 2)
                x_mean = xs.mean()
                ss_x = np.sum((xs - x_mean) ** 2)
                residuals = ys - y_hat
                s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
                ci = t_crit * s_err * np.sqrt(1 / n + (xs - x_mean) ** 2 / ss_x)
                ax.fill_between(xs, y_hat - ci, y_hat + ci, alpha=0.15, color=_RED, label="95% CI")
                ax.legend(fontsize=8)
            except Exception:
                pass

            if log_scale or ax_idx == 1:
                try:
                    ax.set_xscale("log")
                    ax.set_yscale("log")
                    suffix = " (log scale)"
                except Exception:
                    suffix = ""
            else:
                suffix = ""

            ax.set_xlabel(x_col, fontsize=9)
            ax.set_ylabel(y_col, fontsize=9)
            ax.set_title(("Linear scale" if ax_idx == 0 else "Log scale") + suffix, fontsize=9)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))

        _save_or_show(fig, save_path, f"06_scatter_{x_col}_vs_{y_col}")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Time-series
    # ──────────────────────────────────────────────────────────────────────────

    def plot_time_series(
        self,
        cols: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Line charts of metrics over time, with one line per campaign.

        Requires date_col to be set on the CampaignExplorer instance.

        Parameters
        ----------
        cols : list of str, optional  – metrics to plot; defaults to all numeric
        save_path : str, optional
        """
        if not self.date_col or self.date_col not in self.df.columns:
            print("  [plot_time_series] No date_col set — skipping.")
            return

        plot_cols = cols or self.numeric_cols
        n = len(plot_cols)
        fig, axes = plt.subplots(n, 1, figsize=(self.figsize[0], 4 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle("Metrics Over Time (per Campaign)", fontsize=13, fontweight="bold")

        df_sorted = self.df.sort_values(self.date_col)

        for ax, col in zip(axes, plot_cols):
            if self.campaign_col and self.campaign_col in self.df.columns:
                groups = df_sorted[self.campaign_col].unique()
                for grp, color in zip(groups, PALETTE * 10):
                    gdf = df_sorted[df_sorted[self.campaign_col] == grp]
                    ax.plot(
                        gdf[self.date_col],
                        gdf[col],
                        marker="o",
                        ms=4,
                        lw=1.8,
                        color=color,
                        label=str(grp),
                        alpha=0.85,
                    )
                ax.legend(fontsize=8, title=self.campaign_col)
            else:
                ax.plot(
                    df_sorted[self.date_col], df_sorted[col], marker="o", ms=4, lw=1.8, color=_BLUE
                )

            ax.set_ylabel(col, fontsize=9)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax.grid(axis="y", alpha=0.4)

        axes[-1].set_xlabel(self.date_col, fontsize=9)
        fig.autofmt_xdate(rotation=30)
        _save_or_show(fig, save_path, "07_time_series")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Outlier detection
    # ──────────────────────────────────────────────────────────────────────────

    def plot_outliers(
        self,
        z_threshold: float = 3.0,
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """
        Identify and visualise outliers using both IQR and z-score methods.

        Each column gets two side-by-side plots:
          - Boxplot with outlier points highlighted in red
          - Strip plot with z-score threshold lines

        Parameters
        ----------
        z_threshold : float
            Z-score cutoff for flagging outliers. Default 3.0.
        save_path : str, optional

        Returns
        -------
        pd.DataFrame  – table of detected outliers (row index, column, value, method)
        """
        cols = self.numeric_cols
        n = len(cols)
        fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
        if n == 1:
            axes = axes.reshape(1, 2)
        fig.suptitle(
            f"Outlier Detection (IQR + Z-score >{z_threshold}σ)", fontsize=13, fontweight="bold"
        )

        outlier_records = []

        for row, col in enumerate(cols):
            data = self.df[col].dropna()
            vals = data.values.astype(float)

            # IQR method
            q1, q3 = np.percentile(vals, 25), np.percentile(vals, 75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_outliers = (vals < lower) | (vals > upper)

            # Z-score method
            zscores = np.abs(stats.zscore(vals))
            z_outliers = zscores > z_threshold

            # Record outliers
            for idx, (v, is_iqr, is_z) in enumerate(zip(vals, iqr_outliers, z_outliers)):
                if is_iqr:
                    outlier_records.append(
                        {"column": col, "value": v, "method": "IQR", "index": data.index[idx]}
                    )
                if is_z:
                    outlier_records.append(
                        {
                            "column": col,
                            "value": v,
                            "method": f"Z>{z_threshold}",
                            "index": data.index[idx],
                        }
                    )

            # ── Boxplot ──
            ax1 = axes[row][0]
            bp = ax1.boxplot(
                vals,
                patch_artist=True,
                medianprops=dict(color="white", lw=2),
                whiskerprops=dict(color=_GREY),
                capprops=dict(color=_GREY),
                flierprops=dict(marker="o", color=_RED, markersize=6, alpha=0.7),
            )
            bp["boxes"][0].set_facecolor(_BLUE)
            bp["boxes"][0].set_alpha(0.6)
            ax1.axhline(lower, color=_ORANGE, lw=1.2, ls="--", label=f"IQR lower={lower:,.0f}")
            ax1.axhline(upper, color=_ORANGE, lw=1.2, ls="--", label=f"IQR upper={upper:,.0f}")
            ax1.set_title(f"{col} – IQR outliers: {iqr_outliers.sum()}", fontsize=9)
            ax1.set_xticklabels([])
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax1.legend(fontsize=7)

            # ── Z-score strip plot ──
            ax2 = axes[row][1]
            x_jitter = np.random.uniform(-0.2, 0.2, len(vals))
            normal_mask = ~z_outliers
            ax2.scatter(
                x_jitter[normal_mask],
                vals[normal_mask],
                s=20,
                color=_BLUE,
                alpha=0.5,
                label="Normal",
            )
            ax2.scatter(
                x_jitter[z_outliers],
                vals[z_outliers],
                s=40,
                color=_RED,
                alpha=0.8,
                zorder=5,
                label=f"Z-outlier ({z_outliers.sum()})",
            )
            ax2.axhline(
                vals.mean() + z_threshold * vals.std(),
                color=_RED,
                lw=1.2,
                ls="--",
                label=f"+{z_threshold}σ",
            )
            ax2.axhline(
                vals.mean() - z_threshold * vals.std(),
                color=_RED,
                lw=1.2,
                ls="--",
                label=f"-{z_threshold}σ",
            )
            ax2.set_title(f"{col} – Z-score outliers: {z_outliers.sum()}", fontsize=9)
            ax2.set_xlim(-1, 1)
            ax2.set_xticklabels([])
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
            ax2.legend(fontsize=7)

        _save_or_show(fig, save_path, "08_outliers")

        outlier_df = pd.DataFrame(outlier_records).drop_duplicates()
        if not outlier_df.empty:
            print(f"\n  Outliers detected ({len(outlier_df)} flags):")
            print(outlier_df.to_string(index=False))
        else:
            print("\n  No outliers detected.")
        return outlier_df

    # ──────────────────────────────────────────────────────────────────────────
    # 9. Distribution fitting visualisation
    # ──────────────────────────────────────────────────────────────────────────

    def plot_distribution_fits(
        self,
        col: str,
        top_n: int = 5,
        candidates: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Comprehensive 4-panel distribution fitting chart for a single column.

        Panels
        ------
        Top-left  : Histogram + top-N fitted PDFs overlaid
        Top-right : ECDF + top-N fitted CDFs overlaid
        Bot-left  : Q-Q plot vs best-fitting distribution
        Bot-right : AIC / BIC bar chart for all candidates

        Parameters
        ----------
        col : str
            Column to analyse.
        top_n : int
            Number of best distributions to overlay. Default 5.
        candidates : list of str, optional
            Distributions to test. Defaults to the standard adsat candidate list.
        save_path : str, optional
        """
        from adsat.distribution import CANDIDATE_DISTRIBUTIONS

        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        data = self.df[col].dropna().values.astype(float)
        if len(data) < 5:
            print(f"  Too few observations in '{col}' to fit distributions.")
            return

        dist_names = candidates or CANDIDATE_DISTRIBUTIONS

        # ── Fit all candidates ────────────────────────────────────────────────
        fit_results = []
        n = len(data)

        for dist_name in dist_names:
            try:
                dist_obj = getattr(stats, dist_name)
                shifted = data.copy()

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
                    shift = max(0, -data.min() + 1e-6)
                    shifted = data + shift
                elif dist_name == "beta":
                    shifted = (data - data.min() + 1e-6) / (data.max() - data.min() + 2e-6)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    params = dist_obj.fit(shifted)

                ks_stat, ks_p = stats.kstest(shifted, dist_name, args=params)
                log_lik = np.sum(dist_obj.logpdf(shifted, *params))
                k = len(params)
                aic = 2 * k - 2 * log_lik
                bic = k * np.log(n) - 2 * log_lik

                fit_results.append(
                    {
                        "distribution": dist_name,
                        "params": params,
                        "shifted": shifted,
                        "ks_stat": ks_stat,
                        "ks_pvalue": ks_p,
                        "aic": aic,
                        "bic": bic,
                        "acceptable": ks_p > 0.05,
                    }
                )
            except Exception:
                pass

        if not fit_results:
            print(f"  No distributions could be fitted to '{col}'.")
            return

        fit_results.sort(key=lambda r: r["aic"])
        top_fits = fit_results[:top_n]
        best = fit_results[0]

        # ── Figure ────────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle(
            f"Distribution Fitting: {col}  "
            f"(best fit: {best['distribution']}, "
            f"AIC={best['aic']:.1f}, KS p={best['ks_pvalue']:.3f})",
            fontsize=12,
            fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
        ax_hist = fig.add_subplot(gs[0, 0])
        ax_ecdf = fig.add_subplot(gs[0, 1])
        ax_qq = fig.add_subplot(gs[1, 0])
        ax_aic = fig.add_subplot(gs[1, 1])

        colors = plt.cm.tab10(np.linspace(0, 0.9, top_n))
        x_range = np.linspace(data.min(), data.max(), 400)

        # Panel 1 – Histogram + PDFs
        ax_hist.hist(
            data,
            bins="auto",
            density=True,
            alpha=0.35,
            color=_GREY,
            edgecolor="white",
            label="Data",
        )
        try:
            kde = stats.gaussian_kde(data)
            ax_hist.plot(x_range, kde(x_range), color="black", lw=1.5, ls=":", label="KDE")
        except Exception:
            pass

        for fit, color in zip(top_fits, colors):
            try:
                dist_obj = getattr(stats, fit["distribution"])
                shifted_range = x_range.copy()
                if fit["distribution"] == "beta":
                    shifted_range = (x_range - data.min() + 1e-6) / (data.max() - data.min() + 2e-6)
                elif fit["distribution"] in (
                    "lognorm",
                    "expon",
                    "gamma",
                    "weibull_min",
                    "weibull_max",
                    "pareto",
                    "powerlaw",
                ):
                    shift = max(0, -data.min() + 1e-6)
                    shifted_range = x_range + shift

                pdf = dist_obj.pdf(shifted_range, *fit["params"])
                # Scale back to original data range for beta
                if fit["distribution"] == "beta":
                    pdf = pdf / (data.max() - data.min() + 2e-6)
                lw = 3 if fit == best else 1.5
                label = f"{fit['distribution']} ★" if fit == best else fit["distribution"]
                ax_hist.plot(x_range, pdf, lw=lw, color=color, label=label)
            except Exception:
                pass

        ax_hist.set_title("Histogram + Fitted PDFs", fontsize=9)
        ax_hist.set_xlabel(col, fontsize=8)
        ax_hist.set_ylabel("Density", fontsize=8)
        ax_hist.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
        ax_hist.legend(fontsize=7)

        # Panel 2 – ECDF + fitted CDFs
        sorted_data = np.sort(data)
        ecdf_y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax_ecdf.step(sorted_data, ecdf_y, where="post", color="black", lw=2, label="ECDF", zorder=5)

        for fit, color in zip(top_fits, colors):
            try:
                dist_obj = getattr(stats, fit["distribution"])
                shifted_range = x_range.copy()
                if fit["distribution"] == "beta":
                    shifted_range = (x_range - data.min() + 1e-6) / (data.max() - data.min() + 2e-6)
                elif fit["distribution"] in (
                    "lognorm",
                    "expon",
                    "gamma",
                    "weibull_min",
                    "weibull_max",
                    "pareto",
                    "powerlaw",
                ):
                    shift = max(0, -data.min() + 1e-6)
                    shifted_range = x_range + shift
                cdf = dist_obj.cdf(shifted_range, *fit["params"])
                lw = 2.5 if fit == best else 1.2
                ax_ecdf.plot(
                    x_range,
                    cdf,
                    lw=lw,
                    color=color,
                    label=fit["distribution"] + (" ★" if fit == best else ""),
                )
            except Exception:
                pass

        ax_ecdf.set_title("ECDF + Fitted CDFs", fontsize=9)
        ax_ecdf.set_xlabel(col, fontsize=8)
        ax_ecdf.set_ylabel("Cumulative Probability", fontsize=8)
        ax_ecdf.set_ylim(0, 1.05)
        ax_ecdf.xaxis.set_major_formatter(plt.FuncFormatter(_fmt_large))
        ax_ecdf.legend(fontsize=7)

        # Panel 3 – Q-Q plot vs best fit
        try:
            dist_obj = getattr(stats, best["distribution"])
            sparams = best["params"][:-2] if len(best["params"]) > 2 else best["params"]
            (osm, osr), (slope, intercept, r) = stats.probplot(
                best["shifted"], dist=dist_obj, sparams=sparams, fit=True
            )
            ax_qq.scatter(osm, osr, s=20, color=_BLUE, alpha=0.6, zorder=4)
            x_line = np.array([osm.min(), osm.max()])
            ax_qq.plot(x_line, slope * x_line + intercept, color=_RED, lw=2)
            ax_qq.fill_between(osm, osr, slope * osm + intercept, alpha=0.12, color=_ORANGE)
            ax_qq.set_title(f"Q-Q: {best['distribution']} (R={r:.3f})", fontsize=9)
            ax_qq.set_xlabel("Theoretical Quantiles", fontsize=8)
            ax_qq.set_ylabel("Sample Quantiles", fontsize=8)
        except Exception:
            ax_qq.text(
                0.5, 0.5, "Q-Q unavailable", ha="center", va="center", transform=ax_qq.transAxes
            )

        # Panel 4 – AIC / BIC ranking
        names = [r["distribution"] for r in fit_results]
        aics = [r["aic"] for r in fit_results]
        bics = [r["bic"] for r in fit_results]
        bar_colors = [_GREEN if r["acceptable"] else _RED for r in fit_results]

        y_pos = np.arange(len(names))
        ax_aic.barh(y_pos - 0.2, aics, height=0.35, color=bar_colors, alpha=0.75, label="AIC")
        ax_aic.barh(
            y_pos + 0.2, bics, height=0.35, color=bar_colors, alpha=0.45, label="BIC", hatch="//"
        )
        ax_aic.set_yticks(y_pos)
        ax_aic.set_yticklabels(names, fontsize=8)
        ax_aic.set_xlabel("Information Criterion (lower = better)", fontsize=8)
        ax_aic.set_title("AIC / BIC Ranking\n(green = KS acceptable, red = rejected)", fontsize=9)
        ax_aic.legend(fontsize=8)
        ax_aic.axvline(0, color="black", lw=0.5)

        _save_or_show(fig, save_path, f"09_dist_fits_{col}")


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience one-liner
# ═══════════════════════════════════════════════════════════════════════════════


def explore(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    x_col: str | None = None,
    y_col: str | None = None,
    campaign_col: str | None = None,
    date_col: str | None = None,
    save_path: str | None = None,
) -> CampaignExplorer:
    """
    One-function EDA shortcut — creates a CampaignExplorer and runs all plots.

    Parameters
    ----------
    df : pd.DataFrame
    numeric_cols : list of str, optional  – columns to analyse
    x_col : str, optional  – independent variable for scatter plot
    y_col : str, optional  – dependent variable for scatter plot
    campaign_col : str, optional  – campaign identifier column
    date_col : str, optional  – date/period column
    save_path : str, optional  – base path for saving all plots

    Returns
    -------
    CampaignExplorer  – the configured explorer instance (for further use)

    Examples
    --------
    >>> from adsat.exploratory import explore
    >>> explorer = explore(
    ...     df,
    ...     numeric_cols=["impressions", "conversions", "revenue"],
    ...     x_col="impressions",
    ...     y_col="conversions",
    ...     campaign_col="campaign_id",
    ...     date_col="date",
    ... )
    """
    explorer = CampaignExplorer(
        df=df,
        numeric_cols=numeric_cols,
        campaign_col=campaign_col,
        date_col=date_col,
    )
    explorer.explore(x_col=x_col, y_col=y_col, save_path=save_path)
    return explorer
