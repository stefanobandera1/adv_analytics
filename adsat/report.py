"""
adsat.report
============
Automated HTML report generation for advertising saturation analyses.

This module takes the outputs of the other adsat modules and renders a
polished, self-contained HTML report that can be shared with clients or
stakeholders who do not use Python.  The report embeds all charts as
base64 images, so it requires no external files or internet connection
to display correctly.

Key classes & functions
-----------------------
ReportBuilder       – main class; collects results and renders HTML
generate_report()   – one-liner convenience function

Typical workflow
----------------
>>> from adsat.report import ReportBuilder
>>>
>>> builder = ReportBuilder(title="Q3 Brand Campaign Analysis")
>>>
>>> # Add whichever results you have
>>> builder.add_campaign_batch(batch)                 # from CampaignSaturationAnalyzer
>>> builder.add_budget_allocation(budget_result)      # from BudgetOptimizer
>>> builder.add_response_curves(curve_results)        # from ResponseCurveAnalyzer
>>> builder.add_diagnostics(diag_reports)             # from ModelDiagnostics
>>> builder.add_seasonality(decomp_result)            # from SeasonalDecomposer
>>> builder.add_simulation(sim_result)                # from ScenarioSimulator
>>>
>>> builder.save("saturation_report.html")
>>> # Opens the file in your default browser automatically
>>>
>>> # One-liner: just needs a batch result
>>> from adsat.report import generate_report
>>> generate_report(batch, output_path="report.html")
"""

from __future__ import annotations

import base64
import io
import os
import webbrowser
from datetime import datetime
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── colour palette ────────────────────────────────────────────────────────────
_BLUE = "#2E86AB"
_ORANGE = "#E07B39"
_GREEN = "#3BB273"
_RED = "#E84855"
_GREY = "#6C757D"
_PURPLE = "#7B2D8B"
PALETTE = [_BLUE, _ORANGE, _GREEN, _RED, _PURPLE, _GREY, "#F4A261", "#264653", "#A8DADC", "#E9C46A"]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return encoded


def _embed_img(b64: str, alt: str = "chart", width: str = "100%") -> str:
    """
    Wrap a base64-encoded image in an <img> HTML tag with optional CSS class.
    """
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'alt="{alt}" style="width:{width};max-width:900px;'
        f'display:block;margin:auto;" />'
    )


def _fmt(v: Any, decimals: int = 2, pct: bool = False, sign: bool = False) -> str:
    """Format a numeric value for HTML table display."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        val = float(v)
        if pct:
            s = f"{val:+.{decimals}f}%" if sign else f"{val:.{decimals}f}%"
        elif sign:
            s = f"{val:+,.{decimals}f}"
        else:
            s = f"{val:,.{decimals}f}"
        return s
    except Exception:
        return str(v)


def _status_badge(status: str) -> str:
    """Return an HTML badge element for a saturation status string."""
    colour_map = {
        "below": ("#D1FAE5", "#065F46"),  # green
        "approaching": ("#FEF3C7", "#92400E"),  # amber
        "at": ("#FEE2E2", "#991B1B"),  # red
        "beyond": ("#F3E8FF", "#5B21B6"),  # purple
    }
    for key, (bg, fg) in colour_map.items():
        if key in str(status).lower():
            return (
                f'<span style="background:{bg};color:{fg};'
                f"padding:2px 8px;border-radius:12px;font-size:0.8em;"
                f'font-weight:600;">{status}</span>'
            )
    return f'<span style="color:{_GREY};">{status}</span>'


def _df_to_html(df: pd.DataFrame, highlight_col: str = None) -> str:
    """Convert a DataFrame to a styled HTML table."""
    rows_html = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            cell_style = "padding:6px 12px;text-align:right;white-space:nowrap;"
            if col in ("campaign_id", "model", "model_name"):
                cell_style = "padding:6px 12px;text-align:left;font-weight:600;"
            if col == "saturation_status":
                cells.append(f'<td style="padding:6px 12px;">{_status_badge(str(val))}</td>')
            elif isinstance(val, bool):
                icon = "✓" if val else "✗"
                colour = _GREEN if val else _RED
                cells.append(f'<td style="{cell_style}color:{colour};">{icon}</td>')
            elif val is None or (isinstance(val, float) and np.isnan(float(val))):
                cells.append(f'<td style="{cell_style}color:{_GREY};">—</td>')
            else:
                cells.append(f'<td style="{cell_style}">{val}</td>')
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    headers = "".join(
        f'<th style="padding:8px 12px;text-align:{"left" if c in ("campaign_id","model","model_name") else "right"}'
        f';border-bottom:2px solid #E5E7EB;white-space:nowrap;">{c}</th>'
        for c in df.columns
    )
    table = f"""
    <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;font-size:0.88em;font-family:monospace;">
      <thead style="background:#F9FAFB;">
        <tr>{headers}</tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
    </div>"""
    return table


# ─────────────────────────────────────────────────────────────────────────────
# HTML template helpers
# ─────────────────────────────────────────────────────────────────────────────


def _section(title: str, content: str, icon: str = "📊") -> str:
    """
    Wrap content in a <section> HTML block with a heading and optional collapsible toggle.
    """
    return f"""
    <section style="margin:2rem 0;padding:1.5rem;background:white;
                    border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,0.08);">
      <h2 style="margin:0 0 1rem;font-size:1.25rem;color:#111827;
                 border-bottom:2px solid #E5E7EB;padding-bottom:0.5rem;">
        {icon} {title}
      </h2>
      {content}
    </section>"""


def _metric_card(label: str, value: str, sub: str = "", colour: str = _BLUE) -> str:
    """
    Render a single KPI metric card: label on top, large value below.
    """
    return f"""
    <div style="background:{colour}10;border-left:4px solid {colour};
                border-radius:8px;padding:1rem 1.5rem;flex:1;min-width:140px;">
      <div style="font-size:0.8rem;color:{_GREY};text-transform:uppercase;
                  letter-spacing:0.05em;">{label}</div>
      <div style="font-size:1.8rem;font-weight:700;color:#111827;margin:4px 0;">{value}</div>
      <div style="font-size:0.78rem;color:{_GREY};">{sub}</div>
    </div>"""


def _page_css() -> str:
    """
    Return the full inline CSS stylesheet used by the generated HTML report.
    """
    return """
    <style>
      * { box-sizing: border-box; margin: 0; padding: 0; }
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     Helvetica, Arial, sans-serif;
        background: #F3F4F6;
        color: #1F2937;
        line-height: 1.6;
      }
      .container { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem; }
      .header {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E86AB 100%);
        color: white; padding: 2.5rem 2rem; border-radius: 16px;
        margin-bottom: 2rem;
      }
      .header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
      .header .subtitle { font-size: 1rem; opacity: 0.8; }
      .header .meta { font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem; }
      .metric-row {
        display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem;
      }
      .callout {
        background: #EFF6FF; border-left: 4px solid #2563EB;
        border-radius: 6px; padding: 0.75rem 1rem;
        font-size: 0.88rem; color: #1E40AF; margin: 0.75rem 0;
      }
      .warn-callout {
        background: #FFF7ED; border-left: 4px solid #EA580C;
        border-radius: 6px; padding: 0.75rem 1rem;
        font-size: 0.88rem; color: #9A3412; margin: 0.75rem 0;
      }
      table tr:hover { background: #F9FAFB; }
      footer {
        text-align: center; font-size: 0.78rem; color: #9CA3AF;
        margin-top: 3rem; padding: 1.5rem;
      }
      @media print {
        body { background: white; }
        .container { max-width: 100%; }
        section { box-shadow: none; border: 1px solid #E5E7EB; }
      }
    </style>"""


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class ReportBuilder:
    """
    Collect analysis results from multiple adsat modules and render
    a polished, self-contained HTML report.

    All charts are embedded as base64 PNG images — no external files needed.
    The report is print-friendly and displays cleanly in all modern browsers.

    Parameters
    ----------
    title : str
        Report title shown in the header. Default 'Advertising Saturation Report'.
    subtitle : str, optional
        Optional subtitle / client name shown below the title.
    author : str, optional
        Name shown in the report footer.
    dpi : int
        Resolution for embedded charts. Default 110.

    Examples
    --------
    >>> builder = ReportBuilder("Q3 Brand Campaign – Saturation Analysis",
    ...                         subtitle="Acme Corp")
    >>> builder.add_campaign_batch(batch)
    >>> builder.add_budget_allocation(budget_result)
    >>> builder.save("report.html")
    """

    def __init__(
        self,
        title: str = "Advertising Saturation Report",
        subtitle: str = "",
        author: str = "",
        dpi: int = 110,
    ):
        """
        Initialise an empty report builder with a title and optional subtitle.
        Sections are added incrementally via add_* methods before calling save().
        """
        self.title = title
        self.subtitle = subtitle
        self.author = author
        self.dpi = dpi

        # Accumulated sections (list of HTML strings, in insertion order)
        self._sections: list[str] = []

        # Summary stats gathered from added results
        self._summary_stats: dict[str, Any] = {}

    # ── public: add results ────────────────────────────────────────────────────

    def add_campaign_batch(
        self,
        batch,
        title: str = "Campaign Saturation Analysis",
    ) -> ReportBuilder:
        """
        Add a saturation analysis section from a CampaignBatchResult.

        Parameters
        ----------
        batch : CampaignBatchResult
            Output of CampaignSaturationAnalyzer.run().
        title : str

        Returns
        -------
        self  (for method chaining)
        """

        rows = []
        for cid in batch.campaign_results:
            cr = batch.get(cid)
            rows.append(
                {
                    "campaign_id": cr.campaign_id,
                    "n_obs": cr.n_observations,
                    "best_model": cr.best_model or "—",
                    "r2": _fmt(cr.r2, 4) if cr.r2 is not None else "—",
                    "rmse": _fmt(cr.rmse, 2) if cr.rmse is not None else "—",
                    "saturation_point": (
                        _fmt(cr.saturation_point, 0) if cr.saturation_point else "—"
                    ),
                    "pct_of_saturation": (
                        _fmt(cr.pct_of_saturation, 1) if cr.pct_of_saturation else "—"
                    ),
                    "saturation_status": (
                        cr.saturation_status
                        if hasattr(cr, "saturation_status") and cr.saturation_status
                        else "—"
                    ),
                    "succeeded": cr.succeeded,
                }
            )

        df = pd.DataFrame(rows)

        # Summary metrics
        n_total = batch.n_total if hasattr(batch, "n_total") else len(rows)
        n_succ = batch.n_succeeded if hasattr(batch, "n_succeeded") else df["succeeded"].sum()
        n_fail = batch.n_failed if hasattr(batch, "n_failed") else (n_total - n_succ)

        self._summary_stats.update(
            {
                "n_campaigns": n_total,
                "n_succeeded": n_succ,
                "n_failed": n_fail,
            }
        )

        # Metric cards
        metric_cards = f"""
        <div class="metric-row">
          {_metric_card("Campaigns", str(n_total), "total", _BLUE)}
          {_metric_card("Succeeded", str(n_succ), "fitted OK", _GREEN)}
          {_metric_card("Failed", str(n_fail), "insufficient data", _RED if n_fail > 0 else _GREY)}
        </div>"""

        # Saturation status summary
        if "saturation_status" in df.columns:
            status_counts = df["saturation_status"].value_counts()
            status_items = " · ".join(f"<strong>{v}</strong> {k}" for k, v in status_counts.items())
            callout = f'<div class="callout">Status summary: {status_items}</div>'
        else:
            callout = ""

        # Chart: best model distribution
        model_fig = self._plot_model_distribution(df)
        model_b64 = _fig_to_base64(model_fig, self.dpi)
        plt.close(model_fig)

        # Chart: saturation status bar
        status_fig = self._plot_saturation_overview(df)
        status_b64 = _fig_to_base64(status_fig, self.dpi)
        plt.close(status_fig)

        charts = f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1rem 0;">
          <div>{_embed_img(model_b64, "Model distribution")}</div>
          <div>{_embed_img(status_b64, "Saturation overview")}</div>
        </div>"""

        content = metric_cards + callout + charts + _df_to_html(df)
        self._sections.append(_section(title, content, "🎯"))
        return self

    def add_budget_allocation(
        self,
        result,
        title: str = "Budget Optimisation",
    ) -> ReportBuilder:
        """
        Add a budget optimisation section from a BudgetAllocation result.

        Parameters
        ----------
        result : BudgetAllocation
            Output of BudgetOptimizer.optimise().
        title : str
        """
        df = result.allocations.copy()

        uplift_pct = result.total_outcome_lift_pct
        uplift_col = _GREEN if uplift_pct >= 0 else _RED

        metric_cards = f"""
        <div class="metric-row">
          {_metric_card("Total Budget", _fmt(result.total_budget, 0), "allocated", _BLUE)}
          {_metric_card("Outcome Uplift",
                        f"{uplift_pct:+.1f}%",
                        _fmt(result.total_outcome_lift, 0, sign=True) + " units",
                        uplift_col)}
          {_metric_card("Converged", "Yes" if result.converged else "No",
                        "solver status", _GREEN if result.converged else _RED)}
        </div>"""

        if not result.converged:
            note = (
                f'<div class="warn-callout">⚠ Solver did not fully converge: '
                f"{result.notes}. Results should be treated as approximate.</div>"
            )
        else:
            note = (
                '<div class="callout">✓ Optimisation converged successfully. '
                "Marginal returns are equalised across campaigns at the optimum.</div>"
            )

        # Chart: budget allocation comparison
        fig = self._plot_budget_comparison(df, result.total_budget, uplift_pct)
        b64 = _fig_to_base64(fig, self.dpi)
        plt.close(fig)

        # Format the allocation table for display
        disp_df = df.copy()
        for col in ("current_spend", "optimal_spend", "spend_change"):
            if col in disp_df.columns:
                disp_df[col] = disp_df[col].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
        for col in ("spend_change_pct", "outcome_lift_pct"):
            if col in disp_df.columns:
                disp_df[col] = disp_df[col].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else "—")
        for col in ("current_outcome", "optimal_outcome", "outcome_lift"):
            if col in disp_df.columns:
                disp_df[col] = disp_df[col].apply(lambda v: f"{v:,.1f}" if pd.notna(v) else "—")
        for col in ("pct_of_saturation_before", "pct_of_saturation_after"):
            if col in disp_df.columns:
                disp_df[col] = disp_df[col].apply(lambda v: f"{v:.1f}%" if pd.notna(v) else "—")

        content = (
            metric_cards
            + note
            + _embed_img(b64, "Budget allocation")
            + "<br>"
            + _df_to_html(disp_df)
        )
        self._sections.append(_section(title, content, "💰"))
        return self

    def add_response_curves(
        self,
        results: dict,
        title: str = "Response Curve Analysis",
    ) -> ReportBuilder:
        """
        Add a response curves section from ResponseCurveAnalyzer results.

        Parameters
        ----------
        results : dict {campaign_id: ResponseCurveResult}
            Output of ResponseCurveAnalyzer.analyse().
        title : str
        """
        rows = []
        for cid, res in results.items():
            rows.append(
                {
                    "campaign_id": cid,
                    "asymptote": _fmt(res.asymptote, 0) if res.asymptote else "—",
                    "saturation_point": (
                        _fmt(res.saturation_point, 0) if res.saturation_point else "—"
                    ),
                    "current_spend": _fmt(res.current_x, 0) if res.current_x else "—",
                    "current_outcome": _fmt(res.current_y, 1) if res.current_y else "—",
                    "marginal_return": (
                        _fmt(res.current_marginal_return, 5)
                        if res.current_marginal_return is not None
                        else "—"
                    ),
                    "elasticity": (
                        _fmt(res.current_elasticity, 3)
                        if res.current_elasticity is not None
                        else "—"
                    ),
                    "pct_saturation": (
                        _fmt(
                            res.saturation_point
                            and res.current_x
                            and res.current_x / res.saturation_point * 100,
                            1,
                        )
                        if res.saturation_point and res.current_x
                        else "—"
                    ),
                    "inflection_pt": (
                        _fmt(res.inflection_point_x, 0) if res.inflection_point_x else "—"
                    ),
                }
            )

        df = pd.DataFrame(rows)

        # Chart: multi-panel response curves
        fig = self._plot_response_curves_for_report(results)
        b64 = _fig_to_base64(fig, self.dpi)
        plt.close(fig)

        callout = (
            '<div class="callout">Elasticity &lt; 1 indicates diminishing returns. '
            "Marginal return shows the incremental outcome per unit of additional spend "
            "at the current spend level.</div>"
        )

        content = _embed_img(b64, "Response curves") + "<br>" + callout + _df_to_html(df)
        self._sections.append(_section(title, content, "📈"))
        return self

    def add_diagnostics(
        self,
        reports: dict,
        title: str = "Model Diagnostics",
    ) -> ReportBuilder:
        """
        Add a model diagnostics section from ModelDiagnostics.run_all() results.

        Parameters
        ----------
        reports : dict {model_name: DiagnosticsReport}
            Output of ModelDiagnostics.run_all().
        title : str
        """
        rows = []
        for name, rep in reports.items():
            rows.append(
                {
                    "model": name,
                    "n": rep.n,
                    "shapiro_p": _fmt(rep.shapiro_pvalue, 4),
                    "ks_p": _fmt(rep.ks_pvalue, 4),
                    "jb_p": _fmt(rep.jb_pvalue, 4),
                    "normality_ok": rep.normality_ok,
                    "durbin_watson": _fmt(rep.durbin_watson, 3),
                    "autocorr_ok": rep.autocorrelation_ok,
                    "levene_p": _fmt(rep.levene_pvalue, 4),
                    "homo_ok": rep.homoscedasticity_ok,
                    "n_influential": rep.n_high_influence,
                    "overall_ok": rep.overall_ok,
                }
            )

        df = pd.DataFrame(rows)
        n_pass = sum(r.overall_ok for r in reports.values())
        n_fail = len(reports) - n_pass

        metric_cards = f"""
        <div class="metric-row">
          {_metric_card("Models Tested", str(len(reports)), "", _BLUE)}
          {_metric_card("Passed", str(n_pass), "diagnostics OK", _GREEN)}
          {_metric_card("Issues", str(n_fail), "review warnings", _RED if n_fail > 0 else _GREY)}
        </div>"""

        # Collect warnings
        all_warnings = []
        for name, rep in reports.items():
            for w in rep.warnings:
                all_warnings.append(f"<strong>{name}</strong>: {w}")

        warn_html = ""
        if all_warnings:
            items = "".join(f"<li style='margin:0.3rem 0;'>{w}</li>" for w in all_warnings)
            warn_html = (
                f'<div class="warn-callout"><ul style="margin:0;'
                f'padding-left:1.2rem;">{items}</ul></div>'
            )

        content = metric_cards + warn_html + _df_to_html(df)
        self._sections.append(_section(title, content, "🔬"))
        return self

    def add_seasonality(
        self,
        result,
        title: str = "Seasonality Analysis",
    ) -> ReportBuilder:
        """
        Add a seasonality decomposition section from SeasonalDecomposer.

        Parameters
        ----------
        result : SeasonalDecomposition
            Output of SeasonalDecomposer.fit().
        title : str
        """
        metric_cards = f"""
        <div class="metric-row">
          {_metric_card("Model", result.model.title(), "", _BLUE)}
          {_metric_card("Period", str(result.period), "observations", _PURPLE)}
          {_metric_card("Seasonal Strength",
                        f"{result.strength_of_seasonality:.1%}",
                        "of total variance", _ORANGE)}
          {_metric_card("Observations", str(result.n), "", _GREY)}
        </div>"""

        strength = result.strength_of_seasonality
        if strength > 0.3:
            callout_cls = "warn-callout"
            callout_msg = (
                f"⚠ Strong seasonality detected ({strength:.1%} of variance). "
                "It is recommended to fit saturation models on the seasonally-adjusted "
                "series to avoid biased saturation estimates."
            )
        else:
            callout_cls = "callout"
            callout_msg = (
                f"✓ Mild seasonality ({strength:.1%} of variance). "
                "Direct saturation modelling may be acceptable, but using the "
                "adjusted series will still improve estimate quality."
            )

        callout = f'<div class="{callout_cls}">{callout_msg}</div>'

        # Chart: decomposition
        fig = self._plot_decomp_for_report(result)
        b64 = _fig_to_base64(fig, self.dpi)
        plt.close(fig)

        # Seasonal factors table
        factor_df = pd.DataFrame(
            {
                "period_position": result.period_labels,
                "seasonal_factor": [_fmt(f, 3, sign=True) for f in result.seasonal_factors],
            }
        )

        content = (
            metric_cards
            + callout
            + _embed_img(b64, "Decomposition")
            + "<br>"
            + "<h3 style='margin:1rem 0 0.5rem;font-size:1rem;'>Seasonal Factors</h3>"
            + _df_to_html(factor_df)
        )
        self._sections.append(_section(title, content, "🌊"))
        return self

    def add_simulation(
        self,
        result,
        title: str = "Scenario Simulation",
    ) -> ReportBuilder:
        """
        Add a scenario simulation section from ScenarioSimulator results.

        Parameters
        ----------
        result : SimulationResult
            Output of ScenarioSimulator.simulate().
        title : str
        """
        # Render the scenarios summary table
        df = result.summary_table if hasattr(result, "summary_table") else pd.DataFrame()

        metric_cards = f"""
        <div class="metric-row">
          {_metric_card("Scenarios", str(len(result.scenarios)) if hasattr(result, "scenarios") else "—", "evaluated", _BLUE)}
          {_metric_card("Best Scenario",
                        str(result.best_scenario_name) if hasattr(result, "best_scenario_name") else "—",
                        "highest outcome", _GREEN)}
        </div>"""

        # Chart: scenario comparison
        if hasattr(result, "_fig_b64") and result._fig_b64:
            chart_html = _embed_img(result._fig_b64, "Scenario comparison")
        else:
            fig = self._plot_scenario_for_report(result)
            b64 = _fig_to_base64(fig, self.dpi)
            plt.close(fig)
            chart_html = _embed_img(b64, "Scenario comparison")

        content = metric_cards + chart_html + "<br>" + _df_to_html(df)
        self._sections.append(_section(title, content, "🔮"))
        return self

    def add_custom_section(
        self,
        title: str,
        content_html: str,
        icon: str = "📋",
    ) -> ReportBuilder:
        """
        Add a fully custom HTML section to the report.

        Parameters
        ----------
        title : str
        content_html : str  — raw HTML for the section body
        icon : str          — emoji icon shown next to the title
        """
        self._sections.append(_section(title, content_html, icon))
        return self

    def add_figure(
        self,
        fig: plt.Figure,
        title: str,
        description: str = "",
        icon: str = "📊",
    ) -> ReportBuilder:
        """
        Embed any matplotlib Figure as a section in the report.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        title : str
        description : str — optional text shown above the chart
        icon : str
        """
        b64 = _fig_to_base64(fig, self.dpi)
        desc_html = (
            (
                f'<p style="color:{_GREY};font-size:0.9rem;margin-bottom:0.75rem;">'
                f"{description}</p>"
            )
            if description
            else ""
        )
        content = desc_html + _embed_img(b64, title)
        self._sections.append(_section(title, content, icon))
        return self

    def save(
        self,
        output_path: str = "adsat_report.html",
        open_browser: bool = False,
    ) -> str:
        """
        Render and save the report to an HTML file.

        Parameters
        ----------
        output_path : str
            File path for the HTML output. Default 'adsat_report.html'.
        open_browser : bool
            If True, opens the file in the default web browser after saving.

        Returns
        -------
        str — absolute path to the saved file.
        """
        html = self._render_html()
        abs_path = os.path.abspath(output_path)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[ReportBuilder] Report saved → {abs_path}")
        if open_browser:
            webbrowser.open(f"file://{abs_path}")
        return abs_path

    def get_html(self) -> str:
        """Return the rendered HTML as a string (without saving to disk)."""
        return self._render_html()

    # ── internal rendering ────────────────────────────────────────────────────

    def _render_html(self) -> str:
        """
        Assemble the complete HTML report string from all accumulated sections.
        Includes the page CSS, header, all section blocks, and a footer timestamp.
        """
        now = datetime.now().strftime("%d %b %Y, %H:%M")
        author_txt = f" · {self.author}" if self.author else ""
        subtitle_html = f'<div class="subtitle">{self.subtitle}</div>' if self.subtitle else ""

        header = f"""
        <div class="header">
          <h1>{self.title}</h1>
          {subtitle_html}
          <div class="meta">Generated {now}{author_txt} · adsat</div>
        </div>"""

        sections_html = "\n".join(self._sections)

        footer = f"""
        <footer>
          Generated by <strong>adsat</strong> — Advertising Saturation Toolkit ·
          {now}{author_txt}
        </footer>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{self.title}</title>
  {_page_css()}
</head>
<body>
  <div class="container">
    {header}
    {sections_html}
    {footer}
  </div>
</body>
</html>"""

    # ── internal chart builders ───────────────────────────────────────────────

    def _plot_model_distribution(self, df: pd.DataFrame) -> plt.Figure:
        """Bar chart of model type usage across campaigns."""
        fig, ax = plt.subplots(figsize=(5, 3))
        if "best_model" in df.columns:
            counts = df["best_model"].value_counts()
            colors = (PALETTE * ((len(counts) // len(PALETTE)) + 1))[: len(counts)]
            ax.bar(
                counts.index.astype(str), counts.values, color=colors, alpha=0.85, edgecolor="white"
            )
            ax.set_title("Best Model Distribution", fontsize=10, fontweight="bold")
            ax.set_ylabel("Count", fontsize=9)
            ax.tick_params(axis="x", rotation=20, labelsize=8)
        else:
            ax.text(0.5, 0.5, "No model data", ha="center", va="center", transform=ax.transAxes)
        plt.tight_layout()
        return fig

    def _plot_saturation_overview(self, df: pd.DataFrame) -> plt.Figure:
        """Horizontal bar chart of saturation % per campaign."""
        fig, ax = plt.subplots(figsize=(5, max(3, len(df) * 0.5)))
        if "pct_of_saturation" in df.columns:
            pcts = []
            labels = []
            for _, row in df.iterrows():
                try:
                    v = float(str(row["pct_of_saturation"]).replace("%", "").replace("—", ""))
                    pcts.append(min(v, 200))
                    labels.append(str(row["campaign_id"]))
                except Exception:
                    pass
            if pcts:
                colors = [_RED if v >= 90 else (_ORANGE if v >= 70 else _GREEN) for v in pcts]
                ax.barh(labels, pcts, color=colors, alpha=0.85, edgecolor="white")
                ax.axvline(90, color=_RED, lw=1.5, ls="--", label="90% (near sat.)")
                ax.axvline(100, color="darkred", lw=1.2, ls=":", alpha=0.7, label="100% (sat.)")
                ax.set_xlabel("% of saturation point reached", fontsize=9)
                ax.set_title("Saturation Level per Campaign", fontsize=10, fontweight="bold")
                ax.legend(fontsize=7)
        else:
            ax.text(
                0.5, 0.5, "No saturation % data", ha="center", va="center", transform=ax.transAxes
            )
        plt.tight_layout()
        return fig

    def _plot_budget_comparison(
        self, df: pd.DataFrame, total_budget: float, uplift_pct: float
    ) -> plt.Figure:
        """Grouped bar: current vs optimal spend and outcome."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(
            f"Budget Optimisation  |  Outcome uplift: {uplift_pct:+.1f}%",
            fontsize=11,
            fontweight="bold",
        )
        ids = df["campaign_id"].astype(str).tolist()
        x = np.arange(len(ids))
        w = 0.38

        # Spend
        def _try_float(col):
            """
            Attempt to convert x to float; return fallback value on failure.
            """
            vals = []
            for v in df[col]:
                try:
                    vals.append(float(str(v).replace(",", "")))
                except Exception:
                    vals.append(0.0)
            return vals

        cur_s = _try_float("current_spend")
        opt_s = _try_float("optimal_spend")
        ax1.bar(x - w / 2, cur_s, w, color=_GREY, alpha=0.8, label="Current")
        ax1.bar(x + w / 2, opt_s, w, color=_BLUE, alpha=0.85, label="Optimal")
        ax1.set_xticks(x)
        ax1.set_xticklabels(ids, rotation=20, ha="right", fontsize=8)
        ax1.set_ylabel("Spend", fontsize=9)
        ax1.set_title("Budget Allocation", fontsize=10)
        ax1.legend(fontsize=8)
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}")
        )

        # Outcome
        cur_o = _try_float("current_outcome")
        opt_o = _try_float("optimal_outcome")
        ax2.bar(x - w / 2, cur_o, w, color=_GREY, alpha=0.8, label="Current")
        ax2.bar(x + w / 2, opt_o, w, color=_GREEN, alpha=0.85, label="Optimal")
        ax2.set_xticks(x)
        ax2.set_xticklabels(ids, rotation=20, ha="right", fontsize=8)
        ax2.set_ylabel("Outcome", fontsize=9)
        ax2.set_title("Outcome Comparison", fontsize=10)
        ax2.legend(fontsize=8)

        plt.tight_layout()
        return fig

    def _plot_response_curves_for_report(self, results: dict) -> plt.Figure:
        """Grid of response curves for report embedding."""

        n = len(results)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows))
        axes = np.array(axes).flatten()
        fig.suptitle("Response Curves", fontsize=12, fontweight="bold")

        colors = PALETTE * ((n // len(PALETTE)) + 1)
        for ax, (cid, res), color in zip(axes, results.items(), colors):
            xs = res.x_values
            ys = res.y_values
            ax.plot(xs, ys, color=color, lw=2)
            if res.current_x and res.current_y:
                ax.scatter(
                    [res.current_x], [res.current_y], s=60, color="black", zorder=5, label="Current"
                )
            if res.saturation_point:
                ax.axvline(
                    res.saturation_point,
                    color=_RED,
                    lw=1.2,
                    ls="--",
                    alpha=0.7,
                    label="Saturation pt",
                )
            ax.set_title(str(cid), fontsize=9, fontweight="bold")
            ax.set_xlabel("Spend", fontsize=7)
            ax.set_ylabel("Outcome", fontsize=7)
            ax.legend(fontsize=7)
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v >= 1000 else f"{v:.0f}")
            )

        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        return fig

    def _plot_decomp_for_report(self, result) -> plt.Figure:
        """Compact 2-panel decomposition chart for report embedding."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
        idx = np.arange(result.n)

        # Original + trend
        ax1.plot(idx, result.original, color=_BLUE, lw=1.5, alpha=0.6, label="Original")
        ax1.plot(idx, result.trend, color=_RED, lw=2, label="Trend")
        ax1.set_title("Original + Trend", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=8)
        ax1.set_xlabel("Time index", fontsize=8)

        # Seasonal factors bar chart
        factors = result.seasonal_factors
        colors = [_GREEN if f >= 0 else _RED for f in factors]
        ax2.bar(np.arange(len(factors)), factors, color=colors, alpha=0.8, edgecolor="white")
        ref = 0 if result.model == "additive" else 1
        ax2.axhline(ref, color=_GREY, lw=1, ls="--")
        ax2.set_title(
            f"Seasonal Factors  (period={result.period}  "
            f"strength={result.strength_of_seasonality:.1%})",
            fontsize=10,
            fontweight="bold",
        )
        ax2.set_xlabel("Period position", fontsize=8)
        ax2.set_ylabel("Factor", fontsize=8)

        plt.tight_layout()
        return fig

    def _plot_scenario_for_report(self, result) -> plt.Figure:
        """Fallback scenario comparison chart."""
        fig, ax = plt.subplots(figsize=(8, 4))
        if hasattr(result, "summary_table") and not result.summary_table.empty:
            df = result.summary_table
            if "scenario_name" in df.columns and "total_outcome" in df.columns:
                colors = [_GREEN if i == 0 else _BLUE for i in range(len(df))]
                ax.bar(
                    df["scenario_name"].astype(str),
                    df["total_outcome"],
                    color=colors,
                    alpha=0.85,
                    edgecolor="white",
                )
                ax.set_ylabel("Total Outcome", fontsize=9)
                ax.set_title("Scenario Comparison", fontsize=11, fontweight="bold")
                ax.tick_params(axis="x", rotation=20)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Scenario data unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "No scenario data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────


def generate_report(
    batch,
    output_path: str = "adsat_report.html",
    title: str = "Advertising Saturation Report",
    subtitle: str = "",
    author: str = "",
    budget_result=None,
    curve_results: dict = None,
    diag_reports: dict = None,
    seasonality_result=None,
    simulation_result=None,
    open_browser: bool = False,
) -> str:
    """
    One-function report generator.

    Parameters
    ----------
    batch : CampaignBatchResult
        Required. Output of CampaignSaturationAnalyzer.run().
    output_path : str
        Output file path. Default 'adsat_report.html'.
    title : str
    subtitle : str
    author : str
    budget_result : BudgetAllocation, optional
    curve_results : dict, optional
        {campaign_id: ResponseCurveResult}
    diag_reports : dict, optional
        {model_name: DiagnosticsReport}
    seasonality_result : SeasonalDecomposition, optional
    simulation_result : SimulationResult, optional
    open_browser : bool

    Returns
    -------
    str — absolute path to the saved HTML file.

    Examples
    --------
    >>> from adsat.report import generate_report
    >>> path = generate_report(batch, title="Q3 Analysis", output_path="report.html")
    """
    builder = ReportBuilder(title=title, subtitle=subtitle, author=author)
    builder.add_campaign_batch(batch)

    if budget_result is not None:
        builder.add_budget_allocation(budget_result)
    if curve_results:
        builder.add_response_curves(curve_results)
    if diag_reports:
        builder.add_diagnostics(diag_reports)
    if seasonality_result is not None:
        builder.add_seasonality(seasonality_result)
    if simulation_result is not None:
        builder.add_simulation(simulation_result)

    return builder.save(output_path, open_browser=open_browser)
