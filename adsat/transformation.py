"""
adsat.transformation
====================
Apply and reverse data transformations to prepare campaign metrics for modeling.

Key class: DataTransformer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

SUPPORTED_TRANSFORMS = [
    "none",
    "log",
    "log1p",
    "sqrt",
    "cbrt",
    "reflect_log",
    "reflect_sqrt",
    "boxcox",
    "yeo_johnson",
    "quantile_normal",
    "standard",
    "minmax",
    "robust",
]


@dataclass
class TransformRecord:
    """Stores metadata needed to reverse a transformation."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    # fitted sklearn transformer (if any)
    transformer: Any = None
    # shift applied before log/sqrt to ensure positivity
    shift: float = 0.0
    # reflection constant (max + 1) for reflect_* methods
    reflect_const: float = 0.0


class DataTransformer:
    """
    Apply the appropriate transformation to make data suitable for saturation modeling.

    Transformations can be specified manually or selected automatically based on
    the output of DistributionAnalyzer (via the ``recommended_transform`` field
    of each ``ColumnDistributionReport``).

    Parameters
    ----------
    strategy : str or dict
        Either a single strategy applied to all columns (e.g. 'log') or a
        column-level mapping (e.g. {'impressions': 'log', 'revenue': 'boxcox'}).
        Use 'auto' to delegate to DistributionAnalyzer recommendations.
    epsilon : float
        Small constant added before log/sqrt to prevent log(0).

    Examples
    --------
    >>> from adsat import DataTransformer
    >>> transformer = DataTransformer(strategy='log')
    >>> df_t = transformer.fit_transform(df, columns=['impressions', 'conversions'])
    >>> df_orig = transformer.inverse_transform(df_t, columns=['impressions', 'conversions'])
    """

    def __init__(
        self,
        strategy="auto",
        epsilon: float = 1e-6,
    ):
        """
        Store the transformation strategy and epsilon for numerical stability.

        strategy : "auto" (use DistributionAnalyzer recommendations), a single method
                   name ("log", "log1p", "sqrt", "yeo_johnson", "box_cox", "none"),
                   or a dict mapping column name -> method.
        epsilon  : small value added before log transforms to avoid log(0).
        """
        self.strategy = strategy
        self.epsilon = epsilon
        self._records: dict[str, TransformRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
        distribution_reports: dict | None = None,
    ) -> pd.DataFrame:
        """
        Fit transformations and return transformed DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list of str, optional  — defaults to all numeric columns.
        distribution_reports : dict, optional
            Output of DistributionAnalyzer.analyze(). Required when strategy='auto'.

        Returns
        -------
        pd.DataFrame with transformed columns (suffixed with ``_t``).
        """
        cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        df_out = df.copy()

        for col in cols:
            method = self._resolve_method(col, distribution_reports)
            series = df[col].dropna().values.astype(float)
            transformed, record = self._apply(series, method, fit=True)
            self._records[col] = record
            df_out[f"{col}_t"] = np.nan
            df_out.loc[df[col].notna(), f"{col}_t"] = transformed

        return df_out

    def transform(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Apply previously fitted transformations to new data."""
        cols = columns or list(self._records.keys())
        df_out = df.copy()

        for col in cols:
            if col not in self._records:
                raise KeyError(f"No fitted transform for column '{col}'. Call fit_transform first.")
            record = self._records[col]
            series = df[col].dropna().values.astype(float)
            transformed, _ = self._apply(series, record.method, fit=False, record=record)
            df_out[f"{col}_t"] = np.nan
            df_out.loc[df[col].notna(), f"{col}_t"] = transformed

        return df_out

    def inverse_transform(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Reverse the transformation to return predictions back to the original scale.

        Parameters
        ----------
        df : pd.DataFrame  — contains columns with ``_t`` suffix.
        columns : list of str  — original column names (without ``_t`` suffix).
        """
        cols = columns or list(self._records.keys())
        df_out = df.copy()

        for col in cols:
            t_col = f"{col}_t"
            if t_col not in df.columns:
                raise KeyError(f"Column '{t_col}' not found in DataFrame.")
            record = self._records[col]
            series = df[t_col].dropna().values.astype(float)
            original = self._reverse(series, record)
            df_out[f"{col}_inv"] = np.nan
            df_out.loc[df[t_col].notna(), f"{col}_inv"] = original

        return df_out

    def get_transform_summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising the transform applied to each column."""
        rows = []
        for col, rec in self._records.items():
            rows.append(
                {
                    "column": col,
                    "method": rec.method,
                    "shift": rec.shift,
                    "reflect_const": rec.reflect_const,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_method(self, col: str, distribution_reports) -> str:
        """
        Determine which transformation to apply to col.

        Resolution priority: (1) per-column dict override, (2) DistributionAnalyzer
        recommendation when strategy="auto", (3) global strategy string.
        """
        if isinstance(self.strategy, dict):
            return self.strategy.get(col, "none")
        if self.strategy == "auto":
            if distribution_reports and col in distribution_reports:
                return distribution_reports[col].recommended_transform
            return "yeo_johnson"
        return self.strategy  # single global method

    def _apply(
        self,
        data: np.ndarray,
        method: str,
        fit: bool = True,
        record: TransformRecord | None = None,
    ) -> tuple[np.ndarray, TransformRecord]:
        """Apply a transformation; return (transformed_data, TransformRecord)."""

        if record is None:
            record = TransformRecord(method=method)

        if method == "none":
            return data.copy(), record

        elif method == "log":
            shift = max(0, -data.min() + self.epsilon) if fit else record.shift
            record.shift = shift
            return np.log(data + shift), record

        elif method == "log1p":
            shift = max(0, -data.min() + self.epsilon) if fit else record.shift
            record.shift = shift
            return np.log1p(data + shift), record

        elif method == "sqrt":
            shift = max(0, -data.min() + self.epsilon) if fit else record.shift
            record.shift = shift
            return np.sqrt(data + shift), record

        elif method == "cbrt":
            return np.cbrt(data), record

        elif method == "reflect_log":
            if fit:
                record.reflect_const = float(data.max()) + 1.0
            reflected = record.reflect_const - data
            shift = max(0, -reflected.min() + self.epsilon)
            record.shift = shift
            return np.log(reflected + shift), record

        elif method == "reflect_sqrt":
            if fit:
                record.reflect_const = float(data.max()) + 1.0
            reflected = record.reflect_const - data
            shift = max(0, -reflected.min() + self.epsilon)
            record.shift = shift
            return np.sqrt(reflected + shift), record

        elif method == "boxcox":
            shift = max(0, -data.min() + self.epsilon) if fit else record.shift
            record.shift = shift
            if fit:
                transformer = PowerTransformer(method="box-cox", standardize=False)
                transformed = transformer.fit_transform((data + shift).reshape(-1, 1)).ravel()
                record.transformer = transformer
            else:
                transformed = record.transformer.transform((data + shift).reshape(-1, 1)).ravel()
            return transformed, record

        elif method == "yeo_johnson":
            if fit:
                transformer = PowerTransformer(method="yeo-johnson", standardize=False)
                transformed = transformer.fit_transform(data.reshape(-1, 1)).ravel()
                record.transformer = transformer
            else:
                transformed = record.transformer.transform(data.reshape(-1, 1)).ravel()
            return transformed, record

        elif method == "quantile_normal":
            if fit:
                transformer = QuantileTransformer(output_distribution="normal", random_state=42)
                transformed = transformer.fit_transform(data.reshape(-1, 1)).ravel()
                record.transformer = transformer
            else:
                transformed = record.transformer.transform(data.reshape(-1, 1)).ravel()
            return transformed, record

        elif method == "standard":
            if fit:
                transformer = StandardScaler()
                transformed = transformer.fit_transform(data.reshape(-1, 1)).ravel()
                record.transformer = transformer
            else:
                transformed = record.transformer.transform(data.reshape(-1, 1)).ravel()
            return transformed, record

        elif method == "minmax":
            if fit:
                transformer = MinMaxScaler()
                transformed = transformer.fit_transform(data.reshape(-1, 1)).ravel()
                record.transformer = transformer
            else:
                transformed = record.transformer.transform(data.reshape(-1, 1)).ravel()
            return transformed, record

        elif method == "robust":
            if fit:
                transformer = RobustScaler()
                transformed = transformer.fit_transform(data.reshape(-1, 1)).ravel()
                record.transformer = transformer
            else:
                transformed = record.transformer.transform(data.reshape(-1, 1)).ravel()
            return transformed, record

        else:
            raise ValueError(
                f"Unknown transformation '{method}'. " f"Supported: {SUPPORTED_TRANSFORMS}"
            )

    def _reverse(self, data: np.ndarray, record: TransformRecord) -> np.ndarray:
        """
        Invert a previously-applied transformation using the stored TransformRecord.

        Supports: log, log1p, sqrt, yeo_johnson, box_cox, standard_scaler, none.
        """
        method = record.method

        if method == "none":
            return data

        elif method == "log":
            return np.exp(data) - record.shift

        elif method == "log1p":
            return np.expm1(data) - record.shift

        elif method == "sqrt":
            return data**2 - record.shift

        elif method == "cbrt":
            return data**3

        elif method == "reflect_log":
            return record.reflect_const - (np.exp(data) - record.shift)

        elif method == "reflect_sqrt":
            return record.reflect_const - (data**2 - record.shift)

        elif method in ("boxcox", "yeo_johnson", "quantile_normal", "standard", "minmax", "robust"):
            inv = record.transformer.inverse_transform(data.reshape(-1, 1)).ravel()
            if method == "boxcox":
                inv = inv - record.shift
            return inv

        else:
            raise ValueError(f"Cannot reverse transform '{method}'.")
