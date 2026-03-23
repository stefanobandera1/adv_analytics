"""
adsat.modeling
==============
Fit saturation curve models to impression-vs-outcome data.

Supported models
----------------
- Hill function          (Bayesian via PyMC or frequentist fallback)
- Negative Exponential   (frequentist via scipy curve_fit)
- Power function         (frequentist via scipy curve_fit)
- Michaelis-Menten       (special case of Hill with n=1)
- Logistic / S-curve     (frequentist)

Key class: SaturationModeler
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Model functional forms
# ---------------------------------------------------------------------------


def hill_function(x: np.ndarray, a: float, k: float, n: float) -> np.ndarray:
    """
    Hill / sigmoidal saturation curve.

    Parameters
    ----------
    x : impressions (or any input metric)
    a : maximum asymptote (saturation ceiling)
    k : half-saturation constant (EC50) – the x value at which y = a/2
    n : Hill coefficient – controls steepness of the curve
    """
    return a * (x**n) / (k**n + x**n)


def negative_exponential(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Negative exponential: y = a * (1 - exp(-b * x))

    Parameters
    ----------
    a : saturation ceiling
    b : decay rate / speed of saturation
    """
    return a * (1.0 - np.exp(-b * x))


def power_function(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Power function: y = a * x^b  (b < 1 = concave / diminishing returns)
    """
    return a * np.power(np.maximum(x, 1e-10), b)


def michaelis_menten(x: np.ndarray, vmax: float, km: float) -> np.ndarray:
    """
    Michaelis-Menten (Hill with n=1): y = Vmax * x / (Km + x)
    """
    return vmax * x / (km + x)


def logistic_curve(x: np.ndarray, L: float, x0: float, k: float) -> np.ndarray:
    """
    Logistic / S-curve: y = L / (1 + exp(-k*(x - x0)))

    Parameters
    ----------
    L  : curve's maximum value
    x0 : midpoint (inflection point)
    k  : steepness
    """
    return L / (1.0 + np.exp(-k * (x - x0)))


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "hill": {
        "func": hill_function,
        "param_names": ["a_max", "k_half", "n_hill"],
        "p0_strategy": "hill",
        "bounds_strategy": "hill",
    },
    "negative_exponential": {
        "func": negative_exponential,
        "param_names": ["a_max", "b_rate"],
        "p0_strategy": "exp",
        "bounds_strategy": "exp",
    },
    "power": {
        "func": power_function,
        "param_names": ["a_scale", "b_exp"],
        "p0_strategy": "power",
        "bounds_strategy": "power",
    },
    "michaelis_menten": {
        "func": michaelis_menten,
        "param_names": ["vmax", "km"],
        "p0_strategy": "mm",
        "bounds_strategy": "mm",
    },
    "logistic": {
        "func": logistic_curve,
        "param_names": ["L_max", "x0_mid", "k_steep"],
        "p0_strategy": "logistic",
        "bounds_strategy": "logistic",
    },
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelFitResult:
    """Fitted model with metrics and saturation point estimate."""

    model_name: str
    params: dict[str, float]
    r2: float
    rmse: float
    mae: float
    mape: float | None
    aic: float
    bic: float
    y_pred: np.ndarray
    y_true: np.ndarray
    x_values: np.ndarray
    saturation_point: float | None = None
    saturation_y: float | None = None
    saturation_threshold: float = 0.90  # fraction of asymptote used to define saturation
    converged: bool = True
    notes: str = ""
    # For Bayesian models
    posterior_samples: dict[str, np.ndarray] | None = None
    credible_intervals: dict[str, tuple[float, float]] | None = None

    def summary(self) -> dict[str, Any]:
        """
        Return a dict of key model metrics suitable for building a comparison DataFrame.
        """
        return {
            "model": self.model_name,
            "r2": round(self.r2, 4),
            "rmse": round(self.rmse, 4),
            "mae": round(self.mae, 4),
            # Always include mape (even as None) so the column exists in evaluation DataFrames.
            # ModelEvaluator.evaluate() iterates _METRIC_DIRECTION which includes 'mape';
            # a missing column causes a KeyError in the composite rank calculation.
            "mape": round(self.mape, 4) if self.mape is not None else None,
            "aic": round(self.aic, 2),
            "bic": round(self.bic, 2),
            "saturation_point": self.saturation_point,
            "saturation_y": self.saturation_y,
            "converged": self.converged,
            "params": self.params,
        }

    def __repr__(self) -> str:
        """
        Short one-line representation: model name, R², RMSE, and saturation point.
        """
        return (
            f"ModelFitResult(model={self.model_name!r}, "
            f"r2={self.r2:.4f}, rmse={self.rmse:.4f}, "
            f"saturation_point={self.saturation_point})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SaturationModeler:
    """
    Fit saturation curve models to impression-vs-conversion (or revenue) data.

    Parameters
    ----------
    models : list of str
        Models to fit. Defaults to all: hill, negative_exponential, power,
        michaelis_menten, logistic.
    saturation_threshold : float
        Fraction of the asymptote used to define saturation point.
        Default 0.90 means: saturation occurs when y >= 90% of the maximum.
    use_bayesian_hill : bool
        If True and PyMC is installed, use Bayesian regression for the Hill model.
    verbose : bool

    Examples
    --------
    >>> from adsat import SaturationModeler
    >>> modeler = SaturationModeler(models=['hill', 'negative_exponential', 'power'])
    >>> results = modeler.fit(df, x_col='impressions', y_col='conversions')
    >>> print(results['hill'])
    """

    def __init__(
        self,
        models: list[str] | None = None,
        saturation_threshold: float = 0.90,
        use_bayesian_hill: bool = False,
        verbose: bool = True,
    ):
        """
        Configure which models to fit, the saturation threshold, and whether to use
        Bayesian Hill regression.  No fitting occurs here; call fit() to run the models.
        """
        self.models = models or list(MODEL_REGISTRY.keys())
        self.saturation_threshold = saturation_threshold
        self.use_bayesian_hill = use_bayesian_hill
        self.verbose = verbose
        self._results: dict[str, ModelFitResult] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        weights_col: str | None = None,
    ) -> dict[str, ModelFitResult]:
        """
        Fit all specified saturation models.

        Parameters
        ----------
        df : pd.DataFrame
        x_col : str    — independent variable (e.g. impressions, ad_spend)
        y_col : str    — dependent variable (e.g. conversions, revenue)
        weights_col : str, optional — column to use as fitting weights

        Returns
        -------
        dict mapping model name -> ModelFitResult
        """
        clean = df[[x_col, y_col]].dropna().copy()
        if weights_col and weights_col in df.columns:
            clean["_w"] = df.loc[clean.index, weights_col].values
        else:
            clean["_w"] = 1.0

        # Sort by x for cleaner curves
        clean = clean.sort_values(x_col).reset_index(drop=True)
        x = clean[x_col].values.astype(float)
        y = clean[y_col].values.astype(float)
        w = clean["_w"].values.astype(float)

        self._results = {}

        for model_name in self.models:
            if model_name not in MODEL_REGISTRY:
                warnings.warn(f"Unknown model '{model_name}' – skipping.")
                continue

            if self.verbose:
                print(f"[SaturationModeler] Fitting '{model_name}'…")

            if model_name == "hill" and self.use_bayesian_hill:
                result = self._fit_bayesian_hill(x, y, w)
            else:
                result = self._fit_frequentist(model_name, x, y, w)

            self._results[model_name] = result

        return self._results

    def predict(
        self,
        model_name: str,
        x_values: np.ndarray,
    ) -> np.ndarray:
        """Generate predictions from a fitted model."""
        if model_name not in self._results:
            raise KeyError(f"Model '{model_name}' has not been fitted.")
        result = self._results[model_name]
        # hill_bayesian is not in MODEL_REGISTRY (it uses the same function as hill)
        registry_name = "hill" if model_name == "hill_bayesian" else model_name
        if registry_name not in MODEL_REGISTRY:
            raise KeyError(f"No prediction function found for model '{model_name}'.")
        func = MODEL_REGISTRY[registry_name]["func"]
        param_vals = list(result.params.values())
        return func(x_values, *param_vals)

    def summary_table(self) -> pd.DataFrame:
        """Return a comparison DataFrame of all fitted models."""
        rows = [r.summary() for r in self._results.values()]
        return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Frequentist fitting
    # ------------------------------------------------------------------

    def _fit_frequentist(
        self,
        model_name: str,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> ModelFitResult:
        """
        Fit one saturation model using scipy.optimize.curve_fit (weighted nonlinear least squares).

        Computes R², RMSE, MAE, MAPE, AIC, BIC, and saturation point.
        If curve_fit fails to converge, falls back to the initial parameter guess and sets
        converged=False on the returned ModelFitResult.
        """
        spec = MODEL_REGISTRY[model_name]
        func = spec["func"]
        param_names = spec["param_names"]

        p0, bounds = self._get_initial_params(model_name, x, y)

        converged = True
        notes = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, pcov = curve_fit(
                    func,
                    x,
                    y,
                    p0=p0,
                    bounds=bounds,
                    sigma=1.0 / (w + 1e-8),
                    absolute_sigma=True,
                    maxfev=50000,
                )
        except RuntimeError as e:
            converged = False
            notes = str(e)
            popt = np.array(p0)

        params = dict(zip(param_names, popt))
        y_pred = func(x, *popt)

        # Metrics
        r2 = r2_score(y, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        mape = self._mape(y, y_pred)
        n = len(y)
        k = len(popt)
        ss_res = np.sum((y - y_pred) ** 2)
        # Bug fix: clamp ss_res to avoid log(0) = -inf when fit is near-perfect
        ss_res_clamped = max(ss_res, 1e-10)
        aic = n * np.log(ss_res_clamped / n) + 2 * k
        bic = n * np.log(ss_res_clamped / n) + k * np.log(n)

        sat_x, sat_y = self._compute_saturation_point(model_name, params, x)

        return ModelFitResult(
            model_name=model_name,
            params=params,
            r2=float(r2),
            rmse=rmse,
            mae=mae,
            mape=mape,
            aic=float(aic),
            bic=float(bic),
            y_pred=y_pred,
            y_true=y,
            x_values=x,
            saturation_point=sat_x,
            saturation_y=sat_y,
            saturation_threshold=self.saturation_threshold,
            converged=converged,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Bayesian Hill model (optional – requires PyMC)
    # ------------------------------------------------------------------

    def _fit_bayesian_hill(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> ModelFitResult:
        """
        Fit the Hill model with Bayesian inference via PyMC (requires pymc installed).

        Uses HalfNormal priors on a and k, a Gamma prior on n, and a Normal likelihood.
        Falls back to frequentist fitting with a UserWarning if PyMC is not available.
        Returns a ModelFitResult with posterior_samples and credible_intervals populated.
        """
        try:
            import pymc as pm
        except ImportError:
            warnings.warn(
                "PyMC not installed. Falling back to frequentist Hill fitting. "
                "Install with: pip install pymc",
                UserWarning,
            )
            return self._fit_frequentist("hill", x, y, w)

        a_init = float(y.max())
        k_init = float(np.percentile(x, 50))

        with pm.Model():
            # Priors
            a = pm.HalfNormal("a", sigma=a_init * 2)
            k = pm.HalfNormal("k", sigma=k_init * 2)
            n = pm.Gamma("n", alpha=2, beta=1)
            sigma = pm.HalfNormal("sigma", sigma=a_init * 0.5)

            mu = a * (x**n) / (k**n + x**n)
            pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trace = pm.sample(
                    1000,
                    tune=1000,
                    target_accept=0.9,
                    progressbar=self.verbose,
                    random_seed=42,
                    return_inferencedata=True,
                )

        # Point estimates from posterior means
        a_est = float(trace.posterior["a"].mean())
        k_est = float(trace.posterior["k"].mean())
        n_est = float(trace.posterior["n"].mean())

        params = {"a_max": a_est, "k_half": k_est, "n_hill": n_est}
        y_pred = hill_function(x, a_est, k_est, n_est)

        r2 = float(r2_score(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        mae = float(mean_absolute_error(y, y_pred))
        mape = self._mape(y, y_pred)
        n_obs = len(y)
        k_params = 3
        ss_res = np.sum((y - y_pred) ** 2)
        ss_res_clamped = max(ss_res, 1e-10)
        aic = float(n_obs * np.log(ss_res_clamped / n_obs) + 2 * k_params)
        bic = float(n_obs * np.log(ss_res_clamped / n_obs) + k_params * np.log(n_obs))

        # Credible intervals
        ci = {
            "a_max": (
                float(trace.posterior["a"].quantile(0.025)),
                float(trace.posterior["a"].quantile(0.975)),
            ),
            "k_half": (
                float(trace.posterior["k"].quantile(0.025)),
                float(trace.posterior["k"].quantile(0.975)),
            ),
            "n_hill": (
                float(trace.posterior["n"].quantile(0.025)),
                float(trace.posterior["n"].quantile(0.975)),
            ),
        }

        sat_x, sat_y = self._compute_saturation_point("hill", params, x)

        return ModelFitResult(
            model_name="hill_bayesian",
            params=params,
            r2=r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            aic=aic,
            bic=bic,
            y_pred=y_pred,
            y_true=y,
            x_values=x,
            saturation_point=sat_x,
            saturation_y=sat_y,
            saturation_threshold=self.saturation_threshold,
            converged=True,
            credible_intervals=ci,
        )

    # ------------------------------------------------------------------
    # Saturation point calculation
    # ------------------------------------------------------------------

    def _compute_saturation_point(
        self,
        model_name: str,
        params: dict[str, float],
        x: np.ndarray,
    ) -> tuple[float | None, float | None]:
        """
        Find x at which y reaches `saturation_threshold` * asymptote.

        The asymptote is extracted analytically where possible (Hill, Michaelis-Menten,
        Negative Exponential, Logistic all have known ceilings as model parameters).
        Power function has no true asymptote so saturation is undefined — returns None.
        """
        try:
            func = MODEL_REGISTRY[model_name]["func"]
            param_vals = list(params.values())

            # --- Analytically derive the true asymptote per model type ---
            if model_name in ("hill", "hill_bayesian"):
                # Hill: asymptote = a_max (first parameter)
                asymptote = param_vals[0]
            elif model_name == "michaelis_menten":
                # MM: asymptote = vmax (first parameter)
                asymptote = param_vals[0]
            elif model_name == "negative_exponential":
                # NegExp: asymptote = a_max (first parameter)
                asymptote = param_vals[0]
            elif model_name == "logistic":
                # Logistic: asymptote = L_max (first parameter)
                asymptote = param_vals[0]
            elif model_name == "power":
                # Power y = a * x^b has NO finite asymptote when b > 0.
                # Saturation point is therefore undefined.
                return None, None
            else:
                # Fallback: evaluate curve far beyond observed range
                x_far = np.linspace(x.min(), x.max() * 10, 50000)
                y_far = func(x_far, *param_vals)
                asymptote = float(y_far[-1])

            if asymptote <= 0:
                return None, None

            target = self.saturation_threshold * asymptote

            # Estimate how far x needs to go to reach the target.
            # For well-behaved curves, solve analytically where possible,
            # otherwise extend the grid until y exceeds the target.
            x.max() * 10  # start with 10× observed range

            for multiplier in [10, 50, 200, 1000]:
                x_fine = np.linspace(x.min(), x.max() * multiplier, 50000)
                y_fine = func(x_fine, *param_vals)
                if y_fine.max() >= target:
                    break
                x.max() * multiplier
            else:
                # Target never reached — saturation is beyond practical range
                return None, None

            # Ensure y_fine is monotone enough for searchsorted
            idx = np.searchsorted(y_fine, target)
            if idx >= len(x_fine):
                return None, None

            return float(x_fine[idx]), float(y_fine[idx])

        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Parameter initialisation heuristics
    # ------------------------------------------------------------------

    def _get_initial_params(
        self,
        model_name: str,
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[list[float], tuple]:
        """
        Return data-driven starting parameters (p0) and bounds for each model type.

        Good initialisations are critical for curve_fit convergence.  All bounds are
        strictly positive to match the physical interpretation of saturation curves.
        """
        # Clamp a_init: when y is all-zeros y.max()=0, which collapses the
        # upper bounds to 0 and makes curve_fit raise a ValueError.
        a_init = max(float(y.max()) * 1.2, 1.0)
        k_init = max(float(np.median(x)), 1e-6)

        if model_name == "hill":
            p0 = [a_init, k_init, 1.5]
            bounds = ([0, 1e-6, 0.1], [a_init * 10, x.max() * 10, 10])
        elif model_name == "negative_exponential":
            b_init = 1.0 / k_init if k_init > 0 else 1e-5
            p0 = [a_init, b_init]
            bounds = ([0, 1e-12], [a_init * 10, 1e3])
        elif model_name == "power":
            p0 = [a_init * 0.5, 0.5]
            bounds = ([0, 0.01], [a_init * 50, 0.99])
        elif model_name == "michaelis_menten":
            p0 = [a_init, k_init]
            bounds = ([0, 1e-6], [a_init * 10, x.max() * 10])
        elif model_name == "logistic":
            p0 = [a_init, k_init, 1.0 / k_init]
            bounds = ([0, 0, 1e-12], [a_init * 10, x.max() * 5, 100])
        else:
            p0 = [a_init, k_init]
            bounds = (0, np.inf)

        return p0, bounds

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
        """
        Compute Mean Absolute Percentage Error, ignoring zero-valued true observations.
        Returns None when no non-zero observations are available.
        """
        mask = y_true != 0
        if mask.sum() == 0:
            return None
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
