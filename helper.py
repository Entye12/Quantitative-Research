from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as st
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

if TYPE_CHECKING:
    from matplotlib.axes import Axes

__all__ = [
    "make_cv_splits",
    "basic_feature_stats",
    "correlation_screen",
    "plot_binned_target",
    "univariate_ols_screen",
    "ols_assumption_diagnostics",
    "joint_ols_report",
    "univariate_oos_screen",
    "feature_stability_report",
    "redundancy_screen",
    "incremental_gain_report",
    "build_research_table",
    "run_tabular_numeric_pipeline",
]

def _print_header(title: str) -> None:
    bar = "─" * len(title)
    print(f"\n{title}\n{bar}")


def _format_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    if df is None or df.empty:
        return "(none)"
    if max_rows is not None:
        df = df.head(max_rows)
    # to_string is simple and robust (works in scripts, logs, terminals)
    return df.to_string(index=True)


def _ensure_numeric_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"All columns must be numeric. Non-numeric columns: {non_numeric}")

    return X.copy()


def _ensure_numeric_series(y: pd.Series, name: str = "y") -> pd.Series:
    return pd.Series(y, name=name).astype(float)


def _align_xy(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    df = pd.concat(
        [pd.Series(x, name="x").astype(float), pd.Series(y, name="y").astype(float)],
        axis=1,
    )
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def _significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.1:
        return "."
    return ""



def _score_regression(y_true, y_pred, metric):
    if metric == "rmse":
        return np.sqrt(mean_squared_error(y_true, y_pred))
    elif metric == "mse":
        return mean_squared_error(y_true, y_pred)
    elif metric == "mae":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "r2":
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _higher_is_better(metric: str) -> bool:
    return metric.lower() == "r2"


def _build_linear_pipeline(alpha: float = 1.0, standardize: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("ridge", Ridge(alpha=alpha)))
    return Pipeline(steps)


def _r_style_coefficient_table(
    result: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "Estimate": result.params.values,
            "Std. Error": result.bse.values,
            "t value": result.tvalues.values,
            "Pr(>|t|)": result.pvalues.values,
        }
    ).set_index("term")
    table["Signif."] = table["Pr(>|t|)"].map(_significance_stars)
    return table


def _model_summary_frame(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    cov_type: str,
) -> pd.DataFrame:
    resid_std_error = np.sqrt(result.scale) if pd.notna(result.scale) else np.nan
    return pd.DataFrame(
        {
            "value": {
                "Observations": int(result.nobs),
                "Df Residuals": float(result.df_resid),
                "Df Model": float(result.df_model),
                "R-squared": float(result.rsquared),
                "Adj. R-squared": float(result.rsquared_adj),
                "Residual Std. Error": float(resid_std_error),
                "F-statistic": float(result.fvalue) if result.fvalue is not None else np.nan,
                "Prob (F-statistic)": float(result.f_pvalue) if result.f_pvalue is not None else np.nan,
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "Covariance Type": cov_type,
            }
        }
    )


def make_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    """Freeze one CV scheme and reuse it across the whole research loop."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    indices = np.arange(n_samples)
    return [(train_idx, val_idx) for train_idx, val_idx in kf.split(indices)]


def basic_feature_stats(X: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for numerical features."""
    X = _ensure_numeric_frame(X)
    rows = []

    for col in X.columns:
        s = pd.Series(X[col], dtype=float)
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()

        if finite.empty:
            rows.append(
                {
                    "feature": col,
                    "n_obs": 0,
                    "missing_rate": 1.0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "p25": np.nan,
                    "median": np.nan,
                    "p75": np.nan,
                    "max": np.nan,
                    "iqr": np.nan,
                    "pct_zero": np.nan,
                    "skew": np.nan,
                    "kurtosis": np.nan,
                    "n_unique": 0,
                    "is_constant": True,
                }
            )
            continue

        x = finite.to_numpy()
        rows.append(
            {
                "feature": col,
                "n_obs": int(len(x)),
                "missing_rate": float(1.0 - len(x) / len(s)),
                "mean": float(np.mean(x)),
                "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                "min": float(np.min(x)),
                "p25": float(np.quantile(x, 0.25)),
                "median": float(np.median(x)),
                "p75": float(np.quantile(x, 0.75)),
                "max": float(np.max(x)),
                "iqr": float(np.quantile(x, 0.75) - np.quantile(x, 0.25)),
                "pct_zero": float(np.mean(x == 0)),
                "skew": float(st.skew(x, bias=False)) if len(x) > 2 else np.nan,
                "kurtosis": float(st.kurtosis(x, fisher=True, bias=False)) if len(x) > 3 else np.nan,
                "n_unique": int(finite.nunique()),
                "is_constant": bool(finite.nunique() <= 1),
            }
        )

    return pd.DataFrame(rows).set_index("feature").sort_index()


def correlation_screen(
    X: pd.DataFrame,
    y: pd.Series,
    methods: Sequence[str] = ("pearson", "spearman"),
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Marginal linear and monotone signal screen with FDR-adjusted p-values."""
    X = _ensure_numeric_frame(X)
    y = _ensure_numeric_series(y)

    supported = {"pearson", "spearman"}
    unknown = set(methods) - supported
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")

    rows = []
    for col in X.columns:
        row = {"feature": col}
        df = _align_xy(X[col], y)
        row["n"] = int(len(df))

        for method in methods:
            if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
                corr, p_value = np.nan, np.nan
            elif method == "pearson":
                corr, p_value = st.pearsonr(df["x"], df["y"])
            else:
                corr, p_value = st.spearmanr(df["x"], df["y"])

            row[f"{method}_corr"] = float(corr) if pd.notna(corr) else np.nan
            row[f"{method}_abs_corr"] = abs(float(corr)) if pd.notna(corr) else np.nan
            row[f"{method}_p_value"] = float(p_value) if pd.notna(p_value) else np.nan

        rows.append(row)

    result = pd.DataFrame(rows).set_index("feature")
    for method in methods:
        pvals = result[f"{method}_p_value"]
        mask = pvals.notna()
        qvals = pd.Series(np.nan, index=result.index, dtype=float)
        reject = pd.Series(False, index=result.index, dtype=bool)

        if mask.any():
            rej, qv, _, _ = multipletests(pvals[mask], method=fdr_method)
            qvals.loc[mask] = qv
            reject.loc[mask] = rej

        result[f"{method}_q_value"] = qvals
        result[f"{method}_signif"] = reject

    sort_col = "pearson_abs_corr" if "pearson" in methods else f"{methods[0]}_abs_corr"
    return result.sort_values(sort_col, ascending=False)

def plot_binned_target(
    feature: pd.Series,
    y: pd.Series,
    n_bins: int = 20,
    strategy: str = "quantile",
    min_bin_size: int = 5,
    ci: float = 0.95,
    show_median: bool = True,
    color: str = "#1f77b4",
    ax=None,
):
    """
    Binned response plot: shows mean/median trend of y vs feature.
    No summary table returned (visual diagnostic only).
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats as st
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plot_binned_target requires matplotlib, seaborn and scipy to be installed."
        ) from exc

    fname = feature.name or "feature"
    yname = y.name or "target"

    df = pd.concat([feature.astype(float), y.astype(float)], axis=1).dropna()
    df.columns = [fname, yname]

    if len(df) < max(n_bins, 10):
        raise ValueError("Not enough data for stable binning.")

    # --- binning
    if strategy == "quantile":
        q = min(n_bins, df[fname].nunique())
        df["bin"] = pd.qcut(df[fname], q=q, duplicates="drop")
    elif strategy == "uniform":
        df["bin"] = pd.cut(df[fname], bins=n_bins, duplicates="drop")
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    summary = (
        df.groupby("bin", observed=True)
        .agg(
            feature_mean=(fname, "mean"),
            y_mean=(yname, "mean"),
            y_median=(yname, "median"),
            y_std=(yname, "std"),
            count=(yname, "size"),
        )
        .reset_index()
    )

    summary = summary[summary["count"] >= min_bin_size]
    summary = summary.sort_values("feature_mean")

    if summary.empty:
        raise ValueError("All bins dropped by min_bin_size.")

    # --- CI for mean
    summary["y_sem"] = summary["y_std"].fillna(0.0) / np.sqrt(summary["count"])
    z = st.norm.ppf(0.5 + ci / 2.0)
    summary["y_low"] = summary["y_mean"] - z * summary["y_sem"]
    summary["y_high"] = summary["y_mean"] + z * summary["y_sem"]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    # Mean line
    sns.lineplot(
        data=summary,
        x="feature_mean",
        y="y_mean",
        marker="o",
        linewidth=2,
        color=color,
        ax=ax,
        label="bin mean",
    )

    # CI band
    ax.fill_between(
        summary["feature_mean"],
        summary["y_low"],
        summary["y_high"],
        alpha=0.18,
        color=color,
        label=f"{int(ci * 100)}% CI",
    )

    # Median line
    if show_median:
        sns.lineplot(
            data=summary,
            x="feature_mean",
            y="y_median",
            linestyle="--",
            linewidth=1.5,
            color="#d62728",
            ax=ax,
            label="bin median",
        )

    ax.set_title(f"Binned response: {yname} vs {fname}")
    ax.set_xlabel(fname)
    ax.set_ylabel(yname)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    return ax


def plot_scatter_target(
    feature: pd.Series,
    y: pd.Series,
    max_points: int = 5000,
    smooth: bool = True,
    frac: float = 0.25,
    alpha: float = 0.35,
    ax=None,
):
    """
    Seaborn-based scatter diagnostic with optional LOWESS smoothing.
    """

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Requires matplotlib, seaborn, statsmodels"
        ) from exc

    fname = feature.name or "feature"
    yname = y.name or "target"

    df = pd.concat([feature.astype(float), y.astype(float)], axis=1).dropna()
    df.columns = [fname, yname]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    # --- scatter or density depending on size
    if len(df) <= max_points:
        sns.scatterplot(
            data=df,
            x=fname,
            y=yname,
            alpha=alpha,
            s=20,
            edgecolor=None,
            ax=ax,
            label="data",
        )
    else:
        # Density view for large datasets
        sns.kdeplot(
            data=df,
            x=fname,
            y=yname,
            fill=True,
            levels=30,
            thresh=0.05,
            cmap="Blues",
            ax=ax,
        )

    # --- LOWESS smooth
    if smooth and len(df) > 50:
        smoothed = lowess(df[yname], df[fname], frac=frac, return_sorted=True)
        ax.plot(
            smoothed[:, 0],
            smoothed[:, 1],
            linewidth=2.5,
            label="LOWESS",
        )

    ax.set_title(f"Scatter: {yname} vs {fname}")
    ax.set_xlabel(fname)
    ax.set_ylabel(yname)
    ax.grid(alpha=0.25)

    if smooth:
        ax.legend(frameon=False)

    return ax


def univariate_ols_screen(
    X: pd.DataFrame,
    y: pd.Series,
    robust: str = "HC3",
    add_constant: bool = True,
    fdr_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Fit y on each feature separately and return an R-style coefficient table."""
    X = _ensure_numeric_frame(X)
    y = _ensure_numeric_series(y)
    rows = []

    for col in X.columns:
        df = _align_xy(X[col], y)
        if len(df) < 8 or df["x"].nunique() < 2:
            rows.append(
                {
                    "feature": col,
                    "n": int(len(df)),
                    "Estimate": np.nan,
                    "Std. Error": np.nan,
                    "t value": np.nan,
                    "Pr(>|t|)": np.nan,
                    "Signif.": "",
                    "Robust Std. Error": np.nan,
                    "Robust t value": np.nan,
                    "Robust Pr(>|t|)": np.nan,
                    "Robust Signif.": "",
                    "Intercept": np.nan,
                    "R-squared": np.nan,
                    "Adj. R-squared": np.nan,
                    "F-statistic": np.nan,
                    "Prob (F-statistic)": np.nan,
                    "AIC": np.nan,
                    "BIC": np.nan,
                }
            )
            continue

        design = df[["x"]]
        if add_constant:
            design = sm.add_constant(design, has_constant="add")

        model = sm.OLS(df["y"], design).fit()
        robust_model = model.get_robustcov_results(cov_type=robust)
        robust_table = pd.DataFrame(
            {
                "term": robust_model.model.exog_names,
                "Estimate": robust_model.params,
                "Std. Error": robust_model.bse,
                "t value": robust_model.tvalues,
                "Pr(>|t|)": robust_model.pvalues,
            }
        ).set_index("term")

        rows.append(
            {
                "feature": col,
                "n": int(model.nobs),
                "Estimate": float(model.params.get("x", np.nan)),
                "Std. Error": float(model.bse.get("x", np.nan)),
                "t value": float(model.tvalues.get("x", np.nan)),
                "Pr(>|t|)": float(model.pvalues.get("x", np.nan)),
                "Signif.": _significance_stars(model.pvalues.get("x", np.nan)),
                "Robust Std. Error": float(robust_table.loc["x", "Std. Error"]),
                "Robust t value": float(robust_table.loc["x", "t value"]),
                "Robust Pr(>|t|)": float(robust_table.loc["x", "Pr(>|t|)"]),
                "Robust Signif.": _significance_stars(robust_table.loc["x", "Pr(>|t|)"]),
                "Intercept": float(model.params.get("const", np.nan)),
                "R-squared": float(model.rsquared),
                "Adj. R-squared": float(model.rsquared_adj),
                "F-statistic": float(model.fvalue) if model.fvalue is not None else np.nan,
                "Prob (F-statistic)": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
                "AIC": float(model.aic),
                "BIC": float(model.bic),
            }
        )

    result = pd.DataFrame(rows).set_index("feature")
    for source_col, target_col in [
        ("Pr(>|t|)", "q_value"),
        ("Robust Pr(>|t|)", "Robust q_value"),
    ]:
        mask = result[source_col].notna()
        qvals = pd.Series(np.nan, index=result.index, dtype=float)
        if mask.any():
            _, qv, _, _ = multipletests(result.loc[mask, source_col], method=fdr_method)
            qvals.loc[mask] = qv
        result[target_col] = qvals

    return result.sort_values("Robust t value", key=lambda s: s.abs(), ascending=False)


def ols_assumption_diagnostics(
    result: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    """Diagnostics for the main OLS assumptions. These are evidence, not proofs."""
    resid = pd.Series(result.resid).astype(float)
    exog = result.model.exog
    rows = []

    jb_stat, jb_pvalue, _, _ = jarque_bera(resid)
    rows.append(
        {
            "test": "Jarque-Bera",
            "statistic": float(jb_stat),
            "p_value": float(jb_pvalue),
            "null_hypothesis": "Residuals are normally distributed",
            "flag": "reject" if jb_pvalue < 0.05 else "do_not_reject",
        }
    )

    bp_stat, bp_pvalue, _, _ = het_breuschpagan(resid, exog)
    rows.append(
        {
            "test": "Breusch-Pagan",
            "statistic": float(bp_stat),
            "p_value": float(bp_pvalue),
            "null_hypothesis": "Errors are homoskedastic",
            "flag": "reject" if bp_pvalue < 0.05 else "do_not_reject",
        }
    )

    try:
        white_stat, white_pvalue, _, _ = het_white(resid, exog)
    except Exception:
        white_stat, white_pvalue = np.nan, np.nan
    rows.append(
        {
            "test": "White",
            "statistic": float(white_stat) if pd.notna(white_stat) else np.nan,
            "p_value": float(white_pvalue) if pd.notna(white_pvalue) else np.nan,
            "null_hypothesis": "Errors are homoskedastic",
            "flag": "reject" if pd.notna(white_pvalue) and white_pvalue < 0.05 else "do_not_reject",
        }
    )

    try:
        reset = linear_reset(result, power=2, use_f=True)
        reset_stat = float(reset.fvalue)
        reset_pvalue = float(reset.pvalue)
        reset_flag = "reject" if reset_pvalue < 0.05 else "do_not_reject"
    except Exception:
        reset_stat, reset_pvalue, reset_flag = np.nan, np.nan, "not_available"
    rows.append(
        {
            "test": "RESET",
            "statistic": reset_stat,
            "p_value": reset_pvalue,
            "null_hypothesis": "Linear specification is adequate",
            "flag": reset_flag,
        }
    )

    rows.append(
        {
            "test": "Durbin-Watson",
            "statistic": float(durbin_watson(resid)),
            "p_value": np.nan,
            "null_hypothesis": "No first-order autocorrelation in residuals",
            "flag": "inspect_stat",
        }
    )
    rows.append(
        {
            "test": "Condition Number",
            "statistic": float(np.linalg.cond(exog)),
            "p_value": np.nan,
            "null_hypothesis": "Design matrix is well conditioned",
            "flag": "inspect_stat",
        }
    )

    return pd.DataFrame(rows).set_index("test")


def joint_ols_report(
    X: pd.DataFrame,
    y: pd.Series,
    robust: str = "HC3",
    add_constant: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Joint OLS report for a shortlisted set of features."""
    X = _ensure_numeric_frame(X)
    y = _ensure_numeric_series(y)

    df = pd.concat([X.astype(float), y.rename("y")], axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.shape[0] < max(10, X.shape[1] + 2):
        raise ValueError("Not enough complete rows to fit a stable joint OLS model.")

    design = df[X.columns]
    if add_constant:
        design = sm.add_constant(design, has_constant="add")

    model = sm.OLS(df["y"], design).fit()
    robust_model = model.get_robustcov_results(cov_type=robust)
    robust_coefficients = pd.DataFrame(
        {
            "term": robust_model.model.exog_names,
            "Estimate": robust_model.params,
            "Std. Error": robust_model.bse,
            "t value": robust_model.tvalues,
            "Pr(>|t|)": robust_model.pvalues,
        }
    ).set_index("term")
    robust_coefficients["Signif."] = robust_coefficients["Pr(>|t|)"].map(_significance_stars)

    vif_rows = []
    if design.shape[1] > 2:
        vif_design = pd.DataFrame(design, columns=design.columns)
        for i, col in enumerate(vif_design.columns):
            if col == "const":
                continue
            vif_rows.append({"feature": col, "VIF": float(variance_inflation_factor(vif_design.values, i))})
    vif_table = pd.DataFrame(vif_rows).set_index("feature") if vif_rows else pd.DataFrame(columns=["VIF"])

    return {
        "coefficients": _r_style_coefficient_table(model),
        "robust_coefficients": robust_coefficients,
        "model_summary": _model_summary_frame(model, cov_type=robust),
        "diagnostics": ols_assumption_diagnostics(model),
        "vif": vif_table,
    }


def univariate_oos_screen(
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    metric: str = "rmse",
    ridge_alpha: float = 1.0,
    standardize: bool = True,
    corr_like_higher_is_better: bool = True,  # ignored unless you extend metrics
    decimals: int = 4,
    top_k: int = 20,
    show_bottom: int = 0,
    min_train_points: int = 5,
    min_val_points: int = 3,
    min_valid_folds: int = 1,
    return_table: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Out-of-sample univariate screen vs an intercept-only baseline.
    Prints a clean report (R-ish). Optionally returns the full results table.

    Parameters
    ----------
    splits : iterable of (train_idx, val_idx)
        Your CV folds indices (e.g., from TimeSeriesSplit.split(...))
    metric : {"rmse","mse","mae","r2"}
    """

    # ---------- coerce numeric + clean ----------
    Xn = X.copy()
    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce")
    yn = pd.to_numeric(y, errors="coerce")

    y_arr = yn.to_numpy(dtype=float)
    metric = metric.lower()

    # ---------- scoring + direction ----------
    def score_regression(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if metric == "rmse":
            return float(np.sqrt(mean_squared_error(y_true, y_pred)))
        if metric == "mse":
            return float(mean_squared_error(y_true, y_pred))
        if metric == "mae":
            return float(mean_absolute_error(y_true, y_pred))
        if metric == "r2":
            return float(r2_score(y_true, y_pred))
        raise ValueError(f"Unsupported metric: {metric}")

    higher_is_better = metric in {"r2"}  # for the supported metrics above

    # ---------- model pipeline ----------
    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("ridge", Ridge(alpha=ridge_alpha)))
    base_model = Pipeline(steps)

    # If splits might be a generator, materialize once (so we can iterate for each feature)
    splits_list = list(splits)
    n_folds = len(splits_list)

    # ---------- compute ----------
    rows = []

    for col in Xn.columns:
        x = Xn[col].to_numpy(dtype=float).reshape(-1, 1)

        fold_model_scores = []
        fold_baseline_scores = []
        fold_improvements = []

        for train_idx, val_idx in splits_list:
            y_tr = y_arr[train_idx]
            y_va = y_arr[val_idx]

            # finite masks on y; also require finite x
            x_tr = x[train_idx].reshape(-1)
            x_va = x[val_idx].reshape(-1)

            tr_mask = np.isfinite(y_tr) & np.isfinite(x_tr)
            va_mask = np.isfinite(y_va) & np.isfinite(x_va)

            if tr_mask.sum() < min_train_points or va_mask.sum() < min_val_points:
                fold_model_scores.append(np.nan)
                fold_baseline_scores.append(np.nan)
                fold_improvements.append(np.nan)
                continue

            X_tr = x_tr[tr_mask].reshape(-1, 1)
            X_va = x_va[va_mask].reshape(-1, 1)
            y_tr2 = y_tr[tr_mask]
            y_va2 = y_va[va_mask]

            # fit + predict
            model = base_model
            model.fit(X_tr, y_tr2)
            pred = model.predict(X_va)

            # intercept-only baseline
            baseline_pred = np.full(shape=len(y_va2), fill_value=float(np.mean(y_tr2)))

            ms = score_regression(y_va2, pred)
            bs = score_regression(y_va2, baseline_pred)

            # positive improvement means "better than baseline"
            improvement = (ms - bs) if higher_is_better else (bs - ms)

            fold_model_scores.append(ms)
            fold_baseline_scores.append(bs)
            fold_improvements.append(improvement)

        ms = np.array(fold_model_scores, dtype=float)
        bs = np.array(fold_baseline_scores, dtype=float)
        im = np.array(fold_improvements, dtype=float)

        valid = np.isfinite(ms)
        v = int(valid.sum())

        rows.append(
            {
                "feature": col,
                "valid_folds": v,
                f"uni_{metric}_mean": float(np.nanmean(ms)) if v > 0 else np.nan,
                f"uni_{metric}_std": float(np.nanstd(ms, ddof=1)) if v > 1 else np.nan,
                f"baseline_{metric}_mean": float(np.nanmean(bs)) if np.isfinite(bs).any() else np.nan,
                "oos_improvement_mean": float(np.nanmean(im)) if np.isfinite(im).any() else np.nan,
                "oos_improvement_std": float(np.nanstd(im, ddof=1)) if np.isfinite(im).sum() > 1 else np.nan,
                "improves_in_pct_folds": float(np.nanmean(im > 0)) if np.isfinite(im).any() else np.nan,
                # keep folds in case you want to inspect later
                f"uni_{metric}_folds": fold_model_scores,
                f"baseline_{metric}_folds": fold_baseline_scores,
                "oos_improvement_folds": fold_improvements,
            }
        )

    res = pd.DataFrame(rows).set_index("feature")

    # filter low fold validity
    if min_valid_folds > 1:
        res = res[res["valid_folds"] >= min_valid_folds]

    # sort: best improvement first
    res = res.sort_values("oos_improvement_mean", ascending=False)

    # ---------- pretty printing ----------
    def header(title: str) -> None:
        print("\n" + title)
        print("─" * len(title))

    header("Univariate OOS screen (vs intercept-only baseline)")
    print(f"n_samples={len(yn)}  n_features={Xn.shape[1]}  n_folds={n_folds}")
    print(f"metric={metric}  ridge_alpha={ridge_alpha:g}  standardize={standardize}")
    print("improvement direction:", "higher is better" if higher_is_better else "lower is better (reported as positive gain)")
    if min_valid_folds > 1:
        print(f"filtered: min_valid_folds >= {min_valid_folds}")

    if res.empty:
        print("\n(no results)")
        return res if return_table else None

    # print view table
    view_cols = [
        "oos_improvement_mean",
        "oos_improvement_std",
        "improves_in_pct_folds",
        "valid_folds",
        f"uni_{metric}_mean",
        f"uni_{metric}_std",
        f"baseline_{metric}_mean",
    ]
    view = res[view_cols].copy()
    for c in view.columns:
        if c != "valid_folds":
            view[c] = view[c].astype(float).round(decimals)

    header(f"Top {min(top_k, len(view))} features by mean OOS improvement")
    print(view.head(top_k).to_string())

    if show_bottom:
        header(f"Bottom {min(show_bottom, len(view))} features by mean OOS improvement")
        print(view.tail(show_bottom).to_string())

    # summary
    header("Summary")
    best_feat = view.index[0]
    best_imp = view.iloc[0]["oos_improvement_mean"]
    n_pos = int((view["oos_improvement_mean"] > 0).sum())
    print(f"best_feature={best_feat}  best_mean_improvement={best_imp}")
    print(f"features_with_positive_mean_improvement={n_pos}/{len(view)}")

    return res if return_table else None


def feature_stability_report(
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    robust: str = "HC3",
) -> pd.DataFrame:
    """Measure how stable the sign and strength of each feature are across folds."""
    X = _ensure_numeric_frame(X)
    y = _ensure_numeric_series(y)
    rows = []

    for col in X.columns:
        pearson_values = []
        spearman_values = []
        beta_values = []
        robust_t_values = []

        for train_idx, _ in splits:
            df = _align_xy(X.iloc[train_idx][col], y.iloc[train_idx])
            if len(df) < 8 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
                pearson_values.append(np.nan)
                spearman_values.append(np.nan)
                beta_values.append(np.nan)
                robust_t_values.append(np.nan)
                continue

            pearson_values.append(st.pearsonr(df["x"], df["y"]).statistic)
            spearman_values.append(st.spearmanr(df["x"], df["y"]).statistic)
            model = sm.OLS(df["y"], sm.add_constant(df[["x"]], has_constant="add")).fit()
            robust_model = model.get_robustcov_results(cov_type=robust)
            robust_table = pd.DataFrame(
                {
                    "term": robust_model.model.exog_names,
                    "Estimate": robust_model.params,
                    "t value": robust_model.tvalues,
                }
            ).set_index("term")
            beta_values.append(float(model.params["x"]))
            robust_t_values.append(float(robust_table.loc["x", "t value"]))

        pearson_values = np.array(pearson_values, dtype=float)
        spearman_values = np.array(spearman_values, dtype=float)
        beta_values = np.array(beta_values, dtype=float)
        robust_t_values = np.array(robust_t_values, dtype=float)
        rows.append(
            {
                "feature": col,
                "pearson_fold_values": pearson_values.tolist(),
                "spearman_fold_values": spearman_values.tolist(),
                "beta_fold_values": beta_values.tolist(),
                "robust_t_fold_values": robust_t_values.tolist(),
                "pearson_mean": float(np.nanmean(pearson_values)) if np.isfinite(pearson_values).any() else np.nan,
                "pearson_std": float(np.nanstd(pearson_values, ddof=1))
                if np.isfinite(pearson_values).sum() > 1
                else np.nan,
                "spearman_mean": float(np.nanmean(spearman_values)) if np.isfinite(spearman_values).any() else np.nan,
                "spearman_std": float(np.nanstd(spearman_values, ddof=1))
                if np.isfinite(spearman_values).sum() > 1
                else np.nan,
                "beta_mean": float(np.nanmean(beta_values)) if np.isfinite(beta_values).any() else np.nan,
                "beta_std": float(np.nanstd(beta_values, ddof=1)) if np.isfinite(beta_values).sum() > 1 else np.nan,
                "robust_t_mean": float(np.nanmean(robust_t_values)) if np.isfinite(robust_t_values).any() else np.nan,
                "robust_t_std": float(np.nanstd(robust_t_values, ddof=1))
                if np.isfinite(robust_t_values).sum() > 1
                else np.nan,
                "pearson_sign_consistency": float(
                    np.nanmean(np.sign(pearson_values) == np.sign(np.nanmedian(pearson_values)))
                )
                if np.isfinite(pearson_values).any()
                else np.nan,
                "spearman_sign_consistency": float(
                    np.nanmean(np.sign(spearman_values) == np.sign(np.nanmedian(spearman_values)))
                )
                if np.isfinite(spearman_values).any()
                else np.nan,
                "beta_sign_consistency": float(
                    np.nanmean(np.sign(beta_values) == np.sign(np.nanmedian(beta_values)))
                )
                if np.isfinite(beta_values).any()
                else np.nan,
            }
        )

    return pd.DataFrame(rows).set_index("feature").sort_values("beta_sign_consistency", ascending=False)


def redundancy_screen(
    X: pd.DataFrame,
    correlation_method: str = "spearman",
    corr_threshold: float = 0.8,
    vif_threshold: float = 5.0,
    decimals: int = 3,
    max_pairs: int | None = 50,
    return_tables: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Prints a 'R-like' redundancy report:
    - Correlated pairs above corr_threshold
    - VIF table with flags/recommendations
    - Suggested drops (naive heuristic: HIGH VIF features sorted by VIF desc)

    By default prints and returns None.
    If return_tables=True, returns {"pairwise": ..., "vif": ..., "suggested_drop": ...}.
    """
    X = _ensure_numeric_frame(X)

    # ---------- pairwise correlated pairs ----------
    corr = X.corr(method=correlation_method)

    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            v = corr.loc[left, right]
            if pd.notna(v) and abs(v) >= corr_threshold:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        f"{correlation_method}_corr": float(v),
                        "abs_corr": abs(float(v)),
                    }
                )

    if rows:
        pairwise = pd.DataFrame(rows).sort_values("abs_corr", ascending=False)

        pairwise["strength"] = pd.cut(
            pairwise["abs_corr"],
            bins=[0, 0.8, 0.9, 0.95, 1.0000001],
            labels=["high", "very high", "extreme", "near-perfect"],
            include_lowest=True,
        )
        pairwise["recommendation"] = "consider dropping one of the pair"

        pairwise[f"{correlation_method}_corr"] = pairwise[f"{correlation_method}_corr"].round(decimals)
        pairwise["abs_corr"] = pairwise["abs_corr"].round(decimals)

        if max_pairs is not None:
            pairwise = pairwise.head(max_pairs)
    else:
        pairwise = pd.DataFrame(
            columns=["feature_left", "feature_right", f"{correlation_method}_corr", "abs_corr", "strength", "recommendation"]
        )

    # ---------- VIF table ----------
    complete = X.replace([np.inf, -np.inf], np.nan).dropna()

    vif_rows = []
    if not complete.empty and complete.shape[1] >= 2 and complete.shape[0] > complete.shape[1]:
        design = sm.add_constant(complete, has_constant="add")
        for i, col in enumerate(design.columns):
            if col == "const":
                continue
            vif_rows.append({"feature": col, "VIF": float(variance_inflation_factor(design.values, i))})

    if vif_rows:
        vif_table = pd.DataFrame(vif_rows).set_index("feature")
        vif_table["VIF"] = vif_table["VIF"].round(decimals)
        vif_table["flag"] = np.where(vif_table["VIF"] >= vif_threshold, "HIGH", "ok")
        vif_table["recommendation"] = np.where(
            vif_table["VIF"] >= vif_threshold,
            f"consider removing (VIF ≥ {vif_threshold:g})",
            "keep",
        )
        vif_table = vif_table.sort_values("VIF", ascending=False)
    else:
        vif_table = pd.DataFrame(columns=["VIF", "flag", "recommendation"])

    # ---------- suggested drops ----------
    suggested_drop = (
        vif_table.index[vif_table.get("flag", pd.Series(dtype=str)) == "HIGH"].tolist()
        if not vif_table.empty
        else []
    )

    # ---------- printing ----------
    _print_header("Redundancy screen")
    print(f"n_samples={X.shape[0]}  n_features={X.shape[1]}")
    print(f"correlation_method={correlation_method}  corr_threshold={corr_threshold:g}  vif_threshold={vif_threshold:g}")

    _print_header(f"Correlated pairs (|corr| ≥ {corr_threshold:g})")
    if pairwise.empty:
        print("(none found)")
    else:
        print(_format_table(pairwise, max_rows=max_pairs))
        print(f"\nshown={len(pairwise)} pair(s)")

    _print_header("VIF (multicollinearity)")
    if vif_table.empty:
        print("(VIF not computed: need at least 2 features and more rows than columns after dropna)")
    else:
        print(_format_table(vif_table))
        n_high = int((vif_table["flag"] == "HIGH").sum()) if "flag" in vif_table else 0
        print(f"\nHIGH VIF features={n_high}")

    _print_header("Suggested drops (naive: HIGH VIF first)")
    if suggested_drop:
        print(", ".join(suggested_drop))
    else:
        print("(none)")

    if return_tables:
        return {"pairwise": pairwise, "vif": vif_table, "suggested_drop": suggested_drop}

    return None


def incremental_gain_report(
    X_base: Optional[pd.DataFrame],
    X_candidates: pd.DataFrame,
    y: pd.Series,
    splits,
    metric: str = "rmse",
    ridge_alpha: float = 1.0,
    standardize: bool = True,
) -> pd.DataFrame:
    """
    Evaluate candidate features by comparing augmented model performance 
    against a pre-calculated baseline across all folds.
    """
    y = _ensure_numeric_series(y)
    X_candidates = _ensure_numeric_frame(X_candidates)
    
    # Initialize empty baseline if None provided
    if X_base is None:
        X_base = pd.DataFrame(index=X_candidates.index)
    else:
        X_base = _ensure_numeric_frame(X_base)
        
    y_arr = y.to_numpy()
    higher_is_better = _higher_is_better(metric)
    
    # --- STEP 1: PRE-CALCULATE BASELINE SCORES PER FOLD ---
    # We do this once to avoid O(N_candidates * N_folds) complexity
    fold_baselines = []
    
    for train_idx, val_idx in splits:
        y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
        mask_tr, mask_va = np.isfinite(y_tr), np.isfinite(y_va)
        
        # Guard against empty/small folds
        if mask_tr.sum() < 5 or mask_va.sum() < 3:
            fold_baselines.append(None)
            continue
            
        y_tr_f, y_va_f = y_tr[mask_tr], y_va[mask_va]
        Xb_tr, Xb_va = X_base.iloc[train_idx].loc[mask_tr], X_base.iloc[val_idx].loc[mask_va]
        
        if X_base.shape[1] == 0:
            # Baseline is just the global mean of the training fold
            base_pred = np.repeat(np.mean(y_tr_f), len(y_va_f))
            base_score = _score_regression(y_va_f, base_pred, metric)
        else:
            # Fit the baseline linear model
            model = _build_linear_pipeline(alpha=ridge_alpha, standardize=standardize)
            model.fit(Xb_tr, y_tr_f)
            base_score = _score_regression(y_va_f, model.predict(Xb_va), metric)
            
        fold_baselines.append({
            "score": base_score,
            "indices": (train_idx, val_idx),
            "masks": (mask_tr, mask_va),
            "y_train": y_tr_f,
            "y_val": y_va_f,
            "Xb_train": Xb_tr,
            "Xb_val": Xb_va
        })

    # --- STEP 2: EVALUATE CANDIDATES ---
    rows = []
    for col in X_candidates.columns:
        improvements = []
        baseline_scores = []
        augmented_scores = []
        
        candidate_col_data = X_candidates[col]

        for entry in fold_baselines:
            if entry is None:
                for lst in [improvements, baseline_scores, augmented_scores]: 
                    lst.append(np.nan)
                continue
            
            # Extract fold data
            tr_idx, va_idx = entry["indices"]
            m_tr, m_va = entry["masks"]
            
            # Efficiently augment the baseline features with the single candidate
            # We use loc[mask] to ensure alignment with the target variable
            Xa_tr = pd.concat([entry["Xb_train"], candidate_col_data.iloc[tr_idx].loc[m_tr]], axis=1)
            Xa_va = pd.concat([entry["Xb_val"], candidate_col_data.iloc[va_idx].loc[m_va]], axis=1)
            
            # Fit Augmented Model
            aug_model = _build_linear_pipeline(alpha=ridge_alpha, standardize=standardize)
            aug_model.fit(Xa_tr, entry["y_train"])
            aug_score = _score_regression(entry["y_val"], aug_model.predict(Xa_va), metric)
            
            # Calculate gain
            gain = (aug_score - entry["score"]) if higher_is_better else (entry["score"] - aug_score)
            
            baseline_scores.append(entry["score"])
            augmented_scores.append(aug_score)
            improvements.append(gain)

        # Summarize results for this feature
        imp_arr = np.array(improvements, dtype=float)
        rows.append({
            "feature": col,
            f"baseline_{metric}_mean": np.nanmean(baseline_scores),
            f"augmented_{metric}_mean": np.nanmean(augmented_scores),
            "incremental_gain_mean": np.nanmean(imp_arr),
            "incremental_gain_std": np.nanstd(imp_arr, ddof=1) if np.isfinite(imp_arr).sum() > 1 else np.nan,
            "improves_in_pct_folds": np.nanmean(imp_arr > 0),
            "valid_folds": int(np.isfinite(imp_arr).sum()),
        })

    return pd.DataFrame(rows).set_index("feature").sort_values("incremental_gain_mean", ascending=False)


def build_research_table(
    stats: pd.DataFrame,
    correlation: pd.DataFrame,
    univariate_ols: pd.DataFrame,
    univariate_oos: pd.DataFrame,
    stability: pd.DataFrame,
    redundancy: Optional[Dict[str, pd.DataFrame]] = None,
    incremental_gain: Optional[pd.DataFrame] = None,
    corr_alpha: float = 0.10,
    ols_alpha: float = 0.10,
    min_oos_improvement: float = 0.0,
    min_sign_consistency: float = 0.60,
    max_missing_rate: float = 0.40,
    redundancy_corr_threshold: float = 0.80,
) -> pd.DataFrame:
    """Merge all screens into one research table with suggested actions."""
    table = (
        stats.join(correlation, how="outer")
        .join(univariate_ols, how="outer")
        .join(univariate_oos, how="outer")
        .join(stability, how="outer")
    )

    if redundancy is not None:
        max_pair_corr = pd.Series(0.0, index=table.index, dtype=float)
        pairwise = redundancy.get("pairwise", pd.DataFrame())
        if not pairwise.empty:
            for _, row in pairwise.iterrows():
                max_pair_corr.loc[row["feature_left"]] = max(max_pair_corr.loc[row["feature_left"]], row["abs_corr"])
                max_pair_corr.loc[row["feature_right"]] = max(max_pair_corr.loc[row["feature_right"]], row["abs_corr"])
        table["max_abs_pair_corr"] = max_pair_corr

        vif = redundancy.get("vif", pd.DataFrame())
        if not vif.empty:
            table = table.join(vif.rename(columns={"VIF": "joint_vif"}), how="left")

    if incremental_gain is not None:
        table = table.join(incremental_gain, how="left")

    decisions = []
    notes_all = []
    for _, row in table.iterrows():
        notes = []
        corr_signal = (
            (pd.notna(row.get("pearson_q_value")) and row.get("pearson_q_value") <= corr_alpha)
            or (pd.notna(row.get("spearman_q_value")) and row.get("spearman_q_value") <= corr_alpha)
        )
        ols_signal = pd.notna(row.get("Robust Pr(>|t|)")) and row.get("Robust Pr(>|t|)") <= ols_alpha
        oos_signal = pd.notna(row.get("oos_improvement_mean")) and row.get("oos_improvement_mean") > min_oos_improvement
        stable = pd.notna(row.get("beta_sign_consistency")) and row.get("beta_sign_consistency") >= min_sign_consistency

        if pd.notna(row.get("missing_rate")) and row.get("missing_rate") > max_missing_rate:
            notes.append("high_missingness")
        if corr_signal:
            notes.append("marginal_corr_signal")
        if ols_signal:
            notes.append("ols_signal")
        if oos_signal:
            notes.append("oos_signal")
        if stable:
            notes.append("stable_sign")
        if pd.notna(row.get("max_abs_pair_corr")) and row.get("max_abs_pair_corr") >= redundancy_corr_threshold:
            notes.append("redundant_pairwise")
        if pd.notna(row.get("joint_vif")) and row.get("joint_vif") >= 10:
            notes.append("high_vif")
        if incremental_gain is not None:
            if pd.notna(row.get("incremental_gain_mean")) and row.get("incremental_gain_mean") > 0:
                notes.append("incremental_gain")
            elif pd.notna(row.get("incremental_gain_mean")):
                notes.append("no_incremental_gain")

        signal_count = sum([corr_signal, ols_signal, oos_signal, stable])
        if signal_count >= 3 and "redundant_pairwise" not in notes and "high_vif" not in notes:
            decision = "keep_candidate"
        elif signal_count <= 1 and not oos_signal:
            decision = "drop_candidate"
        else:
            decision = "watch"

        decisions.append(decision)
        notes_all.append(", ".join(notes) if notes else "weak_evidence")

    table["research_decision"] = decisions
    table["research_notes"] = notes_all

    preferred_columns = [
        "research_decision",
        "research_notes",
        "missing_rate",
        "pearson_corr",
        "pearson_q_value",
        "spearman_corr",
        "spearman_q_value",
        "Estimate",
        "Robust t value",
        "Robust Pr(>|t|)",
        "R-squared",
        "oos_improvement_mean",
        "improves_in_pct_folds",
        "beta_sign_consistency",
        "max_abs_pair_corr",
        "joint_vif",
        "incremental_gain_mean",
    ]
    ordered = [col for col in preferred_columns if col in table.columns]
    remaining = [col for col in table.columns if col not in ordered]
    table = table[ordered + remaining]

    rank_map = {"keep_candidate": 0, "watch": 1, "drop_candidate": 2}
    table["_decision_rank"] = table["research_decision"].map(rank_map).fillna(99)
    table = table.sort_values(
        ["_decision_rank", "oos_improvement_mean", "Robust t value"],
        ascending=[True, False, False],
        na_position="last",
    )
    return table.drop(columns="_decision_rank")


def run_tabular_numeric_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    splits=None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    metric: str = "rmse",
    ridge_alpha: float = 1.0,
    standardize: bool = True,
    joint_ols_features: Optional[Sequence[str]] = None,
    base_features: Optional[Sequence[str]] = None,
    redundancy_features: Optional[Sequence[str]] = None,
    redundancy_corr_threshold: float = 0.8,
) -> Dict[str, Any]:
    """End-to-end tabular numerical feature research pipeline."""
    X = _ensure_numeric_frame(X)
    y = _ensure_numeric_series(y)
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if splits is None:
        splits = make_cv_splits(len(X), n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    results: Dict[str, Any] = {"splits": splits}
    results["stats"] = basic_feature_stats(X)
    results["correlation"] = correlation_screen(X, y)
    results["univariate_ols"] = univariate_ols_screen(X, y)
    results["univariate_oos"] = univariate_oos_screen(
        X,
        y,
        splits=splits,
        metric=metric,
        ridge_alpha=ridge_alpha,
        standardize=standardize,
    )
    results["stability"] = feature_stability_report(X, y, splits=splits)

    redundancy_cols = list(redundancy_features) if redundancy_features is not None else list(X.columns)
    results["redundancy"] = redundancy_screen(
        X[redundancy_cols],
        corr_threshold=redundancy_corr_threshold,
    )

    if joint_ols_features:
        results["joint_ols"] = joint_ols_report(X[list(joint_ols_features)], y)
    else:
        results["joint_ols"] = None

    if base_features:
        base_features = list(base_features)
        candidate_cols = [col for col in X.columns if col not in base_features]
        results["incremental_gain"] = incremental_gain_report(
            X[base_features],
            X[candidate_cols],
            y,
            splits=splits,
            metric=metric,
            ridge_alpha=ridge_alpha,
            standardize=standardize,
        )
    else:
        results["incremental_gain"] = None

    results["research_table"] = build_research_table(
        stats=results["stats"],
        correlation=results["correlation"],
        univariate_ols=results["univariate_ols"],
        univariate_oos=results["univariate_oos"],
        stability=results["stability"],
        redundancy=results["redundancy"],
        incremental_gain=results["incremental_gain"],
        redundancy_corr_threshold=redundancy_corr_threshold,
    )
    return results
