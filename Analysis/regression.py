from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

__all__ = [
    "make_cv_splits",
    "basic_feature_stats",
    "correlation_screen",
    "plot_binned_target",
    "plot_scatter_target",
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


def _num_frame(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")


def _num_series(y: pd.Series, name: str = "y") -> pd.Series:
    return pd.to_numeric(pd.Series(y, name=name), errors="coerce")


def _xy(x: pd.Series, y: pd.Series) -> pd.DataFrame:
    return pd.concat([_num_series(x, "x"), _num_series(y, "y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()


def _stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def _fdr(p_values: pd.Series, method: str) -> tuple[pd.Series, pd.Series]:
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    reject = pd.Series(False, index=p_values.index, dtype=bool)
    mask = p_values.notna()
    if mask.any():
        rej, qv, _, _ = multipletests(p_values.loc[mask], method=method)
        q_values.loc[mask] = qv
        reject.loc[mask] = rej
    return q_values, reject


def _score_regression(y_true, y_pred, metric: str) -> float:
    metric = metric.lower()
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mse":
        return float(mean_squared_error(y_true, y_pred))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    return float(r2_score(y_true, y_pred))


def _higher_is_better(metric: str) -> bool:
    return metric.lower() == "r2"


def _linear_pipeline(alpha: float = 1.0, standardize: bool = True) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("ridge", Ridge(alpha=alpha)))
    return Pipeline(steps)


def _sign_consistency(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan
    ref = np.sign(np.nanmedian(finite))
    return float(np.mean(np.sign(finite) == ref))


def _vif_table(X: pd.DataFrame, decimals: int = 3, threshold: float = 5.0) -> pd.DataFrame:
    complete = _num_frame(X).replace([np.inf, -np.inf], np.nan).dropna()
    if complete.shape[1] < 2 or complete.shape[0] <= complete.shape[1]:
        return pd.DataFrame(columns=["VIF", "flag", "recommendation"])

    design = sm.add_constant(complete, has_constant="add")
    rows = [
        {"feature": col, "VIF": float(variance_inflation_factor(design.values, i))}
        for i, col in enumerate(design.columns)
        if col != "const"
    ]
    vif = pd.DataFrame(rows).set_index("feature").sort_values("VIF", ascending=False)
    vif["VIF"] = vif["VIF"].round(decimals)
    vif["flag"] = np.where(vif["VIF"] >= threshold, "HIGH", "ok")
    vif["recommendation"] = np.where(vif["VIF"] >= threshold, f"consider removing (VIF >= {threshold:g})", "keep")
    return vif


def make_cv_splits(
    n_samples: int,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> list[Tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(kf.split(np.arange(n_samples)))


def basic_feature_stats(X: pd.DataFrame) -> pd.DataFrame:
    X = _num_frame(X).replace([np.inf, -np.inf], np.nan)
    if X.shape[1] == 0:
        return pd.DataFrame()

    desc = (
        X.describe(percentiles=[0.25, 0.5, 0.75]).T.reindex(X.columns).rename(
            columns={"count": "n_obs", "25%": "p25", "50%": "median", "75%": "p75"}
        )
    )

    out = desc.loc[:, ["n_obs", "mean", "std", "min", "p25", "median", "p75", "max"]]
    out["n_obs"] = out["n_obs"].fillna(0).astype(int)
    out["missing_rate"] = 1.0 - out["n_obs"] / len(X)
    out["iqr"] = out["p75"] - out["p25"]
    out["pct_zero"] = X.eq(0).sum().div(X.notna().sum()).replace([np.inf], np.nan)
    out["skew"] = X.skew()
    out["kurtosis"] = X.kurt()
    out["n_unique"] = X.nunique(dropna=True).astype(int)
    out["is_constant"] = out["n_unique"] <= 1

    cols = [
        "n_obs",
        "missing_rate",
        "mean",
        "std",
        "min",
        "p25",
        "median",
        "p75",
        "max",
        "iqr",
        "pct_zero",
        "skew",
        "kurtosis",
        "n_unique",
        "is_constant",
    ]
    return out[cols].sort_index()


def correlation_screen(
    X: pd.DataFrame,
    y: pd.Series,
    methods: Sequence[str] = ("pearson", "spearman"),
    fdr_method: str = "fdr_bh",
    *,
    title: str = "Correlation screen",
    max_rows: int | None = 30,
    spearman_minus_pearson_threshold: float = 0.20,
    min_abs_spearman_for_hint: float = 0.30,
) -> None:

    X = _num_frame(X)
    y = _num_series(y)

    methods = tuple(dict.fromkeys(m.lower() for m in methods))
    corr_fn = {"pearson": st.pearsonr, "spearman": st.spearmanr}

    rows = []
    for col in X.columns:
        df = _xy(X[col], y)

        row = {"feature": col, "n": len(df)}

        for method in methods:
            if len(df) < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
                corr, p_value = np.nan, np.nan
            else:
                res = corr_fn[method](df["x"], df["y"])
                corr, p_value = float(res.statistic), float(res.pvalue)

            row[f"{method}_corr"] = corr
            row[f"{method}_p_value"] = p_value

        rows.append(row)

    out = pd.DataFrame(rows).set_index("feature")

    # FDR correction + stars (per method)
    for method in methods:
        q, _ = _fdr(out[f"{method}_p_value"], fdr_method)
        out[f"{method}_q_value"] = q
        out[f"{method}_sig"] = out[f"{method}_q_value"].map(_stars)  # stars on q-values

    # hint: Spearman much stronger than Pearson (monotone/nonlinear suspicion)
    if "pearson" in methods and "spearman" in methods:
        delta = out["spearman_corr"].abs() - out["pearson_corr"].abs()
        out["hint"] = np.where(
            (delta >= spearman_minus_pearson_threshold)
            & (out["spearman_corr"].abs() >= min_abs_spearman_for_hint),
            "maybe transformation needed",
            "",
        )
    else:
        out["hint"] = ""

    # Sorting: by best q-value across methods, tie-break by larger |corr| (internal)
    q_cols = [f"{m}_q_value" for m in methods]
    out["_min_q"] = out[q_cols].min(axis=1)

    corr_abs = pd.concat([out[f"{m}_corr"].abs() for m in methods], axis=1)
    out["_max_abs_corr"] = corr_abs.max(axis=1)

    out = out.sort_values(["_min_q", "_max_abs_corr"], ascending=[True, False]).drop(
        columns=["_min_q", "_max_abs_corr"]
    )

    # display columns
    cols = ["n"]
    for m in methods:
        cols += [f"{m}_corr", f"{m}_p_value", f"{m}_q_value", f"{m}_sig"]
    cols += ["hint"]

    view = out[cols]
    if max_rows is not None:
        view = view.head(max_rows)

    # HEADER
    _print_header(title)
    print(f"methods     : {', '.join(methods)}")
    print(f"fdr_method  : {fdr_method}")
    print(f"n_features  : {out.shape[0]}")
    print(f"max_rows    : {max_rows if max_rows is not None else 'all'}")

    print()
    print(view.to_string(float_format=lambda x: f"{x:.3g}"))

    if max_rows is not None and len(out) > max_rows:
        print(f"\n… ({len(out) - max_rows} more rows)")

    # NOTES
    _print_header("Notes")
    print("- pearson_corr   : linear correlation coefficient.")
    print("- spearman_corr  : rank correlation (detects monotone nonlinear relationships).")
    print("- p_value        : raw test p-value for corr != 0 (per feature, per method).")
    print("- q_value        : FDR-corrected p-value across features (per method).")
    print("- *_sig          : Computed from q_value.")
    print(f"- Stars legend   : *** <0.001  ** <0.01  * <0.05  . <0.1")
    print("- hint           : Spearman much stronger than Pearson (possible monotone nonlinearity / outliers).")


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
    feature = _num_series(feature, feature.name or "feature")
    y = _num_series(y, y.name or "target")
    df = pd.concat([feature, y], axis=1).dropna()
    fname, yname = df.columns

    q = min(n_bins, max(1, df[fname].nunique()))
    df["bin"] = {
        "quantile": pd.qcut(df[fname], q=q, duplicates="drop"),
        "uniform": pd.cut(df[fname], bins=n_bins, duplicates="drop"),
    }[strategy]

    summary = (
        df.groupby("bin", observed=True)
        .agg(
            feature_mean=(fname, "mean"),
            y_mean=(yname, "mean"),
            y_median=(yname, "median"),
            y_std=(yname, "std"),
            count=(yname, "size"),
        )
        .query("count >= @min_bin_size")
        .sort_values("feature_mean")
        .reset_index(drop=True)
    )

    summary["y_sem"] = summary["y_std"].fillna(0.0) / np.sqrt(summary["count"])
    z = st.norm.ppf(0.5 + ci / 2.0)
    summary["y_low"] = summary["y_mean"] - z * summary["y_sem"]
    summary["y_high"] = summary["y_mean"] + z * summary["y_sem"]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    sns.lineplot(data=summary, x="feature_mean", y="y_mean", marker="o", linewidth=2, color=color, ax=ax, label="bin mean")
    ax.fill_between(summary["feature_mean"], summary["y_low"], summary["y_high"], alpha=0.18, color=color, label=f"{int(ci * 100)}% CI")

    if show_median:
        sns.lineplot(data=summary, x="feature_mean", y="y_median", linestyle="--", linewidth=1.5, color="#d62728", ax=ax, label="bin median")

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
    feature = _num_series(feature, feature.name or "feature")
    y = _num_series(y, y.name or "target")
    df = pd.concat([feature, y], axis=1).dropna()
    fname, yname = df.columns

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if len(df) <= max_points:
        sns.scatterplot(data=df, x=fname, y=yname, alpha=alpha, s=20, edgecolor=None, ax=ax, label="data")
    else:
        sns.kdeplot(data=df, x=fname, y=yname, fill=True, levels=30, thresh=0.05, cmap="Blues", ax=ax)

    if smooth and len(df) > 50:
        smoothed = lowess(df[yname], df[fname], frac=frac, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2.5, label="LOWESS")

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
    X = _num_frame(X)
    y = _num_series(y)
    rows = []

    for col in X.columns:
        df = _xy(X[col], y)
        if len(df) < 8 or df["x"].nunique() < 2:
            rows.append(
                {
                    "feature": col,
                    "n": len(df),
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

        design = sm.add_constant(df[["x"]], has_constant="add") if add_constant else df[["x"]]
        model = sm.OLS(df["y"], design).fit()
        robust_model = model.get_robustcov_results(cov_type=robust)

        names = model.model.exog_names
        params = pd.Series(np.asarray(model.params, dtype=float), index=names)
        bse = pd.Series(np.asarray(model.bse, dtype=float), index=names)
        tvalues = pd.Series(np.asarray(model.tvalues, dtype=float), index=names)
        pvalues = pd.Series(np.asarray(model.pvalues, dtype=float), index=names)

        robust_names = robust_model.model.exog_names
        robust_bse = pd.Series(np.asarray(robust_model.bse, dtype=float), index=robust_names)
        robust_tvalues = pd.Series(np.asarray(robust_model.tvalues, dtype=float), index=robust_names)
        robust_pvalues = pd.Series(np.asarray(robust_model.pvalues, dtype=float), index=robust_names)

        rows.append(
            {
                "feature": col,
                "n": int(model.nobs),
                "Estimate": float(params.get("x", np.nan)),
                "Std. Error": float(bse.get("x", np.nan)),
                "t value": float(tvalues.get("x", np.nan)),
                "Pr(>|t|)": float(pvalues.get("x", np.nan)),
                "Signif.": _stars(pvalues.get("x", np.nan)),
                "Robust Std. Error": float(robust_bse.get("x", np.nan)),
                "Robust t value": float(robust_tvalues.get("x", np.nan)),
                "Robust Pr(>|t|)": float(robust_pvalues.get("x", np.nan)),
                "Robust Signif.": _stars(robust_pvalues.get("x", np.nan)),
                "Intercept": float(params.get("const", np.nan)),
                "R-squared": float(model.rsquared),
                "Adj. R-squared": float(model.rsquared_adj),
                "F-statistic": float(model.fvalue) if model.fvalue is not None else np.nan,
                "Prob (F-statistic)": float(model.f_pvalue) if model.f_pvalue is not None else np.nan,
                "AIC": float(model.aic),
                "BIC": float(model.bic),
            }
        )

    out = pd.DataFrame(rows).set_index("feature")
    out["q_value"], _ = _fdr(out["Pr(>|t|)"], fdr_method)
    out["Robust q_value"], _ = _fdr(out["Robust Pr(>|t|)"], fdr_method)
    return out.sort_values("Robust t value", key=lambda s: s.abs(), ascending=False)


def ols_assumption_diagnostics(result) -> pd.DataFrame:
    resid = pd.Series(result.resid, dtype=float)
    exog = result.model.exog

    jb_stat, jb_p = jarque_bera(resid)[:2]
    bp_stat, bp_p = het_breuschpagan(resid, exog)[:2]
    white_stat, white_p = het_white(resid, exog)[:2]
    reset = linear_reset(result, power=2, use_f=True)
    dw = durbin_watson(resid)
    cond = np.linalg.cond(exog)

    rows = [
        ("Jarque-Bera", jb_stat, jb_p, "Residuals are normally distributed", "reject" if jb_p < 0.05 else "do_not_reject"),
        ("Breusch-Pagan", bp_stat, bp_p, "Errors are homoskedastic", "reject" if bp_p < 0.05 else "do_not_reject"),
        ("White", white_stat, white_p, "Errors are homoskedastic", "reject" if white_p < 0.05 else "do_not_reject"),
        ("RESET", float(reset.fvalue), float(reset.pvalue), "Linear specification is adequate", "reject" if float(reset.pvalue) < 0.05 else "do_not_reject"),
        ("Durbin-Watson", dw, np.nan, "No first-order autocorrelation in residuals", "inspect_stat"),
        ("Condition Number", cond, np.nan, "Design matrix is well conditioned", "inspect_stat"),
    ]
    return pd.DataFrame(rows, columns=["test", "statistic", "p_value", "null_hypothesis", "flag"]).set_index("test")


def joint_ols_report(
    X: pd.DataFrame,
    y: pd.Series,
    robust: str = "HC3",
    add_constant: bool = True,
) -> Dict[str, Any]:

    X = _num_frame(X)
    y = _num_series(y)

    df = pd.concat([X, y.rename("y")], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    design = sm.add_constant(df[X.columns], has_constant="add") if add_constant else df[X.columns]

    base_model = sm.OLS(df["y"], design).fit()
    result = base_model.get_robustcov_results(cov_type=robust) if robust else base_model

    diagnostics = ols_assumption_diagnostics(base_model)
    vif = _vif_table(pd.DataFrame(design, columns=design.columns).drop(columns="const", errors="ignore"))

    _print_header(f"Joint OLS ({robust if robust else 'nonrobust'})")
    print(result.summary())

    _print_header("Diagnostics")
    print(diagnostics.to_string())

    _print_header("VIF")
    print("(none)" if vif.empty else vif.to_string())

    # IMPORTANT: do NOT return result.summary() (Summary objects get auto-rendered)
    return {
        "model": result,
        "base_model": base_model,
        "diagnostics": diagnostics,
        "vif": vif,
        # if you want the text, return it as plain string:
        "summary_text": result.summary().as_text(),
    }


def univariate_oos_screen(
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    metric: str = "rmse",
    ridge_alpha: float = 1.0,
    standardize: bool = True,
    corr_like_higher_is_better: bool = True,
    decimals: int = 4,
    top_k: int = 20,
    show_bottom: int = 0,
    min_train_points: int = 5,
    min_val_points: int = 3,
    min_valid_folds: int = 1,
    return_table: bool = False,
) -> Optional[pd.DataFrame]:
    X = _num_frame(X).replace([np.inf, -np.inf], np.nan)
    y = _num_series(y)
    splits = list(splits)
    metric = metric.lower()
    model = _linear_pipeline(alpha=ridge_alpha, standardize=standardize)
    higher_is_better = _higher_is_better(metric)

    rows = []
    for col in X.columns:
        fold_model_scores, fold_baseline_scores, fold_improvements = [], [], []
        x = X[[col]].to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)

        for train_idx, val_idx in splits:
            y_tr, y_va = y_arr[train_idx], y_arr[val_idx]
            tr_mask, va_mask = np.isfinite(y_tr), np.isfinite(y_va)

            if tr_mask.sum() < min_train_points or va_mask.sum() < min_val_points:
                fold_model_scores.append(np.nan)
                fold_baseline_scores.append(np.nan)
                fold_improvements.append(np.nan)
                continue

            X_tr, X_va = x[train_idx][tr_mask], x[val_idx][va_mask]
            y_tr2, y_va2 = y_tr[tr_mask], y_va[va_mask]

            model.fit(X_tr, y_tr2)
            pred = model.predict(X_va)
            baseline = np.full(len(y_va2), float(np.mean(y_tr2)))

            score = _score_regression(y_va2, pred, metric)
            base = _score_regression(y_va2, baseline, metric)
            gain = (score - base) if higher_is_better else (base - score)

            fold_model_scores.append(score)
            fold_baseline_scores.append(base)
            fold_improvements.append(gain)

        ms = np.array(fold_model_scores, dtype=float)
        bs = np.array(fold_baseline_scores, dtype=float)
        im = np.array(fold_improvements, dtype=float)
        valid = int(np.isfinite(ms).sum())

        rows.append(
            {
                "feature": col,
                "valid_folds": valid,
                f"uni_{metric}_mean": np.nanmean(ms) if valid else np.nan,
                f"uni_{metric}_std": np.nanstd(ms, ddof=1) if valid > 1 else np.nan,
                f"baseline_{metric}_mean": np.nanmean(bs) if np.isfinite(bs).any() else np.nan,
                "oos_improvement_mean": np.nanmean(im) if np.isfinite(im).any() else np.nan,
                "oos_improvement_std": np.nanstd(im, ddof=1) if np.isfinite(im).sum() > 1 else np.nan,
                "improves_in_pct_folds": np.nanmean(im > 0) if np.isfinite(im).any() else np.nan,
                f"uni_{metric}_folds": fold_model_scores,
                f"baseline_{metric}_folds": fold_baseline_scores,
                "oos_improvement_folds": fold_improvements,
            }
        )

    out = pd.DataFrame(rows).set_index("feature")
    if min_valid_folds > 1:
        out = out[out["valid_folds"] >= min_valid_folds]
    out = out.sort_values("oos_improvement_mean", ascending=False)

    _print_header("Univariate OOS Screen")
    print(f"observations={len(y)}  features={X.shape[1]}  folds={len(splits)}")
    print(f"metric={metric}  ridge_alpha={ridge_alpha}  standardize={standardize}")
    if min_valid_folds > 1:
        print(f"filtered: min_valid_folds >= {min_valid_folds}")

    view_cols = [
        "oos_improvement_mean",
        "oos_improvement_std",
        "improves_in_pct_folds",
        "valid_folds",
        f"uni_{metric}_mean",
        f"uni_{metric}_std",
        f"baseline_{metric}_mean",
    ]
    view = out[view_cols].round(decimals) if not out.empty else out

    _print_header("Top features by mean OOS improvement")
    print("(none)" if view.empty else view.head(top_k).to_string())

    if show_bottom:
        _print_header("Bottom features by mean OOS improvement")
        print("(none)" if view.empty else view.tail(show_bottom).to_string())

    if not out.empty:
        _print_header("Summary")
        print(f"best_feature={out.index[0]}")
        print(f"positive_mean_improvement={(out['oos_improvement_mean'] > 0).sum()}/{len(out)}")
    return out if return_table else None


def feature_stability_report(
    X: pd.DataFrame,
    y: pd.Series,
    splits,
    method: str = "spearman",
    robust: str = "HC3",
    decimals: int = 3,
    top_k: int = 8,
    show_fragile: int = 5,
    min_valid_folds: int = 1,
    return_table: bool = False,
) -> Optional[pd.DataFrame]:
    X = _num_frame(X)
    y = _num_series(y)
    splits = list(splits)
    corr_fn = {"pearson": st.pearsonr, "spearman": st.spearmanr}[method.lower()]

    rows = []
    for col in X.columns:
        corr_values, beta_values, robust_t_values = [], [], []

        for train_idx, _ in splits:
            df = _xy(X.iloc[train_idx][col], y.iloc[train_idx])
            if len(df) < 8 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
                corr_values.append(np.nan)
                beta_values.append(np.nan)
                robust_t_values.append(np.nan)
                continue

            corr_values.append(float(corr_fn(df["x"], df["y"]).statistic))
            model = sm.OLS(df["y"], sm.add_constant(df[["x"]], has_constant="add")).fit()
            robust_model = model.get_robustcov_results(cov_type=robust)
            beta_values.append(float(model.params["x"]))
            robust_t_values.append(float(robust_model.tvalues[1]))

        corr_values = np.array(corr_values, dtype=float)
        beta_values = np.array(beta_values, dtype=float)
        robust_t_values = np.array(robust_t_values, dtype=float)

        rows.append(
            {
                "feature": col,
                "valid_folds": int(np.isfinite(corr_values).sum()),
                f"{method}_fold_values": corr_values.tolist(),
                f"{method}_mean": np.nanmean(corr_values) if np.isfinite(corr_values).any() else np.nan,
                f"{method}_std": np.nanstd(corr_values, ddof=1) if np.isfinite(corr_values).sum() > 1 else np.nan,
                f"{method}_sign_consistency": _sign_consistency(corr_values),
                "beta_fold_values": beta_values.tolist(),
                "beta_mean": np.nanmean(beta_values) if np.isfinite(beta_values).any() else np.nan,
                "beta_std": np.nanstd(beta_values, ddof=1) if np.isfinite(beta_values).sum() > 1 else np.nan,
                "beta_sign_consistency": _sign_consistency(beta_values),
                "robust_t_fold_values": robust_t_values.tolist(),
                "robust_t_mean": np.nanmean(robust_t_values) if np.isfinite(robust_t_values).any() else np.nan,
                "robust_t_std": np.nanstd(robust_t_values, ddof=1) if np.isfinite(robust_t_values).sum() > 1 else np.nan,
            }
        )

    out = pd.DataFrame(rows).set_index("feature")
    if min_valid_folds > 1:
        out = out[out["valid_folds"] >= min_valid_folds]
    out = out.sort_values([f"{method}_sign_consistency", "beta_sign_consistency", f"{method}_mean"], ascending=[False, False, False])

    stable = out[(out[f"{method}_sign_consistency"] >= 0.80) & (out["beta_sign_consistency"] >= 0.80)]
    stable_positive = stable[stable[f"{method}_mean"] >= 0].sort_values(f"{method}_mean", ascending=False)
    stable_negative = stable[stable[f"{method}_mean"] < 0].sort_values(f"{method}_mean", ascending=True)
    fragile = out[(out[f"{method}_sign_consistency"] < 0.60) | (out["beta_sign_consistency"] < 0.60)].sort_values(
        [f"{method}_sign_consistency", "beta_sign_consistency", f"{method}_mean"], ascending=[True, True, True]
    )

    _print_header(f"Feature Stability Report ({method.title()})")
    print(f"observations={len(y)}  features={X.shape[1]}  folds={len(splits)}")
    print(f"correlation={method}  robust_covariance={robust}")
    if min_valid_folds > 1:
        print(f"filtered: min_valid_folds >= {min_valid_folds}")

    def section(frame: pd.DataFrame, title: str, limit: int) -> None:
        _print_header(title)
        display = frame[
            [
                f"{method}_mean",
                f"{method}_std",
                f"{method}_sign_consistency",
                "beta_mean",
                "beta_sign_consistency",
                "robust_t_mean",
                "valid_folds",
            ]
        ].copy()
        display = display.rename(
            columns={
                f"{method}_mean": f"{method.title()} Corr",
                f"{method}_std": "Corr SD",
                f"{method}_sign_consistency": "Corr Sign",
                "beta_mean": "Beta",
                "beta_sign_consistency": "Beta Sign",
                "robust_t_mean": "Robust t",
                "valid_folds": "Valid Folds",
            }
        )
        if not display.empty:
            display[["Corr Sign", "Beta Sign"]] = display[["Corr Sign", "Beta Sign"]] * 100.0
        print("(none)" if display.empty else display.head(limit).round(decimals).to_string())

    section(stable_positive, f"Most stable positive {method.title()} signals", top_k)
    section(stable_negative, f"Most stable negative {method.title()} signals", top_k)
    section(fragile, "Fragile / sign-flipping features", show_fragile)
    if not out.empty:
        _print_header("Summary")
        print(f"median_abs_{method}={np.nanmedian(np.abs(out[f'{method}_mean'])):.{decimals}f}")
    return out if return_table else None


def redundancy_screen(
    X: pd.DataFrame,
    correlation_method: str = "spearman",
    corr_threshold: float = 0.8,
    vif_threshold: float = 5.0,
    decimals: int = 3,
    max_pairs: int | None = 50,
    return_tables: bool = False,
) -> Optional[Dict[str, Any]]:
    X = _num_frame(X)
    corr = X.corr(method=correlation_method.lower())

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairwise = (
        upper.stack()
        .rename(f"{correlation_method}_corr")
        .reset_index()
        .rename(columns={"level_0": "feature_left", "level_1": "feature_right"})
    )
    if not pairwise.empty:
        pairwise["abs_corr"] = pairwise[f"{correlation_method}_corr"].abs()
        pairwise = pairwise[pairwise["abs_corr"] >= corr_threshold].sort_values("abs_corr", ascending=False)
        pairwise["strength"] = pd.cut(
            pairwise["abs_corr"],
            bins=[0, 0.8, 0.9, 0.95, 1.0000001],
            labels=["high", "very high", "extreme", "near-perfect"],
            include_lowest=True,
        )
        pairwise["recommendation"] = "consider dropping one of the pair"
        pairwise = pairwise.round(decimals)
        if max_pairs is not None:
            pairwise = pairwise.head(max_pairs)

    vif = _vif_table(X, decimals=decimals, threshold=vif_threshold)
    suggested_drop = vif.index[vif["flag"] == "HIGH"].tolist() if not vif.empty else []

    _print_header("Redundancy Screen")
    print(f"observations={X.shape[0]}  features={X.shape[1]}")
    print(f"correlation={correlation_method}  corr_threshold={corr_threshold}  vif_threshold={vif_threshold}")

    _print_header(f"Correlated pairs (|corr| >= {corr_threshold:g})")
    print("(none)" if pairwise.empty else pairwise.to_string(index=False))

    _print_header("VIF (multicollinearity)")
    print("(none)" if vif.empty else vif.to_string())

    _print_header("Suggested drops")
    print(", ".join(suggested_drop) if suggested_drop else "(none)")

    if return_tables:
        return {"pairwise": pairwise, "vif": vif, "suggested_drop": suggested_drop}
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
    y = _num_series(y)
    X_candidates = _num_frame(X_candidates)
    X_base = pd.DataFrame(index=X_candidates.index) if X_base is None else _num_frame(X_base)
    splits = list(splits)
    metric = metric.lower()
    higher_is_better = _higher_is_better(metric)

    baseline_scores = []
    for train_idx, val_idx in splits:
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
        tr_mask, va_mask = y_tr.notna().to_numpy(), y_va.notna().to_numpy()

        if tr_mask.sum() < 5 or va_mask.sum() < 3:
            baseline_scores.append(np.nan)
            continue

        y_tr2, y_va2 = y_tr.iloc[tr_mask].to_numpy(), y_va.iloc[va_mask].to_numpy()
        if X_base.shape[1] == 0:
            baseline = np.full(len(y_va2), float(np.mean(y_tr2)))
            baseline_scores.append(_score_regression(y_va2, baseline, metric))
        else:
            model = _linear_pipeline(alpha=ridge_alpha, standardize=standardize)
            model.fit(X_base.iloc[train_idx].iloc[tr_mask], y_tr2)
            baseline_scores.append(_score_regression(y_va2, model.predict(X_base.iloc[val_idx].iloc[va_mask]), metric))

    rows = []
    for col in X_candidates.columns:
        gains, augmented = [], []

        for fold_id, (train_idx, val_idx) in enumerate(splits):
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            tr_mask, va_mask = y_tr.notna().to_numpy(), y_va.notna().to_numpy()

            if tr_mask.sum() < 5 or va_mask.sum() < 3 or not np.isfinite(baseline_scores[fold_id]):
                gains.append(np.nan)
                augmented.append(np.nan)
                continue

            X_tr = pd.concat([X_base.iloc[train_idx], X_candidates[[col]].iloc[train_idx]], axis=1).iloc[tr_mask]
            X_va = pd.concat([X_base.iloc[val_idx], X_candidates[[col]].iloc[val_idx]], axis=1).iloc[va_mask]
            y_tr2, y_va2 = y_tr.iloc[tr_mask].to_numpy(), y_va.iloc[va_mask].to_numpy()

            model = _linear_pipeline(alpha=ridge_alpha, standardize=standardize)
            model.fit(X_tr, y_tr2)
            score = _score_regression(y_va2, model.predict(X_va), metric)
            gain = (score - baseline_scores[fold_id]) if higher_is_better else (baseline_scores[fold_id] - score)

            augmented.append(score)
            gains.append(gain)

        gains = np.array(gains, dtype=float)
        augmented = np.array(augmented, dtype=float)
        baseline = np.array(baseline_scores, dtype=float)
        rows.append(
            {
                "feature": col,
                f"baseline_{metric}_mean": np.nanmean(baseline) if np.isfinite(baseline).any() else np.nan,
                f"augmented_{metric}_mean": np.nanmean(augmented) if np.isfinite(augmented).any() else np.nan,
                "incremental_gain_mean": np.nanmean(gains) if np.isfinite(gains).any() else np.nan,
                "incremental_gain_std": np.nanstd(gains, ddof=1) if np.isfinite(gains).sum() > 1 else np.nan,
                "improves_in_pct_folds": np.nanmean(gains > 0) if np.isfinite(gains).any() else np.nan,
                "valid_folds": int(np.isfinite(gains).sum()),
            }
        )

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
    table = stats.join(correlation, how="outer").join(univariate_ols, how="outer").join(univariate_oos, how="outer").join(stability, how="outer")

    if redundancy is not None:
        pairwise = redundancy.get("pairwise", pd.DataFrame())
        if not pairwise.empty:
            pair_max = pd.concat(
                [
                    pairwise[["feature_left", "abs_corr"]].rename(columns={"feature_left": "feature"}).set_index("feature"),
                    pairwise[["feature_right", "abs_corr"]].rename(columns={"feature_right": "feature"}).set_index("feature"),
                ]
            ).groupby(level=0)["abs_corr"].max()
            table["max_abs_pair_corr"] = pair_max.reindex(table.index)
        else:
            table["max_abs_pair_corr"] = np.nan

        vif = redundancy.get("vif", pd.DataFrame())
        if not vif.empty:
            table = table.join(vif.rename(columns={"VIF": "joint_vif"}), how="left")

    if incremental_gain is not None:
        table = table.join(incremental_gain, how="left")

    def col(name: str) -> pd.Series:
        return table[name] if name in table.columns else pd.Series(np.nan, index=table.index)

    flags = pd.DataFrame(index=table.index)
    flags["high_missingness"] = col("missing_rate") > max_missing_rate
    flags["marginal_corr_signal"] = (col("pearson_q_value") <= corr_alpha) | (col("spearman_q_value") <= corr_alpha)
    flags["ols_signal"] = col("Robust Pr(>|t|)") <= ols_alpha
    flags["oos_signal"] = col("oos_improvement_mean") > min_oos_improvement
    flags["stable_sign"] = col("beta_sign_consistency") >= min_sign_consistency
    flags["redundant_pairwise"] = col("max_abs_pair_corr") >= redundancy_corr_threshold
    flags["high_vif"] = col("joint_vif") >= 10

    if incremental_gain is not None:
        flags["incremental_gain"] = col("incremental_gain_mean") > 0
        flags["no_incremental_gain"] = col("incremental_gain_mean").notna() & ~flags["incremental_gain"]

    signal_count = flags[["marginal_corr_signal", "ols_signal", "oos_signal", "stable_sign"]].sum(axis=1)
    table["research_decision"] = np.select(
        [
            (signal_count >= 3) & ~flags["redundant_pairwise"] & ~flags["high_vif"],
            (signal_count <= 1) & ~flags["oos_signal"],
        ],
        ["keep_candidate", "drop_candidate"],
        default="watch",
    )
    table["research_notes"] = flags.apply(lambda row: ", ".join(row.index[row.to_numpy()]) or "weak_evidence", axis=1)

    preferred = [
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
    ordered = [c for c in preferred if c in table.columns]
    table = table[ordered + [c for c in table.columns if c not in ordered]]

    rank = pd.Series({"keep_candidate": 0, "watch": 1, "drop_candidate": 2})
    return (
        table.assign(_rank=table["research_decision"].map(rank).fillna(99))
        .sort_values(["_rank", "oos_improvement_mean", "Robust t value"], ascending=[True, False, False], na_position="last")
        .drop(columns="_rank")
    )


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
    X = _num_frame(X)
    y = _num_series(y)
    splits = make_cv_splits(len(X), n_splits=n_splits, shuffle=shuffle, random_state=random_state) if splits is None else list(splits)

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
        return_table=True,
    )
    results["stability"] = feature_stability_report(X, y, splits=splits, return_table=True)

    redundancy_cols = list(redundancy_features) if redundancy_features is not None else list(X.columns)
    results["redundancy"] = redundancy_screen(
        X[redundancy_cols],
        corr_threshold=redundancy_corr_threshold,
        return_tables=True,
    )

    results["joint_ols"] = joint_ols_report(X[list(joint_ols_features)], y) if joint_ols_features else None

    if base_features:
        base_features = list(base_features)
        candidate_cols = [c for c in X.columns if c not in base_features]
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
