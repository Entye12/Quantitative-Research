import pandas as pd
import numpy as np
from scipy import stats
from math import gcd
from functools import reduce
from scipy.stats import norm
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import gaussian_kde


import matplotlib.pyplot as plt



def _print_header(title: str) -> None:
    bar = "─" * len(title)
    print(f"\n{title}\n{bar}")


def _print_kv(data: dict, width: int = 28) -> None:
    for k, v in data.items():
        print(f"{k:<{width}} {v}")


def _print_trades_summary(df: pd.DataFrame, qc: dict) -> None:
    _print_header("Trades Data Summary")

    print("Shape")
    print("-----")
    print(f"Raw rows        : {qc['n_rows_raw']}")
    print(f"Rows after prep : {qc['n_rows_after']}")

    print("\nMissing & invalid values")
    print("------------------------")
    print(f"Missing ts      : {qc['n_ts_na']}")
    print(f"Missing price   : {qc['n_price_na']}")
    print(f"Missing qty     : {qc['n_qty_na']}")
    print(f"price <= 0      : {qc['n_price_leq_0_raw']}")
    print(f"qty <= 0        : {qc['n_qty_leq_0_raw']}")
    print(f"Bad rows total  : {qc['n_bad_rows']}")

    print("\nTimestamp quality")
    print("-----------------")
    print(f"Decreasing ts (raw)   : {qc['n_ts_decreasing_raw']}")
    print(f"Decreasing ts (after) : {qc['n_ts_decreasing_after']}")
    print(f"Equal ts (after)      : {qc['n_ts_equal_after']}")

    if len(df) > 0:
        print("\nPrepared data overview")
        print("----------------------")
        print(f"First timestamp : {df['ts'].min()}")
        print(f"Last timestamp  : {df['ts'].max()}")
        print(f"Price range     : [{df['price'].min():.6g}, {df['price'].max():.6g}]")
        print(f"Qty range       : [{df['qty'].min():.6g}, {df['qty'].max():.6g}]")

    print()


def _fmt_stat_value(x):
    if pd.isna(x):
        return "NA"
    if isinstance(x, pd.Timestamp):
        return x.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(x, (int, np.integer)):
        return f"{int(x)}"
    if isinstance(x, (float, np.floating)):
        return f"{x:.6g}"
    return str(x)


def print_tick_summary(res: dict, title: str = "Tick Size Estimation") -> None:
    _print_header(title)

    tick = res.get("tick_est", np.nan)
    fit = res.get("multiple_fit_rate", np.nan)

    print(f"{'Estimated tick':<22}: {tick}")
    
    if not pd.isna(fit):
        print(f"{'Multiple fit rate':<22}: {100*fit:.2f} %")

    if "small_cutoff" in res:
        print(f"{'Small move cutoff':<22}: {res['small_cutoff']}")

    if "n_nonzero_deltas" in res:
        print(f"{'Non-zero price jumps':<22}: {res['n_nonzero_deltas']:,}")

    if "n_samples" in res:
        print(f"{'Samples analysed':<22}: {res['n_samples']:,}")

    if "most_frequent_jump" in res:
        print(f"{'Most frequent jump':<22}: {res['most_frequent_jump']}")

    # Print top deltas table if available
    top = res.get("top_deltas")
    if top:
        print("\nMost frequent deltas")
        print("--------------------")
        print(f"{'delta':<10}{'count'}")
        for d, c in top:
            print(f"{d:<10}{c}")

    print()


def print_basic_stats_summary(stats: dict) -> None:
    _print_header("Basic Trade Statistics")

    table = pd.Series(stats, name="value").to_frame()
    table["value"] = table["value"].map(_fmt_stat_value)

    print(table.to_string())
    print()


def print_vol_volume_summary(res: dict, title: str = "Volatility–Volume Link") -> None:
    _print_header(title)

    print(f"{'Bin size':<24}: {res.get('bin_size')}")
    print(f"{'Volume measure':<24}: {res.get('volume_measure')}")
    print(f"{'Volatility proxy':<24}: {res.get('vol_proxy')}")
    print(f"{'Bins used':<24}: {res.get('n_bins_used'):,}")

    print("\nCorrelations")
    print("------------")

    if "pearson_corr" in res:
        print(f"{'Pearson corr':<24}: {res.get('pearson_corr'):.4f}")
        print(f"{'Pearson p-value':<24}: {res.get('pearson_pvalue'):.3g}")

    if "spearman_corr" in res:
        print(f"{'Spearman corr':<24}: {res.get('spearman_corr'):.4f}")
        print(f"{'Spearman p-value':<24}: {res.get('spearman_pvalue'):.3g}")

    if "pearson_corr_loglog" in res:
        print(f"{'Log–log Pearson corr':<24}: {res.get('pearson_corr_loglog'):.4f}")
        print(f"{'Log–log p-value':<24}: {res.get('loglog_pvalue'):.3g}")

    print()


def prepare_trades_df(
    df_raw: pd.DataFrame,
    *,
    drop_bad_rows: bool = False,
    print_summary: bool = True
):
    """
    Step 1: standardize a single stock-day trades dataframe.

    Input:
      df_raw with columns ['price', 'qty', 'ts']

    Output:
      df: cleaned & sorted dataframe, with derived columns ['price', 'qty', 'ts','dt','dprice','logp','dlogp']
      qc: dict of quality-control metrics
    """
    qc = {}

    df = df_raw.loc[:, ["price", "qty", "ts"]].copy()

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")

    qc["n_rows_raw"] = len(df)
    qc["n_ts_na"] = int(df["ts"].isna().sum())
    qc["n_price_na"] = int(df["price"].isna().sum())
    qc["n_qty_na"] = int(df["qty"].isna().sum())

    ts_raw = df["ts"]
    qc["n_ts_decreasing_raw"] = int((ts_raw.diff() < pd.Timedelta(0)).sum())

    qc["n_price_leq_0_raw"] = int((df["price"] <= 0).sum(skipna=True))
    qc["n_qty_leq_0_raw"] = int((df["qty"] <= 0).sum(skipna=True))

    bad_mask = (
        df["ts"].isna()
        | df["price"].isna()
        | df["qty"].isna()
        | (df["price"] <= 0)
        | (df["qty"] <= 0)
    )
    qc["n_bad_rows"] = int(bad_mask.sum())

    if drop_bad_rows:
        df = df.loc[~bad_mask].copy()

    df.sort_values(["ts"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    qc["n_rows_after"] = len(df)
    qc["n_ts_decreasing_after"] = int((df["ts"].diff() < pd.Timedelta(0)).sum())
    qc["n_ts_equal_after"] = int((df["ts"].diff() == pd.Timedelta(0)).sum())

    df["dt"] = df["ts"].diff()
    df["dprice"] = df["price"].diff()
    df["logp"] = np.log(df["price"].astype(float))
    df["dlogp"] = df["logp"].diff()

    if print_summary:
        _print_trades_summary(df, qc)

    return df, qc


def compute_basic_stats(df: pd.DataFrame, *, print_summary: bool = True) -> dict:
    """
    Compute general descriptive statistics for ONE stock-day.

    Input
    -----
    df : pd.DataFrame
        Trades dataframe sorted by timestamp.
        Expected columns:
            - 'ts'     : trade timestamp (datetime-like)
            - 'price'  : trade price (strictly positive numeric)
            - 'qty'    : trade quantity (strictly positive numeric)

        Optional precomputed columns:
            - 'dt'     : inter-trade time difference, typically df['ts'].diff()
            - 'dprice' : price increment, typically df['price'].diff()
            - 'dlogp'  : log-price increment, typically np.log(df['price']).diff()

    Returns
    -------
    dict
        Dictionary of descriptive statistics for this stock-day.
    """
    out = {}
    out["n_trades"] = int(len(df))

    if len(df) == 0:
        for k in [
            "ts_min", "ts_max", "span_seconds",
            "n_duplicate_ts", "dt_median_ms", "dt_p90_ms", "dt_p99_ms", "dt_max_ms",
            "price_min", "price_median", "price_max", "frac_zero_dprice",
            "ret_mean", "ret_std", "ret_p95_abs", "ret_p99_abs",
            "qty_sum", "qty_mean", "qty_median", "qty_p95", "qty_p99", "qty_max",
            "notional_sum"
        ]:
            out[k] = np.nan

        if print_summary:
            print_basic_stats_summary(out)
        return out

    # --- Time span ---
    ts_min = df["ts"].iloc[0]
    ts_max = df["ts"].iloc[-1]
    out["ts_min"] = ts_min
    out["ts_max"] = ts_max
    out["span_seconds"] = float((ts_max - ts_min) / pd.Timedelta(seconds=1))

    # --- Inter-trade durations ---
    dt = df["dt"] if "dt" in df.columns else df["ts"].diff()
    dt_ms = (dt / pd.Timedelta(milliseconds=1)).to_numpy()
    dt_ms = dt_ms[~np.isnan(dt_ms)]

    out["n_duplicate_ts"] = int(np.sum(dt_ms == 0)) if dt_ms.size else 0

    if dt_ms.size:
        out["dt_median_ms"] = float(np.quantile(dt_ms, 0.50))
        out["dt_p90_ms"] = float(np.quantile(dt_ms, 0.90))
        out["dt_p99_ms"] = float(np.quantile(dt_ms, 0.99))
        out["dt_max_ms"] = float(np.max(dt_ms))
    else:
        out["dt_median_ms"] = np.nan
        out["dt_p90_ms"] = np.nan
        out["dt_p99_ms"] = np.nan
        out["dt_max_ms"] = np.nan

    # --- Price stats ---
    price = df["price"].to_numpy(dtype=float)
    out["price_min"] = float(np.nanmin(price))
    out["price_median"] = float(np.nanmedian(price))
    out["price_max"] = float(np.nanmax(price))

    dprice = df["dprice"] if "dprice" in df.columns else df["price"].diff()
    dprice = dprice.to_numpy(dtype=float)
    dprice = dprice[~np.isnan(dprice)]
    out["frac_zero_dprice"] = float(np.mean(dprice == 0)) if dprice.size else np.nan

    # --- Event-time log return stats ---
    dlogp = df["dlogp"] if "dlogp" in df.columns else np.log(df["price"]).diff()
    r = dlogp.to_numpy(dtype=float)
    r = r[~np.isnan(r)]

    if r.size:
        out["ret_mean"] = float(np.mean(r))
        out["ret_std"] = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        abs_r = np.abs(r)
        out["ret_p95_abs"] = float(np.quantile(abs_r, 0.95))
        out["ret_p99_abs"] = float(np.quantile(abs_r, 0.99))
    else:
        out["ret_mean"] = np.nan
        out["ret_std"] = np.nan
        out["ret_p95_abs"] = np.nan
        out["ret_p99_abs"] = np.nan

    # --- Quantity stats ---
    qty = df["qty"].to_numpy(dtype=float)
    out["qty_sum"] = float(np.nansum(qty))
    out["qty_mean"] = float(np.nanmean(qty))
    out["qty_median"] = float(np.nanmedian(qty))
    out["qty_p95"] = float(np.nanquantile(qty, 0.95))
    out["qty_p99"] = float(np.nanquantile(qty, 0.99))
    out["qty_max"] = float(np.nanmax(qty))

    # --- Notional ---
    out["notional_sum"] = float(np.nansum(df["price"].to_numpy(dtype=float) * qty))

    if print_summary:
        print_basic_stats_summary(out)

    return out


def updates_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("DataFrame is empty")
        return

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)

    # -------------------------
    # Basic information
    # -------------------------
    n_rows, n_cols = df.shape
    ts_min, ts_max = df["ts"].iloc[0], df["ts"].iloc[-1]

    _print_header("Basic information")
    _print_kv({
        "n_rows": n_rows,
        "n_cols": n_cols,
        "ts_min": ts_min,
        "ts_max": ts_max,
        "span_seconds": (ts_max - ts_min).total_seconds()
    })

    # -------------------------
    # Opening and activity
    # -------------------------
    valid_mask = df["bp"].notna() & df["ap"].notna()

    if valid_mask.any():
        opening_ts = df.loc[valid_mask, "ts"].iloc[0]
        olb = df[df["ts"] >= opening_ts]
        empty_ratio = (olb["ap"].isna() & olb["bp"].isna()).mean()
    else:
        opening_ts = None
        empty_ratio = 1.0

    _print_header("Opening and activity")
    _print_kv({
        "opening_time": opening_ts,
        "empty OLB in activity (%)": f"{empty_ratio:.2%}"
    })

    # -------------------------
    # Missing values
    # -------------------------
    miss = {}
    for col in ["ap", "bp", "aq", "bq"]:
        if col in df.columns:
            miss[f"missing {col} (%)"] = f"{df[col].isna().mean():.2%}"

    _print_header("Missing data ratios")
    _print_kv(miss)

    # -------------------------
    # Inter-update timing
    # -------------------------
    dt = df["dt"] if "dt" in df.columns else df["ts"].diff()
    dt_ms = dt.dt.total_seconds().to_numpy() * 1000 if hasattr(dt, "dt") else dt.to_numpy()
    dt_ms = dt_ms[~np.isnan(dt_ms)]

    timing = {
        "n_duplicate_ts": int(np.sum(dt_ms == 0)) if dt_ms.size else 0
    }

    if dt_ms.size:
        timing.update({
            "dt_median_ms": float(np.median(dt_ms)),
            "dt_p90_ms": float(np.percentile(dt_ms, 90)),
            "dt_p99_ms": float(np.percentile(dt_ms, 99)),
            "dt_max_ms": float(np.max(dt_ms)),
        })

    _print_header("Inter-update timing (ms)")
    _print_kv(timing)

    # -------------------------
    # Price statistics
    # -------------------------
    price_stats = {}

    for col in ["ap", "bp"]:
        vals = df[col].to_numpy(dtype=float)
        if not np.all(np.isnan(vals)):
            price_stats[f"{col}_min"] = float(np.nanmin(vals))
            price_stats[f"{col}_median"] = float(np.nanmedian(vals))
            price_stats[f"{col}_max"] = float(np.nanmax(vals))
        else:
            price_stats[f"{col}_min"] = np.nan
            price_stats[f"{col}_median"] = np.nan
            price_stats[f"{col}_max"] = np.nan

    df["spread"] = df["ap"] - df["bp"]
    price_stats["spread_positive"] = not (df["spread"].round(4) < 0).any()

    _print_header("Price statistics")
    _print_kv(price_stats)


def estimate_tick_size_quantile(
    df: pd.DataFrame,
    *,
    rounding_decimals: int = 6,
    top_k: int = 10,
    print_summary: bool = True,
) -> dict:
    """
    Estimate tick size from trade prices using frequent small price changes.
    """

    out = {}

    if len(df) < 2:
        out.update({
            "tick_est": np.nan,
            "n_nonzero_deltas": 0,
            "top_deltas": [],
            "multiple_fit_rate": np.nan
        })
        if print_summary:
            print_tick_summary(out, "Tick Size Estimation (Quantile)")
        return out

    # Absolute non-zero price changes
    d = np.abs(df["price"].diff().to_numpy(dtype=float))
    d = d[~np.isnan(d)]
    d = d[d > 0]

    out["n_nonzero_deltas"] = int(d.size)

    if d.size == 0:
        out.update({
            "tick_est": np.nan,
            "top_deltas": [],
            "multiple_fit_rate": np.nan
        })
        if print_summary:
            print_tick_summary(out, "Tick Size Estimation (Quantile)")
        return out

    # Reduce float noise
    d_rounded = np.round(d, rounding_decimals)

    # Focus on small price moves
    cutoff = np.quantile(d_rounded, 0.20)
    small = d_rounded[d_rounded <= cutoff]
    if small.size == 0:
        small = d_rounded

    # Most frequent small delta = tick estimate
    vals, counts = np.unique(small, return_counts=True)
    tick = float(vals[np.argmax(counts)])

    # Diagnostics: most frequent deltas
    vals_all, counts_all = np.unique(d_rounded, return_counts=True)
    order = np.argsort(counts_all)[::-1]
    top = [(float(vals_all[i]), int(counts_all[i])) for i in order[:top_k]]

    # Check if deltas are multiples of tick
    ratio = d_rounded / tick
    nearest = np.round(ratio)
    ok = np.isclose(ratio, nearest, rtol=1e-4, atol=1e-8)

    out.update({
        "tick_est": tick,
        "top_deltas": top,
        "multiple_fit_rate": float(np.mean(ok)),
        "small_cutoff": float(cutoff)
    })

    if print_summary:
        print_tick_summary(out, "Tick Size Estimation (Quantile)")

    return out


def estimate_tick_size_gcd(
    df: pd.DataFrame,
    precision: int = 10,
    print_summary: bool = True
) -> dict:
    """
    Estimate tick size using the GCD of frequent price jumps.
    """

    prices = df["price"].dropna().to_numpy()
    deltas = np.abs(np.diff(prices))
    deltas = deltas[deltas > 1e-9]

    if len(deltas) == 0:
        res = {"tick_est": np.nan, "confidence": 0}
        if print_summary:
            print_tick_summary(res, "Tick Size Estimation (GCD)")
        return res

    # Convert to integers to avoid float noise
    multiplier = 10**6
    d_int = np.round(deltas * multiplier).astype(np.int64)

    # Unique jumps and frequencies
    vals, counts = np.unique(d_int, return_counts=True)

    order = np.argsort(counts)[::-1]
    vals = vals[order]
    counts = counts[order]

    # GCD of most frequent jumps
    top_n = min(5, len(vals))
    representative = vals[:top_n]

    estimated_tick_int = reduce(gcd, representative)

    tick_est = estimated_tick_int / multiplier

    # Check proportion of multiples
    fit_rate = np.mean((d_int % estimated_tick_int) == 0)

    res = {
        "tick_est": tick_est,
        "multiple_fit_rate": fit_rate,
        "most_frequent_jump": vals[0] / multiplier,
        "n_samples": len(deltas),
    }

    if print_summary:
        print_tick_summary(res, "Tick Size Estimation (GCD)")

    return res


def binned_lastprice_returns(df: pd.DataFrame, *, bin_size: str = "1min") -> pd.Series:
    """
    Compute log returns of the last traded price in each time bin.

    Input
    -----
    df : pd.DataFrame
        Trades dataframe containing columns: 'ts', 'price'.

    Output
    ------
    pd.Series
        Log returns indexed by bin time.

    Notes
    -----
    Returns are computed separately within each day to avoid
    overnight jumps between consecutive trading days.
    """
    if len(df) == 0:
        return pd.Series(dtype=float)

    # Prepare clean time-indexed price series
    d = df[["ts", "price"]].copy()
    d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
    d = d.dropna(subset=["ts", "price"]).set_index("ts").sort_index()

    # Last traded price in each calendar-time bin
    last_p = d["price"].resample(bin_size).last()

    # Log returns computed day by day
    log_p = np.log(last_p)
    r = log_p.groupby(log_p.index.date).diff()

    r.name = "ret"
    return r


def intraday_activity(df: pd.DataFrame, *, bin_size: str = "5min") -> pd.DataFrame:
    """
    Compute intraday activity curves for ONE stock-day.

    Input
    -----
    df : pd.DataFrame
        Trades dataframe containing columns: 'ts', 'price', 'qty'.

    Output
    ------
    DataFrame indexed by time bins with:
        n_trades, volume, notional, vwap, last_price.

    Notes
    -----
    The resulting dataframe is designed to feed plotting functions
    such as `plot_volume_hist`.
    """

    if len(df) == 0:
        return pd.DataFrame(columns=["n_trades", "volume", "notional", "vwap", "last_price"])

    # Prepare clean time-indexed dataframe
    d = df[["ts", "price", "qty"]].copy()
    d["ts"] = pd.to_datetime(d["ts"], errors="coerce")
    d = d.dropna(subset=["ts", "price", "qty"]).set_index("ts").sort_index()

    # Aggregate core activity measures per time bin
    n_trades = d["price"].resample(bin_size).count().rename("n_trades")
    volume = d["qty"].resample(bin_size).sum().rename("volume")
    notional = (d["price"] * d["qty"]).resample(bin_size).sum().rename("notional")

    # Volume-weighted average price
    vwap = (notional / volume).rename("vwap").where(volume > 0)

    # Last trade price in each bin
    last_price = d["price"].resample(bin_size).last().rename("last_price")

    out = pd.concat([n_trades, volume, notional, vwap, last_price], axis=1)

    return out


def vol_volume_link(
    df: pd.DataFrame,
    *,
    bin_size: str = "1min",
    use: str = "volume",
    vol_proxy: str = "sq_ret",
    min_bins: int = 30,
    eps: float = 1e-16,
    print_summary: bool = True,
) -> dict:
    """
    Compute volatility-volume link for ONE stock-day.
    
    Inputs:
      df: cleaned trades df
      activity: output of intraday_activity(df, bin_size)
    Output:
      dict with correlations and counts
    """
    out = {}

    # 1) returns on same bins
    activity = intraday_activity(df,bin_size=bin_size)
    r = binned_lastprice_returns(df, bin_size=bin_size)

    # 2) volatility proxy per bin
    if vol_proxy == "sq_ret":
        v = (r ** 2).rename("vol")
    elif vol_proxy == "abs_ret":
        v = (r.abs()).rename("vol")
    else:
        raise ValueError("vol_proxy must be 'sq_ret' or 'abs_ret'")

    # 3) choose volume measure
    if use not in activity.columns:
        raise ValueError(f"activity must contain column '{use}'. Got {list(activity.columns)}")
    x = activity[use].rename(use)

    # 4) align and drop NaNs / empty bins
    joined = pd.concat([x, v], axis=1).dropna()
   
    joined = joined[(joined[use] > 0) & (joined["vol"] >= 0)]

    out["n_bins_used"] = int(len(joined))
    out["bin_size"] = bin_size
    out["vol_proxy"] = vol_proxy
    out["volume_measure"] = use


    if len(joined) < min_bins:
        out["pearson_corr"] = np.nan
        out["spearman_corr"] = np.nan
        out["pearson_corr_loglog"] = np.nan
        return out

    # 5) correlations
    corr_p, pval_p = pearsonr(joined[use], joined["vol"])
    out["pearson_corr"] = float(corr_p)
    out["pearson_pvalue"] = float(pval_p)

    corr_s, pval_s = spearmanr(joined[use], joined["vol"])
    out["spearman_corr"] = float(corr_s)
    out["spearman_pvalue"] = float(pval_s)

    log_x = np.log(joined[use])
    log_v = np.log(joined["vol"] + eps)
    corr_l, pval_l = pearsonr(log_x, log_v)
    out["pearson_corr_loglog"] = float(corr_l)
    out["loglog_pvalue"] = float(pval_l)

    if print_summary:
        print_vol_volume_summary(out)


    return out


def event_time_returns(df: pd.DataFrame, *, step: int = 1) -> np.ndarray:
    """
    Event-time log returns.
    
    step=1  → every trade
    step=5  → every 5th trade, etc.
    """
    if len(df) <= step:
        return np.array([])

    prices = df["price"].to_numpy(dtype=float)

    # sample every `step` trades
    sampled = prices[::step]

    # log returns in event time
    r = np.diff(np.log(sampled))

    return r


def calendar_time_returns(df: pd.DataFrame, *, freq: str = "1min"):
    """
    Calendar-time log returns with interval weights.

    Returns:
      r      : array of log returns
      w      : array of weights (interval lengths in seconds)
    """
    if len(df) == 0:
        return np.array([]), np.array([])

    d = df[["ts", "price"]].copy().set_index("ts").sort_index()

    # last traded price on regular grid
    p = d["price"].resample(freq).last()

    # log returns
    logp = np.log(p)
    r = logp.diff()

    # weights = length of each interval (seconds)
    dt = p.index.to_series().diff().dt.total_seconds()

    # drop NaNs
    mask = (~r.isna()) & (~dt.isna())
    r = r[mask].to_numpy(dtype=float)
    w = dt[mask].to_numpy(dtype=float)

    return r, w


def weighted_return_stats(r: np.ndarray, w: np.ndarray) -> dict:
    """
    Compute weighted moments for calendar-time returns.
    """
    out = {}
    if len(r) == 0:
        out["var"] = np.nan
        out["std"] = np.nan
        out["mean"] = np.nan
        out["kurtosis"] = np.nan
        return out

    w = w / np.sum(w)  # normalize weights

    mean = np.sum(w * r)
    var = np.sum(w * (r - mean) ** 2)
    std = np.sqrt(var)

    # weighted kurtosis
    kurt = np.sum(w * (r - mean) ** 4) / (var ** 2) if var > 0 else np.nan

    out["mean"] = float(mean)
    out["var"] = float(var)
    out["std"] = float(std)
    out["kurtosis"] = float(kurt)

    return out


# Groupby the prices that occurs on the same timestamps as to avoid to bias the returns toward 0.
def group_by_tolerance(df, tolerance="1s", price_col="price", qty_col="qty", ts_col="ts",
                          method="vwap"):
    """
    Collapse multiple rows into tolerance-sized time buckets.
    tolerance examples: "1s", "100ms", "500ms", "1min"
    method: "mean" or "vwap"
    """
    d = df[[ts_col, price_col, qty_col]].copy()
    d[ts_col] = pd.to_datetime(d[ts_col], utc=True, errors="coerce")
    d = d.dropna(subset=[ts_col, price_col]).sort_values(ts_col)

    if method == "mean":
        out = (
            d.set_index(ts_col)
             .groupby(pd.Grouper(freq=tolerance))[price_col]
             .mean()
             .dropna()
             .to_frame("price")
        )
    elif method == "vwap":
        d = d.set_index(ts_col)
        def _vwap(g):
            q = g[qty_col].to_numpy()
            p = g[price_col].to_numpy()
            # if qty missing/zero, fallback to mean
            if np.nansum(q) <= 0:
                return np.nanmean(p)
            return np.nansum(p * q) / np.nansum(q)

        out = (
            d.groupby(pd.Grouper(freq=tolerance))
             .apply(_vwap)
             .dropna()
             .to_frame("price")
        )
    else:
        raise ValueError("method must be 'mean' or 'vwap'")

    out = out.reset_index().rename(columns={ts_col: "ts"})
    return out


def prepare_updates(df: pd.DataFrame, *, drop_bad_rows: bool = False, ticksize: float):
    """
    Clean order book updates and compute basic microstructure features.

    Adds:
    - mid price
    - bid–ask spread
    - order book imbalance
    - weighted mid price
    """
    
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])

    if drop_bad_rows:
        df = df.dropna(subset=["bp", "ap", "bq", "aq"])
        df = df[(df["bq"] > 0) & (df["aq"] > 0)]
        df = df[df["ap"] >= df["bp"]]

    df["mid"] = (df["ap"]+df["bp"])/2
    df["spread"] = df["ap"] - df["bp"]
    df["imbalance"] = (df["bq"]-df["aq"])/(df["bq"]+df["aq"])
    df["wmid"] = (df["ap"]*df["bq"] + df["bp"]*df["aq"])/(df["aq"] + df["bq"])
    
    return df


