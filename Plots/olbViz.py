from __future__ import annotations

from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

from Analysis.olb import (
    binned_lastprice_returns,
    calendar_time_returns,
    intraday_activity,
    weighted_return_stats,
)


sns.set_theme(style="whitegrid", context="notebook")


def _get_ax(ax: Optional[plt.Axes], figsize: Tuple[int, int] = (10, 4)) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _safe_log_bins(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if vmin <= 0:
        raise ValueError("Log-scale bins require strictly positive values.")

    if np.isclose(vmin, vmax):
        vmin = max(vmin * 0.9, np.finfo(float).eps)
        vmax = vmax * 1.1

    return np.logspace(np.log10(vmin), np.log10(vmax), bins)


def plot_price_event_time(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    max_points: int = 200_000,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plot trade price in event time (trade index).

    Input
    -----
    df : pd.DataFrame
        Trades dataframe containing a 'price' column.

    Output
    ------
    Displays price as a function of trade index.
    Returns the matplotlib axes used for the plot.
    """

    n = len(df)
    if n == 0:
        return None

    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        plot_df = pd.DataFrame(
            {"trade_index": idx, "price": df["price"].to_numpy(dtype=float)[idx]}
        )
    else:
        plot_df = pd.DataFrame(
            {
                "trade_index": np.arange(n),
                "price": df["price"].to_numpy(dtype=float),
            }
        )

    ax = _get_ax(ax)
    sns.lineplot(data=plot_df, x="trade_index", y="price", ax=ax, linewidth=1.2)
    ax.set_xlabel("Trade index (event time)")
    ax.set_ylabel("Price")
    ax.set_title(title or "Price in Event Time")
    ax.figure.tight_layout()
    return ax


def plot_dt_hist(
    df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    bins: int = 100,
    log_x: bool = True,
    ax: Optional[plt.Axes] = None,
) -> Optional[plt.Axes]:
    """
    Plot histogram of inter-trade time intervals.

    Input
    -----
    df : pd.DataFrame
        Trades dataframe containing:
        - 'ts' : timestamps
        - optional 'dt' : precomputed df['ts'].diff()

    Output
    ------
    Displays a histogram of positive dt values (in milliseconds).
    Returns the matplotlib axes used for the plot.
    """
    
    dt = df["dt"] if "dt" in df.columns else df["ts"].diff()

    if pd.api.types.is_timedelta64_dtype(dt):
        dt_ms = (dt.dt.total_seconds() * 1000.0).to_numpy(dtype=float)
    else:
        dt_ms = pd.to_numeric(dt, errors="coerce").to_numpy(dtype=float)

    dt_ms = dt_ms[np.isfinite(dt_ms)]
    dt_ms = dt_ms[dt_ms > 0]
    if dt_ms.size == 0:
        return None

    ax = _get_ax(ax)
    hist_bins = _safe_log_bins(dt_ms, bins) if log_x else bins
    sns.histplot(x=dt_ms, bins=hist_bins, ax=ax, color="#4C72B0",edgecolor ="black",linewidth=0.5)
    

    if log_x:
        ax.set_xscale("log")

    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("Count")
    ax.set_title(title or "Inter-trade Duration Distribution")
    ax.figure.tight_layout()
    return ax


def plot_qty_hist(df, *, title="Distribution of Trade Quantities", bins=100, log_x=True, label="Trade Sizes"):
    """
    Plot histogram of trade quantities.

    Input
    -----
    df : pd.DataFrame
        Trades dataframe containing a 'qty' column.

    Output
    ------
    Displays the distribution of positive trade quantities.
    """

    # Extract quantity values and remove NaNs / non-positive trades
    qty = df["qty"].to_numpy(dtype=float)
    qty = qty[(np.isfinite(qty)) & (qty > 0)]

    if qty.size == 0:
        print("No valid quantity data found.")
        return

    
    plt.figure(figsize=(10, 6))
    
    # Bins
    hist_bins = bins
    if log_x:
        # Create geometrically spaced bins for the log scale
        hist_bins = np.logspace(np.log10(qty.min()), np.log10(qty.max()), bins)

    # histplot
    ax = sns.histplot(
        data=qty,
        bins=hist_bins,
        color="#4C72B0",
        edgecolor="black",  
        linewidth=0.5,     
        alpha=0.7,
        label=label
    )

    if log_x:
        ax.set_xscale("log")

    # Formatting
    ax.set_xlabel("Quantity of shares")
    ax.set_ylabel("Frequency (count)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
    return ax


def plot_volume_hist(df, *, title="Trade Volume Evolution", color="steelblue", label="Volume per bin"):
    """
    Plot trade volume over time using the intraday activity dataframe.

    Input
    -----
    df : pd.DataFrame
        Output of `intraday_activity()` containing a 'volume' column
        indexed by time bins.

    Output
    ------
    Displays the volume traded in each time bin.
    """

    if df.empty:
        print("Dataframe is empty.")
        return

    plt.figure(figsize=(12, 6))

    # Estimate bin width from index spacing
    try:
        bin_width = df.index.to_series().diff().median()
    except Exception:
        bin_width = pd.Timedelta(minutes=1)

    # Bar plot with visible bin edges
    plt.bar(
        df.index,
        df["volume"],
        width=bin_width,
        color=color,
        edgecolor="black",   
        linewidth=0.5,
        alpha=0.85,
        label=label,
    )

    plt.xlabel("Time")
    plt.ylabel("Volume (Number of shares)")
    plt.title(title)

    plt.legend()

    # Grid only on volume axis
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_activity_and_variance(
    df,
    bin_size="5min",
    activity_col="n_trades",
    vol_proxy="sq_ret",
    var_mode="cumsum",
):
    """
    Plot cumulative trading activity and return variation through the day.

    Input
    -----
    df : pd.DataFrame
        Trades dataframe used to compute intraday activity and binned returns.

    Output
    ------
    Displays a two-axis plot:
        - left axis: cumulative activity
        - right axis: cumulative or running return variation
    """


    # Activity per bin
    act = intraday_activity(df, bin_size=bin_size)
    if activity_col not in act.columns:
        raise ValueError(f"{activity_col=} not in intraday_activity output")

    a = act[activity_col].dropna()

    # Returns per bin
    r = binned_lastprice_returns(df, bin_size=bin_size).dropna()

    # Volatility proxy
    if vol_proxy == "sq_ret":
        v = (r**2).rename("vol")
    elif vol_proxy == "abs_ret":
        v = r.abs().rename("vol")
    else:
        raise ValueError("vol_proxy must be 'sq_ret' or 'abs_ret'")

    # Align the two series
    joined = pd.concat([a.rename("act"), v], axis=1).dropna()

    # Curves
    act_curve = joined["act"].cumsum()

    if var_mode == "cumsum":
        var_curve = joined["vol"].cumsum()
        var_label = "Cumulative realized variance (Σ r²)"
    else:
        var_curve = joined["vol"].expanding().mean()
        var_label = "Running variance proxy (mean r²)"

    # Plot
    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.plot(
        act_curve.index,
        act_curve.values,
        color=sns.color_palette()[3],
        linewidth=2,
        label=f"Cumulative {activity_col.replace('_',' ')}",
    )

    ax1.set_ylabel(activity_col.replace("_", " ").title())

    ax2 = ax1.twinx()

    ax2.plot(
        var_curve.index,
        var_curve.values,
        color=sns.color_palette()[0],
        linewidth=2,
        label=var_label,
    )

    ax2.set_ylabel(var_label)

    ax1.set_title(f"Trading Activity and Return Variance | Bin: {bin_size}")
    ax1.set_xlabel("Time")

    # Merge legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    sns.despine(ax=ax1, right=False)

    plt.tight_layout()
    plt.show()


def plot_return_distributions(df, frequencies=("1min", "5min", "15min", "60min")):
    """
    Plot standardized calendar-time return distributions across sampling
    frequencies and compare them with a standard normal benchmark.

    Returns a DataFrame containing weighted summary statistics for each
    frequency.
    """
    plt.figure(figsize=(12, 7))

    stats_list = []
    x = np.linspace(-5, 5, 500)

    for freq in frequencies:
        r, w = calendar_time_returns(df, freq=freq)

        if len(r) < 2 or np.sum(w) == 0:
            continue

        stats = weighted_return_stats(r, w)
        mean = stats["mean"]
        var = stats["var"]

        if not np.isfinite(var) or var <= 0:
            continue

        std_r = (r - mean) / np.sqrt(var)

        sns.kdeplot(
            std_r,
            bw_adjust=1.0,
            fill=False,
            clip=(-5, 5),
            label=f"{freq} (kurt={stats['kurtosis']:.2f})",
        )

        stats["freq"] = freq
        stats_list.append(stats)

    plt.plot(x, norm.pdf(x), "k--", alpha=0.7, label="Standard normal")

    plt.title("Distribution of Log-Returns Across Sampling Frequencies")
    plt.xlabel("Standardized Log-Return")
    plt.ylabel("Density")
    plt.xlim(-5, 5)
    plt.legend()
    plt.tight_layout()

    return pd.DataFrame(stats_list)


def plot_pie(df, column_name: str):
    
    plt.figure(figsize=(3, 3))

    counts = df[column_name].value_counts()
    counts.plot(
        kind='pie', 
        autopct='%1.1f%%',     
        startangle=140,         
        shadow=False, 
        colors=["#66dbff", '#99ff99', "#99ffd1", "#ffcc99", "#ffb3e6"] 
    )


    plt.title(f'Distribution of {column_name.capitalize()} in Dataset')
    plt.legend(counts.index, title=column_name, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.ylabel('') 
    plt.tight_layout()
    plt.show()


def plot_OLB(df, freq="1S"):
    """
    Plot bid, ask, mid, and weighted mid prices on a regular time grid.

    Bid and ask are shown lightly, while mid and weighted mid are emphasized.
    """

    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    d = d.set_index("ts").sort_index()
    d = d.resample(freq).last()
    d = d.dropna(subset=["ap", "bp", "mid", "wmid"])

    if d.empty:
        raise ValueError("No valid data available after resampling and filtering.")

    fig, ax = plt.subplots(figsize=(13, 7))

    # Spread band
    ax.fill_between(
        d.index,
        d["bp"].to_numpy(),
        d["ap"].to_numpy(),
        alpha=0.10,
        label="Spread"
    )

    # Order book levels
    sns.lineplot(x=d.index, y=d["bp"], ax=ax, linewidth=1.2, alpha=0.55, label="Bid")
    sns.lineplot(x=d.index, y=d["ap"], ax=ax, linewidth=1.2, alpha=0.55, label="Ask")
    sns.lineplot(x=d.index, y=d["mid"], ax=ax, linewidth=2.5, label="Mid")
    sns.lineplot(x=d.index, y=d["wmid"], ax=ax, linewidth=2.5, label="Weighted Mid")

    ax.set_title("Order Book Prices")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(frameon=True)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()



def plot_spread_dist(df: pd.DataFrame, ticksize, calendar_time=False, freq='1min'):
    """
    Plot the spread distribution with a detailed bar chart and a grouped pie chart.
    """
    if calendar_time:
        # We use .last() 
        data_to_plot = df.set_index("ts")["spread"].resample(freq).last().dropna()
        analysis_label = f"Calendar Time (Last value, Freq: {freq})"
    else:
        data_to_plot = df["spread"].dropna()
        analysis_label = "Event-Based"

    # Normalize the distribution
    event_dist = data_to_plot.round(2).value_counts(normalize=True).sort_index()

    # Pie chart grouping logic
    threshold = 2 * ticksize
    mask = event_dist < threshold
    pie_data = event_dist[~mask].copy()
    if event_dist[mask].sum() > 0:
        pie_data['Other'] = event_dist[mask].sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Bar Chart 
    event_dist.plot(kind='barh', ax=ax1, color='#66dbff', edgecolor='black')
    ax1.set_title(f'Detailed Distribution\n({analysis_label})')
    ax1.invert_yaxis()
    ax1.set_xlabel('Proportion (0.0 to 1.0)')
    ax1.set_ylabel('Spread Value (Ticks)') 

    # Adding data labels for precision
    for i, v in enumerate(event_dist):
        ax1.text(v + 0.005, i, f'{v:.2%}', va='center', fontweight='bold')

    # Pie Chart 
    pie_data.plot.pie(
        ax=ax2, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=["#66dbff", '#99ff99', "#99ffd1", "#ffcc99"]
    )
    ax2.set_title(f'Global Overview\n({analysis_label})')
    ax2.legend(pie_data.index, title="Spread (Ticks)", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax2.set_ylabel('') 

    plt.suptitle(f'Microstructure Analysis: Spread Distribution', fontsize=16)
    plt.tight_layout()
    plt.show()


def display_spread(df: pd.DataFrame, freq="5min"):
    """
    Plot the evolution of the bid–ask spread using resampled statistics.

    Displays the mean, median, and quantile bands (5–95% and 25–75%) over time
    to visualize spread stability and variability.
    """
    stats = (
        df.set_index("ts")["spread"]
        .resample(freq)
        .agg(
            mean='mean',
            median='median',
            q05=lambda x: x.quantile(0.05),
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
            q95=lambda x: x.quantile(0.95)
        )
    )

    plt.figure(figsize=(15, 8))

    # Zone 5% - 95% 
    plt.fill_between(stats.index, stats['q05'], stats['q95'], 
                    color='gray', alpha=0.15, label='Intervalle 5% - 95%')

    # Zone 25% - 75% 
    plt.fill_between(stats.index, stats['q25'], stats['q75'], 
                    color='royalblue', alpha=0.35, label='Intervalle 25% - 75% (IQR)')

    # Tracer la Médiane 
    plt.plot(stats.index, stats['median'], color='darkblue', 
            linewidth=2, label='stable median')

    # Tracer la Moyenne 
    plt.plot(stats.index, stats['mean'], color='crimson', 
            linestyle='--', linewidth=1.5, label='sensible mean')

    plt.title("Analyse de Stationnarité du Spread : Moyenne, Médiane et Quantiles", fontsize=14, pad=20)
    plt.xlabel("Temps (Minute par minute)", fontsize=12)
    plt.ylabel("Valeur du Spread", fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.gcf().autofmt_xdate() 

    plt.tight_layout()
    plt.show()


def plot_imbalance_x(df: pd.DataFrame, x="mid", bins=100):
    """
    Plot conditional probabilities of upward and downward price moves
    given the order book imbalance.

    The imbalance is discretized into bins and the probabilities
    P(Δx > 0|Imbalance(t)) and P(Δx < 0|Imbalance(t)) are estimated for each bin.
    """
    df = df.copy()
    
    # Define bins and categorize imbalance
    bin_edges = np.linspace(-1, 1, bins + 1)
    df['imbalance_bins'] = pd.cut(df['imbalance'], bins=bin_edges)

    # Compute delta
    df["delta"] = df[x].shift(-1) - df[x]

    # Grouping and calculating probabilities
    grouped = df.groupby('imbalance_bins', observed=False)
    pos_counts = grouped['delta'].apply(lambda val: (val > 0).sum())
    neg_counts = grouped['delta'].apply(lambda val: (val < 0).sum())
    total_counts = pos_counts + neg_counts

    # Probabilities
    p_plus = (pos_counts / total_counts).fillna(0)
    p_neg = (neg_counts / total_counts).fillna(0)

    bin_midpoints = np.array([interval.mid for interval in p_plus.index])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(13, 6))
    width = (bin_edges[1] - bin_edges[0]) * 0.8 
    
    ax.bar(bin_midpoints, p_plus.values, width=width, 
           label=f'P(Δ{x} > 0) [Upward Move]', color='royalblue', alpha=0.8)
    ax.bar(bin_midpoints, p_neg.values, width=width, 
           label=f'P(Δ{x} < 0) [Downward Move]', color='indianred', alpha=0.4, bottom=0)
    
    neutral_line = ax.axvline(0, color="black", linestyle="--", lw=1, label="Neutral Imbalance")
    threshold_line = ax.axhline(0.5, color="gray", linestyle=":", lw=1, label="50% Threshold")

    
    ax.set_title(f'Conditional Probability of Price Movement Given Order Book Imbalance ({x})', fontsize=14)
    ax.set_xlabel('Order Book Imbalance (Unitless ratio [-1, 1])', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    
    
    ax.grid(axis="y", alpha=0.3)

    # Figure-level legend below the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.02),
    fontsize=15,        
    frameon=True,
    columnspacing=1.0,  
    handlelength=1.2,   
    handletextpad=0.4,  
    borderpad=0.3       
    )

    plt.subplots_adjust(bottom=0.22)
    plt.show()


def plot_imbalance_sign(df, sign_col="sign", bins=50):
    """
    Plot conditional probabilities of trade signs (+1 / -1) given order book imbalance.

    The imbalance is discretized into bins and the probabilities
    P(sign = +1) and P(sign = -1) are estimated for each bin.
    """
    d = df.copy()

    edges = np.linspace(-1, 1, bins + 1)
    d["imb_bin"] = pd.cut(d["imbalance"], bins=edges, include_lowest=True)

    g = d.groupby("imb_bin", observed=False)[sign_col]
    n_plus = g.apply(lambda s: (s == 1).sum())
    n_minus = g.apply(lambda s: (s == -1).sum())
    n_total = n_plus + n_minus

    
    p_plus = (n_plus / n_total).fillna(0)
    p_minus = (n_minus / n_total).fillna(0)

    mids = np.array([iv.mid for iv in p_plus.index])
    w = (edges[1] - edges[0]) * 0.42

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(mids - w/2, p_plus.values, width=w, label="P(sign=+1)", color="tab:blue", alpha=0.85)
    ax.bar(mids + w/2, p_minus.values, width=w, label="P(sign=-1)", color="tab:red", alpha=0.85)

    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Imbalance")
    ax.set_ylabel("Probability")
    ax.set_title(f"Trade Sign vs Imbalance ({sign_col})")
    ax.axvline(0, color="black", lw=1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.show()



__all__ = [
    "plot_price_event_time",
    "plot_dt_hist",
    "plot_qty_hist",
    "plot_trades_per_minute",
    "plot_volume_hist",
    "plot_activity_and_variance",
    "plot_return_distributions",
    "plot_pie",
    "plot_OLB",
    "plot_spread_hist",
    "display_spread",
    "plot_imbalance_x",
    "plot_imbalance_sign",
]
