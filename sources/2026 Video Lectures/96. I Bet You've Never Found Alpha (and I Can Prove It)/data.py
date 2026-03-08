#!/usr/bin/env python3
"""
Fetch daily returns (2020 - present) for AAPL, GOOG, AMZN, NVDA, SPY using IB API.
Saves each symbol to its own CSV: AAPL.csv, GOOG.csv, AMZN.csv, NVDA.csv, SPY.csv.
Requires TWS or IB Gateway running with API connections enabled (default port 7497).
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time
import threading
import os

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData


# Output directory (same folder as this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SYMBOLS = ["AAPL", "GOOG", "AMZN", "NVDA", "SPY"]
START_DATE = "2020-01-01"


def create_stock_contract(symbol: str) -> Contract:
    """Create a US stock/ETF contract for the given symbol."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


class IBApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data: dict[int, list] = {}
        self.data_received: dict[int, bool] = {}
        self.req_id_to_symbol: dict[int, str] = {}
        self._lock = threading.Lock()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
        print(f"Error [reqId={reqId}] {errorCode}: {errorString}")
        if errorCode in (162, 200, 354, 366):  # HMDS / no security / pacing
            with self._lock:
                self.data_received[reqId] = True

    def historicalData(self, reqId: int, bar: BarData):
        with self._lock:
            if reqId not in self.data:
                self.data[reqId] = []
            self.data[reqId].append({
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        with self._lock:
            self.data_received[reqId] = True
            sym = self.req_id_to_symbol.get(reqId, str(reqId))
            n = len(self.data.get(reqId, []))
            print(f"  Historical data end for {sym}: {n} bars.")


def fetch_daily_returns(
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 1,
    duration_str: str = "6 Y",
    out_dir: str | None = None,
) -> None:
    """
    Connect to IB, request daily bars from 2020 to present for each symbol,
    compute daily returns, and save one CSV per symbol.
    """
    out_dir = out_dir or SCRIPT_DIR

    app = IBApp()
    try:
        print("Connecting to Interactive Brokers...")
        app.connect(host, port, clientId=client_id)
        api_thread = threading.Thread(target=app.run, daemon=True)
        api_thread.start()
        time.sleep(2)

        if not app.isConnected():
            print("Failed to connect. Ensure TWS or IB Gateway is running and API is enabled.")
            return

        print("Connected. Requesting daily bars (2020 - present)...")

        for i, symbol in enumerate(SYMBOLS):
            req_id = i + 1
            app.req_id_to_symbol[req_id] = symbol
            contract = create_stock_contract(symbol)
            app.reqHistoricalData(
                reqId=req_id,
                contract=contract,
                endDateTime="",
                durationStr=duration_str,
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=1,
                formatDate=1,
                keepUpToDate=False,
                chartOptions=[],
            )
            time.sleep(0.6)  # avoid pacing violations

        # Wait for all requests to finish
        timeout = 90
        deadline = time.time() + timeout
        while not all(app.data_received.get(rid, False) for rid in range(1, len(SYMBOLS) + 1)):
            if time.time() > deadline:
                print("Timeout waiting for historical data.")
                break
            time.sleep(0.5)

        # Build and save one CSV per symbol
        for req_id, symbol in enumerate(SYMBOLS, start=1):
            rows = app.data.get(req_id, [])
            if not rows:
                print(f"No data for {symbol}, skipping.")
                continue

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)
            # Filter from 2020-01-01
            df = df.loc[df["date"] >= START_DATE].copy()
            df["daily_return"] = df["close"].pct_change()
            df = df.dropna(subset=["daily_return"])

            out_df = df[["date", "close", "daily_return"]].copy()
            out_df.columns = ["Date", "Close", "Daily_Return"]
            path = os.path.join(out_dir, f"{symbol}.csv")
            out_df.to_csv(path, index=False)
            print(f"Saved {path} ({len(out_df)} rows, {out_df['Date'].min().date()} to {out_df['Date'].max().date()})")

    finally:
        if app.isConnected():
            app.disconnect()
        print("Disconnected from IB.")


if __name__ == "__main__":
    fetch_daily_returns()
