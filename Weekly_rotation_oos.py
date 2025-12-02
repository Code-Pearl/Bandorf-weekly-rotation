"""
Bensdorp Weekly Rotation - Full (Parquet + SQLite, Next-Day OPEN execution, OOS)

What this script does
---------------------
- Implements the Bensdorp / Bandorf Weekly Rotation trading logic exactly (as in your C# referenced file).
- Signals are formed on the rebalance day (Monday) using Close data; orders are executed at the NEXT TRADING DAY OPEN.
- Performs Out-Of-Sample (OOS) evaluation: you set a warmup period (in days or years) and the OOS period is evaluated only after warmup.
- Saves price data in two formats:
    1) Single Parquet file: 'ticker_data/all_tickers.parquet' (fast columnar)
    2) Lightweight SQLite DB: 'ticker_data/ticker_data.db' with 'prices' table (useful to store price + future signals)
- Prints a final-week "ADVICE" (BUY / SELL / HOLD) — printed once at the end for the last rebalance in the tested period (no weekly spamming).
- Plots OOS NAV vs SPY OOS return.

Usage
-----
- pip install pandas numpy yfinance matplotlib pyarrow
- python Bensdorp_WeeklyRotation_full.py

Notes
-----
- Only execution semantics changed from previous file: next-day OPEN execution (closer to TuringTrader's next-bar semantics).
- No "next-bar limit/stop" simulation here; stops/profits from MRL/MRS would still need next-bar intraday data; for WR we only need open/close.
- You can choose to load prices from the parquet file or sqlite in future runs if you prefer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import os
import sqlite3
import matplotlib.pyplot as plt

# ---------------------------
# PARAMETERS (tweak here)
# ---------------------------
UNIVERSE = ["AAPL","MSFT","AMZN","GOOGL","TSLA","NVDA","JPM","V","DIS","SPY"]
INITIAL_CAPITAL = 100000
MAX_ENTRIES = 6             # same as your previous config
MAX_RSI = 50
OOS_WARMUP_YEARS = 2        # warmup/in-sample in years; OOS starts after this many years from start
DATA_FOLDER = "ticker_data"
PARQUET_PATH = os.path.join(DATA_FOLDER, "all_tickers.parquet")
SQLITE_PATH = os.path.join(DATA_FOLDER, "ticker_data.db")

# ---------------------------
# Helpers / Indicators
# ---------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def momentum(series: pd.Series, lookback: int) -> pd.Series:
    return series / series.shift(lookback) - 1.0

def rsi(series: pd.Series, period: int = 3) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi_val = 100 - 100 / (1 + rs)
    return rsi_val.fillna(50.0)

# ---------------------------
# Data fetch + storage
# ---------------------------
def fetch_prices_open_close(universe: List[str], start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download data from yfinance with auto_adjust=True and return two DataFrames:
        - opens: adjusted Open prices (columns = tickers)
        - closes: adjusted Close prices (columns = tickers)
    """
    data = yf.download(universe, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=True)

    # yfinance returns MultiIndex columns if multiple tickers; otherwise flat.
    if isinstance(data.columns, pd.MultiIndex):
        # prefer 'Open' and 'Close' levels
        if "Open" in data.columns.levels[0] and "Close" in data.columns.levels[0]:
            opens = data["Open"].copy()
            closes = data["Close"].copy()
        else:
            # fallback: try first level as 'Close' like previous logic
            # We will try to extract columns named 'Open' or 'Close' if present
            lvl0 = data.columns.levels[0]
            if "Close" in lvl0:
                closes = data["Close"].copy()
            else:
                closes = data.iloc[:, data.columns.get_level_values(0) == lvl0[-1]]
            if "Open" in lvl0:
                opens = data["Open"].copy()
            else:
                opens = closes.copy()  # best-effort fallback
    else:
        # single ticker case -> flat columns
        if "Open" in data.columns and "Close" in data.columns:
            opens = data["Open"].to_frame(name=universe[0])
            closes = data["Close"].to_frame(name=universe[0])
        else:
            # If something weird, treat last column as close and copy to open
            closes = data.iloc[:, -1].to_frame(name=universe[0])
            opens = closes.copy()

    # normalize index & forward-fill to reduce missing days
    opens = opens.dropna(how='all').ffill()
    closes = closes.dropna(how='all').ffill()
    # align columns (some tickers may be missing)
    common_cols = sorted(list(set(opens.columns).intersection(set(closes.columns))))
    opens = opens[common_cols]
    closes = closes[common_cols]
    return opens, closes

def save_all_parquet(opens: pd.DataFrame, closes: pd.DataFrame, folder=DATA_FOLDER, parquet_path=PARQUET_PATH):
    os.makedirs(folder, exist_ok=True)
    # Save a single parquet with a MultiIndex columns: ('Open'/'Close', ticker)
    combined = pd.concat({"Open": opens, "Close": closes}, axis=1)
    combined.to_parquet(parquet_path)
    print(f"[OK] Saved combined parquet -> {parquet_path}")

def save_all_sqlite(opens: pd.DataFrame, closes: pd.DataFrame, sqlite_path=SQLITE_PATH):
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    conn = sqlite3.connect(sqlite_path)
    # Build a single table 'prices' with columns: date, ticker, open, close
    # Flatten to long format for ease of queries and small size
    df_list = []
    for t in closes.columns:
        df = pd.DataFrame({
            "date": closes.index,
            "ticker": t,
            "open": opens[t].values,
            "close": closes[t].values
        })
        df_list.append(df)
    big = pd.concat(df_list, ignore_index=True)
    # Save to SQL (replace)
    big.to_sql("prices", conn, if_exists="replace", index=False)
    # Create index for faster queries
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_date ON prices (ticker, date);")
        conn.commit()
    except Exception:
        pass
    conn.close()
    print(f"[OK] Saved SQLite DB -> {sqlite_path} (table: prices)")

# ---------------------------
# Next-day OPEN execution backtester (signals formed on day T, orders executed day T+1 open)
# ---------------------------
@dataclass
class Order:
    ticker: str
    shares: int
    comment: str

@dataclass
class Position:
    ticker: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp

class NextOpenBacktester:
    def __init__(self, opens: pd.DataFrame, closes: pd.DataFrame, initial_capital: float = INITIAL_CAPITAL):
        # opens, closes are dataframes indexed by date, columns tickers
        self.opens = opens
        self.closes = closes
        self.dates = closes.index
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []   # orders scheduled to execute on next trading day open
        self.nav = pd.Series(index=self.dates, dtype=float)
        self.history: List[Tuple[pd.Timestamp, str, str, int, float, str]] = []  # (date, action, ticker, shares, price, comment)

    def submit_order(self, ticker: str, shares: int, comment: str = ""):
        # schedule order to run at next open (we just append; it will be executed on next date processed)
        self.pending_orders.append(Order(ticker, shares, comment))

    def run_pending_orders_at_open(self, date: pd.Timestamp):
        # Execute all pending orders at today's OPEN price
        if not self.pending_orders:
            return

        for order in self.pending_orders:
            # check price existence
            if order.ticker not in self.opens.columns:
                # ticker not available
                continue
            # If today's open is NaN, skip this order (can't execute)
            price = self.opens.at[date, order.ticker]
            if pd.isna(price):
                # skip execution (order is dropped)
                continue
            price = float(price)
            if order.shares > 0:
                cost = order.shares * price
                if cost <= self.cash + 1e-8:
                    # buy
                    if order.ticker in self.positions:
                        p = self.positions[order.ticker]
                        total_shares = p.shares + order.shares
                        p.entry_price = (p.entry_price * p.shares + price * order.shares) / total_shares
                        p.shares = total_shares
                    else:
                        self.positions[order.ticker] = Position(order.ticker, order.shares, price, date)
                    self.cash -= cost
                    self.history.append((date, "BUY", order.ticker, order.shares, price, order.comment))
                else:
                    # scale down to maximum affordable shares
                    max_shares = int(self.cash // price)
                    if max_shares > 0:
                        if order.ticker in self.positions:
                            p = self.positions[order.ticker]
                            total_shares = p.shares + max_shares
                            p.entry_price = (p.entry_price * p.shares + price * max_shares) / total_shares
                            p.shares = total_shares
                        else:
                            self.positions[order.ticker] = Position(order.ticker, max_shares, price, date)
                        self.cash -= max_shares * price
                        self.history.append((date, "BUY", order.ticker, max_shares, price, order.comment + " (scaled)"))
            else:
                # SELL
                shares_to_sell = abs(order.shares)
                if order.ticker in self.positions:
                    p = self.positions[order.ticker]
                    sellshares = min(shares_to_sell, p.shares)
                    self.cash += sellshares * price
                    p.shares -= sellshares
                    self.history.append((date, "SELL", order.ticker, sellshares, price, order.comment))
                    if p.shares == 0:
                        del self.positions[order.ticker]
        # clear pending
        self.pending_orders.clear()

    def record_nav(self, date: pd.Timestamp):
        total = self.cash
        for p in self.positions.values():
            # use close price for valuation to avoid intraday lookahead; if close missing, try open
            if date in self.closes.index and p.ticker in self.closes.columns:
                price = self.closes.at[date, p.ticker]
                if pd.isna(price) and date in self.opens.index:
                    price = self.opens.at[date, p.ticker]
            else:
                price = None
            if price is None or pd.isna(price):
                # skip valuation for this instrument this date (conservative)
                continue
            total += p.shares * float(price)
        self.nav.at[date] = total

# ---------------------------
# Weekly Rotation Strategy (Bensdorp_WR) using closes for signals and next-open execution
# ---------------------------
class Bensdorp_WR:
    def __init__(self, universe: List[str], benchmark: str = "SPY", max_rsi: int = MAX_RSI, max_entries: int = MAX_ENTRIES):
        self.universe = universe
        self.benchmark = benchmark
        self.max_rsi = max_rsi
        self.max_entries = max_entries

    def run(self, opens: pd.DataFrame, closes: pd.DataFrame, bt: NextOpenBacktester, oos_start: Optional[pd.Timestamp] = None):
        """
        Runs the weekly rotation:
         - Compute indicators from closes
         - On every Monday (weekday==0) form signals based on close data
         - Submit orders which will be executed on the next trading day's OPEN
        Returns:
            nav: Series of NAV (full history)
            advice: dict with BUY/SELL/HOLD for the last rebalance inside the OOS testing window
            last_rebalance_date: pd.Timestamp of last rebalance considered (for advice)
        """
        dates = closes.index
        rsi3 = closes.apply(lambda s: rsi(s, 3))
        mom200 = closes.apply(lambda s: momentum(s, 200))
        # benchmark sma200 using closes
        if self.benchmark in closes.columns:
            bench_sma200 = sma(closes[self.benchmark], 200)
        else:
            bench_sma200 = pd.Series(0.0, index=dates)
        sma_band = bench_sma200 * 0.98

        last_advice = {"BUY": [], "SELL": [], "HOLD": []}
        last_rebalance_date = None

        # iterate through all trading dates, run pending orders at today's open BEFORE making today's signals
        for idx, date in enumerate(dates):
            # 1) Execute pending orders at today's OPEN
            bt.run_pending_orders_at_open(date)
            # 2) Record NAV after execution
            bt.record_nav(date)

            # 3) Only form signals on rebalance day — Monday
            if date.weekday() != 0:
                continue

            # Skip if benchmark below SMA band (no new entries)
            if self.benchmark in closes.columns and closes.at[date, self.benchmark] <= sma_band.at[date]:
                # still consider this a rebalance day but skip new entries (we might still want to sell non-top positions)
                # We'll still compute topN and sell anything not in final_list
                pass

            # Build rank by momentum (at close)
            # Ensure we only consider tickers present in closes.columns
            valid_universe = [t for t in self.universe if t in closes.columns]
            if not valid_universe:
                continue

            momentums = mom200.loc[date, valid_universe].dropna()
            rsi_vals = rsi3.loc[date, valid_universe].dropna()

            # If there are fewer than 1 valid momentum entries, skip
            if momentums.empty:
                continue

            topN = list(momentums.sort_values(ascending=False).head(self.max_entries).index)

            # keep existing positions that remain in topN
            keep = [t for t in topN if t in bt.positions]
            # enter: in topN, not currently held, RSI < threshold
            enter = [t for t in topN if (t not in bt.positions) and (rsi_vals.get(t, 999) < self.max_rsi)]

            final_list = keep + enter

            # Build last_advice from final_list vs current positions
            last_advice = {"BUY": [], "SELL": [], "HOLD": []}
            for t in valid_universe:
                if t in final_list and t not in bt.positions:
                    last_advice["BUY"].append(t)
                elif t not in final_list and t in bt.positions:
                    last_advice["SELL"].append(t)
                elif t in final_list and t in bt.positions:
                    last_advice["HOLD"].append(t)
            last_rebalance_date = date

            # Position sizing: equal weight across max_entries (use NAV from most recent recorded)
            nav_now = bt.nav.loc[:date].dropna().iloc[-1] if not bt.nav.loc[:date].dropna().empty else bt.cash
            target_weight = 1.0 / self.max_entries

            # Sell anything currently held but not in final_list (schedule sells for next open)
            for t in list(bt.positions.keys()):
                if t not in final_list:
                    bt.submit_order(t, -bt.positions[t].shares, comment="WR Exit")

            # For entries and keeps, compute target_shares and submit delta order
            for t in final_list:
                # Ensure today's next-open exists: we submit orders to be executed next trading day open
                # Compute target_shares using nav_now and today's close price (signal formed using close)
                # If close is missing on signal day, skip
                if pd.isna(closes.at[date, t]):
                    continue
                price_close = float(closes.at[date, t])
                if price_close <= 0:
                    continue
                target_value = nav_now * target_weight
                # use target_shares integer floor
                target_shares_f = target_value / price_close
                if np.isnan(target_shares_f) or target_shares_f <= 0:
                    continue
                target_shares = int(target_shares_f)

                current_shares = bt.positions[t].shares if t in bt.positions else 0
                delta = target_shares - current_shares
                # Submit delta to execute at next open
                if delta > 0:
                    bt.submit_order(t, delta, comment="WR Buy")
                elif delta < 0:
                    bt.submit_order(t, delta, comment="WR Reduce")
            # end of Monday processing

        # end for dates
        return bt.nav, last_advice, last_rebalance_date

# ---------------------------
# Utilities & main
# ---------------------------
def run_backtest_and_report(universe: List[str], start: pd.Timestamp, end: pd.Timestamp, use_parquet_save: bool = True, use_sqlite_save: bool = True):
    # 1) fetch data (Open & Close)
    opens, closes = fetch_prices_open_close(universe, start, end)

    # 2) save combined parquet and sqlite if requested
    if use_parquet_save:
        save_all_parquet(opens, closes)
    if use_sqlite_save:
        save_all_sqlite(opens, closes)

    # 3) prepare OOS boundary
    oos_start = start + pd.Timedelta(days=int(365 * OOS_WARMUP_YEARS))
    if oos_start not in closes.index:
        # find first index >= oos_start
        oos_idx = closes.index.searchsorted(oos_start)
        if oos_idx >= len(closes.index):
            oos_start = closes.index[-1]
        else:
            oos_start = closes.index[oos_idx]

    # 4) init backtester
    bt = NextOpenBacktester(opens, closes, initial_capital=INITIAL_CAPITAL)
    wr = Bensdorp_WR(universe=[t for t in universe if t != "SPY"], benchmark="SPY", max_rsi=MAX_RSI, max_entries=MAX_ENTRIES)

    # 5) run strategy (forms signals from close, executes next open)
    nav_series, advice, last_rebalance_date = wr.run(opens, closes, bt, oos_start=oos_start)

    # 6) extract OOS NAV & SPY returns
    # Ensure nav_series is filled forward for dates where no record exists
    nav_series = nav_series.fillna(method="ffill").dropna()
    # OOS NAV start index >= oos_start
    nav_oos = nav_series[nav_series.index >= oos_start].copy()
    if nav_oos.empty:
        print("[WARN] OOS NAV is empty (OOS may be beyond data range). Falling back to full-sample NAV.")
        nav_oos = nav_series.copy()

    # SPY OOS series
    if "SPY" in closes.columns:
        spy_oos = closes["SPY"].reindex(nav_oos.index).ffill()
        spy_oos = spy_oos / spy_oos.iloc[0]
    else:
        spy_oos = None

    # 7) Print final-week advice (based on last_rebalance_date within backtest)
    print("\n============== FINAL WEEK ADVICE ==============")
    if last_rebalance_date is None:
        print("No rebalance occurred during run.")
    else:
        print(f"Last rebalance date (signals formed): {last_rebalance_date.date()}")
        print("BUY:  ", ", ".join(advice["BUY"]) if advice["BUY"] else "(none)")
        print("SELL: ", ", ".join(advice["SELL"]) if advice["SELL"] else "(none)")
        print("HOLD: ", ", ".join(advice["HOLD"]) if advice["HOLD"] else "(none)")
    print("==============================================\n")

    # 8) Plot OOS NAV vs SPY
    plt.figure(figsize=(10,6))
    plt.plot(nav_oos.index, nav_oos / nav_oos.iloc[0], label="WR OOS NAV (norm)")
    if spy_oos is not None:
        plt.plot(spy_oos.index, spy_oos.values, label="SPY OOS (norm)")
    plt.legend()
    plt.title(f"Bensdorp Weekly Rotation - OOS from {oos_start.date()}")
    plt.show()

    # 9) return key objects
    return {
        "nav": nav_series,
        "nav_oos": nav_oos,
        "spy_oos": spy_oos,
        "advice": advice,
        "last_rebalance_date": last_rebalance_date,
        "backtester": bt,
        "opens": opens,
        "closes": closes
    }

# ---------------------------
# __main__
# ---------------------------
if __name__ == "__main__":
    # set timeframe
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365 * 5)   # 5 years data by default

    results = run_backtest_and_report(UNIVERSE, start, end, use_parquet_save=True, use_sqlite_save=True)

    # Optional: print last few trades
    bt = results["backtester"]
    print("\nRecent trades (last 40):")
    for row in bt.history[-40:]:
        # row: (date, action, ticker, shares, price, comment)
        print(row)
