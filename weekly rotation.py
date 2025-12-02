"""
Bandorf weekly rotation (fixed yfinance + runnable)
Translated to Python and kept the trading logic as-is (from your C#-inspired port).
This file fixes yfinance changes (auto_adjust) and robustly extracts prices.

Requirements:
    pip install pandas numpy yfinance matplotlib

Usage:
    python "Bandorf weekly rotation.py"
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# ---------------------------
# Indicator helpers
# ---------------------------
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def momentum(series: pd.Series, lookback: int):
    return series / series.shift(lookback) - 1.0

def rsi(series: pd.Series, period: int = 3) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)

def atr_from_close(close: pd.Series, window: int = 10) -> pd.Series:
    tr = (close - close.shift(1)).abs()
    return tr.rolling(window=window, min_periods=1).mean()

# ---------------------------
# Simple backtester (close-execution)
# ---------------------------
@dataclass
class Position:
    ticker: str
    shares: int
    entry_price: float
    entry_date: pd.Timestamp
    stop_loss: Optional[float] = None
    profit_target: Optional[float] = None

class SimpleBacktest:
    def __init__(self, prices: pd.DataFrame, initial_capital: float = 100000.0):
        self.prices = prices.sort_index()
        self.dates = self.prices.index
        self.tickers = list(self.prices.columns)
        self.capital = initial_capital
        self.nav = pd.Series(index=self.dates, dtype=float)
        self.positions: Dict[str, Position] = {}
        self.cash = initial_capital
        self.history = []
        self.commission_per_share = 0.0

    def _value(self, date):
        total = self.cash
        for pos in self.positions.values():
            # if ticker missing on date, forward-fill last known price
            if date not in self.prices.index:
                continue
            p = self.prices.at[date, pos.ticker]
            total += pos.shares * p
        return total

    def record_nav(self, date):
        self.nav.at[date] = self._value(date)

    def buy(self, date: pd.Timestamp, ticker: str, shares: int, comment: str = ""):
        price = float(self.prices.at[date, ticker])
        cost = shares * price + abs(shares) * self.commission_per_share
        if cost > self.cash + 1e-8:
            max_shares = int((self.cash) / (price + self.commission_per_share))
            shares = max(0, max_shares)
            cost = shares * price + abs(shares) * self.commission_per_share
        if shares == 0:
            return
        self.cash -= cost
        pos = self.positions.get(ticker)
        if pos is None:
            self.positions[ticker] = Position(ticker, shares, price, date)
        else:
            total_shares = pos.shares + shares
            pos.entry_price = (pos.entry_price * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
        self.history.append((date, 'BUY', ticker, shares, price, comment))

    def sell(self, date: pd.Timestamp, ticker: str, shares: int, comment: str = ""):
        price = float(self.prices.at[date, ticker])
        pos = self.positions.get(ticker)
        if pos is None:
            return
        shares = min(shares, pos.shares)
        proceeds = shares * price - shares * self.commission_per_share
        self.cash += proceeds
        pos.shares -= shares
        self.history.append((date, 'SELL', ticker, shares, price, comment))
        if pos.shares == 0:
            del self.positions[ticker]

    def close_all(self, date: pd.Timestamp):
        for t in list(self.positions.keys()):
            self.sell(date, t, self.positions[t].shares, comment='close_all')

# ---------------------------
# Bensdorp Weekly Rotation (keeps original logic)
# ---------------------------
class Bensdorp_WR:
    def __init__(self,
                 universe: List[str],
                 benchmark: Optional[str] = 'SPY',
                 max_rsi: int = 50,
                 max_entries: int = 10):
        self.universe = universe
        self.benchmark = benchmark
        self.max_rsi = max_rsi
        self.max_entries = max_entries

    def run(self, prices: pd.DataFrame, bt: SimpleBacktest):
        p = prices
        dates = p.index
        rsi3 = p.apply(lambda s: rsi(s, period=3))
        mom200 = p.apply(lambda s: momentum(s, 200))

        if self.benchmark and self.benchmark in p.columns:
            bench_sma200 = sma(p[self.benchmark], 200)
            sma_band = bench_sma200 * 0.98
        else:
            sma_band = pd.Series(0.0, index=dates)

        for date in dates:
            bt.record_nav(date)

            # weekly rebalance: Monday == 0
            if date.weekday() != 0:
                continue

            if self.benchmark and p.at[date, self.benchmark] <= sma_band.at[date]:
                # market below band -> skip new entries (keep existing)
                continue

            momentums = mom200.loc[date, self.universe].dropna()
            rsi_values = rsi3.loc[date, self.universe].dropna()
            topN = list(momentums.sort_values(ascending=False).head(self.max_entries).index)

            keep = [t for t in topN if t in bt.positions]
            enter = [t for t in topN if t not in bt.positions and rsi_values.get(t, 1000) < self.max_rsi]

            next_holdings = keep + enter
            target_pct = {t: 1.0 / self.max_entries for t in next_holdings}
            nav = bt._value(date)

            # close positions not in target
            for t in list(bt.positions.keys()):
                if t not in next_holdings and t in self.universe:
                    bt.sell(date, t, bt.positions[t].shares, comment='WR exit')

            # set / adjust positions for target holdings
            for t in next_holdings:
                price = float(p.at[date, t])
                target_value = nav * target_pct[t]
                target_shares = int(target_value // price)
                current_shares = bt.positions[t].shares if t in bt.positions else 0
                delta = target_shares - current_shares
                if delta > 0:
                    bt.buy(date, t, delta, comment='WR buy')
                elif delta < 0:
                    bt.sell(date, t, -delta, comment='WR sell')

        for date in dates:
            bt.record_nav(date)
        return bt.nav

# ---------------------------
# Mean-reversion base (MRL / MRS)
# ---------------------------
class Bensdorp_MRx:
    def __init__(self,
                 universe: List[str],
                 entry_dir: int = 1,
                 sma_days: int = 150,
                 min_adx: int = 45,
                 min_atr: int = 400,
                 minmax_rsi: int = 30,
                 stop_loss: int = 250,
                 profit_target: int = 300,
                 max_cap: int = 100,
                 max_risk: int = 20,
                 max_entries: int = 10,
                 max_hold_days: int = 4):
        self.universe = universe
        self.entry_dir = entry_dir
        self.sma_days = sma_days
        self.min_adx = min_adx
        self.min_atr = min_atr
        self.minmax_rsi = minmax_rsi
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.max_cap = max_cap
        self.max_risk = max_risk
        self.max_entries = max_entries
        self.max_hold_days = max_hold_days

    def run(self, prices: pd.DataFrame, bt: SimpleBacktest):
        p = prices
        dates = p.index

        sma_series = p.apply(lambda s: sma(s, self.sma_days))
        rsi3 = p.apply(lambda s: rsi(s, 3))
        atr10 = p.apply(lambda s: (atr_from_close(s, 10) / s).replace([np.inf, -np.inf], np.nan).fillna(0.0))

        entry_meta: Dict[str, Dict] = {}

        for i, date in enumerate(dates):
            # manage existing positions
            for t in list(bt.positions.keys()):
                pos = bt.positions[t]
                days_held = (date - pos.entry_date).days
                if days_held >= self.max_hold_days:
                    bt.sell(date, t, bt.positions[t].shares, comment='time exit')
                    continue
                close = float(p.at[date, t])
                if self.entry_dir > 0:
                    if pos.stop_loss is not None and close <= pos.stop_loss:
                        bt.sell(date, t, bt.positions[t].shares, comment='stop loss')
                    elif pos.profit_target is not None and close >= pos.profit_target:
                        bt.sell(date, t, bt.positions[t].shares, comment='profit target')
                else:
                    if pos.stop_loss is not None and close >= pos.stop_loss:
                        bt.sell(date, t, bt.positions[t].shares, comment='stop loss')
                    elif pos.profit_target is not None and close <= pos.profit_target:
                        bt.sell(date, t, bt.positions[t].shares, comment='profit target')

            # open new positions based on filters & rankings
            cand = []
            for t in self.universe:
                if pd.isna(p.at[date, t]):
                    continue
                try:
                    sma_ok = p.at[date, t] > sma_series.at[date, t] if self.entry_dir > 0 else True
                    atr_ok = atr10.at[date, t] >= (self.min_atr / 10000.0)
                    rsi_val = rsi3.at[date, t]
                except KeyError:
                    continue
                if self.entry_dir > 0:
                    cond = sma_ok and atr_ok and (rsi_val < self.minmax_rsi)
                else:
                    idx = p.index.get_loc(date)
                    cond = True
                    if idx >= 2:
                        cond = (p.iloc[idx, p.columns.get_loc(t)] > p.iloc[idx-1, p.columns.get_loc(t)]) and \
                               (p.iloc[idx-1, p.columns.get_loc(t)] > p.iloc[idx-2, p.columns.get_loc(t)])
                    cond = cond and atr_ok and (rsi_val > self.minmax_rsi)
                if cond and t not in bt.positions:
                    cand.append((t, rsi_val))

            cand_sorted = sorted(cand, key=lambda x: x[1], reverse=(self.entry_dir < 0))
            num_open = len(bt.positions)
            slots = max(0, self.max_entries - num_open)
            to_enter = [t for t, _ in cand_sorted][:slots]

            nav = bt._value(date)
            for t in to_enter:
                entry_price = float(p.at[date, t]) * (1.0 - (self.min_atr / 10000.0)) if self.entry_dir > 0 else float(p.at[date, t])
                atr_frac = atr10.at[date, t]
                if self.entry_dir > 0:
                    stop_loss = entry_price * (1.0 - (self.stop_loss / 100.0) * atr_frac)
                    profit_target = entry_price * (1.0 + (self.profit_target / 10000.0))
                else:
                    stop_loss = entry_price * (1.0 + (self.stop_loss / 100.0) * atr_frac)
                    profit_target = entry_price * (1.0 - (self.profit_target / 10000.0))

                risk_per_share = max(0.10, abs(entry_price - stop_loss))
                shares_risk_limited = int((self.max_risk / 100.0 / self.max_entries) * nav / risk_per_share) if risk_per_share > 0 else 0
                shares_cap_limited = int((self.max_cap / 100.0 / self.max_entries) * nav / entry_price)
                target_shares = min(shares_risk_limited, shares_cap_limited)
                target_shares = max(0, target_shares)
                if self.entry_dir < 0:
                    target_shares = -target_shares

                if target_shares != 0:
                    if target_shares > 0:
                        bt.buy(date, t, target_shares, comment='MR enter')
                    else:
                        bt.buy(date, t, abs(target_shares), comment='MR (short) enter')

                    if t in bt.positions:
                        pos = bt.positions[t]
                        pos.stop_loss = stop_loss
                        pos.profit_target = profit_target
                        pos.entry_date = date
                    entry_meta[t] = dict(entry_date=date, entry_price=entry_price, stop_loss=stop_loss, profit_target=profit_target)

            bt.record_nav(date)

        return bt.nav

class Bensdorp_MRL(Bensdorp_MRx):
    def __init__(self, universe: List[str], **kwargs):
        super().__init__(universe, entry_dir=1, sma_days=150, min_adx=45, min_atr=400,
                         minmax_rsi=30, stop_loss=250, profit_target=300, max_cap=100, max_risk=20,
                         max_entries=10, max_hold_days=4, **kwargs)

class Bensdorp_MRS(Bensdorp_MRx):
    def __init__(self, universe: List[str], **kwargs):
        super().__init__(universe, entry_dir=-1, sma_days=150, min_adx=50, min_atr=500,
                         minmax_rsi=85, stop_loss=250, profit_target=400, max_cap=100, max_risk=20,
                         max_entries=10, max_hold_days=2, **kwargs)

# ---------------------------
# Helper: robust yfinance loader (fix for Adj Close KeyError)
# ---------------------------
def fetch_prices(universe: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # download with auto_adjust=True (yfinance changed behavior; will produce 'Close' when adjusted)
    data = yf.download(
        universe,
        start=start.strftime('%Y-%m-%d'),
        end=end.strftime('%Y-%m-%d'),
        auto_adjust=True,
        progress=True
    )

    # yfinance returns:
    # - If multiple tickers: MultiIndex columns with levels ('Close', ticker) OR columns named like 'AAPL' under 'Close'
    # - If single ticker: DataFrame columns: ['Open','High','Low','Close','Volume']
    # We want the adjusted close (Close when auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        # try to pick level 0 == 'Close' (or 'Adj Close' fallback)
        if 'Close' in data.columns.levels[0]:
            prices = data['Close'].copy()
        elif 'Adj Close' in data.columns.levels[0]:
            prices = data['Adj Close'].copy()
        else:
            # fallback: take last level (face)
            prices = data.iloc[:, data.columns.get_level_values(0) == data.columns.levels[0][-1]]
    else:
        # normal flat columns -> use Close
        if 'Close' in data.columns:
            prices = data['Close']
        elif 'Adj Close' in data.columns:
            prices = data['Adj Close']
        else:
            # last column as fallback
            prices = data.iloc[:, -1]

    # Ensure DataFrame shape: columns = tickers
    prices = prices.dropna(how='all').ffill().dropna(axis=1, how='all')
    # If single ticker, ensure DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=universe[0])

    return prices

# ---------------------------
# Demo / main
# ---------------------------
if __name__ == "__main__":
    universe = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA', 'NVDA', 'JPM', 'V', 'DIS', 'SPY']  # edit as you like
    end = pd.Timestamp.today()
    start = end - pd.Timedelta(days=365 * 3)

    prices = fetch_prices(universe, start, end)

    # init backtest
    bt = SimpleBacktest(prices, initial_capital=100000)
    wr_universe = [c for c in prices.columns if c != 'SPY']
    wr = Bensdorp_WR(universe=wr_universe, benchmark='SPY', max_rsi=50, max_entries=6)

    # run
    nav = wr.run(prices, bt)

    # plot
    nav_norm = nav / nav.iloc[0]
    spy_ret = (prices['SPY'] / prices['SPY'].iloc[0]).reindex(nav.index).ffill()

    plt.figure(figsize=(10,6))
    plt.plot(nav_norm.index, nav_norm.values, label='WR NAV (norm)')
    plt.plot(spy_ret.index, spy_ret.values, label='SPY B&H')
    plt.legend()
    plt.title("Bandorf weekly rotation - fixed yfinance")
    plt.show()

    # print trades summary
    print("Trades (last 20):")
    for row in bt.history[-40:]:
        print(row)
