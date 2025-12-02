This script implements a Python translation of the Bensdorp
Weekly Rotation strategy as originally written in the TuringTrader C# engine.

WHAT THIS STRATEGY DOES
-----------------------
1. Once per week (Monday), it:
   - Computes 200-day momentum for all tickers.
   - Computes 3-period RSI for dip filtering.
   - Computes SPY 200-day SMA for a market filter.
   - Ranks tickers by momentum.
   - Selects the top N.
   - Only enters new positions if RSI < max_rsi.
   - Keeps existing positions if still in top N.
   - Sells anything not in the new top list.

2. Execution is NEXT-DAY:
   Monday signal  →  Tuesday execution (C# equivalent behavior)

3. Position sizing:
   Equal weight: 1 / max_entries

4. Data storage:
   All prices are saved into 1 fast parquet file:
       ticker_data/all_tickers.parquet

5. Weekly report:
   Prints to terminal what to BUY and SELL for the week.
