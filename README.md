# QRE - systematic trading research

**[RESEARCH.md](RESEARCH.md) is the main document** — full methodology, all six
rejected experiments, statistical validation, and honest limitations. This README
is the summary.

This started as "build a trading strategy for TradingView" and turned into a
longer project about proving whether a strategy actually works. I ended up
with two models, a backtesting pipeline, and a growing list of ideas that
looked great until I tested them properly.

The main model (QRE) is a trend follower. It runs statistical tests on each
stock to decide if it's trending, ranging, or neither, then trades breakouts
in trends and fades extremes in ranges. Trades last a few weeks. The second
model (QRE-ST) does the opposite: it buys sharp dips in healthy stocks and
sells the bounce within days. Both exist twice, once in Python for testing
and once in Pine Script so they run on TradingView.

## Results

Tested on 50+ US stocks from 2012 to 2026 with commissions and slippage
included:

| | QRE (trend) | QRE-ST (dip buyer) |
|---|---|---|
| Typical hold | ~5 weeks | ~1 week |
| Portfolio Sharpe (20 stocks) | 0.93 | 0.91 |
| Win rate | ~55% | 68% |

The 20-stock basket (10 large cap, 10 mid cap) is the fixed test universe used
throughout, picked by structural rules rather than past performance. The 50+
figure is the wider set of names used for cross-sectional checks. Tickers are
listed in [RESEARCH.md](RESEARCH.md).

Out-of-sample results came out better than in-sample for both models, the
edge survives even at 4x assumed costs, and bootstrap confidence intervals
on the Sharpe stay above zero. To be clear about the absolute numbers:
returns are modest because the models sit in cash 70-80% of the time. This
is a research project, not a get-rich claim, and not investment advice.

## Things I learned the hard way

- I implemented six "improvements" that all helped in-sample and then failed
  out-of-sample: a breakeven stop, an ML model that picks which signals to
  take, pyramiding into winners, and a few more. All rejected. The details
  are in [RESEARCH.md](RESEARCH.md).
- The ML one surprised me most. Training a model to predict which trades win
  filtered out exactly the biggest winners, because the best trend trades
  look the worst at entry. In my data the top 5% of trades made 83% of the
  total profit.
- Nothing I tried beat simple diversification. The same model run across 20
  stocks has about triple the Sharpe of the median single stock.
- Picking stocks by their past backtest performance didn't work either (the
  rank correlation between past and future per-stock Sharpe was basically
  zero), so the universe is chosen by boring structural rules instead:
  enough liquidity, enough history, no crazy gap behavior.

The full log of experiments, including the methodology and the honest
limitations (like the fact that repeatedly checking the same out-of-sample
window slowly contaminates it), is in [RESEARCH.md](RESEARCH.md).

## Files

| File | What it is |
|---|---|
| `backtest_qre.py` | The backtester. Single stocks, portfolios, walk-forward sweeps, a tradeability screen, and a daily signal scan |
| `short_term_qre.py` | The QRE-ST model and its validation |
| `meta_layer.py` | The rejected ML experiment, kept as evidence |
| `quant_regime_ensemble.pine` | QRE for TradingView |
| `qre_st.pine` | QRE-ST for TradingView |
| `qre_screener.pine` | Indicator version for TradingView's Pine Screener |

## Running it

```bash
pip install pandas numpy yfinance scikit-learn
python backtest_qre.py SPY                               # backtest one symbol
python backtest_qre.py --stocks --basket                 # the 20-name basket + robustness
python backtest_qre.py "AAPL,MU,TSLA" --stocks --basket  # or your own list
python backtest_qre.py "AAPL,MU,TSLA" --stocks --signals # scan for today's signals
```

On TradingView: paste a .pine file into the Pine Editor and add it to a
chart. The Strategy Tester shows the backtest, and alerts set to "Order
fills only" carry a JSON message with the action, size, and stop.
