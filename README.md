# QRE — a systematic trading research project

I built two daily-bar stock trading models and, more importantly, the testing
machinery to find out whether they actually work. Most of the ideas I tried
didn't survive testing. This repo documents what did, what didn't, and how I
told the difference.

**QRE** is a trend-following strategy that classifies each stock's regime
(trending, ranging, or transitional) with statistical tests, then trades
breakouts in trends and fades extremes in ranges. Holds for weeks.
**QRE-ST** is its short-term sister: it buys sharp dips in healthy stocks and
sells the bounce within days. Both are written twice — in Python for testing,
and in Pine Script so they run on TradingView charts.

## Results

Tested on 50+ US stocks, 2012–2026, with realistic costs:

| | QRE (trend) | QRE-ST (mean reversion) |
|---|---|---|
| Typical hold | ~5 weeks | ~1 week |
| Portfolio Sharpe (20 stocks) | 0.93 | 0.91 |
| Win rate | ~55% | 68% |

Out-of-sample results beat in-sample for both models, the edges survive 4×
transaction costs, and bootstrap confidence intervals on the Sharpe ratios
stay above zero. Returns are modest in absolute terms — the models are only
in the market ~20-30% of the time. This is a research project, not a
money-printing claim, and definitely not investment advice.

## What I learned (the short version)

- Six "improvements" that looked good in backtests — a breakeven stop, an ML
  model that picks which signals to take, pyramiding into winners, and others
  — all failed once I tested them out-of-sample. I rejected all six and kept
  the evidence in [RESEARCH.md](RESEARCH.md).
- The ML failure was the most interesting: predicting which trades would win
  filtered out exactly the big winners, because in trend-following the
  best-paying trades look the ugliest at entry. The top 5% of trades carried
  83% of all profit.
- Diversification beat every clever idea. Running one model across many
  stocks tripled the Sharpe ratio versus any single stock.
- Past per-stock performance told me nothing about future per-stock
  performance, so the stock universe is picked by structural rules
  (liquidity, history length), never by past returns.

The full research log — methodology, every experiment, and the honest
limitations (including how repeated testing contaminates out-of-sample data)
— is in [RESEARCH.md](RESEARCH.md).

## Files

| File | What it is |
|---|---|
| `backtest_qre.py` | Backtester: single stocks, portfolios, walk-forward sweeps, screens, daily signal scan |
| `short_term_qre.py` | QRE-ST model and its validation |
| `meta_layer.py` | The rejected ML experiment, kept as evidence |
| `quant_regime_ensemble.pine` | QRE for TradingView |
| `qre_st.pine` | QRE-ST for TradingView |
| `qre_screener.pine` | Indicator version for TradingView's Pine Screener |

## Run it

```bash
pip install pandas numpy yfinance scikit-learn
python backtest_qre.py SPY                              # backtest one symbol
python backtest_qre.py "AAPL,MU,TSLA" --stocks --basket # portfolio + robustness checks
python backtest_qre.py "AAPL,MU,TSLA" --stocks --signals # scan for today's entry signals
```

For TradingView: paste a `.pine` file into the Pine Editor and add it to a
chart. The Strategy Tester shows the backtest; alerts on "Order fills only"
carry a JSON payload with the action, size, and stop.
