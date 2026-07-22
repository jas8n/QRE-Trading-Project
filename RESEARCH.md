# Research log

Full methodology and findings, including everything that failed. The short version is in the README.

Two daily-bar equity models (multi-week trend following and 1–2 week mean
reversion) built in Python and Pine Script — but the models are the *subject*,
not the point. **The point is the validation methodology**: every design
decision in this repo was accepted or rejected through component ablation,
in-sample/out-of-sample splits, and cross-sectional testing, and the negative
results are documented with the same care as the positive ones.

> Research framework, not investment advice. See [Limitations](#limitations--research-integrity).

## Results at a glance

| | QRE (trend) | QRE-ST (mean reversion) |
|---|---|---|
| Horizon | ~5-week avg holds | 6.6-day avg holds |
| Portfolio Sharpe (20-name basket) | **0.93** | **0.91** |
| 90% block-bootstrap CI | [0.52, 1.37] | [0.48, 1.35] |
| Win rate | ~55% (fat right tail) | 68% (no tail) |
| OOS vs IS | OOS Sharpe *exceeded* IS | OOS *exceeded* IS |
| Cost stress (4× frictions) | Sharpe 0.93 → 0.63 | 0.91 → 0.37 |
| Profit concentration | Top 5% of trades = 83% of P&L | diffuse |

Six plausible "improvements" were built, tested, and **rejected** — including
an ML meta-labeling layer, pyramiding, and universe curation by past
performance. Details below; the rejections are the most instructive part.

## Repo contents

| File | Purpose |
|---|---|
| `quant_regime_ensemble.pine` | QRE strategy — TradingView Pine Editor |
| `qre_st.pine` | QRE-ST strategy (1–2 week mean reversion) |
| `qre_screener.pine` | Indicator twin for TradingView's Pine Screener |
| `backtest_qre.py` | Bar-accurate Python replica: backtests, sweeps, screens, bootstrap, portfolio |
| `short_term_qre.py` | QRE-ST engine + validation |
| `meta_layer.py` | ML meta-labeling (tested, rejected — kept as evidence) |

## Methodology

Every change followed the same gauntlet before shipping:

1. **Component ablation** — each candidate feature tested individually against
   the baseline across a 20-name basket (10 large cap, 10 mid cap), never as
   a bundle.
2. **Temporal validation** — parameters chosen on 2012–2020, evaluated
   untouched on 2020–2026 (contains the COVID crash and the 2022 bear).
   OOS-beats-IS was required, not just OOS-positive.
3. **Cross-sectional regression checks** — changes motivated by one cohort
   (e.g. high-beta names) had to be neutral-or-better on the broad basket
   *and* on indices before changing any default.
4. **Statistical significance** — block-bootstrap CIs on portfolio Sharpe
   (block = 20 days, preserving short-range autocorrelation); a config
   shipped only if the CI cleared zero.
5. **Robustness surface** — ±25% perturbation of every key parameter
   (a knife-edge peak = overfit signature; QRE's surface is a flat plateau,
   Sharpe 0.82–1.03, with some neighbors *better* than the chosen point).
6. **Cost stress** — 2× and 4× commission+slippage.

The backtester is fully causal (signals at bar *t* use only data through *t*,
fills on the signal bar's close, matching Pine's
`process_orders_on_close=true`) and charges 0.05% commission + 2bp slippage
per side.

## Model architecture (QRE)

**Layer 1 — Regime classifier (statistical).** Every bar is classified as
TREND, RANGE, or TRANSITION using three independent statistics:

- **Kaufman Efficiency Ratio** — |net move| ÷ path length over 20 bars.
- **Rolling OLS slope t-statistic** (50 bars) — is the drift statistically
  distinguishable from noise (|t| > 2), not just visually apparent?
- **ADX(14)** — directional-strength confirmation.

A fourth statistic, the **ATR percentile rank over ~1 year**, is a circuit
breaker: above the 97th percentile, flatten and stand aside. (Originally
90th — see the [high-beta case study](#case-study-the-volatility-circuit-breaker)
for why that was a measured bug.)

**Layer 2 — Signal engines, routed by regime.**

- *TREND* → KAMA adaptive baseline + 20-day Donchian breakout in the
  direction of the significant slope.
- *RANGE* → fade 2σ z-score extremes confirmed by RSI(2) exhaustion; exit at
  the mean or a 10-bar time stop.

**Layer 3 — Risk engine.** Volatility-targeted sizing (fixed %-equity risk ÷
ATR stop distance), 5-ATR chandelier trail on trend trades, regime-loss exit,
gap-shock exit (adverse overnight gap > 2 ATR), relative-strength filter vs
SPY, and a notional cap per position.

**QRE-ST** (the 1–2 week sister model) inverts the anatomy: buy washouts
(RSI(2) < 10, z(10) < −1.5) in names above their 200d SMA; exit when the
bounce carries past the mean (z > +0.5) or a 10-bar time stop; 4-ATR
catastrophe stop. 68% win rate, no fat tail. The two models harvest different
effects on different clocks and are complementary, not interchangeable.

## Key positive findings

**Diversification is the edge amplifier.** Median single-name Sharpe is 0.32;
the 20-name equal-weight portfolio is 0.93 with max drawdown −1.1% (at 1%
risk) because staggered trades diversify idiosyncratic noise. Sizing scales
cleanly: Sharpe is invariant from 1% to 8% risk while CAGR and drawdown scale
together, with the Calmar-efficient point at ~5% risk.

**Vol-normalization generalizes across the volatility spectrum.** A 36-name
sweep (median ATR 1.3%–9.9% of price) falsified the hypothesis that high
volatility breaks the model — the 5–7% ATR bucket was the *best* cohort
(median Sharpe 0.61, 4/4 profitable). Because sizing, stops, entries, and
breakers are all expressed in each name's own volatility units, one
calibration serves the whole universe.

**Universe rule: breadth, not selection.** Cross-sectional persistence of
per-name Sharpe is ~zero (Spearman IS→OOS = 0.12); a portfolio of the
top-half names by past Sharpe *underperformed* the full universe OOS
(0.88 vs 0.93). The validated rule: trade everything liquid that passes the
structural tradeability screen; never curate by past performance.

**Holding period is load-bearing.** Forcing 1-week holds on QRE destroys 59%
of net PnL and drops profit factor to 1.11: the top 5% of trades carry 83% of
profit, average 68-day holds (vs 24 for the rest), and time-capping amputates
exactly those. Weekly-scale mandates therefore get a different model (QRE-ST),
not a truncated trend model.

**The tradeability screen** (structural, performance-blind): INSUFFICIENT
(< 750 bars post-warmup or < 15 signals — a 4-year single-name Sharpe has
stderr ≈ ±0.5, so the honest verdict is "don't know"), CAUTION (< $10M/day
liquidity, < $5 price, or > 1.25% of days gapping beyond 2 ATR), else PASS.

## Negative results

**ML meta-labeling — rejected.** A gradient-boosted layer (walk-forward, no
temporal leakage, 816 trades) was tested two ways. Classifier on win/loss:
*actively harmful* — its lowest-confidence quartile had the highest PnL per
trade ($527 vs $226), because in a trend system win probability
anti-correlates with payoff size; the filter would have skipped 42% of
signals worth +$127k net. Regressor on trade return: Spearman rank
correlation with realized returns −0.03 (nothing). Interpretation: the base
model's gates already consume the information in those features; conditional
on passing the gates, residual variation is noise.

**Five signal refinements — rejected.** Breakeven stops improved in-sample
(PF 1.15→1.28) and *reversed* out-of-sample (Sharpe 0.51→0.41) — the
textbook IS mirage. Stagnation exits, HTF trend filters, volume-confirmed
breakouts, and loosened MR gates all failed or were inert.

**Pyramiding — rejected as dominated.** Turtle-style adds raised OOS PnL +19%
but plain base sizing raised to match produced more PnL with better Sharpe
and less drawdown. Adding units mid-trend is badly-shaped leverage.

**Dip-buying high-vol story stocks — refused.** On an 18-name hyper-volatile
cohort (APLD, LUNR, RKLB, IONQ, RGTI, OKLO, …), every QRE-ST configuration
tested was net-negative in-sample (47% win, −$19k pooled); the full-history
numbers only look viable because a late mania window (+$9.5k, 61% win) papers
over the bleed, while the quality basket was positive in *both* windows. In
institutional names a 3σ dip is noise around a stable mean; in story stocks
it is information. Tuning parameters until this cohort backtests well would
certify a regime bet as an edge.

### Case study: the volatility circuit breaker

Testing on MU/APLD/LUNR exposed a real miscalibration: the 90th-percentile
vol breaker (shipped in v1, never individually ablated) was ejecting
structurally volatile names mid-rally — 11 of MU's 35 exits — and vetoing
entries on exactly the breakout days, because high-beta names trend *with*
high volatility. Moving it to the 97th percentile tripled MU's Sharpe
(0.07→0.41), was neutral on indices, and slightly improved the broad basket
(0.92→0.94). Validated in all three directions before shipping.

## Limitations & research integrity

- **Multiple-testing contamination.** Many experiments were evaluated against
  the same 2020–2026 out-of-sample window across this project's iterations.
  Repeated OOS consultation degrades its purity — the reported OOS numbers
  are honest per-experiment but the *final configuration* has seen that
  window indirectly many times. Mitigations: structural (non-performance)
  selection rules, bootstrap CIs, plateau analysis. The uncontaminated test
  is forward paper-trading, which is the recommended next step before any
  capital.
- **Survivorship bias in the test universe.** Baskets were assembled from
  names that are liquid *today*; a 2012 portfolio would have included names
  that failed. Cross-sectional conclusions (breadth, persistence) are less
  affected; absolute return levels are optimistic.
- **Fill approximations.** Daily bars; intrabar stop fills approximated
  (stop price unless gapped through); gap exits fill at the open in Python
  but at the close on TradingView historical bars.
- **Data**: yfinance auto-adjusted daily OHLCV — fine for signal research,
  not exact P&L accounting.
- Past performance ≠ future results. Nothing here is investment advice.

## Usage

```bash
pip install pandas numpy yfinance scikit-learn
python backtest_qre.py SPY                                 # single-name backtest
python backtest_qre.py QQQ --sweep                         # walk-forward sweep + OOS
python backtest_qre.py "AAPL,MU,..." --stocks --screen     # tradeability screen
python backtest_qre.py "AAPL,MU,..." --stocks --basket     # portfolio + bootstrap CI + cost stress + plateau
python backtest_qre.py "AAPL,MU,..." --stocks --signals    # daily watchlist scan (live entry signals)
python backtest_qre.py --synthetic                         # smoke test, no network
python short_term_qre.py                                   # QRE-ST validation
python meta_layer.py                                       # meta-labeling experiment (negative result)
```

## TradingView deployment

1. Pine Editor → paste `quant_regime_ensemble.pine` (and/or `qre_st.pine`) →
   Add to chart. Strategy Tester shows the per-name backtest; the chart shows
   entry/exit markers, regime background, and a live stats table.
2. Screening: favorite `qre_screener.pine` and use it in the Pine Screener on
   a watchlist (paid plans), or run the Python `--signals` scan (free).
3. Alerts: strategy alert → "Order fills only" → message
   `{{strategy.order.alert_message}}` (JSON payload with action/qty/stop;
   webhook-ready on paid plans).
4. Portfolio sizing: per chart, set risk% = (total risk) ÷ (number of names)
   and notional cap = 100 ÷ (number of names).
