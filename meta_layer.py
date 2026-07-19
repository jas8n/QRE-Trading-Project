"""
ML meta-labeling layer for the QRE strategy (López de Prado-style).

The base model decides WHEN to trade; a gradient-boosted classifier learns
WHICH of those signals tend to pay, from the statistical context at entry
(regime stats, vol percentile, relative strength, gap state, ...). Signals
the model distrusts are skipped or downsized. The base edge is never
modified — the ML only re-weights validated signals, which keeps the
overfitting surface small and auditable.

Walk-forward protocol (no temporal leakage):
  for each test year Y:  train on all trades that EXITED before Jan 1 of Y,
  predict trades ENTERED during Y. A trade still open at year-end can never
  appear in its own training set.

Usage:
    python meta_layer.py                 # default 20-name basket
    python meta_layer.py AAPL,MU,...     # custom basket
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from backtest_qre import Params, stock_params, load_yf, run_backtest

FEATURES = ["er", "tstat", "adx", "vol_pctile", "z", "rsi_f", "rs_edge",
            "atr_pct", "dist_kama", "gap_atr", "side", "is_trend_engine"]

DEFAULT_BASKET = ("AAPL,MSFT,NVDA,JPM,XOM,KO,TSLA,UNH,HD,CAT,"
                  "WSM,DKS,TOL,CROX,ANF,TXRH,EME,GGG,WMS,SAIA")


def collect_trades(symbols: list[str], p: Params, bench: pd.Series) -> pd.DataFrame:
    rows = []
    for s in symbols:
        df = load_yf(s, "2012-01-01")
        _, trades = run_backtest(df, p, bench=bench)
        for t in trades:
            rows.append({"symbol": s, "entry_date": t.entry_date,
                         "exit_date": t.exit_date, "pnl": t.pnl,
                         "r_mult": t.pnl / (t.qty * t.entry_px) * 100,  # % return on notional
                         **t.features})
    out = pd.DataFrame(rows).dropna(subset=FEATURES)
    return out.sort_values("entry_date").reset_index(drop=True)


def walk_forward_meta(trades: pd.DataFrame, first_test_year: int = 2017,
                      min_train: int = 100, skip_below: float = 0.42,
                      seed: int = 0):
    """Returns trades with p_win and meta weights attached (test years only)."""
    out = []
    years = range(first_test_year, trades["entry_date"].max().year + 1)
    for y in years:
        cutoff = pd.Timestamp(f"{y}-01-01")
        train = trades[trades["exit_date"] < cutoff]
        test = trades[(trades["entry_date"] >= cutoff) &
                      (trades["entry_date"] < pd.Timestamp(f"{y+1}-01-01"))]
        if len(train) < min_train or len(test) == 0:
            continue
        clf = HistGradientBoostingClassifier(
            max_depth=3, max_iter=120, learning_rate=0.08,
            min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
        clf.fit(train[FEATURES], (train["pnl"] > 0).astype(int))
        t = test.copy()
        t["p_win"] = clf.predict_proba(test[FEATURES])[:, 1]
        out.append(t)
    res = pd.concat(out, ignore_index=True)
    # meta weight: skip clearly distrusted signals, scale the rest 0.5–1.5
    res["take"] = res["p_win"] >= skip_below
    res["weight"] = np.where(res["take"], np.clip(res["p_win"] * 2.0, 0.5, 1.5), 0.0)
    return res


def report(res: pd.DataFrame):
    base_pnl = res["pnl"].sum()
    filt = res[res["take"]]
    filt_pnl = filt["pnl"].sum()
    sized_pnl = (res["pnl"] * res["weight"]).sum()
    n_skip = (~res["take"]).sum()

    print(f"\nWalk-forward meta-labeling on {len(res)} out-of-sample trades "
          f"({res['entry_date'].min().year}–{res['entry_date'].max().year}):")
    print(f"  {'':28} {'net PnL':>10} {'trades':>7} {'win%':>6} {'PnL/trade':>10}")
    for name, d, pnl in (("base (all signals)", res, base_pnl),
                         ("meta-filtered (skip weak)", filt, filt_pnl)):
        wr = (d['pnl'] > 0).mean() * 100
        print(f"  {name:<28} {pnl:10,.0f} {len(d):7d} {wr:6.1f} {pnl/len(d):10.0f}")
    print(f"  {'meta-sized (0.5–1.5x)':<28} {sized_pnl:10,.0f} {len(res):7d} "
          f"{'':6} {sized_pnl/len(res):10.0f}")
    print(f"\n  Skipped {n_skip} signals ({n_skip/len(res)*100:.0f}%); "
          f"their net PnL was {res.loc[~res['take'], 'pnl'].sum():,.0f} "
          f"(negative = filter earned its keep)")

    # calibration: does predicted p_win rank realized outcomes?
    res_ = res.copy()
    res_["bucket"] = pd.qcut(res_["p_win"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"],
                             duplicates="drop")
    print("\n  Calibration (quartiles of predicted p_win → realized):")
    g = res_.groupby("bucket", observed=True)
    for b, d in g:
        print(f"    {str(b):<8} n={len(d):4d}  win {(d['pnl']>0).mean()*100:5.1f}%  "
              f"avg PnL {d['pnl'].mean():8.0f}")


def main():
    syms = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASKET).split(",")
    syms = [s.strip().upper() for s in syms if s.strip()]
    p = stock_params()
    bench = load_yf("SPY", "2010-01-01")["Close"]
    print(f"Collecting trades from {len(syms)} names...")
    trades = collect_trades(syms, p, bench)
    print(f"  {len(trades)} trades, {trades['entry_date'].min().date()} → "
          f"{trades['entry_date'].max().date()}")
    res = walk_forward_meta(trades)
    report(res)


if __name__ == "__main__":
    main()
