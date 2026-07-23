"""
QRE-ST: the short-term sister model (1-2 week holds).

Forcing the trend model into weekly holds destroyed most of its profit, so
this is a separate model built for that horizon instead: buy sharp dips in
stocks that are above their 200d SMA, sell the bounce. High win rate, small
wins, no fat right tail - basically the opposite anatomy of QRE.

Entry: RSI(2) washout plus a stretched 10-day z-score, skip big gap days.
Exit: bounce past the mean (z > +0.5) or a 10-bar time stop. A 4-ATR
catastrophe stop and a gap exit cap the downside.

Usage:
    python short_term_qre.py                 # default 20-stock basket
    python short_term_qre.py "AAPL,MSFT"     # custom basket
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from backtest_qre import (atr, rsi, load_yf, metrics, print_report,
                          bootstrap_sharpe_ci, Trade)


@dataclass(frozen=True)
class STParams:
    trend_len: int = 200        # uptrend filter (SMA)
    rsi_len: int = 2
    rsi_entry: float = 10.0
    rsi_exit: float = 101.0     # disabled: the RSI exit kept selling the bounce
                                # too early, the z exit worked better
    z_len: int = 10
    z_entry: float = -1.5
    z_exit: float = 0.5         # exit slightly past the mean, tested best both
                                # in-sample and out-of-sample
    max_hold: int = 10          # hard time stop, keeps holds under 2 weeks
    cat_stop_atr: float = 4.0   # catastrophe stop (0 = none)
    gap_atr_mult: float = 2.0   # adverse-gap exit / gap-day entry block
    atr_len: int = 14
    risk_pct: float = 1.0
    max_notional_pct: float = 20.0
    commission_pct: float = 0.05
    slippage_bp: float = 2.0


def st_features(df: pd.DataFrame, p: STParams) -> pd.DataFrame:
    f = df.copy()
    c = f["Close"]
    f["sma_trend"] = c.rolling(p.trend_len).mean()
    f["rsi_f"] = rsi(c, p.rsi_len)
    basis = c.rolling(p.z_len).mean()
    sd = c.rolling(p.z_len).std(ddof=1)
    f["z"] = ((c - basis) / sd.replace(0, np.nan)).fillna(0)
    f["atr"] = atr(f, p.atr_len)
    f["gap_atr"] = (f["Open"] - c.shift(1)) / f["atr"].shift(1)
    return f


def run_st(df: pd.DataFrame, p: STParams, capital: float = 100_000.0):
    f = st_features(df, p)
    n = len(f)
    cost = p.commission_pct / 100 + p.slippage_bp / 10_000
    C, O, H, L = (f[k].to_numpy() for k in ("Close", "Open", "High", "Low"))
    A, GAP, Z, RSIF = (f[k].to_numpy() for k in ("atr", "gap_atr", "z", "rsi_f"))
    SMA = f["sma_trend"].to_numpy()

    equity = np.full(n, capital)
    cash, pos, entry_px, stop, bars_in, entry_i = capital, 0.0, np.nan, np.nan, 0, -1
    trades: list[Trade] = []
    warm = p.trend_len + 1

    def close_pos(i, px, reason):
        nonlocal cash, pos, bars_in
        fill = px * (1 - cost)
        pnl = pos * (fill - entry_px)
        cash += pnl
        trades.append(Trade(entry_i, i, 1, "ST-MR", entry_px, fill, pos, pnl, reason,
                            entry_date=f.index[entry_i], exit_date=f.index[i]))
        pos, bars_in = 0.0, 0

    for i in range(n):
        if pos != 0:
            bars_in += 1
            if not np.isnan(GAP[i]) and GAP[i] < -p.gap_atr_mult:
                close_pos(i, O[i], "gap-shock")
        if pos != 0 and p.cat_stop_atr > 0 and L[i] <= stop:
            close_pos(i, min(stop, C[i]) if C[i] < stop else stop, "cat-stop")
        if pos != 0:
            if RSIF[i] > p.rsi_exit or Z[i] > p.z_exit:
                close_pos(i, C[i], "snap-back")
            elif bars_in >= p.max_hold:
                close_pos(i, C[i], "time-stop")

        gap_day = not np.isnan(GAP[i]) and abs(GAP[i]) > p.gap_atr_mult
        if pos == 0 and i >= warm and not gap_day:
            if C[i] > SMA[i] and RSIF[i] < p.rsi_entry and Z[i] < p.z_entry:
                risk_per_sh = (p.cat_stop_atr if p.cat_stop_atr > 0 else 4.0) * A[i]
                qty = np.floor(cash * p.risk_pct / 100 / risk_per_sh) if risk_per_sh > 0 else 0
                qty = min(qty, np.floor(cash * p.max_notional_pct / 100 / C[i]))
                if qty > 0:
                    entry_px = C[i] * (1 + cost)
                    pos, entry_i, bars_in = qty, i, 0
                    stop = C[i] - p.cat_stop_atr * A[i] if p.cat_stop_atr > 0 else -np.inf

        equity[i] = cash + (pos * (C[i] - entry_px) if pos != 0 else 0.0)

    if pos != 0:
        close_pos(n - 1, C[-1], "eod")
        equity[-1] = cash
    return pd.Series(equity, index=f.index), trades


def st_portfolio(datas: dict[str, pd.DataFrame], p: STParams,
                 capital: float = 100_000.0) -> pd.Series:
    slice_cap = capital / len(datas)
    curves = [run_st(d, p, capital=slice_cap)[0].rename(s) for s, d in datas.items()]
    port = pd.concat(curves, axis=1).ffill().fillna(slice_cap).sum(axis=1)
    return port


DEFAULT_BASKET = ("AAPL,MSFT,NVDA,JPM,XOM,KO,TSLA,UNH,HD,CAT,"
                  "WSM,DKS,TOL,CROX,ANF,TXRH,EME,GGG,WMS,SAIA")


def main():
    syms = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASKET).split(",")
    syms = [s.strip().upper() for s in syms if s.strip()]
    p = STParams()
    datas = {s: load_yf(s, "2012-01-01") for s in syms}
    all_tr, ms = [], []
    for s, d in datas.items():
        eq, tr = run_st(d, p)
        all_tr += tr
        ms.append(metrics(eq, tr)["sharpe"])
    holds = [t.exit_i - t.entry_i for t in all_tr]
    wins = [t.pnl for t in all_tr if t.pnl > 0]
    print(f"{len(all_tr)} trades | win rate {len(wins)/len(all_tr)*100:.0f}% | "
          f"avg hold {np.mean(holds):.1f} bars | median name Sharpe {np.median(ms):.2f}")
    port = st_portfolio(datas, p)
    print_report("QRE-ST portfolio", port, all_tr)
    lo, pt, hi = bootstrap_sharpe_ci(port)
    print(f"\n  Sharpe 90% bootstrap CI: [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
