"""
Backtester for the QRE strategy.

Mirrors the Pine Script version bar-for-bar so TradingView results can be
cross-checked, plus the things Pine can't do well: train/test splits,
walk-forward sweeps, bootstrap stats, portfolio runs.

Everything is causal - a signal on bar t only uses data up to bar t, and
fills happen on the close of the signal bar (same as Pine with
process_orders_on_close=true). Costs: 0.05% commission + 2bp slippage per side.

Usage:
    pip install pandas numpy yfinance
    python backtest_qre.py SPY
    python backtest_qre.py QQQ --sweep
    python backtest_qre.py "AAPL,MU" --stocks --basket
    python backtest_qre.py --stocks --basket   # the 20-name evaluation basket
    python backtest_qre.py --synthetic     # smoke test, no internet needed
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


# ---- evaluation basket ----
# The fixed 20-name universe every headline result is measured on (see
# RESEARCH.md). 10 large cap + 10 mid cap, assembled from the structural
# tradeability screen - liquidity, history, gap behaviour - and never from past
# performance. Held constant so component ablations stay comparable; the basket
# SIZE was a test-design constant, not swept, so 20 is not claimed to be optimal.
LARGE_CAP = ("AAPL", "MSFT", "NVDA", "JPM", "XOM", "KO", "TSLA", "UNH", "HD", "CAT")
MID_CAP = ("WSM", "DKS", "TOL", "CROX", "ANF", "TXRH", "EME", "GGG", "WMS", "SAIA")
DEFAULT_BASKET = LARGE_CAP + MID_CAP


def parse_symbols(arg: str | None) -> list[str]:
    """Comma-separated symbols, or the default evaluation basket if omitted."""
    if not arg:
        return list(DEFAULT_BASKET)
    return [s.strip().upper() for s in arg.split(",") if s.strip()]


# ---- parameters ----
@dataclass(frozen=True)
class Params:
    # regime classifier
    er_len: int = 20
    er_trend_th: float = 0.25
    reg_len: int = 50
    t_stat_th: float = 2.0
    adx_len: int = 14
    adx_th: float = 20.0
    vol_lookback: int = 252
    vol_cap_pct: float = 90.0
    # trend engine
    kama_len: int = 20
    kama_fast: int = 2
    kama_slow: int = 30
    don_len: int = 20
    # mean-reversion engine
    z_len: int = 20
    z_entry: float = 2.0
    z_exit: float = 0.5
    rsi_len: int = 2
    rsi_os: float = 10.0
    rsi_ob: float = 90.0
    mr_time_stop: int = 10
    # risk engine
    risk_pct: float = 1.0
    atr_len: int = 14
    atr_stop_mult: float = 2.5
    chand_mult: float = 5.0
    allow_shorts: bool = False
    # costs
    commission_pct: float = 0.05   # per side, %
    slippage_bp: float = 2.0       # per side, basis points
    # single-stock defenses, all off by default (= index mode)
    use_mkt_filter: bool = False   # longs only when benchmark > its 200d SMA
    mkt_ma_len: int = 200
    use_rs_filter: bool = False    # longs only when stock outperforms benchmark
    rs_len: int = 50
    gap_atr_mult: float = 0.0      # >0: exit on adverse open gap > k*ATR, skip entries on gap days
    max_notional_pct: float = 100.0  # cap position notional as % of equity
    adaptive_er: bool = False      # ER trend gate = own 70th pctile instead of fixed
    # refinement experiments (mostly rejected, kept for reproducibility)
    use_htf_filter: bool = False   # trend entries only with rising 100d EMA
    htf_len: int = 100
    use_vol_confirm: bool = False  # breakout volume > vol_confirm_x * 20d avg
    vol_confirm_x: float = 1.2
    mr_loose: bool = False         # drop ADX condition from the ranging gate
    stag_bars: int = 0             # >0: exit trend trade if < stag_atr profit after N bars
    stag_atr: float = 0.5
    use_breakeven: bool = False    # move stop to entry after 1.5*ATR of profit
    breakeven_atr: float = 1.5
    tp_atr: float = 0.0            # >0: fixed profit target at k*ATR (win-rate
                                   # experiment, caps the winners - see RESEARCH.md)
    # structural experiments (also rejected, see RESEARCH.md)
    pyramid_max: int = 1           # max units per position (1 = no pyramiding)
    pyramid_step_atr: float = 1.0  # add a unit every k*ATR of favorable move
    use_pullback: bool = False     # third engine: buy z-dips within uptrends
    pullback_z: float = -1.0
    pullback_rsi: float = 30.0
    max_hold: int = 0              # >0: force-close after N bars (holding-period
                                   # experiment - see RESEARCH.md)


def stock_params(**overrides) -> Params:
    """Preset for individual stocks (tested on 10 large caps + 10 mid caps).

    Kept after ablation: the RS filter, the 2-ATR gap exit, the notional cap.
    Dropped: the SPY 200d market filter (missed recoveries) and the adaptive
    ER threshold (blew up worst-case drawdown). See RESEARCH.md for numbers.
    """
    base = dict(use_rs_filter=True, gap_atr_mult=2.0,
                max_notional_pct=20.0, allow_shorts=False,
                vol_cap_pct=97.0)   # 90 kept kicking out volatile names (MU etc)
                                    # mid-rally; 97 only trips on real tail events
    base.update(overrides)
    return Params(**base)


# ---- indicators ----
def wilder_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return wilder_ema(tr, n)


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = wilder_ema(tr, n)
    pdi = 100 * wilder_ema(pd.Series(plus_dm, index=df.index), n) / atr_
    mdi = 100 * wilder_ema(pd.Series(minus_dm, index=df.index), n) / atr_
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return wilder_ema(dx.fillna(0), n)


def rsi(s: pd.Series, n: int) -> pd.Series:
    d = s.diff()
    gain = wilder_ema(d.clip(lower=0), n)
    loss = wilder_ema((-d).clip(lower=0), n)
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def efficiency_ratio(c: pd.Series, n: int) -> pd.Series:
    change = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n).sum()
    return (change / path.replace(0, np.nan)).fillna(0)


def kama(c: pd.Series, n: int, fast: int, slow: int) -> pd.Series:
    er = efficiency_ratio(c, n)
    fast_sc, slow_sc = 2 / (fast + 1), 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    out = np.full(len(c), np.nan)
    prev = c.iloc[0]
    for i in range(len(c)):
        prev = prev + sc.iloc[i] * (c.iloc[i] - prev)
        out[i] = prev
    return pd.Series(out, index=c.index)


def rolling_slope_tstat(c: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS slope and its t-stat, vectorized with sliding windows."""
    x = np.arange(n, dtype=float)
    x_dm = x - x.mean()
    sxx = float((x_dm ** 2).sum())
    y = c.to_numpy(dtype=float)

    slope = np.full(len(y), np.nan)
    tstat = np.full(len(y), np.nan)
    if len(y) >= n:
        win = np.lib.stride_tricks.sliding_window_view(y, n)      # (m, n)
        y_dm = win - win.mean(axis=1, keepdims=True)
        b = (y_dm * x_dm).sum(axis=1) / sxx
        resid = y_dm - b[:, None] * x_dm
        se = np.sqrt((resid ** 2).sum(axis=1) / (n - 2) / sxx)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, b / se, 0.0)
        slope[n - 1:] = b
        tstat[n - 1:] = t
    return pd.Series(slope, index=c.index), pd.Series(tstat, index=c.index)


def pct_rank(s: pd.Series, n: int) -> pd.Series:
    """Same as Pine's ta.percentrank: where the current value sits in the trailing window."""
    return s.rolling(n).apply(lambda w: (w[:-1] <= w[-1]).mean() * 100, raw=True)


# ---- backtest core ----
@dataclass
class Trade:
    entry_i: int
    exit_i: int
    side: int          # +1 / -1
    engine: str
    entry_px: float
    exit_px: float
    qty: float
    pnl: float
    reason: str
    features: dict = field(default_factory=dict)   # entry-bar stats (meta-labeling)
    entry_date: object = None
    exit_date: object = None


def compute_features(df: pd.DataFrame, p: Params,
                     bench: pd.Series | None = None) -> pd.DataFrame:
    f = df.copy()
    c = f["Close"]
    f["er"] = efficiency_ratio(c, p.er_len)
    f["slope"], f["tstat"] = rolling_slope_tstat(c, p.reg_len)
    f["adx"] = adx(f, p.adx_len)
    f["atr"] = atr(f, p.atr_len)
    f["vol_pctile"] = pct_rank(f["atr"], p.vol_lookback)
    f["kama"] = kama(c, p.kama_len, p.kama_fast, p.kama_slow)
    f["don_hi"] = f["High"].rolling(p.don_len).max().shift(1)
    f["don_lo"] = f["Low"].rolling(p.don_len).min().shift(1)
    basis = c.rolling(p.z_len).mean()
    sd = c.rolling(p.z_len).std(ddof=1)
    f["z"] = ((c - basis) / sd.replace(0, np.nan)).fillna(0)
    f["rsi_f"] = rsi(c, p.rsi_len)

    f["vol_blowout"] = f["vol_pctile"] >= p.vol_cap_pct

    # trend gate: fixed threshold, or self-calibrating (own 70th pctile, 1y)
    if p.adaptive_er:
        er_th = f["er"].rolling(252).quantile(0.70).fillna(p.er_trend_th)
    else:
        er_th = pd.Series(p.er_trend_th, index=f.index)
    f["er_th"] = er_th
    f["is_trending"] = (f["er"] > er_th) & (
        (f["tstat"].abs() > p.t_stat_th) | (f["adx"] > p.adx_th))
    f["is_ranging"] = (f["er"] < er_th * 0.8) & \
                      (f["tstat"].abs() < p.t_stat_th) & (f["adx"] < p.adx_th)

    if p.mr_loose:
        f["is_ranging"] = (f["er"] < er_th * 0.8) & (f["tstat"].abs() < p.t_stat_th)

    # precision refinements
    htf_ema = c.ewm(span=p.htf_len, adjust=False).mean()
    f["htf_up"] = htf_ema > htf_ema.shift(5)
    f["htf_dn"] = htf_ema < htf_ema.shift(5)
    f["vol_ok"] = f["Volume"] > p.vol_confirm_x * f["Volume"].rolling(20).mean()

    # single-stock defenses
    f["gap_atr"] = (f["Open"] - c.shift(1)) / f["atr"].shift(1)   # signed, in ATRs
    if bench is not None:
        b = bench.reindex(f.index).ffill()
        f["mkt_ok"] = b > b.rolling(p.mkt_ma_len).mean()
        rs = c / b
        rs_sma = rs.rolling(p.rs_len).mean()
        f["rs_ok"] = rs > rs_sma
        f["rs_edge"] = (rs / rs_sma - 1) * 100          # % above/below RS trend
    else:
        f["mkt_ok"] = True
        f["rs_ok"] = True
        f["rs_edge"] = 0.0
    if not p.use_mkt_filter:
        f["mkt_ok"] = True
    if not p.use_rs_filter:
        f["rs_ok"] = True
    return f


def run_backtest(df: pd.DataFrame, p: Params, capital: float = 100_000.0,
                 bench: pd.Series | None = None):
    f = compute_features(df, p, bench=bench)
    n = len(f)
    cost = p.commission_pct / 100 + p.slippage_bp / 10_000   # per side, fraction

    equity = np.full(n, capital)
    cash = capital
    pos = 0            # signed qty
    engine = ""
    entry_px = stop = trail = np.nan
    bars_in = 0
    entry_i = -1
    trades: list[Trade] = []

    C, H, L = f["Close"].to_numpy(), f["High"].to_numpy(), f["Low"].to_numpy()
    O = f["Open"].to_numpy()
    A = f["atr"].to_numpy()
    GAP = f["gap_atr"].to_numpy()
    warm = max(p.vol_lookback, p.reg_len, p.don_len + 1, p.z_len, p.kama_len) + 1

    entry_feats: dict = {}
    units = 0
    last_add_px = np.nan

    def close_pos(i, px, reason):
        nonlocal cash, pos, engine, trail, bars_in, entry_i, entry_px, stop, units
        fill = px * (1 - cost * np.sign(pos))
        pnl = pos * (fill - entry_px)
        cash += pnl
        trades.append(Trade(entry_i, i, int(np.sign(pos)), engine,
                            entry_px, fill, abs(pos), pnl, reason,
                            features=dict(entry_feats),
                            entry_date=f.index[entry_i], exit_date=f.index[i]))
        pos, engine, trail, bars_in, units = 0, "", np.nan, 0, 0

    for i in range(n):
        row = f.iloc[i]
        if pos != 0:
            bars_in += 1
            # gap-shock exit: bad overnight gap -> get out at the open.
            # this is the earnings-surprise case, so no pretty fills
            if p.gap_atr_mult > 0 and not np.isnan(GAP[i]):
                if (pos > 0 and GAP[i] < -p.gap_atr_mult) or \
                   (pos < 0 and GAP[i] > p.gap_atr_mult):
                    close_pos(i, O[i], "gap-shock")
        if pos != 0:
            # 1) hard stop (intrabar)
            if pos > 0 and L[i] <= stop:
                close_pos(i, min(stop, C[i]) if C[i] < stop else stop, "stop")
            elif pos < 0 and H[i] >= stop:
                close_pos(i, max(stop, C[i]) if C[i] > stop else stop, "stop")

        if pos != 0 and p.tp_atr > 0:
            # fixed profit target (intrabar fill at the target level)
            tgt = entry_px + np.sign(pos) * p.tp_atr * A[i]
            if (pos > 0 and H[i] >= tgt) or (pos < 0 and L[i] <= tgt):
                close_pos(i, tgt, "target")
        if pos != 0 and engine in ("TREND", "PULL"):
            # breakeven stop once the trade has moved enough in our favor
            if p.use_breakeven:
                if pos > 0 and C[i] - entry_px >= p.breakeven_atr * A[i]:
                    stop = max(stop, entry_px)
                elif pos < 0 and entry_px - C[i] >= p.breakeven_atr * A[i]:
                    stop = min(stop, entry_px)
            # stagnation exit: trade went nowhere, free up the capital
            if p.stag_bars > 0 and bars_in >= p.stag_bars and \
                    np.sign(pos) * (C[i] - entry_px) < p.stag_atr * A[i]:
                close_pos(i, C[i], "stagnation")
        if pos != 0 and engine in ("TREND", "PULL"):
            # chandelier trail update (uses completed bars incl. current)
            hh = f["High"].iloc[max(0, i - p.don_len + 1): i + 1].max()
            ll = f["Low"].iloc[max(0, i - p.don_len + 1): i + 1].min()
            if pos > 0:
                new_trail = hh - p.chand_mult * A[i]
                trail = new_trail if np.isnan(trail) else max(trail, new_trail)
                if L[i] <= trail:
                    close_pos(i, min(trail, C[i]) if C[i] < trail else trail, "trail")
                elif C[i] < row["kama"] and row["er"] < p.er_trend_th * 0.6:
                    close_pos(i, C[i], "regime")
            elif pos < 0:
                new_trail = ll + p.chand_mult * A[i]
                trail = new_trail if np.isnan(trail) else min(trail, new_trail)
                if H[i] >= trail:
                    close_pos(i, max(trail, C[i]) if C[i] > trail else trail, "trail")
                elif C[i] > row["kama"] and row["er"] < p.er_trend_th * 0.6:
                    close_pos(i, C[i], "regime")

        if pos != 0 and engine == "MR":
            if pos > 0 and (row["z"] > -p.z_exit or bars_in >= p.mr_time_stop):
                close_pos(i, C[i], "mr-exit")
            elif pos < 0 and (row["z"] < p.z_exit or bars_in >= p.mr_time_stop):
                close_pos(i, C[i], "mr-exit")

        if pos != 0 and p.max_hold > 0 and bars_in >= p.max_hold:
            close_pos(i, C[i], "max-hold")

        if pos != 0 and row["vol_blowout"]:
            close_pos(i, C[i], "vol-breaker")

        # pyramiding (turtle style): add a unit per step of favorable move,
        # raising the stop with each add
        if pos != 0 and engine in ("TREND", "PULL") and units < p.pyramid_max:
            side_ = int(np.sign(pos))
            if side_ * (C[i] - last_add_px) >= p.pyramid_step_atr * A[i]:
                stop_dist = p.atr_stop_mult * A[i]
                add_qty = np.floor(cash * p.risk_pct / 100 / stop_dist) if stop_dist > 0 else 0
                cap_qty = np.floor(cash * p.max_notional_pct / 100 / C[i])
                add_qty = min(add_qty, max(cap_qty - abs(pos), 0))
                if add_qty > 0:
                    add_px = C[i] * (1 + cost * side_)
                    entry_px = (entry_px * abs(pos) + add_px * add_qty) / (abs(pos) + add_qty)
                    pos += side_ * add_qty
                    last_add_px = C[i]
                    units += 1
                    new_stop = C[i] - side_ * stop_dist
                    stop = max(stop, new_stop) if side_ > 0 else min(stop, new_stop)

        # entries (flat only, after warmup, fills on close of signal bar)
        gap_day = p.gap_atr_mult > 0 and not np.isnan(GAP[i]) and \
            abs(GAP[i]) > p.gap_atr_mult
        if pos == 0 and i >= warm and not row["vol_blowout"] and not gap_day:
            eq = cash
            stop_dist = p.atr_stop_mult * A[i]
            qty = np.floor(eq * p.risk_pct / 100 / stop_dist) if stop_dist > 0 else 0
            qty = min(qty, np.floor(eq * p.max_notional_pct / 100 / C[i]))
            if qty > 0:
                long_ok = row["mkt_ok"] and row["rs_ok"]
                conf = (not p.use_vol_confirm or row["vol_ok"])
                trend_long = long_ok and conf and row["is_trending"] and \
                    (not p.use_htf_filter or row["htf_up"]) and \
                    C[i] > row["kama"] and row["slope"] > 0 and C[i] > row["don_hi"]
                trend_short = conf and row["is_trending"] and \
                    (not p.use_htf_filter or row["htf_dn"]) and \
                    C[i] < row["kama"] and row["slope"] < 0 and C[i] < row["don_lo"]
                mr_long = row["mkt_ok"] and row["is_ranging"] and \
                    row["z"] < -p.z_entry and row["rsi_f"] < p.rsi_os
                mr_short = row["is_ranging"] and row["z"] > p.z_entry and row["rsi_f"] > p.rsi_ob
                pull_long = p.use_pullback and long_ok and row["is_trending"] and \
                    C[i] > row["kama"] and row["slope"] > 0 and \
                    row["z"] < p.pullback_z and row["rsi_f"] < p.pullback_rsi
                side, eng = 0, ""
                if trend_long:
                    side, eng = 1, "TREND"
                elif pull_long:
                    side, eng = 1, "PULL"
                elif mr_long:
                    side, eng = 1, "MR"
                elif p.allow_shorts and trend_short:
                    side, eng = -1, "TREND"
                elif p.allow_shorts and mr_short:
                    side, eng = -1, "MR"
                if side != 0:
                    entry_px = C[i] * (1 + cost * side)
                    pos, engine, entry_i, bars_in = side * qty, eng, i, 0
                    stop = C[i] - side * stop_dist
                    trail = np.nan
                    units, last_add_px = 1, C[i]
                    entry_feats = {
                        "er": row["er"], "tstat": row["tstat"], "adx": row["adx"],
                        "vol_pctile": row["vol_pctile"], "z": row["z"],
                        "rsi_f": row["rsi_f"], "rs_edge": row["rs_edge"],
                        "atr_pct": A[i] / C[i] * 100,
                        "dist_kama": (C[i] - row["kama"]) / A[i],
                        "gap_atr": GAP[i] if not np.isnan(GAP[i]) else 0.0,
                        "side": side, "is_trend_engine": 1 if eng == "TREND" else 0,
                    }

        equity[i] = cash + (pos * (C[i] - entry_px) if pos != 0 else 0.0)

    if pos != 0:
        close_pos(n - 1, C[-1], "eod")
        equity[-1] = cash

    return pd.Series(equity, index=f.index), trades


# ---- metrics ----
def metrics(equity: pd.Series, trades: list[Trade], periods_per_year: int = 252) -> dict:
    r = equity.pct_change().dropna()
    ann = np.sqrt(periods_per_year)
    total_ret = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(equity) / periods_per_year
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = r.mean() / r.std() * ann if r.std() > 0 else 0.0
    downside = r[r < 0].std()
    sortino = r.mean() / downside * ann if downside and downside > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else np.inf
    return dict(total_return=total_ret, cagr=cagr, sharpe=sharpe, sortino=sortino,
                max_dd=dd, n_trades=len(trades),
                win_rate=len(wins) / len(trades) if trades else np.nan,
                profit_factor=pf,
                calmar=cagr / abs(dd) if dd < 0 else np.nan)


def print_report(name: str, equity: pd.Series, trades: list[Trade]):
    m = metrics(equity, trades)
    by_engine = {}
    for t in trades:
        by_engine.setdefault(t.engine, []).append(t.pnl)
    print(f"\n── {name} ─────────────────────────────────")
    print(f"  Total return   {m['total_return']*100:8.1f} %")
    print(f"  CAGR           {m['cagr']*100:8.1f} %")
    print(f"  Sharpe         {m['sharpe']:8.2f}")
    print(f"  Sortino        {m['sortino']:8.2f}")
    print(f"  Max drawdown   {m['max_dd']*100:8.1f} %")
    print(f"  Calmar         {m['calmar']:8.2f}")
    print(f"  Trades         {m['n_trades']:8d}   win rate {m['win_rate']*100:.0f} %"
          if trades else "  Trades              0")
    print(f"  Profit factor  {m['profit_factor']:8.2f}")
    for eng, pnls in sorted(by_engine.items()):
        print(f"    {eng:<6} {len(pnls):4d} trades, net {sum(pnls):+,.0f}")


# ---- data loading ----
def load_yf(symbol: str, start: str = "2010-01-01") -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(symbol, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def synthetic_ohlc(n: int = 3000, seed: int = 7) -> pd.DataFrame:
    """Fake price data (alternating trend and chop blocks) for offline smoke tests."""
    rng = np.random.default_rng(seed)
    px, prices = 100.0, []
    i = 0
    while i < n:
        block = rng.integers(120, 400)
        trending = rng.random() < 0.5
        mu = rng.choice([-0.0012, 0.0012]) if trending else 0.0
        sig = 0.012 if trending else 0.008
        theta = 0.0 if trending else 0.05          # mean reversion pull in chop
        anchor = px
        for _ in range(min(block, n - i)):
            px *= np.exp(mu + theta * (np.log(anchor) - np.log(px)) + sig * rng.standard_normal())
            prices.append(px)
            i += 1
    c = pd.Series(prices)
    h = c * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = c * (1 - np.abs(rng.normal(0, 0.004, n)))
    o = c.shift(1).fillna(c.iloc[0])
    idx = pd.bdate_range("2014-01-01", periods=n)
    return pd.DataFrame({"Open": o.values, "High": h.values, "Low": l.values,
                         "Close": c.values, "Volume": 1e6}, index=idx)


# ---- robustness toolkit ----
def portfolio_backtest(datas: dict[str, pd.DataFrame], p: Params,
                       bench: pd.Series | None = None,
                       capital: float = 100_000.0) -> pd.Series:
    """Equal-weight portfolio: run each name on capital/N and sum the curves."""
    slice_cap = capital / len(datas)
    curves = []
    for s, df in datas.items():
        eq, _ = run_backtest(df, p, capital=slice_cap, bench=bench)
        curves.append(eq.rename(s))
    port = pd.concat(curves, axis=1)
    port = port.ffill().fillna(slice_cap).sum(axis=1)
    return port


def bootstrap_sharpe_ci(equity: pd.Series, n_boot: int = 2000,
                        block: int = 20, seed: int = 0) -> tuple[float, float, float]:
    """Block-bootstrap 90% CI for the annualized Sharpe. Blocks keep the
    short-range autocorrelation that plain resampling would destroy."""
    r = equity.pct_change().dropna().to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)
    point = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    sharpes = np.empty(n_boot)
    n_blocks = int(np.ceil(n / block))
    for b in range(n_boot):
        starts = rng.integers(0, n - block, n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sd = sample.std()
        sharpes[b] = sample.mean() / sd * np.sqrt(252) if sd > 0 else 0.0
    lo, hi = np.percentile(sharpes, [5, 95])
    return lo, point, hi


def cost_stress(datas: dict[str, pd.DataFrame], p: Params,
                bench: pd.Series | None = None):
    """Re-run the portfolio with 2x and 4x costs to see if the edge survives."""
    print("\nCost stress (portfolio Sharpe):")
    for mult in (1.0, 2.0, 4.0):
        pm = replace(p, commission_pct=p.commission_pct * mult,
                     slippage_bp=p.slippage_bp * mult)
        port = portfolio_backtest(datas, pm, bench)
        m = metrics(port, [])
        print(f"  {mult:.0f}x costs ({pm.commission_pct:.2f}% + {pm.slippage_bp:.0f}bp): "
              f"Sharpe {m['sharpe']:5.2f}  maxDD {m['max_dd']*100:5.1f}%")


def plateau_check(datas: dict[str, pd.DataFrame], p: Params,
                  bench: pd.Series | None = None):
    """Nudge each key parameter about 25% either way. If Sharpe falls off a
    cliff at the chosen values, that would be an overfitting red flag."""
    perturbs = {
        "er_trend_th":   [0.20, 0.25, 0.30],
        "atr_stop_mult": [2.0, 2.5, 3.0],
        "chand_mult":    [4.0, 5.0, 6.0],
        "don_len":       [15, 20, 25],
        "kama_len":      [15, 20, 25],
    }
    print("\nParameter plateau (portfolio Sharpe at ±25% perturbations):")
    for key, vals in perturbs.items():
        row = []
        for v in vals:
            port = portfolio_backtest(datas, replace(p, **{key: v}), bench)
            row.append(f"{v}→{metrics(port, [])['sharpe']:.2f}")
        print(f"  {key:<14} " + "   ".join(row))


# ---- tradeability screen ----
def tradeability_screen(df: pd.DataFrame, p: Params,
                        bench: pd.Series | None = None) -> dict:
    """Checks whether the model's assumptions can even operate on a stock.

    On purpose this uses no performance data (that would be selection bias)
    and doesn't screen on volatility either - I tested that and vol level
    didn't predict anything (see RESEARCH.md). Verdicts:
      INSUFFICIENT - not enough history or too few signals to judge
      CAUTION      - liquidity/price/gap behavior breaks the fill assumptions
      PASS         - fine to include, though single-stock results stay noisy
    """
    f = compute_features(df, p, bench=bench)
    warm = max(p.vol_lookback, p.reg_len, p.don_len + 1, p.z_len, p.kama_len) + 1
    post = f.iloc[warm:]
    _, trades = run_backtest(df, p, bench=bench)

    dollar_vol = (post["Close"] * post["Volume"]).median()
    checks = {
        "bars_post_warmup": len(post),
        "n_signals": len(trades),                      # sample sufficiency, not P&L
        "median_dollar_vol_m": dollar_vol / 1e6,
        "median_price": post["Close"].median(),
        "extreme_gap_pct": (post["gap_atr"].abs() > 2).mean() * 100,
        "median_atr_pct": (post["atr"] / post["Close"]).median() * 100,  # informational only
    }
    reasons = []
    if checks["bars_post_warmup"] < 750:
        reasons.append(f"only {checks['bars_post_warmup']} bars post-warmup (<750, ~3y)")
    if checks["n_signals"] < 15:
        reasons.append(f"only {checks['n_signals']} signals — too few to judge")
    verdict = "INSUFFICIENT" if reasons else "PASS"
    if verdict == "PASS":
        if checks["median_dollar_vol_m"] < 10:
            reasons.append(f"median dollar volume ${checks['median_dollar_vol_m']:.1f}M "
                           "(<$10M — slippage assumptions unreliable)")
        if checks["median_price"] < 5:
            reasons.append(f"median price ${checks['median_price']:.2f} (<$5 — microstructure risk)")
        if checks["extreme_gap_pct"] > 1.25:
            reasons.append(f"{checks['extreme_gap_pct']:.2f}% of days gap >2 ATR "
                           "(stops routinely jumped — realized risk exceeds modeled risk)")
        if reasons:
            verdict = "CAUTION"
    return {"verdict": verdict, "reasons": reasons, **checks}


# ---- signal scanner ----
def scan_signals(symbols: list[str], p: Params, bench: pd.Series | None = None,
                 capital: float = 100_000.0, start: str = "2018-01-01"):
    """Check the latest bar of each symbol for a live entry signal.
    Same gates and sizing as run_backtest."""
    print(f"{'sym':>6} {'date':>11} {'regime':>10} {'signal':>11} {'close':>9} "
          f"{'stop':>9} {'qty':>6}  notes")
    for s in symbols:
        try:
            df = load_yf(s, start)
            f = compute_features(df, p, bench=bench)
            r = f.iloc[-1]
            atr_now = r["atr"]
            regime = ("VOL-BLOWOUT" if r["vol_blowout"] else
                      "TREND" if r["is_trending"] else
                      "RANGE" if r["is_ranging"] else "TRANSITION")
            gap_day = p.gap_atr_mult > 0 and abs(r["gap_atr"]) > p.gap_atr_mult
            long_ok = bool(r["mkt_ok"]) and bool(r["rs_ok"])
            trend_long = long_ok and r["is_trending"] and r["Close"] > r["kama"] and \
                r["slope"] > 0 and r["Close"] > r["don_hi"]
            mr_long = bool(r["mkt_ok"]) and r["is_ranging"] and \
                r["z"] < -p.z_entry and r["rsi_f"] < p.rsi_os
            signal = ""
            if not r["vol_blowout"] and not gap_day:
                if trend_long:
                    signal = "TREND-LONG"
                elif mr_long:
                    signal = "MR-LONG"
            stop_dist = p.atr_stop_mult * atr_now
            qty = int(min(np.floor(capital * p.risk_pct / 100 / stop_dist),
                          np.floor(capital * p.max_notional_pct / 100 / r["Close"]))) \
                if stop_dist > 0 else 0
            notes = []
            if gap_day:
                notes.append("gap day — no entries")
            if not r["rs_ok"] and p.use_rs_filter:
                notes.append("RS laggard")
            date = str(f.index[-1].date())
            if signal:
                print(f"{s:>6} {date:>11} {regime:>10} {signal:>11} {r['Close']:9.2f} "
                      f"{r['Close']-stop_dist:9.2f} {qty:6d}  {'; '.join(notes)}")
            else:
                print(f"{s:>6} {date:>11} {regime:>10} {'—':>11} {r['Close']:9.2f} "
                      f"{'':>9} {'':>6}  {'; '.join(notes)}")
        except Exception as e:
            print(f"{s:>6}  ERROR: {e}")


# ---- walk-forward sweep ----
def walk_forward(df: pd.DataFrame, base: Params, train_frac: float = 0.6):
    """Small grid search on the train segment, then report the winner on the
    held-out test segment."""
    split = int(len(df) * train_frac)
    train, test = df.iloc[:split], df.iloc[split - 300:]   # 300-bar warmup overlap
    grid = [replace(base, er_trend_th=e, atr_stop_mult=s, z_entry=z)
            for e in (0.30, 0.35, 0.40)
            for s in (2.0, 2.5, 3.0)
            for z in (1.75, 2.0, 2.25)]
    results = []
    for p in grid:
        eq, tr = run_backtest(train, p)
        m = metrics(eq, tr)
        results.append((m["sharpe"] if m["n_trades"] >= 10 else -9, p, m))
    results.sort(key=lambda x: x[0], reverse=True)
    best_sharpe, best_p, best_m = results[0]
    print(f"\nIn-sample champion: er_th={best_p.er_trend_th} "
          f"atr_stop={best_p.atr_stop_mult} z_entry={best_p.z_entry} "
          f"(IS Sharpe {best_sharpe:.2f}, {best_m['n_trades']} trades)")
    eq, tr = run_backtest(test, best_p)
    print_report("OUT-OF-SAMPLE (held-out)", eq, tr)
    return best_p


# ---- entrypoint ----
def main():
    ap = argparse.ArgumentParser(description="QRE strategy backtester")
    ap.add_argument("symbol", nargs="?", default=None,
                    help="ticker, or comma-separated list for --basket/--screen/"
                         "--signals (defaults to the 20-name evaluation basket)")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--sweep", action="store_true", help="walk-forward parameter sweep")
    ap.add_argument("--synthetic", action="store_true", help="run on synthetic data (no network)")
    ap.add_argument("--stocks", action="store_true",
                    help="use the single-stock preset (RS filter, gap-shock exit, notional cap)")
    ap.add_argument("--basket", action="store_true",
                    help="treat SYMBOL as comma-separated list; run equal-weight "
                         "portfolio + bootstrap CI + cost stress + parameter plateau")
    ap.add_argument("--screen", action="store_true",
                    help="tradeability screen only: can the model's assumptions "
                         "operate on this name? (structural, not performance)")
    ap.add_argument("--signals", action="store_true",
                    help="watchlist scanner: evaluate entry conditions on the "
                         "latest bar of each (comma-separated) symbol")
    args = ap.parse_args()

    if args.signals:
        p = stock_params() if args.stocks else Params()
        bench = load_yf("SPY", "2010-01-01")["Close"] if args.stocks else None
        scan_signals(parse_symbols(args.symbol), p, bench)
        return

    if args.screen:
        p = stock_params() if args.stocks else Params()
        bench = load_yf("SPY", "2010-01-01")["Close"] if args.stocks else None
        for s in parse_symbols(args.symbol):
            r = tradeability_screen(load_yf(s, args.start), p, bench)
            print(f"\n{s}: {r['verdict']}")
            print(f"  {r['bars_post_warmup']} bars post-warmup, {r['n_signals']} signals, "
                  f"${r['median_dollar_vol_m']:.0f}M/day, ATR {r['median_atr_pct']:.1f}% (informational), "
                  f"{r['extreme_gap_pct']:.2f}% extreme-gap days")
            for reason in r["reasons"]:
                print(f"  ⚠ {reason}")
        return

    if args.basket:
        syms = parse_symbols(args.symbol)
        p = stock_params() if args.stocks else Params()
        bench = load_yf("SPY", "2010-01-01")["Close"] if args.stocks else None
        datas = {s: load_yf(s, args.start) for s in syms}
        print("Tradeability screen:")
        for s, d in datas.items():
            r = tradeability_screen(d, p, bench)
            flag = "" if r["verdict"] == "PASS" else f"  ← {'; '.join(r['reasons'])}"
            print(f"  {s:>6} {r['verdict']:<12}{flag}")
        port = portfolio_backtest(datas, p, bench)
        print_report(f"PORTFOLIO ({len(syms)} names, equal weight)", port, [])
        lo, pt, hi = bootstrap_sharpe_ci(port)
        print(f"\n  Sharpe 90% bootstrap CI: [{lo:.2f}, {hi:.2f}]  (point {pt:.2f})")
        cost_stress(datas, p, bench)
        plateau_check(datas, p, bench)
        return

    p = stock_params() if args.stocks else Params()
    bench = None
    if args.stocks and not args.synthetic:
        try:
            bench = load_yf("SPY", "2010-01-01")["Close"]
        except Exception:
            print("Benchmark download failed; RS filter disabled.", file=sys.stderr)
            p = replace(p, use_rs_filter=False)
    if args.synthetic:
        df = synthetic_ohlc()
        name = "SYNTHETIC (regime-switching GBM)"
    else:
        try:
            symbol = args.symbol or "SPY"
            df = load_yf(symbol, args.start)
            name = symbol
        except Exception as e:
            print(f"Data download failed ({e}); falling back to synthetic data.", file=sys.stderr)
            df = synthetic_ohlc()
            name = "SYNTHETIC (fallback)"

    print(f"Loaded {len(df)} bars: {df.index[0].date()} → {df.index[-1].date()}")
    eq, trades = run_backtest(df, p, bench=bench)
    print_report(name, eq, trades)

    bh = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    print(f"\n  Buy & hold same period: {bh*100:+.1f} %")

    if args.sweep:
        walk_forward(df, p)


if __name__ == "__main__":
    main()
