"""
GOLD SNIPER BOT — 31 Setups | Gold + Silver
"""
import os, time
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests

TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT  = os.getenv("TG_CHAT_ID", "")
TD_KEY   = os.getenv("TD_KEY", "")
SYMBOLS = {"GOLD": "XAU/USD"}
IST = timezone(timedelta(hours=5, minutes=30))


def send(text):
    if not TG_TOKEN or not TG_CHAT:
        print("[TG]", text)
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "Markdown"},
            timeout=15,
        )
    except Exception as e:
        print(f"[TG ERROR] {e}")


def fetch(symbol, interval, bars=300):
    r = requests.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": interval,
        "outputsize": bars, "apikey": TD_KEY, "format": "JSON",
    }, timeout=30).json()
    if "values" not in r:
        raise RuntimeError(f"Feed: {r}")
    df = pd.DataFrame(r["values"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df


def ema(s, n): return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    u = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + u / dn.replace(0, np.nan))


def adx(df, n=14):
    up = df['high'].diff()
    dn = -df['low'].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / tr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()


def build_features(df):
    df = df.copy()
    c, h, l, o = df['close'], df['high'], df['low'], df['open']
    df['ema20'] = ema(c, 20)
    df['ema50'] = ema(c, 50)
    df['ema200'] = ema(c, 200)
    df['rsi'] = rsi(c, 14)
    df['atr'] = atr(df, 14)
    df['adx'] = adx(df, 14)

    bb_m = c.rolling(20).mean()
    bb_s = c.rolling(20).std()
    df['bb_pos'] = (c - (bb_m - 2*bb_s)) / ((bb_m + 2*bb_s) - (bb_m - 2*bb_s)).replace(0, np.nan)

    tp = (h + l + c) / 3
    df['vwap'] = tp.cumsum() / pd.Series(1.0, index=df.index).cumsum()

    ts = df['datetime']
    df['hour'] = ts.dt.hour
    df['dow'] = ts.dt.dayofweek
    df['dom'] = ts.dt.day

    df['tokyo']   = (df['hour'] >= 0) & (df['hour'] < 8)
    df['london']  = (df['hour'] >= 7) & (df['hour'] < 16)
    df['ny_am']   = (df['hour'] >= 13) & (df['hour'] < 17)
    df['ny_pm']   = (df['hour'] >= 17) & (df['hour'] < 21)
    df['overlap'] = (df['hour'] >= 13) & (df['hour'] < 16)
    df['all']     = True

    df['tue_thu'] = df['dow'].isin([1, 2, 3])
    df['mon'] = df['dow'] == 0
    df['fri'] = df['dow'] == 4
    df['mid_month'] = df['dom'].between(13, 18)
    df['month_start'] = df['dom'] <= 5
    df['month_end'] = df['dom'] >= 25

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    uw = h - pd.concat([o, c], axis=1).max(axis=1)
    lw = pd.concat([o, c], axis=1).min(axis=1) - l

    df['engulf_bull'] = (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1))
    df['engulf_bear'] = (c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1))
    df['hammer'] = (lw > 2 * body) & (uw < body)
    df['shooting_star'] = (uw > 2 * body) & (lw < body)

    hh = h.rolling(10).max().shift(1)
    ll = l.rolling(10).min().shift(1)
    df['bos_bull'] = c > hh
    df['bos_bear'] = c < ll
    df['choch_bull'] = c > h.rolling(5).max().shift(1)
    df['choch_bear'] = c < l.rolling(5).min().shift(1)
    df['fvg_bull'] = l > h.shift(2)
    df['fvg_bear'] = h < l.shift(2)

    df['date'] = ts.dt.date
    g = df.groupby('date').agg(pd_h=('high', 'max'), pd_l=('low', 'min'))
    g['prev_h'] = g['pd_h'].shift(1)
    g['prev_l'] = g['pd_l'].shift(1)
    df = df.merge(g[['prev_h', 'prev_l']], left_on='date', right_index=True, how='left')

    df['day_hi'] = df.groupby('date')['high'].cummax()
    df['day_lo'] = df.groupby('date')['low'].cummin()
    df['at_day_hi'] = df['high'] >= df['day_hi'].shift(1)
    df['at_day_lo'] = df['low'] <= df['day_lo'].shift(1)
    df['rej_day_hi'] = (df['high'] > df['day_hi'].shift(1)) & (c < df['day_hi'].shift(1))
    df['rej_day_lo'] = (df['low'] < df['day_lo'].shift(1)) & (c > df['day_lo'].shift(1))

    df['up_trend'] = (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
    df['dn_trend'] = (df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])
    df['strong_trend'] = df['adx'] > 25
    df['weak_trend'] = df['adx'] < 20
    df['high_vol'] = df['atr'] > df['atr'].rolling(100).mean() * 1.2
    df['low_vol'] = df['atr'] < df['atr'].rolling(100).mean() * 0.8
    df['rsi_os'] = df['rsi'] < 30
    df['rsi_ob'] = df['rsi'] > 70
    df['bb_os'] = df['bb_pos'] < 0.05
    df['bb_ob'] = df['bb_pos'] > 0.95
    df['above_vwap'] = c > df['vwap']
    df['below_vwap'] = c < df['vwap']
    df['vwap_ext_up'] = (c - df['vwap']) / df['atr'] > 1.5
    df['vwap_ext_dn'] = (c - df['vwap']) / df['atr'] < -1.5

    vol20 = c.pct_change().rolling(20).std()
    df['vol_up'] = vol20 > vol20.rolling(20).mean()
    df['vol_dn'] = vol20 < vol20.rolling(20).mean()

    m50 = c.rolling(50).mean()
    s50 = c.rolling(50).std()
    df['zscore'] = (c - m50) / s50.replace(0, np.nan)
    df['zscore_low'] = df['zscore'].abs() < 1.0

    df['none'] = True
    return df


def pattern_col(pat, side):
    fixed = ('hammer', 'shooting_star', 'at_day_hi', 'at_day_lo',
             'rej_day_hi', 'rej_day_lo')
    if pat in fixed:
        return pat
    return f"{pat}_{'bull' if side == 'BUY' else 'bear'}"


def check_setup(s, df):
    side = str(s['direction']).upper()
    pat = str(s['pattern']).lower()

    col = pattern_col(pat, side)
    if col not in df.columns:
        return False
    if not df[col].iloc[-1]:
        return False

    sess = str(s['session']).replace('sess_', '').lower()
    if sess in df.columns and not df[sess].iloc[-1]:
        return False

    day = str(s['day']).lower()
    if day not in ('all', 'none', ''):
        if day in df.columns and not df[day].iloc[-1]:
            return False

    filt = str(s['filter']).lower()
    if filt != 'none':
        if filt in df.columns and not df[filt].iloc[-1]:
            return False

    return True


def ist_now():
    return datetime.now(IST).strftime("%H:%M IST")


SESSION_IST = {
    'tokyo': '05:30-13:30 IST',
    'london': '12:30-21:30 IST',
    'ny_am': '18:30-22:30 IST',
    'ny_pm': '22:30-02:30 IST',
    'overlap': '18:30-21:30 IST',
    'all': 'Full day',
}


FIRED = {}


def run_once():
    setups = pd.read_csv("setups.csv")
    print(f"📋 {len(setups)} setups | {ist_now()}")

    for metal, symbol in SYMBOLS.items():
        try:
            df_5m = build_features(fetch(symbol, "5min", 300))
            df_15m = build_features(fetch(symbol, "15min", 300))
            metal_setups = setups[setups['instrument'] == metal]

            for _, s in metal_setups.iterrows():
                s = s.to_dict()
                tf = str(s['tf'])
                df = df_5m if tf == '5m' else df_15m

                if not check_setup(s, df):
                    continue

                key = f"{metal}_{s['pattern']}_{s['direction']}_{s['session']}_{s['day']}_{s['filter']}"
                now = time.time()
                if key in FIRED and (now - FIRED[key]) < 1800:
                    continue

                price = df['close'].iloc[-1]
                atr_val = df['atr'].iloc[-1]
                side = str(s['direction']).upper()

                if str(s['tier']) == 'RR2':
                    sl = price - 1.0 * atr_val if side == 'BUY' else price + 1.0 * atr_val
                    tp = price + 2.0 * atr_val if side == 'BUY' else price - 2.0 * atr_val
                    sltp_label = "1xATR / 2xATR (1:2)"
                else:
                    sl = price - 1.0 * atr_val if side == 'BUY' else price + 1.0 * atr_val
                    tp = price + 1.0 * atr_val if side == 'BUY' else price - 1.0 * atr_val
                    sltp_label = "1xATR / 1xATR (1:1)"

                pip = 0.10 if metal == 'GOLD' else 0.001
                limit = price - 5 * pip if side == 'BUY' else price + 5 * pip

                emoji = "🟢" if side == "BUY" else "🔴"
                sess_label = SESSION_IST.get(str(s['session']).replace('sess_', ''), s['session'])

                msg = (
                    f"{emoji} *{metal} {side}* — {s['tier']}-Tier\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"*Pattern:* `{s['pattern']}`\n"
                    f"*Session:* {sess_label}\n"
                    f"*Day:* {s['day']}  |  *Filter:* {s['filter']}\n"
                    f"*TF:* {tf}  |  *WR:* {float(s['wr']):.1%}\n"
                    f"*SL/TP:* {sltp_label}\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"📌 *MARKET* (instant)\n"
                    f"  Entry: `{price:.2f}`\n"
                    f"  SL: `{sl:.2f}`  |  TP: `{tp:.2f}`\n\n"
                    f"⏳ *LIMIT* (30 min timeout)\n"
                    f"  Limit: `{limit:.2f}`\n"
                    f"  SL: `{sl:.2f}`  |  TP: `{tp:.2f}`\n"
                    f"━━━━━━━━━━━━━━━━━━\n"
                    f"🕐 {ist_now()}"
                )
                send(msg)
                FIRED[key] = now
                print(f"✅ FIRE: {key} @ {price}")

        except Exception as e:
            print(f"❌ {metal}: {e}")


if __name__ == "__main__":
    print("🚀 Gold Sniper Bot starting...")
    send("🤖 *Gold Sniper Bot online* — 31 setups loaded")
    run_once()
