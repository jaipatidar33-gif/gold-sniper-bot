"""
GOLD SNIPER BOT v4.1 — FINAL
- 38 setups | Market+Limit alerts | ATR SL/TP
- DST shift | News blackout | 1-min monitor (Twelve Data)
"""
import os, time, traceback
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import requests

# ═══════════════════════════════════════════════════════════════
TG_TOKEN = os.getenv("TG_TOKEN", "")
TG_CHAT  = os.getenv("TG_CHAT_ID", "")
TD_KEY   = os.getenv("TD_KEY", "")

SYMBOLS = {"GOLD": "XAU/USD"}
IST = timezone(timedelta(hours=5, minutes=30))

ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE") or "10000")
RISK_PER_TRADE  = float(os.getenv("RISK_PER_TRADE") or "0.01")
PIP_VALUE_GOLD  = 10.0

SL_ATR_MULT = 1.0
TP_ATR_MULT = 1.0
COOLDOWN_SEC = 1800
MAX_TRADES_WARN = 4
CONSEC_LOSS_WARN = 2
NEWS_BLACKOUT_MIN = 15
SIGNAL_EXPIRY_MIN = 15
RUN_DURATION_SEC = 13 * 60

# ═══════════════════════════════════════════════════════════════
FIRED = {}
LAST_SIGNAL = {}
DAILY_STATS = {'date': None, 'trades': 0, 'losses': 0}

# ═══════════════════════════════════════════════════════════════
def escape_md(t):
    if t is None: return ""
    for ch in ['_', '*', '[', ']', '`']:
        t = str(t).replace(ch, f"\\{ch}")
    return t

def send(text):
    if not TG_TOKEN or not TG_CHAT:
        print("[TG]", text); return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT, "text": text, "parse_mode": "MarkdownV2"},
            timeout=15,
        )
    except Exception as e:
        print(f"[TG ERROR] {e}")

# ═══════════════════════════════════════════════════════════════
# DST AUTO-SHIFT
# ═══════════════════════════════════════════════════════════════
def get_us_dst_offset(dt_utc):
    y = dt_utc.year
    mar = datetime(y, 3, 1, tzinfo=timezone.utc)
    d = (6 - mar.weekday()) % 7
    s2_mar = mar + timedelta(days=d+7)
    nov = datetime(y, 11, 1, tzinfo=timezone.utc)
    d = (6 - nov.weekday()) % 7
    s1_nov = nov + timedelta(days=d)
    return -4 if s2_mar <= dt_utc < s1_nov else -5

def get_uk_dst_offset(dt_utc):
    y = dt_utc.year
    mar = datetime(y, 3, 31, tzinfo=timezone.utc)
    d = (mar.weekday() - 6) % 7
    l_mar = mar - timedelta(days=d)
    oct_ = datetime(y, 10, 31, tzinfo=timezone.utc)
    d = (oct_.weekday() - 6) % 7
    l_oct = oct_ - timedelta(days=d)
    return 1 if l_mar <= dt_utc < l_oct else 0

def get_session_shift():
    now = datetime.now(timezone.utc)
    return (get_us_dst_offset(now) - (-5)), (get_uk_dst_offset(now) - 0)

# ═══════════════════════════════════════════════════════════════
# NEWS BLACKOUT
# ═══════════════════════════════════════════════════════════════
def get_news_events_today():
    now = datetime.now(timezone.utc)
    y, m, d = now.year, now.month, now.day
    events = []
    if d <= 7 and now.weekday() == 4:
        events.append(datetime(y, m, d, 12, 30, tzinfo=timezone.utc))
    if 12 <= d <= 16 and now.weekday() in [1, 2, 3]:
        events.append(datetime(y, m, d, 12, 30, tzinfo=timezone.utc))
    FOMC = [(1,29),(3,19),(5,7),(6,18),(7,30),(9,17),(11,5),(12,17)]
    for fm, fd in FOMC:
        if fm == m and fd == d:
            events.append(datetime(y, m, d, 18, 0, tzinfo=timezone.utc))
    return events

def is_news_blackout():
    now = datetime.now(timezone.utc)
    for ev in get_news_events_today():
        delta = (ev - now).total_seconds() / 60
        if -NEWS_BLACKOUT_MIN <= delta <= NEWS_BLACKOUT_MIN:
            return True, ev, delta
    return False, None, None

# ═══════════════════════════════════════════════════════════════
# DATA FEED
# ═══════════════════════════════════════════════════════════════
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

def get_live_price(symbol="XAU/USD"):
    try:
        r = requests.get("https://api.twelvedata.com/price", params={
            "symbol": symbol, "apikey": TD_KEY
        }, timeout=10).json()
        if "price" in r:
            return float(r["price"])
        print(f"[TD PRICE] {r}")
    except Exception as e:
        print(f"[TD PRICE ERROR] {e}")
    return None

# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def atr(df, n=14):
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def rsi(c, n=14):
    d = c.diff()
    u = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + u / dn.replace(0, np.nan))

def adx(df, n=14):
    up = df['high'].diff(); dn = -df['low'].diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, n)
    pdi = 100 * pd.Series(pdm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / tr
    mdi = 100 * pd.Series(mdm, index=df.index).ewm(alpha=1/n, adjust=False).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean()

# ═══════════════════════════════════════════════════════════════
# FEATURES
# ═══════════════════════════════════════════════════════════════
def build_features(df):
    if df is None or len(df) < 60: return None
    df = df.copy()
    c, h, l, o = df['close'], df['high'], df['low'], df['open']
    df['ema20'] = ema(c, 20); df['ema50'] = ema(c, 50); df['ema200'] = ema(c, 200)
    df['rsi'] = rsi(c, 14); df['atr'] = atr(df, 14); df['adx'] = adx(df, 14)

    bb_m = c.rolling(20).mean(); bb_s = c.rolling(20).std()
    df['bb_pos'] = (c - (bb_m - 2*bb_s)) / ((bb_m + 2*bb_s) - (bb_m - 2*bb_s)).replace(0, np.nan)
    tp = (h + l + c) / 3
    df['vwap'] = tp.cumsum() / pd.Series(1.0, index=df.index).cumsum()

    ts = df['datetime']
    df['hour'] = ts.dt.hour; df['dow'] = ts.dt.dayofweek; df['dom'] = ts.dt.day
    us_shift, uk_shift = get_session_shift()

    df['tokyo']   = (df['hour'] >= 0) & (df['hour'] < 8)
    df['london']  = (df['hour'] >= (7 - uk_shift)) & (df['hour'] < (16 - uk_shift))
    df['ny_am']   = (df['hour'] >= (13 - us_shift)) & (df['hour'] < (17 - us_shift))
    df['ny_pm']   = (df['hour'] >= (17 - us_shift)) & (df['hour'] < (21 - us_shift))
    df['overlap'] = (df['hour'] >= (13 - us_shift)) & (df['hour'] < (16 - us_shift))
    df['ny_all']  = (df['hour'] >= (13 - us_shift)) & (df['hour'] < (21 - us_shift))
    df['all']     = True

    df['tue_thu'] = df['dow'].isin([1, 2, 3])
    df['mon'] = df['dow'] == 0; df['fri'] = df['dow'] == 4
    df['mid_month'] = df['dom'].between(13, 18)
    df['month_start'] = df['dom'] <= 5; df['month_end'] = df['dom'] >= 25

    body = (c - o).abs(); rng = (h - l).replace(0, np.nan)
    uw = h - pd.concat([o, c], axis=1).max(axis=1)
    lw = pd.concat([o, c], axis=1).min(axis=1) - l

    df['engulf_bull'] = (c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1))
    df['engulf_bear'] = (c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1))
    df['hammer'] = (lw > 2 * body) & (uw < body)
    df['shooting_star'] = (uw > 2 * body) & (lw < body)

    hh = h.rolling(10).max().shift(1); ll = l.rolling(10).min().shift(1)
    df['bos_bull'] = c > hh; df['bos_bear'] = c < ll
    df['choch_bull'] = c > h.rolling(5).max().shift(1)
    df['choch_bear'] = c < l.rolling(5).min().shift(1)
    df['fvg_bull'] = l > h.shift(2); df['fvg_bear'] = h < l.shift(2)

    df['date'] = ts.dt.date
    g = df.groupby('date').agg(pd_h=('high', 'max'), pd_l=('low', 'min'))
    g['prev_h'] = g['pd_h'].shift(1); g['prev_l'] = g['pd_l'].shift(1)
    df = df.merge(g[['prev_h', 'prev_l']], left_on='date', right_index=True, how='left')

    df['sweep_pdh'] = (df['high'] > df['prev_h']) & (c < df['prev_h'])
    df['sweep_pdl'] = (df['low'] < df['prev_l']) & (c > df['prev_l'])
    df['break_pdh'] = c > df['prev_h']; df['break_pdl'] = c < df['prev_l']
    df['sweep_pdh_f'] = df['sweep_pdh']; df['sweep_pdl_f'] = df['sweep_pdl']

    df['day_hi'] = df.groupby('date')['high'].cummax()
    df['day_lo'] = df.groupby('date')['low'].cummin()
    df['at_day_hi'] = df['high'] >= df['day_hi'].shift(1)
    df['at_day_lo'] = df['low'] <= df['day_lo'].shift(1)
    df['rej_day_hi'] = (df['high'] > df['day_hi'].shift(1)) & (c < df['day_hi'].shift(1))
    df['rej_day_lo'] = (df['low'] < df['day_lo'].shift(1)) & (c > df['day_lo'].shift(1))
    df['sweep_pwl'] = (df['low'] < df['prev_l'].rolling(5).min().shift(1)) & (c > df['prev_l'].rolling(5).min().shift(1))

    df['up_trend'] = (df['ema20'] > df['ema50']) & (df['ema50'] > df['ema200'])
    df['dn_trend'] = (df['ema20'] < df['ema50']) & (df['ema50'] < df['ema200'])
    df['strong_trend'] = df['adx'] > 25; df['weak_trend'] = df['adx'] < 20
    df['high_vol'] = df['atr'] > df['atr'].rolling(100).mean() * 1.2
    df['low_vol'] = df['atr'] < df['atr'].rolling(100).mean() * 0.8
    df['rsi_os'] = df['rsi'] < 30; df['rsi_ob'] = df['rsi'] > 70
    df['bb_os'] = df['bb_pos'] < 0.05; df['bb_ob'] = df['bb_pos'] > 0.95
    df['above_vwap'] = c > df['vwap']; df['below_vwap'] = c < df['vwap']
    df['vwap_ext_up'] = (c - df['vwap']) / df['atr'] > 1.5
    df['vwap_ext_dn'] = (c - df['vwap']) / df['atr'] < -1.5
    vol20 = c.pct_change().rolling(20).std()
    df['vol_up'] = vol20 > vol20.rolling(20).mean()
    df['vol_dn'] = vol20 < vol20.rolling(20).mean()
    m50 = c.rolling(50).mean(); s50 = c.rolling(50).std()
    df['zscore'] = (c - m50) / s50.replace(0, np.nan)
    df['zscore_low'] = df['zscore'].abs() < 1.0
    df['zscore_high'] = df['zscore'].abs() > 1.5
    df['none'] = True
    return df

# ═══════════════════════════════════════════════════════════════
def pattern_col(pat, side):
    fixed = ('hammer', 'shooting_star', 'at_day_hi', 'at_day_lo', 'rej_day_hi', 'rej_day_lo',
             'break_pdh', 'break_pdl', 'sweep_pdh', 'sweep_pdl', 'sweep_pdh_f', 'sweep_pdl_f', 'sweep_pwl')
    if pat in fixed: return pat
    return f"{pat}_{'bull' if side == 'BUY' else 'bear'}"

def check_setup(s, df):
    side = str(s['direction']).upper()
    pat = str(s['pattern']).lower()
    col = pattern_col(pat, side)
    if col not in df.columns: return False
    if not bool(df[col].iloc[-2]): return False
    sess = str(s['session']).replace('sess_', '').lower()
    if sess in df.columns and not bool(df[sess].iloc[-2]): return False
    day = str(s['day']).lower()
    if day not in ('all', 'none', '', 'nan'):
        if day in df.columns and not bool(df[day].iloc[-2]): return False
    filt = str(s['filter']).lower()
    if filt not in ('none', '', 'nan'):
        if filt in df.columns and not bool(df[filt].iloc[-2]): return False
    return True

def ist_now(): return datetime.now(IST).strftime("%H:%M IST")

def calc_lot(entry, sl):
    risk = ACCOUNT_BALANCE * RISK_PER_TRADE
    dist = abs(entry - sl)
    if dist <= 0: return 0.01
    return max(0.01, round(risk / (dist * PIP_VALUE_GOLD), 2))

SESSION_IST = {
    'tokyo': '05:30-13:30 IST',
    'london': '12:30-21:30 (summer)/13:30-22:30 (winter) IST',
    'ny_am': '18:30-22:30 (summer)/19:30-23:30 (winter) IST',
    'ny_pm': '22:30-02:30 (summer)/23:30-03:30 (winter) IST',
    'overlap': '18:30-21:30 (summer)/19:30-22:30 (winter) IST',
    'ny_all': '18:30-02:30 (summer)/19:30-03:30 (winter) IST',
    'all': 'Full day',
}

# ═══════════════════════════════════════════════════════════════
def detect_signals_once():
    signals = []
    try:
        setups = pd.read_csv("setups.csv")
    except Exception as e:
        send(f"❌ setups.csv error: {escape_md(str(e))}")
        return signals

    for metal, symbol in SYMBOLS.items():
        try:
            df_5m = build_features(fetch(symbol, "5min", 300))
            df_15m = build_features(fetch(symbol, "15min", 300))
            df_30m = build_features(fetch(symbol, "30min", 300))
            if not all([df_5m is not None, df_15m is not None, df_30m is not None]):
                continue
            tf_map = {'5m': df_5m, '15m': df_15m, '30m': df_30m}
            metal_setups = setups[setups['instrument'] == metal]

            for _, s in metal_setups.iterrows():
                s = s.to_dict()
                tf = str(s['tf'])
                if tf not in tf_map: continue
                df = tf_map[tf]
                if not check_setup(s, df): continue

                price = float(df['close'].iloc[-2])
                atr_val = float(df['atr'].iloc[-2])
                if np.isnan(atr_val) or atr_val <= 0: continue
                side = str(s['direction']).upper()

                candle_time = pd.to_datetime(df['datetime'].iloc[-2], utc=True)
                age_min = (datetime.now(timezone.utc) - candle_time).total_seconds() / 60
                if age_min > SIGNAL_EXPIRY_MIN:
                    continue

                sl = price - SL_ATR_MULT * atr_val if side == 'BUY' else price + SL_ATR_MULT * atr_val
                tp = price + TP_ATR_MULT * atr_val if side == 'BUY' else price - TP_ATR_MULT * atr_val
                lot = calc_lot(price, sl)
                pip = 0.10
                limit = price - 5 * pip if side == 'BUY' else price + 5 * pip

                signals.append({
                    'metal': metal, 's': s, 'side': side, 'price': price,
                    'sl': sl, 'tp': tp, 'atr': atr_val, 'lot': lot,
                    'limit': limit, 'freshness_min': age_min, 'tf': tf,
                    'sltp_pips': (atr_val * SL_ATR_MULT) / 0.10,
                    'key': f"{metal}_{s['pattern']}_{s['direction']}_{s['session']}_{s['day']}_{s['filter']}_{candle_time}",
                })
        except Exception as e:
            print(f"❌ {metal}: {e}")

    signals.sort(key=lambda x: x['s'].get('wr', 0), reverse=True)
    return signals

def send_full_signal(sig, current_price=None):
    s = sig['s']; side = sig['side']; price = sig['price']
    tier = str(s.get('tier', 'C')).upper()

    warnings = []
    DAILY_STATS['trades'] += 1
    if DAILY_STATS['trades'] > MAX_TRADES_WARN:
        warnings.append(f"⚠️ Trade limit exceeded ({DAILY_STATS['trades']}/{MAX_TRADES_WARN})")
    if DAILY_STATS['losses'] >= CONSEC_LOSS_WARN:
        warnings.append(f"⚠️ {DAILY_STATS['losses']} consecutive losses")
    if tier == 'A':
        warnings.append("⭐ A-TIER priority")
    elif tier == 'C':
        warnings.append("⚠️ C-TIER (low priority)")

    warn_text = "\n".join(warnings) if warnings else "✅ All clear"
    emoji = "🟢" if side == "BUY" else "🔴"
    sess = SESSION_IST.get(str(s['session']).replace('sess_', ''), s['session'])

    fresh = sig['freshness_min']
    if fresh <= 5:
        age_txt = f"🟢 Fresh ({fresh:.0f} min)"
        market_allowed = True
    else:
        age_txt = f"⚠️ LATE ({fresh:.0f} min old)"
        market_allowed = False

    msg = (
        f"{emoji} *GOLD {side}* \\- {escape_md(tier)}\\-Tier\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"*Pattern:* `{escape_md(s['pattern'])}`\n"
        f"*Session:* {escape_md(sess)}\n"
        f"*Day:* {escape_md(s['day'])} \\| *Filter:* {escape_md(s['filter'])}\n"
        f"*TF:* {escape_md(sig['tf'])} \\| *WR:* {float(s['wr']):.1%}\n"
        f"*Signal:* {escape_md(age_txt)}\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 *SIGNAL CANDLE:* `{price:.2f}`\n"
    )
    if current_price:
        msg += f"💹 *CURRENT PRICE:* `{current_price:.2f}`\n"
    msg += f"*SL/TP:* `{sig['sl']:.2f}` / `{sig['tp']:.2f}` \\({sig['sltp_pips']:.0f} pips\\)\n"
    msg += f"*Lot Size:* `{sig['lot']}` \\(1% risk\\)\n"

    if market_allowed:
        msg += (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"📌 *MARKET ENTRY*\n"
            f"  SL: `{sig['sl']:.2f}` \\| TP: `{sig['tp']:.2f}`\n"
        )

    msg += (
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⏳ *LIMIT ENTRY* \\(5 pip pullback\\)\n"
        f"  Limit: `{sig['limit']:.2f}`\n"
        f"  SL: `{sig['sl']:.2f}` \\| TP: `{sig['tp']:.2f}`\n"
        f"  Timeout: 30 min\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"{escape_md(warn_text)}\n"
        f"📊 Trades today: {DAILY_STATS['trades']}\n"
        f"🕐 {ist_now()}"
    )
    send(msg)

def send_proximity_alert(sig, current_price):
    side = sig['side']
    entry = sig['price']
    pips_away = abs(current_price - entry) / 0.10
    msg = (
        f"🎯 *PRICE APPROACHING — GOLD {side}*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"Current: `{current_price:.2f}`\n"
        f"Entry: `{entry:.2f}`\n"
        f"Distance: *{pips_away:.1f} pips*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"⚠️ _Signal active — prepare to enter_\n"
        f"🕐 {ist_now()}"
    )
    send(msg)

# ═══════════════════════════════════════════════════════════════
def run_bot_loop():
    start_time = time.time()

    blackout, ev, delta = is_news_blackout()
    if blackout:
        if delta > 0:
            send(f"⚠️ *NEWS BLACKOUT* \\- {delta:.0f} min to news\n_No new trades_")
        else:
            send(f"⚠️ *NEWS ACTIVE* \\- {abs(delta):.0f} min since\n_Wait 15 min_")

    print("🔍 Initial scan...")
    signals = detect_signals_once()
    current_price = get_live_price()

    if signals:
        for sig in signals:
            key = sig['key']
            if key in FIRED and (time.time() - FIRED[key]) < COOLDOWN_SEC:
                continue
            send_full_signal(sig, current_price)
            FIRED[key] = time.time()
            LAST_SIGNAL[sig['metal']] = {'side': sig['side'], 'time': datetime.now(IST)}
            print(f"✅ Signal: {sig['s']['pattern']} {sig['side']}")
    else:
        print("   No signals at start")

    if not signals:
        return

    print(f"👁 Monitor loop for {RUN_DURATION_SEC // 60} min...")
    last_check = 0
    alerted = set()

    while time.time() - start_time < RUN_DURATION_SEC:
        elapsed = time.time() - start_time
        if elapsed - last_check < 60:
            time.sleep(5)
            continue
        last_check = elapsed

        cp = get_live_price()
        if cp is None:
            continue

        for sig in signals:
            key = f"{sig['tf']}_{sig['s']['pattern']}_{sig['side']}"
            if key in alerted:
                continue
            if abs(cp - sig['price']) < 0.50:
                send_proximity_alert(sig, cp)
                alerted.add(key)

# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🚀 Gold Sniper v4.1 starting...")
    try:
        setups = pd.read_csv("setups.csv")
        send(f"🤖 Gold Sniper Bot v4\\.1 online\n📋 {len(setups)} setups loaded\n🕐 {ist_now()}")
    except Exception as e:
        send(f"❌ Startup error: {escape_md(str(e))}")
        raise
    try:
        run_bot_loop()
    except Exception as e:
        send(f"❌ Loop error: {escape_md(str(e))}")
        traceback.print_exc()
