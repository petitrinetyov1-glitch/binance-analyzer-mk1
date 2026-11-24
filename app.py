import ccxt
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Crypto USDT-M Futures Analyzer", layout="wide")
st.title("Crypto USDT-M Futures Multi-Timeframe Analyzer")
st.caption("Data publik Futures (OKX) • Lengkap dengan semua indikator dan timeframe default")

TF = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", 
    "H1": "1h", "H2": "2h", "H4": "4h", "H12": "12h", 
    "D1": "1d", "W1": "1w"
}

IND = ["RSI", "MACD", "Volume", "OBV", "PVO", "MFI", "CMF", "CCI",
       "Stochastic RSI", "EMA 9/21", "SMA 20/50", "ADX",
       "Bollinger Bands", "ATR", "ROC"]

@st.cache_resource
def exchange():
    # MENGGUNAKAN OKX karena lebih ramah terhadap IP Cloud Server untuk data publik
    # Opsi "defaultType": "swap" memastikan kita mengambil data Futures Perpetual
    return ccxt.okx({
        "enableRateLimit": True, 
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

@st.cache_data(ttl=300)
def symbols_list():
    markets = exchange().load_markets()
    return sorted(s for s, m in markets.items()
                 if m.get("swap") and m.get("linear")
                 and m.get("quote") == "USDT" and m.get("settle") == "USDT"
                 and m.get("active") is not False)

def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + up / dn.replace(0, float("nan")))
    out = out.mask((dn == 0) & (up > 0), 100)
    return out.mask((dn == 0) & (up == 0), 50)

def calculate(df, chosen):
    c, h, l, v = df.close, df.high, df.low, df.volume
    if "RSI" in chosen: df["RSI"] = rsi(c)
    if "MACD" in chosen:
        line = ema(c, 12) - ema(c, 26)
        sig = ema(line, 9)
        df["MACD"], df["MACD Signal"], df["MACD Hist"] = line, sig, line - sig
    if "Volume" in chosen:
        df["Volume"] = v
        df["Volume SMA20"] = v.rolling(20).mean()
        df["Volume vs SMA20 %"] = (v / df["Volume SMA20"] - 1) * 100
    if "OBV" in chosen:
        df["OBV"] = (v * c.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)).fillna(0)).cumsum()
        df["OBV EMA20"] = ema(df["OBV"], 20)
    if "PVO" in chosen:
        vf, vs = ema(v, 12), ema(v, 26)
        pvo = (vf - vs) / vs.replace(0, float("nan")) * 100
        df["PVO"] = pvo
        df["PVO Signal"] = ema(pvo, 9)
        df["PVO Hist"] = df["PVO"] - df["PVO Signal"]
    if "MFI" in chosen:
        tp = (h + l + c) / 3
        flow = tp * v
        pos = flow.where(tp.diff() > 0, 0).rolling(14).sum()
        neg = flow.where(tp.diff() < 0, 0).rolling(14).sum()
        df["MFI"] = (100 - 100 / (1 + pos / neg.replace(0, float("nan")))).mask((neg == 0) & (pos > 0), 100)
    if "CMF" in chosen:
        mult = ((c - l) - (h - c)) / (h - l).replace(0, float("nan"))
        df["CMF"] = (mult * v).rolling(20).sum() / v.rolling(20).sum().replace(0, float("nan"))
    if "CCI" in chosen:
        tp = (h + l + c) / 3
        mean = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        df["CCI"] = (tp - mean) / (0.015 * mad.replace(0, float("nan")))
    if "Stochastic RSI" in chosen:
        r = rsi(c)
        lo, hi = r.rolling(14).min(), r.rolling(14).max()
        k = ((r - lo) / (hi - lo).replace(0, float("nan")) * 100).rolling(3).mean()
        df["Stoch RSI %K"], df["Stoch RSI %D"] = k, k.rolling(3).mean()
    if "EMA 9/21" in chosen:
        df["EMA 9"], df["EMA 21"] = ema(c, 9), ema(c, 21)
    if "SMA 20/50" in chosen:
        df["SMA 20"], df["SMA 50"] = c.rolling(20).mean(), c.rolling(50).mean()
    if "ADX" in chosen:
        up, down = h.diff(), -l.diff()
        pdm = up.where((up > down) & (up > 0), 0)
        mdm = down.where((down > up) & (down > 0), 0)
        prev = c.shift(1)
        tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        pdi = 100 * pdm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, float("nan"))
        mdi = 100 * mdm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, float("nan"))
        dx = 100 * (pdi-mdi).abs() / (pdi+mdi).replace(0, float("nan"))
