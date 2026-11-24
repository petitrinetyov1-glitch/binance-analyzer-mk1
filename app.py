import ccxt
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Binance USDT-M Futures Analyzer (Full)", layout="wide")
st.title("Binance USDT-M Futures Multi-Timeframe Analyzer (Full Version)")
st.caption("Data publik Binance Futures • Lengkap dengan semua indikator dan timeframe default")

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
    return ccxt.binanceusdm({"enableRateLimit": True, "timeout": 20000})

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
        df["ADX"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    if "Bollinger Bands" in chosen:
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        df["BB Upper"], df["BB Middle"], df["BB Lower"] = mid+2*sd, mid, mid-2*sd
    if "ATR" in chosen:
        prev = c.shift(1)
        tr = pd.concat([h-l, (h-prev).abs(), (l-prev).abs()], axis=1).max(axis=1)
        df["ATR"] = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    if "ROC" in chosen:
        df["ROC %"] = c.pct_change(10) * 100
    return df

with st.sidebar:
    st.header("Pengaturan")
    try:
        all_symbols = symbols_list()
    except Exception as e:
        st.error("Gagal memuat daftar Futures Binance.")
        st.code(str(e))
        st.stop()
        
    query = st.text_input("Cari pair", placeholder="BTC, ETH, SOL...")
    options = [s for s in all_symbols if query.upper() in s.upper()] if query else all_symbols
    
    default_index = 0
    if "BTC/USDT:USDT" in options:
        default_index = options.index("BTC/USDT:USDT")
    
    selected = st.selectbox("Pilih kontrak USDT-M Futures", options, index=default_index if options else 0)
    timeframes = st.multiselect("Timeframe", list(TF), default=["M1", "M5", "M15", "M30", "H1", "H2", "H4", "D1"])
    chosen = st.multiselect("Indikator", IND, default=IND)
    candle_count = st.selectbox("Jumlah candle", [100, 150, 200, 300, 500], index=0)
    closed_only = st.checkbox("Gunakan candle yang sudah tutup", value=True)

st.info("Pilih pair di panel kiri. Semua timeframe dan indikator sudah otomatis terpilih untuk analisis penuh.")

if st.button("Analisis / Refresh", type="primary"):
    if not selected or not timeframes:
        st.warning("Pilih pair dan minimal satu timeframe.")
    else:
        ex = exchange()
        results, errors = [], []
        symbols_to_process = [selected]
        total = len(symbols_to_process) * len(timeframes)
        bar, status = st.progress(0), st.empty()
        count = 0
        
        for symbol in symbols_to_process:
            for label in timeframes:
                count += 1
                status.text(f"Mengambil {symbol} {label} ({count}/{total})")
                try:
                    candles = ex.fetch_ohlcv(symbol, timeframe=TF[label], limit=candle_count)
                    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    
                    if closed_only:
                        duration = ex.parse_timeframe(TF[label]) * 1000
                        now = ex.milliseconds()
                        df = df[(df["timestamp"].astype("int64") // 1_000_000 + duration) <= now].copy()
                        
                    if df.empty:
                        continue
                        
                    df = calculate(df, chosen)
                    last = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else last
                    
                    row = {
                        "Pair": symbol, "Timeframe": label,
                        "Candle UTC": last["timestamp"].strftime("%Y-%m-%d %H:%M"),
                        "Close": last["close"],
                        "Change candle %": (last["close"] / prev["close"] - 1) * 100 if prev["close"] else None,
                        "Volume": last["volume"],
                    }
                    
                    for col in df.columns:
                        if col not in ["timestamp", "open", "high", "low", "close", "volume"]:
                            row[col] = last[col]
                            
                    results.append(row)
                except Exception as e:
                    errors.append(f"{symbol} {label}: {str(e)[:180]}")
                bar.progress(count / total)
                
        bar.empty()
        status.empty()
        st.session_state["results"] = results
        st.session_state["errors"] = errors
        st.session_state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state["selected_pair"] = selected

if "results" in st.session_state:
    rows = st.session_state["results"]
    st.caption(f"Update terakhir: {st.session_state['updated']}")
    if rows:
        result = pd.DataFrame(rows)
        st.dataframe(result, use_container_width=True, hide_index=True)
        
        pair_name = st.session_state.get("selected_pair", "crypto").replace("/", "").replace(":", "_")
        download_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{pair_name}_{download_time}_full.csv"
        
        st.download_button(
            "Download CSV", 
            result.to_csv(index=False).encode("utf-8-sig"),
            filename, 
            "text/csv"
        )
    else:
        st.warning("Tidak ada data yang berhasil dianalisis.")
        
    if st.session_state.get("errors"):
        with st.expander(f"Detail error ({len(st.session_state['errors'])})"):
            st.code("\n".join(st.session_state["errors"][:50]))

st.caption("USDT-M perpetual Futures. Semua indikator aktif secara default.")
