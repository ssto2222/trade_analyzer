"""
╔══════════════════════════════════════════════════════════════╗
║  GOLD (XAU/USD) 売買シグナル分析ダッシュボード               ║
║  MA × RSI テクニカル分析 ─ Exness MT5 / yfinance 対応       ║
╚══════════════════════════════════════════════════════════════╝

【セットアップ】
  pip install streamlit pandas numpy plotly

  ■ Exness MT5 接続 (Windows のみ):
    pip install MetaTrader5
    → MT5ターミナルを起動し、Exnessアカウントでログイン

  ■ yfinance フォールバック (全OS対応):
    pip install yfinance

【起動】
  streamlit run gold_trading_analyzer.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ページ設定
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.set_page_config(
    page_title="Gold Trading Signal Analyzer",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  カスタムCSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@300;400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

.stApp {
    background: linear-gradient(170deg, #07070f 0%, #0d0d1f 40%, #12101e 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a18 0%, #0f0f22 100%);
    border-right: 1px solid #1a1a2e;
}

h1, h2, h3 {
    font-family: 'Noto Sans JP', sans-serif !important;
}

.metric-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(212,160,23,0.15);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(212,160,23,0.4);
    background: rgba(255,255,255,0.05);
}
.metric-label {
    font-size: 12px;
    color: #6b7094;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
    font-family: 'Noto Sans JP', sans-serif;
}
.metric-value {
    font-size: 28px;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
}
.gold { color: #D4A017; }
.green { color: #00E676; }
.red { color: #FF1744; }
.purple { color: #AB47BC; }
.cyan { color: #00D2FF; }

.signal-buy {
    background: rgba(0,230,118,0.06);
    border: 1px solid rgba(0,230,118,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
}
.signal-sell {
    background: rgba(255,23,68,0.06);
    border: 1px solid rgba(255,23,68,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
}
.signal-strong {
    border-width: 2px;
    box-shadow: 0 0 15px rgba(212,160,23,0.1);
}

.header-title {
    text-align: center;
    padding: 10px 0 20px;
}
.header-title h1 {
    font-size: 2.2em;
    font-weight: 900;
    background: linear-gradient(135deg, #D4A017, #FFD54F, #D4A017);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.header-sub {
    color: #6b7094;
    font-size: 13px;
    letter-spacing: 4px;
    text-transform: uppercase;
}

.disclaimer {
    background: rgba(212,160,23,0.05);
    border: 1px solid rgba(212,160,23,0.15);
    border-radius: 8px;
    padding: 14px;
    font-size: 12px;
    color: #8b8fa8;
    line-height: 1.7;
    margin-top: 20px;
}

div[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  データ取得関数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fetch_mt5_data(symbol: str, timeframe_str: str, num_bars: int,
                   login: int = None, password: str = None, server: str = None) -> pd.DataFrame:
    """Exness MT5 ターミナル経由でデータ取得"""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        st.error("MetaTrader5 パッケージ未インストール: `pip install MetaTrader5`")
        return pd.DataFrame()

    if not mt5.initialize():
        st.error("MT5 初期化失敗。MT5ターミナルが起動しているか確認してください。")
        mt5.shutdown()
        return pd.DataFrame()

    # ログイン (認証情報が入力された場合)
    if login and password and server:
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            st.error(f"MT5 ログイン失敗 (アカウント: {login})")
            mt5.shutdown()
            return pd.DataFrame()

    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
    }
    timeframe = tf_map.get(timeframe_str, mt5.TIMEFRAME_H1)

    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        st.error(f"MT5 からデータ取得失敗。シンボル '{symbol}' を確認してください。")
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"time": "Date", "open": "Open", "high": "High",
                        "low": "Low", "close": "Close", "tick_volume": "Volume"}, inplace=True)
    df.set_index("Date", inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]]


def fetch_yfinance_data(period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """yfinance 経由でゴールドデータ取得 (フォールバック)"""
    try:
        import yfinance as yf
    except ImportError:
        st.error("yfinance 未インストール: `pip install yfinance`")
        return pd.DataFrame()

    ticker = yf.Ticker("GC=F")
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        st.error("yfinance からデータ取得失敗")
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    return df


def generate_demo_data(days: int = 180) -> pd.DataFrame:
    """デモ用シミュレーションデータ"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
    price = 2350.0
    data = []
    for i in range(days):
        volatility = 15 + np.random.rand() * 25
        trend = np.sin(i / 30) * 8 + np.cos(i / 15) * 5
        price += trend + (np.random.rand() - 0.48) * volatility
        price = max(2100, min(2700, price))
        high = price + np.random.rand() * 25
        low = price - np.random.rand() * 25
        open_p = price + (np.random.rand() - 0.5) * 15
        vol = int(np.random.rand() * 100000 + 50000)
        data.append([open_p, high, low, price, vol])
    df = pd.DataFrame(data, index=dates, columns=["Open", "High", "Low", "Close", "Volume"])
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  テクニカル指標計算
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def calc_sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_bollinger(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return sma, upper, lower

def calc_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  シグナル検出
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_signals(df: pd.DataFrame, ma_short: pd.Series, ma_long: pd.Series,
                   rsi: pd.Series, rsi_ob: float, rsi_os: float) -> pd.DataFrame:
    signals = []
    for i in range(1, len(df)):
        if pd.isna(ma_short.iloc[i]) or pd.isna(ma_long.iloc[i]) or pd.isna(rsi.iloc[i]):
            continue
        if pd.isna(ma_short.iloc[i-1]) or pd.isna(ma_long.iloc[i-1]):
            continue

        cross_up = ma_short.iloc[i-1] <= ma_long.iloc[i-1] and ma_short.iloc[i] > ma_long.iloc[i]
        cross_down = ma_short.iloc[i-1] >= ma_long.iloc[i-1] and ma_short.iloc[i] < ma_long.iloc[i]
        rsi_oversold = rsi.iloc[i] < rsi_os
        rsi_overbought = rsi.iloc[i] > rsi_ob

        sig_type = None
        reason = ""

        if cross_up and rsi_oversold:
            sig_type, reason = "STRONG_BUY", "ゴールデンクロス + RSI売られすぎ"
        elif cross_up:
            sig_type, reason = "BUY", "ゴールデンクロス (短期MAが長期MAを上抜け)"
        elif cross_down and rsi_overbought:
            sig_type, reason = "STRONG_SELL", "デッドクロス + RSI買われすぎ"
        elif cross_down:
            sig_type, reason = "SELL", "デッドクロス (短期MAが長期MAを下抜け)"
        elif i >= 2 and not pd.isna(rsi.iloc[i-1]):
            if rsi.iloc[i-1] >= rsi_os and rsi.iloc[i] < rsi_os:
                sig_type, reason = "BUY", "RSI が売られすぎゾーンに突入"
            elif rsi.iloc[i-1] <= rsi_ob and rsi.iloc[i] > rsi_ob:
                sig_type, reason = "SELL", "RSI が買われすぎゾーンに突入"

        if sig_type:
            signals.append({
                "日付": df.index[i],
                "タイプ": sig_type,
                "価格": df["Close"].iloc[i],
                "RSI": round(rsi.iloc[i], 1),
                "短期MA": round(ma_short.iloc[i], 2),
                "長期MA": round(ma_long.iloc[i], 2),
                "理由": reason,
            })

    return pd.DataFrame(signals)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  メインアプリ
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    # ─── ヘッダー ───
    st.markdown("""
    <div class="header-title">
        <div class="header-sub">GOLD TRADING SIGNAL ANALYZER</div>
        <h1>🥇 XAU/USD 分析ダッシュボード</h1>
        <div class="header-sub">移動平均線 × RSI シグナル検出システム</div>
    </div>
    """, unsafe_allow_html=True)

    # ─── サイドバー設定 ───
    with st.sidebar:
        st.markdown("## ⚙️ 設定")

        st.markdown("---")
        st.markdown("### 📡 データソース")
        data_source = st.radio(
            "データ取得方法",
            ["🎮 デモデータ", "📊 yfinance (GC=F)", "🔗 Exness MT5"],
            index=0,
            help="MT5はWindows + MT5ターミナル起動が必要です"
        )

        # MT5 設定
        mt5_login = None
        mt5_password = None
        mt5_server = None
        mt5_symbol = "XAUUSD"
        mt5_timeframe = "H1"
        mt5_bars = 500

        if data_source == "🔗 Exness MT5":
            st.markdown("#### Exness MT5 接続設定")
            mt5_login = st.number_input("アカウント番号", value=0, step=1)
            mt5_password = st.text_input("パスワード", type="password")
            mt5_server = st.text_input("サーバー", value="Exness-MT5Real",
                                        help="例: Exness-MT5Real, Exness-MT5Trial")
            mt5_symbol = st.text_input("シンボル", value="XAUUSD",
                                        help="Exnessでのゴールドシンボル名")
            mt5_timeframe = st.selectbox("時間足",
                ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"], index=4)
            mt5_bars = st.slider("取得バー数", 100, 2000, 500, 50)

        yf_period = "1mo"
        yf_interval = "5m"
        if data_source == "📊 yfinance (GC=F)":
            yf_period = st.selectbox("期間", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
            yf_interval = st.selectbox("間隔", ["1d", "1h", "5m"], index=0,
                                        help="1h以下は直近の短い期間のみ対応")

        demo_days = 180
        if data_source == "🎮 デモデータ":
            demo_days = st.slider("デモデータ日数", 60, 365, 180, 10)

        st.markdown("---")
        st.markdown("### 📈 移動平均線 (MA)")
        ma_type = st.radio("MAタイプ", ["SMA", "EMA"], horizontal=True)
        short_period = st.slider("短期MA 期間", 3, 50, 5, 1)
        long_period = st.slider("長期MA 期間", 10, 200, 25, 1)

        if short_period >= long_period:
            st.warning("⚠️ 短期MAは長期MAより小さく設定してください")

        st.markdown("---")
        st.markdown("### 📊 RSI")
        rsi_period = st.slider("RSI 期間", 5, 30, 14, 1)
        rsi_ob = st.slider("買われすぎライン", 60, 90, 70, 1)
        rsi_os = st.slider("売られすぎライン", 10, 40, 30, 1)

        st.markdown("---")
        st.markdown("### 🎨 追加指標")
        show_bb = st.checkbox("ボリンジャーバンド", value=False)
        show_macd = st.checkbox("MACD", value=False)
        show_volume = st.checkbox("出来高", value=True)

    # ─── データ取得 ───
    df = pd.DataFrame()
    source_label = ""

    with st.spinner("データ取得中..."):
        if data_source == "🔗 Exness MT5":
            df = fetch_mt5_data(mt5_symbol, mt5_timeframe, mt5_bars,
                                mt5_login if mt5_login else None,
                                mt5_password if mt5_password else None,
                                mt5_server if mt5_server else None)
            source_label = f"Exness MT5 ({mt5_symbol} / {mt5_timeframe})"
        elif data_source == "📊 yfinance (GC=F)":
            df = fetch_yfinance_data(yf_period, yf_interval)
            source_label = f"yfinance (GC=F / {yf_period} / {yf_interval})"
        else:
            df = generate_demo_data(demo_days)
            source_label = f"デモデータ ({demo_days}日分)"

    if df.empty:
        st.error("データが取得できませんでした。データソース設定を確認してください。")
        st.stop()

    # ─── 指標計算 ───
    calc_ma = calc_sma if ma_type == "SMA" else calc_ema
    df["MA_Short"] = calc_ma(df["Close"], short_period)
    df["MA_Long"] = calc_ma(df["Close"], long_period)
    df["RSI"] = calc_rsi(df["Close"], rsi_period)

    if show_bb:
        df["BB_Mid"], df["BB_Upper"], df["BB_Lower"] = calc_bollinger(df["Close"])
    if show_macd:
        df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calc_macd(df["Close"])

    # ─── シグナル検出 ───
    signals_df = detect_signals(df, df["MA_Short"], df["MA_Long"], df["RSI"], rsi_ob, rsi_os)

    # ─── メトリクスカード ───
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = latest["Close"] - prev["Close"]
    price_pct = (price_change / prev["Close"]) * 100

    current_rsi = latest["RSI"] if not pd.isna(latest["RSI"]) else 0
    trend = "上昇" if (not pd.isna(latest["MA_Short"]) and not pd.isna(latest["MA_Long"])
                        and latest["MA_Short"] > latest["MA_Long"]) else "下降"

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">現在価格</div>
            <div class="metric-value gold">${latest['Close']:.2f}</div>
            <div style="color:{'#00E676' if price_change>=0 else '#FF1744'}; font-size:13px; margin-top:4px;">
                {'▲' if price_change>=0 else '▼'} {abs(price_change):.2f} ({price_pct:+.2f}%)
            </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        rsi_color = "red" if current_rsi > rsi_ob else ("green" if current_rsi < rsi_os else "purple")
        rsi_label = "買われすぎ" if current_rsi > rsi_ob else ("売られすぎ" if current_rsi < rsi_os else "中立")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RSI ({rsi_period})</div>
            <div class="metric-value {rsi_color}">{current_rsi:.1f}</div>
            <div style="color:#6b7094; font-size:12px; margin-top:4px;">{rsi_label}</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        trend_color = "green" if trend == "上昇" else "red"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">トレンド</div>
            <div class="metric-value {trend_color}">{trend}</div>
            <div style="color:#6b7094; font-size:12px; margin-top:4px;">{ma_type} {short_period}/{long_period}</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">シグナル数</div>
            <div class="metric-value cyan">{len(signals_df)}</div>
            <div style="color:#6b7094; font-size:12px; margin-top:4px;">
                買:{len(signals_df[signals_df['タイプ'].str.contains('BUY')])} /
                売:{len(signals_df[signals_df['タイプ'].str.contains('SELL')])}
            </div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">データソース</div>
            <div style="font-size:14px; font-weight:700; color:#D4A017; margin-top:12px;">
                {source_label.split('(')[0].strip()}
            </div>
            <div style="color:#6b7094; font-size:11px; margin-top:4px;">{len(df)}本</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── メインチャート ───
    num_subplots = 2 + int(show_macd) + int(show_volume)
    row_heights = [0.5, 0.15]
    subplot_titles = ["XAU/USD 価格チャート", "RSI"]
    if show_macd:
        row_heights.append(0.15)
        subplot_titles.append("MACD")
    if show_volume:
        row_heights.append(0.1)
        subplot_titles.append("出来高")

    # 比率正規化
    total = sum(row_heights)
    row_heights = [h / total for h in row_heights]

    fig = make_subplots(
        rows=num_subplots, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="XAU/USD",
        increasing_line_color="#00E676", decreasing_line_color="#FF1744",
        increasing_fillcolor="#00E676", decreasing_fillcolor="#FF1744",
    ), row=1, col=1)

    # MA
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA_Short"], name=f"{ma_type} {short_period}",
        line=dict(color="#00D2FF", width=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA_Long"], name=f"{ma_type} {long_period}",
        line=dict(color="#FF6B8A", width=1.5),
    ), row=1, col=1)

    # ボリンジャーバンド
    if show_bb and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="#FFD54F", width=1, dash="dot"), opacity=0.5,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="#FFD54F", width=1, dash="dot"), opacity=0.5,
            fill="tonexty", fillcolor="rgba(255,213,79,0.05)",
        ), row=1, col=1)

    # シグナルマーカー
    if not signals_df.empty:
        buy_signals = signals_df[signals_df["タイプ"].str.contains("BUY")]
        sell_signals = signals_df[signals_df["タイプ"].str.contains("SELL")]

        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals["日付"], y=buy_signals["価格"],
                mode="markers", name="買いシグナル",
                marker=dict(
                    symbol="triangle-up", size=14, color="#00E676",
                    line=dict(width=1, color="#00E676"),
                ),
                text=buy_signals["理由"],
                hovertemplate="<b>買い</b><br>日付: %{x}<br>価格: $%{y:.2f}<br>%{text}<extra></extra>",
            ), row=1, col=1)

        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals["日付"], y=sell_signals["価格"],
                mode="markers", name="売りシグナル",
                marker=dict(
                    symbol="triangle-down", size=14, color="#FF1744",
                    line=dict(width=1, color="#FF1744"),
                ),
                text=sell_signals["理由"],
                hovertemplate="<b>売り</b><br>日付: %{x}<br>価格: $%{y:.2f}<br>%{text}<extra></extra>",
            ), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#AB47BC", width=1.5),
    ), row=2, col=1)

    fig.add_hline(y=rsi_ob, line_dash="dash", line_color="#FF1744",
                  line_width=1, opacity=0.5, row=2, col=1)
    fig.add_hline(y=rsi_os, line_dash="dash", line_color="#00E676",
                  line_width=1, opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#333",
                  line_width=1, row=2, col=1)

    # RSI ゾーン塗りつぶし
    fig.add_hrect(y0=rsi_ob, y1=100, fillcolor="rgba(255,23,68,0.05)",
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=rsi_os, fillcolor="rgba(0,230,118,0.05)",
                  line_width=0, row=2, col=1)

    current_row = 3

    # MACD
    if show_macd and "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#00D2FF", width=1.2),
        ), row=current_row, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color="#FF6B8A", width=1.2),
        ), row=current_row, col=1)
        colors = ["#00E676" if v >= 0 else "#FF1744" for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"], name="Histogram",
            marker_color=colors, opacity=0.5,
        ), row=current_row, col=1)
        current_row += 1

    # 出来高
    if show_volume:
        colors = ["#00E676" if df["Close"].iloc[i] >= df["Open"].iloc[i]
                  else "#FF1744" for i in range(len(df))]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=colors, opacity=0.4,
        ), row=current_row, col=1)

    # レイアウト
    fig.update_layout(
        height=700 + 120 * (num_subplots - 2),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,10,26,0.8)",
        font=dict(family="JetBrains Mono, Noto Sans JP, sans-serif", color="#e0e0e8"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0.3)"),
        xaxis_rangeslider_visible=False,
        margin=dict(l=60, r=20, t=60, b=20),
        hovermode="x unified",
    )

    fig.update_xaxes(gridcolor="rgba(26,26,46,0.8)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(26,26,46,0.8)", zeroline=False)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ─── シグナル一覧 ───
    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 📋 売買シグナル一覧")

        if signals_df.empty:
            st.info("現在のパラメータではシグナルが検出されませんでした。設定を調整してみてください。")
        else:
            for _, sig in signals_df.iloc[::-1].head(15).iterrows():
                is_buy = "BUY" in sig["タイプ"]
                is_strong = "STRONG" in sig["タイプ"]
                css_class = f"signal-{'buy' if is_buy else 'sell'} {'signal-strong' if is_strong else ''}"
                icon = "🟢" if is_buy else "🔴"
                strength = "◆ 強" if is_strong else ""
                label = "買い" if is_buy else "売り"

                st.markdown(f"""
                <div class="{css_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span style="font-size:16px;">{icon}</span>
                            <span style="font-weight:700; color:{'#00E676' if is_buy else '#FF1744'}; font-size:14px;">
                                {strength}{label}
                            </span>
                            <span style="color:#6b7094; font-size:12px; margin-left:12px;">
                                {sig['日付'].strftime('%Y-%m-%d %H:%M') if hasattr(sig['日付'], 'strftime') else sig['日付']}
                            </span>
                        </div>
                        <div>
                            <span style="font-family:'JetBrains Mono',monospace; font-weight:700; color:#D4A017; font-size:15px;">
                                ${sig['価格']:.2f}
                            </span>
                            <span style="color:#AB47BC; font-size:12px; margin-left:8px;">
                                RSI {sig['RSI']}
                            </span>
                        </div>
                    </div>
                    <div style="color:#8b8fa8; font-size:12px; margin-top:6px;">
                        {sig['理由']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 📊 シグナル統計")

        if not signals_df.empty:
            stats = {
                "買い": len(signals_df[signals_df["タイプ"] == "BUY"]),
                "強い買い": len(signals_df[signals_df["タイプ"] == "STRONG_BUY"]),
                "売り": len(signals_df[signals_df["タイプ"] == "SELL"]),
                "強い売り": len(signals_df[signals_df["タイプ"] == "STRONG_SELL"]),
            }

            stat_cols = st.columns(2)
            for i, (label, count) in enumerate(stats.items()):
                color = "#00E676" if "買" in label else "#FF1744"
                with stat_cols[i % 2]:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:12px;">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value" style="color:{color};">{count}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # パフォーマンスシミュレーション
            st.markdown("### 💰 簡易バックテスト")

            if len(signals_df) >= 2:
                trades = []
                position = None
                for _, sig in signals_df.iterrows():
                    if "BUY" in sig["タイプ"] and position is None:
                        position = sig["価格"]
                    elif "SELL" in sig["タイプ"] and position is not None:
                        pnl = sig["価格"] - position
                        trades.append(pnl)
                        position = None

                if trades:
                    total_pnl = sum(trades)
                    win_trades = [t for t in trades if t > 0]
                    win_rate = len(win_trades) / len(trades) * 100

                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:12px;">
                        <div class="metric-label">トレード回数</div>
                        <div class="metric-value cyan">{len(trades)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    pnl_color = "green" if total_pnl >= 0 else "red"
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:12px;">
                        <div class="metric-label">合計損益</div>
                        <div class="metric-value {pnl_color}">
                            {'+'if total_pnl>=0 else ''}${total_pnl:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    wr_color = "green" if win_rate >= 50 else "red"
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:12px;">
                        <div class="metric-label">勝率</div>
                        <div class="metric-value {wr_color}">{win_rate:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("完結したトレードがありません")
            else:
                st.info("バックテストに十分なシグナルがありません")

        else:
            st.info("シグナルが検出されるとここに統計が表示されます")

    # ─── データテーブル (折りたたみ) ───
    with st.expander("📄 生データを表示"):
        st.dataframe(
            df.tail(50).style.format({
                "Open": "${:.2f}", "High": "${:.2f}",
                "Low": "${:.2f}", "Close": "${:.2f}",
                "RSI": "{:.1f}", "MA_Short": "${:.2f}", "MA_Long": "${:.2f}",
            }),
            use_container_width=True,
        )

        if not signals_df.empty:
            st.markdown("#### シグナルデータ")
            st.dataframe(signals_df, use_container_width=True)

        # CSVダウンロード
        csv_data = df.to_csv()
        st.download_button(
            label="📥 価格データをCSVでダウンロード",
            data=csv_data,
            file_name=f"gold_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

        if not signals_df.empty:
            sig_csv = signals_df.to_csv(index=False)
            st.download_button(
                label="📥 シグナルデータをCSVでダウンロード",
                data=sig_csv,
                file_name=f"gold_signals_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

    # ─── 判定ロジック説明 ───
    with st.expander("📖 判定ロジック詳細"):
        st.markdown("""
        #### 🟢 買いシグナル
        | 条件 | 説明 |
        |------|------|
        | **ゴールデンクロス** | 短期MAが長期MAを下から上に抜ける |
        | **RSI売られすぎ突入** | RSIが設定した売られすぎラインを下回る |
        | **◆ 強い買い** | ゴールデンクロス + RSI売られすぎが同時発生 |

        #### 🔴 売りシグナル
        | 条件 | 説明 |
        |------|------|
        | **デッドクロス** | 短期MAが長期MAを上から下に抜ける |
        | **RSI買われすぎ突入** | RSIが設定した買われすぎラインを上回る |
        | **◆ 強い売り** | デッドクロス + RSI買われすぎが同時発生 |

        #### パラメータの目安
        - **短期MA 5-10**: スキャルピング〜デイトレード向け
        - **短期MA 10-20**: スイングトレード向け
        - **RSI 14**: 一般的な設定値
        - **RSI 70/30**: 標準的な買われすぎ/売られすぎ閾値
        """)

    # ─── MT5セットアップガイド ───
    with st.expander("🔧 Exness MT5 セットアップガイド"):
        st.markdown("""
        #### 手順
        1. **Exness アカウント作成**: [exness.com](https://www.exness.com) でアカウント登録
        2. **MT5 ダウンロード**: Exness の Personal Area から MT5 をダウンロード
        3. **Python パッケージ インストール**:
           ```bash
           pip install MetaTrader5
           ```
        4. **MT5 ターミナル起動**: Windows上でMT5を起動しログイン
        5. **アルゴ取引を有効化**: MT5 → ツール → オプション → エキスパートアドバイザ → アルゴリズム取引を許可
        6. **本アプリを起動**:
           ```bash
           streamlit run gold_trading_analyzer.py
           ```
        7. **サイドバーでMT5を選択** して接続情報を入力

        #### サーバー名の例
        - `Exness-MT5Real` (リアル口座)
        - `Exness-MT5Real2` ~ `Exness-MT5Real15`
        - `Exness-MT5Trial` (デモ口座)

        #### シンボル名
        - ゴールド: `XAUUSD` (一般), `XAUUSDm` (マイクロ口座)
        - シルバー: `XAGUSD`

        ⚠️ **注意**: MetaTrader5 パッケージは **Windows のみ** 対応です。
        macOS / Linux では yfinance またはデモデータをご利用ください。
        """)

    # ─── 免責事項 ───
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>免責事項</b>: 本ツールはテクニカル分析の学習・参考用です。
        実際の投資判断は必ずご自身の責任で行ってください。
        過去のデータに基づくシグナルは将来の利益を保証するものではありません。
        CFD取引にはリスクが伴い、投資元本を失う可能性があります。
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
