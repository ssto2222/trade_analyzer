"""
app.py  —  Trade Signal Dashboard (Streamlit)
"""
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    ZONE_BINS, ZONE_LABELS, _ZONE_DATA,
    calc_rsi, evaluate_signal, get_zone_info,
    HOUR_FORBIDDEN_UTC, HOUR_CAUTION_UTC,
    DOW_FORBIDDEN, DOW_CAUTION,
)

# ─── ページ設定 ──────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Signal Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── カスタムCSS ─────────────────────────────────────────────
st.markdown("""
<style>
  /* dark base */
  .stApp { background-color: #0d1117; color: #c9d1d9; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  /* signal cards */
  .signal-card {
    border-radius: 12px; padding: 20px 24px;
    text-align: center; font-weight: 700;
    font-size: 2.2rem; letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .signal-buy    { background:#0d2818; color:#3fb950; border:2px solid #3fb950; }
  .signal-sell   { background:#2d0b0b; color:#f85149; border:2px solid #f85149; }
  .signal-wait   { background:#1c1a0e; color:#e3b341; border:2px solid #e3b341; }
  .signal-strong { background:#0a3320; color:#3fb950; border:2px solid #3fb950; }

  /* zone badge */
  .zone-forbidden { background:#3d1212; color:#f85149;
    border-radius:6px; padding:3px 10px; font-size:0.85rem; }
  .zone-caution   { background:#3d2800; color:#e3b341;
    border-radius:6px; padding:3px 10px; font-size:0.85rem; }
  .zone-ok        { background:#0d2818; color:#3fb950;
    border-radius:6px; padding:3px 10px; font-size:0.85rem; }
  .zone-good      { background:#0a3320; color:#58d68d;
    border-radius:6px; padding:3px 10px; font-size:0.85rem; }

  /* metric override */
  [data-testid="stMetricValue"] { color: #58a6ff; }
  [data-testid="stMetricLabel"] { color: #8b949e; }
</style>
""", unsafe_allow_html=True)


# ─── ヘルパー ────────────────────────────────────────────────

SUPPORTED_SYMBOLS = ["BTCUSD", "XAUUSD"]
DOW_NAMES = ["月", "火", "水", "木", "金", "土", "日"]

VERDICT_LABEL = {
    "forbidden": "🔴 禁止",
    "caution":   "🟡 注意",
    "ok":        "🟢 可",
    "good":      "⭐ 優良",
}

def verdict_color(v):
    return {"forbidden": "#f85149", "caution": "#e3b341",
            "ok": "#3fb950", "good": "#58d68d"}.get(v, "#8b949e")


# ─── RSIゲージ ───────────────────────────────────────────────

def rsi_gauge(rsi_val: float, symbol: str) -> go.Figure:
    zones = _ZONE_DATA.get(symbol, _ZONE_DATA["BTCUSD"])
    color_map = {
        "forbidden": "#f85149", "caution": "#e3b341",
        "ok": "#3fb950", "good": "#58d68d",
    }

    steps = []
    for label, lo, hi in zip(ZONE_LABELS, ZONE_BINS[:-1], ZONE_BINS[1:]):
        v = zones.get(label, {}).get("verdict", "ok")
        steps.append({"range": [lo, hi], "color": color_map[v]})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_val,
        number={"font": {"size": 36, "color": "#c9d1d9"}, "suffix": ""},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#8b949e",
                "tickvals": [0, 20, 30, 40, 50, 60, 70, 80, 100],
                "ticktext": ["0", "20", "30", "40", "50", "60", "70", "80", "100"],
                "tickfont": {"color": "#8b949e", "size": 10},
            },
            "bar": {"color": "#58a6ff", "thickness": 0.25},
            "bgcolor": "#161b22",
            "bordercolor": "#21262d",
            "steps": steps,
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.8,
                "value": rsi_val,
            },
        },
        title={"text": f"RSI(14) — {symbol}", "font": {"color": "#8b949e", "size": 13}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=260, margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="#0d1117", font_color="#c9d1d9",
    )
    return fig


# ─── 時間帯ヒートマップ ──────────────────────────────────────

def hour_heatmap() -> go.Figure:
    avg_profits_utc = {
        0: -1796, 1: -5990, 2: -132,  3:  2091, 4:  1184, 5:   470,
        6: -1919, 7: -5937, 8:  -89,  9:  1395,10: -2051,11:  1154,
       12: -5503,13:   -27,14:  2318,15: -1298,16:   595,17:  3956,
       18:  3427,19:  6909,20:  5310,21:-12470,22:  5447,23:  3124,
    }
    hours_utc = list(range(24))
    profits   = [avg_profits_utc[h] for h in hours_utc]
    jst_labels = [f"{(h+9)%24}時 JST" for h in hours_utc]

    colors = []
    for h in hours_utc:
        if h in HOUR_FORBIDDEN_UTC:
            colors.append("#f85149")
        elif h in HOUR_CAUTION_UTC:
            colors.append("#e3b341")
        elif avg_profits_utc[h] > 0:
            colors.append("#3fb950")
        else:
            colors.append("#8b949e")

    fig = go.Figure(go.Bar(
        x=jst_labels,
        y=profits,
        marker_color=colors,
        hovertemplate="<b>%{x}</b><br>平均損益: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    fig.update_layout(
        title={"text": "時間帯別 平均損益（JST換算）", "font": {"color": "#8b949e", "size": 13}},
        height=240,
        margin=dict(t=40, b=60, l=50, r=10),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis={"tickangle": -45, "tickfont": {"size": 9, "color": "#8b949e"},
               "gridcolor": "#21262d"},
        yaxis={"tickfont": {"size": 9, "color": "#8b949e"}, "gridcolor": "#21262d",
               "tickprefix": "$"},
        showlegend=False,
    )
    return fig


# ─── ゾーンテーブル ──────────────────────────────────────────

def zone_table(symbol: str) -> go.Figure:
    zones = _ZONE_DATA.get(symbol, _ZONE_DATA["BTCUSD"])
    labels = ZONE_LABELS
    wrs    = [zones[l]["wr"] for l in labels]
    aps    = [zones[l]["avg_profit"] for l in labels]
    vds    = [zones[l]["verdict"] for l in labels]

    fill_colors = [[
        {"forbidden": "#3d1212", "caution": "#3d2800",
         "ok": "#0d2818", "good": "#0a3320"}.get(v, "#161b22")
        for v in vds
    ]] * 5

    font_colors_vd = [verdict_color(v) for v in vds]
    font_colors_ap = ["#3fb950" if a >= 0 else "#f85149" for a in aps]
    fmt_ap = [f"+${a:,}" if a >= 0 else f"-${abs(a):,}" for a in aps]
    fmt_vd = [VERDICT_LABEL[v] for v in vds]

    fig = go.Figure(go.Table(
        header=dict(
            values=["RSI帯域", "判定", "勝率", "平均損益"],
            fill_color="#161b22",
            font=dict(color="#8b949e", size=12),
            align="center",
            height=30,
        ),
        cells=dict(
            values=[labels, fmt_vd, [f"{w}%" for w in wrs], fmt_ap],
            fill_color=[["#21262d"] * len(labels)] * 4,
            font=dict(
                color=[
                    ["#c9d1d9"] * len(labels),
                    font_colors_vd,
                    ["#c9d1d9"] * len(labels),
                    font_colors_ap,
                ],
                size=12,
            ),
            align=["center", "center", "center", "right"],
            height=26,
        ),
    ))
    fig.update_layout(
        height=420, margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="#0d1117",
    )
    return fig


# ─── メイン UI ──────────────────────────────────────────────

def main():
    # ── サイドバー
    with st.sidebar:
        st.markdown("## ⚙️ 入力パラメータ")

        symbol = st.selectbox("銘柄", SUPPORTED_SYMBOLS, index=0)
        direction = st.radio("方向", ["buy", "sell"], horizontal=True)

        st.markdown("---")
        st.markdown("**RSI値**")
        rsi_input = st.slider(
            "現在のRSI(14) — 1H足",
            min_value=0.0, max_value=100.0, value=65.0, step=0.5,
        )

        st.markdown("---")
        st.markdown("**日時 (JST)**")
        now_jst = datetime.datetime.now() + datetime.timedelta(hours=9)
        date_in = st.date_input("日付", value=now_jst.date())
        time_in = st.time_input("時刻", value=now_jst.time().replace(second=0, microsecond=0))

        dt_jst   = datetime.datetime.combine(date_in, time_in)
        dt_utc   = dt_jst - datetime.timedelta(hours=9)
        hour_utc = dt_utc.hour
        dow      = dt_jst.weekday()

        st.caption(f"UTC: {dt_utc.strftime('%Y-%m-%d %H:%M')} | {DOW_NAMES[dow]}曜日")

        st.markdown("---")
        st.markdown("### 🗂 凡例")
        st.markdown("🔴 **禁止** — 絶対エントリー不可  \n"
                    "🟡 **注意** — 慎重に判断  \n"
                    "🟢 **可** — エントリー可  \n"
                    "⭐ **優良** — 積極的に狙う")
        st.markdown("---")
        st.caption("データ: 12,080件 (2020-07〜2026-04)")

    # ── シグナル評価
    result = evaluate_signal(symbol, rsi_input, direction, hour_utc, dow)
    sig    = result["signal"]
    stre   = result["strength"]
    verd   = result["verdict"]
    zone   = result["zone_info"]

    # ── ヘッダー
    st.markdown("# 📊 Trade Signal Dashboard")
    st.markdown(f"**{symbol}** | 方向: `{direction.upper()}` | "
                f"RSI: `{rsi_input:.1f}` | {DOW_NAMES[dow]}曜日 "
                f"JST {dt_jst.strftime('%H:%M')}")

    st.divider()

    # ── 上段: シグナル + ゲージ + 時間帯
    col_sig, col_gauge, col_hour = st.columns([1, 1.4, 1.6])

    with col_sig:
        st.markdown("### シグナル")
        if sig == "WAIT":
            st.markdown(
                f'<div class="signal-card signal-wait">⛔ WAIT</div>',
                unsafe_allow_html=True,
            )
            label = "🔴 禁止" if verd == "forbidden" else "🟡 注意"
            st.markdown(f"<center>{label}</center>", unsafe_allow_html=True)
        elif stre == "strong":
            st.markdown(
                f'<div class="signal-card signal-strong">⚡ {sig}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<center>⭐ 強シグナル</center>", unsafe_allow_html=True)
        elif sig == "BUY":
            st.markdown(
                f'<div class="signal-card signal-buy">▲ {sig}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="signal-card signal-sell">▼ {sig}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**判定理由**")
        for r in result["reasons"]:
            st.markdown(f"- {r}")

        # ゾーンメトリクス
        st.markdown("---")
        st.markdown("**ゾーン統計**")
        mc1, mc2 = st.columns(2)
        mc1.metric("勝率", f"{zone['wr']}%")
        mc2.metric("平均損益", f"${zone['avg_profit']:,}")
        mc1.metric("Buy勝率", f"{zone.get('buy_wr', '—')}%" if zone.get('buy_wr') is not None else "—")
        mc2.metric("Sell勝率", f"{zone.get('sell_wr', '—')}%" if zone.get('sell_wr') is not None else "—")

    with col_gauge:
        st.plotly_chart(rsi_gauge(rsi_input, symbol), use_container_width=True, config={"displayModeBar": False})

        # RSIゾーン凡例バー
        st.markdown("**RSIゾーン凡例**")
        leg_cols = st.columns(4)
        for col, (lbl, cls) in zip(leg_cols, [
            ("禁止", "zone-forbidden"), ("注意", "zone-caution"),
            ("可",   "zone-ok"),        ("優良", "zone-good"),
        ]):
            col.markdown(f'<span class="{cls}">{lbl}</span>', unsafe_allow_html=True)

    with col_hour:
        st.plotly_chart(hour_heatmap(), use_container_width=True, config={"displayModeBar": False})

        # 現在時刻のハイライト
        h_jst = dt_jst.hour
        if hour_utc in HOUR_FORBIDDEN_UTC:
            st.error(f"⚠️ 現在 JST {h_jst}時 は禁止時間帯です")
        elif hour_utc in HOUR_CAUTION_UTC:
            st.warning(f"⚠️ 現在 JST {h_jst}時 は注意時間帯です")
        else:
            st.success(f"✅ 現在 JST {h_jst}時 は時間帯OK")

    st.divider()

    # ── 下段: ゾーンテーブル + 曜日カード
    col_tbl, col_dow = st.columns([2, 1])

    with col_tbl:
        st.markdown(f"### {symbol} RSIゾーン一覧")
        st.plotly_chart(zone_table(symbol), use_container_width=True, config={"displayModeBar": False})

    with col_dow:
        st.markdown("### 曜日別判定")
        dow_data = {
            0: ("月", "ok",        "+$6.86M"),
            1: ("火", "ok",        "+$0.10M"),
            2: ("水", "ok",        "+$1.34M"),
            3: ("木", "caution",   "-$4.16M"),
            4: ("金", "caution",   "-$6.38M"),
            5: ("土", "forbidden", "-$3.85M"),
            6: ("日", "ok",        "+$3.65M"),
        }
        for d, (name, verdict, pnl) in dow_data.items():
            cls = {
                "forbidden": "zone-forbidden",
                "caution":   "zone-caution",
                "ok":        "zone-ok",
            }[verdict]
            active = "→ **今日**" if d == dow else ""
            st.markdown(
                f'<span class="{cls}">{name}曜日 {pnl}</span> {active}',
                unsafe_allow_html=True,
            )
            st.markdown("")

        st.markdown("---")
        st.markdown("### ⛔ 最重要禁止ルール")
        st.error("🔴 RSI 40〜55 (BTC / XAU 共通)")
        st.error("🔴 RSI 65〜70 (BTC)")
        st.error("🔴 RSI >80 + Sell (XAU)")
        st.warning("🟡 JST 6時・10時・16時・21時")
        st.warning("🟡 土曜エントリー")


if __name__ == "__main__":
    main()
