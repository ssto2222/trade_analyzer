"""
app.py  —  Trade Signal Dashboard v2.0  (H1 + D1 RSI)
"""
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    ZONE_BINS, ZONE_LABELS,
    _H1_ZONE_DATA, _D1_ZONE_DATA, _CROSS_VERDICT,
    calc_rsi, evaluate_signal,
    get_h1_zone, get_d1_zone, get_cross_verdict,
    HOUR_FORBIDDEN_UTC, HOUR_CAUTION_UTC,
    DOW_FORBIDDEN, DOW_CAUTION,
    _cross_bucket,
)

# ─── ページ設定 ──────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Signal Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #c9d1d9; }
  section[data-testid="stSidebar"] { background-color: #161b22; }

  .signal-card {
    border-radius: 12px; padding: 18px 20px;
    text-align: center; font-weight: 700;
    font-size: 2rem; letter-spacing: 1px; margin-bottom: 6px;
  }
  .sig-buy    { background:#0d2818; color:#3fb950; border:2px solid #3fb950; }
  .sig-sell   { background:#2d0b0b; color:#f85149; border:2px solid #f85149; }
  .sig-wait   { background:#1c1a0e; color:#e3b341; border:2px solid #e3b341; }
  .sig-strong { background:#0a3320; color:#3fb950; border:2px solid #3fb950; }

  .score-bar-wrap { background:#21262d; border-radius:8px; height:14px; overflow:hidden; margin:6px 0 2px; }
  .score-bar      { height:14px; border-radius:8px; transition:width .3s; }

  .cross-cell { border-radius:5px; padding:4px 6px; text-align:center;
                font-size:11px; font-weight:600; }
  .cross-best     { background:#0a3320; color:#58d68d; }
  .cross-good     { background:#0d2818; color:#3fb950; }
  .cross-ok       { background:#1c2a1c; color:#8fbe8f; }
  .cross-caution  { background:#3d2800; color:#e3b341; }
  .cross-forbidden{ background:#3d1212; color:#f85149; }
  .cross-current  { outline:2px solid #58a6ff; }

  [data-testid="stMetricValue"] { color:#58a6ff; }
  [data-testid="stMetricLabel"] { color:#8b949e; }
</style>
""", unsafe_allow_html=True)


# ─── 定数 ────────────────────────────────────────────────────
SUPPORTED_SYMBOLS = ["BTCUSD", "XAUUSD"]
DOW_NAMES = ["月","火","水","木","金","土","日"]
VERDICT_COLOR = {
    "forbidden":"#f85149","caution":"#e3b341","ok":"#3fb950","good":"#58d68d","best":"#a371f7"
}


# ─── RSIゲージ ───────────────────────────────────────────────

def rsi_gauge(rsi_val: float, symbol: str, tf: str, zone_data: dict) -> go.Figure:
    color_map = {"forbidden":"#f85149","caution":"#e3b341","ok":"#3fb950","good":"#58d68d"}
    steps = []
    for label, lo, hi in zip(ZONE_LABELS, ZONE_BINS[:-1], ZONE_BINS[1:]):
        v = zone_data.get(symbol, zone_data.get("BTCUSD",{})).get(label, {}).get("verdict","ok")
        steps.append({"range":[lo,hi],"color":color_map.get(v,"#30363d")})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_val,
        number={"font":{"size":32,"color":"#c9d1d9"}},
        gauge={
            "axis":{"range":[0,100],"tickvals":[0,20,30,40,50,60,70,80,100],
                    "tickfont":{"color":"#8b949e","size":9},"tickcolor":"#8b949e"},
            "bar":{"color":"#58a6ff","thickness":0.22},
            "bgcolor":"#161b22","bordercolor":"#21262d",
            "steps":steps,
            "threshold":{"line":{"color":"#fff","width":3},"thickness":0.8,"value":rsi_val},
        },
        title={"text":f"RSI(14) {tf}","font":{"color":"#8b949e","size":12}},
        domain={"x":[0,1],"y":[0,1]},
    ))
    fig.update_layout(height=230, margin=dict(t=36,b=8,l=16,r=16),
                      paper_bgcolor="#0d1117")
    return fig


# ─── 時間帯バー ──────────────────────────────────────────────

def hour_bar(hour_utc_current: int) -> go.Figure:
    avg_profits = {
        0:-1796,1:-5990,2:-132,3:2091,4:1184,5:470,
        6:-1919,7:-5937,8:-89,9:1395,10:-2051,11:1154,
        12:-5503,13:-27,14:2318,15:-1298,16:595,17:3956,
        18:3427,19:6909,20:5310,21:-12470,22:5447,23:3124,
    }
    h_jst  = [(h+9)%24 for h in range(24)]
    profits = [avg_profits[h] for h in range(24)]
    colors  = []
    for h in range(24):
        if h == hour_utc_current:
            colors.append("#58a6ff")
        elif h in HOUR_FORBIDDEN_UTC:
            colors.append("#f85149")
        elif h in HOUR_CAUTION_UTC:
            colors.append("#e3b341")
        elif avg_profits[h] > 0:
            colors.append("#3fb950")
        else:
            colors.append("#484f58")

    fig = go.Figure(go.Bar(
        x=[f"{j}時" for j in h_jst],
        y=profits,
        marker_color=colors,
        hovertemplate="JST %{x}<br>平均損益: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#30363d", line_width=1)
    fig.update_layout(
        title={"text":"時間帯別 平均損益（JST） — 青が現在時刻","font":{"color":"#8b949e","size":12}},
        height=210, margin=dict(t=36,b=50,l=50,r=10),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis={"tickangle":-50,"tickfont":{"size":8,"color":"#8b949e"},"gridcolor":"#21262d"},
        yaxis={"tickfont":{"size":9,"color":"#8b949e"},"gridcolor":"#21262d","tickprefix":"$"},
        showlegend=False,
    )
    return fig


# ─── クロスヒートマップ ──────────────────────────────────────

def cross_heatmap(symbol: str, rsi_h1: float, rsi_d1: float) -> go.Figure:
    buckets = ["<40","40-50","50-60","60-70",">70"]
    data    = _CROSS_VERDICT.get(symbol, _CROSS_VERDICT["BTCUSD"])

    z, text = [], []
    val_map = {"best":4,"good":3,"ok":2,"caution":1,"forbidden":0}
    for d1b in buckets:
        row_z, row_t = [], []
        for h1b in buckets:
            entry = data.get((d1b,h1b))
            if entry:
                v, ap, wr = entry
                row_z.append(val_map[v])
                row_t.append(f"{v}<br>${ap:,}<br>{wr}%")
            else:
                row_z.append(2)
                row_t.append("—")
        z.append(row_z)
        text.append(row_t)

    colorscale = [
        [0.0,  "#3d1212"],
        [0.25, "#3d2800"],
        [0.5,  "#1c2a1c"],
        [0.75, "#0d2818"],
        [1.0,  "#0a3320"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=buckets, y=buckets,
        text=text, texttemplate="%{text}",
        colorscale=colorscale,
        showscale=False,
        hovertemplate="D1: %{y}<br>H1: %{x}<br>%{text}<extra></extra>",
        textfont={"size":9,"color":"#c9d1d9"},
        zmin=0, zmax=4,
    ))

    # 現在地マーカー
    d1b = _cross_bucket(rsi_d1)
    h1b = _cross_bucket(rsi_h1)
    if d1b in buckets and h1b in buckets:
        fig.add_shape(
            type="rect",
            x0=buckets.index(h1b)-0.5, x1=buckets.index(h1b)+0.5,
            y0=buckets.index(d1b)-0.5, y1=buckets.index(d1b)+0.5,
            line=dict(color="#58a6ff", width=3),
        )

    fig.update_layout(
        title={"text":f"{symbol} H1×D1 RSI クロスマップ（青枠=現在地）",
               "font":{"color":"#8b949e","size":12}},
        height=300, margin=dict(t=40,b=10,l=60,r=10),
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        xaxis={"title":"H1 RSIゾーン","tickfont":{"size":10,"color":"#8b949e"},
               "gridcolor":"#21262d"},
        yaxis={"title":"D1 RSIゾーン","tickfont":{"size":10,"color":"#8b949e"},
               "gridcolor":"#21262d"},
    )
    return fig


# ─── D1ゾーンテーブル ────────────────────────────────────────

def d1_zone_table(symbol: str) -> go.Figure:
    zones = _D1_ZONE_DATA.get(symbol, _D1_ZONE_DATA["BTCUSD"])
    color_map = {"forbidden":"#f85149","caution":"#e3b341","ok":"#3fb950","good":"#58d68d"}
    vd_label  = {"forbidden":"🔴 禁止","caution":"🟡 注意","ok":"🟢 可","good":"⭐ 優良"}

    labels = ZONE_LABELS
    vds    = [zones.get(l,{}).get("verdict","ok") for l in labels]
    wrs    = [zones.get(l,{}).get("wr") for l in labels]
    aps    = [zones.get(l,{}).get("avg_profit") for l in labels]

    fmt_wr = [f"{w}%" if w is not None else "—" for w in wrs]
    fmt_ap = [(f"+${a:,}" if a >= 0 else f"-${abs(a):,}") if a is not None else "—" for a in aps]
    ap_colors = [("#3fb950" if (a or 0)>=0 else "#f85149") if a is not None else "#8b949e" for a in aps]

    fig = go.Figure(go.Table(
        header=dict(values=["RSI帯域","判定","勝率","平均損益"],
                    fill_color="#161b22",font=dict(color="#8b949e",size=11),
                    align="center",height=28),
        cells=dict(
            values=[labels,[vd_label[v] for v in vds],fmt_wr,fmt_ap],
            fill_color=[["#21262d"]*len(labels)]*4,
            font=dict(color=[["#c9d1d9"]*len(labels),
                              [color_map[v] for v in vds],
                              ["#c9d1d9"]*len(labels),
                              ap_colors],size=11),
            align=["center","center","center","right"],height=24,
        ),
    ))
    fig.update_layout(height=400,margin=dict(t=4,b=4,l=0,r=0),
                      paper_bgcolor="#0d1117")
    return fig


# ─── スコアバー ──────────────────────────────────────────────

def score_html(score: int) -> str:
    color = "#f85149" if score < 35 else "#e3b341" if score < 60 else "#3fb950"
    label = "エントリー不適" if score < 35 else "要注意" if score < 60 else "エントリー適性あり"
    return f"""
<div style="margin:8px 0">
  <div style="display:flex;justify-content:space-between;font-size:12px;color:#8b949e;margin-bottom:4px">
    <span>エントリー適性スコア</span><span style="color:{color};font-weight:700">{score}/100 — {label}</span>
  </div>
  <div class="score-bar-wrap">
    <div class="score-bar" style="width:{score}%;background:{color}"></div>
  </div>
</div>"""


# ─── メイン ─────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.markdown("## ⚙️ 入力パラメータ")
        symbol    = st.selectbox("銘柄", SUPPORTED_SYMBOLS)
        direction = st.radio("方向", ["buy","sell"], horizontal=True)
        st.markdown("---")

        st.markdown("**H1 RSI(14) — 1時間足**")
        rsi_h1 = st.slider("H1 RSI", 0.0, 100.0, 65.0, 0.5, key="h1")

        st.markdown("**D1 RSI(14) — 日足**")
        rsi_d1 = st.slider("D1 RSI", 0.0, 100.0, 72.0, 0.5, key="d1")

        st.markdown("---")
        st.markdown("**日時 (JST)**")
        now_jst  = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
        date_in  = st.date_input("日付", value=now_jst.date())
        time_in  = st.time_input("時刻", value=now_jst.time().replace(second=0,microsecond=0))
        dt_jst   = datetime.datetime.combine(date_in, time_in)
        dt_utc   = dt_jst - datetime.timedelta(hours=9)
        hour_utc = dt_utc.hour
        dow      = dt_jst.weekday()
        st.caption(f"UTC: {dt_utc.strftime('%Y-%m-%d %H:%M')} | {DOW_NAMES[dow]}曜日")

        st.markdown("---")
        st.markdown("### 凡例")
        st.markdown("⭐ **最良** — 積極的に狙う  \n"
                    "🟢 **可/優良** — エントリー可  \n"
                    "🟡 **注意** — 慎重に  \n"
                    "🔴 **禁止** — エントリー不可")
        st.caption("データ: 12,080件 (2020-07〜2026-04)")

    # ── 評価
    result = evaluate_signal(symbol, rsi_h1, rsi_d1, direction, hour_utc, dow)
    sig    = result["signal"]
    stre   = result["strength"]
    h1z    = result["h1_zone"]
    d1z    = result["d1_zone"]
    cross  = result["cross"]
    score  = result["score"]

    st.markdown("# 📊 Trade Signal Dashboard  <small style='font-size:14px;color:#8b949e'>v2.0 — H1 + D1 RSI</small>", unsafe_allow_html=True)
    st.markdown(
        f"**{symbol}** | `{direction.upper()}` | "
        f"H1 RSI `{rsi_h1:.1f}` | D1 RSI `{rsi_d1:.1f}` | "
        f"{DOW_NAMES[dow]}曜日 JST {dt_jst.strftime('%H:%M')}"
    )
    st.divider()

    # ── 上段: シグナル / H1ゲージ / D1ゲージ / 時間帯
    c_sig, c_h1, c_d1, c_hour = st.columns([1, 1.1, 1.1, 1.5])

    with c_sig:
        st.markdown("### シグナル")
        if sig == "WAIT":
            cls = "sig-wait"
            txt = "⛔ WAIT"
        elif stre == "strong":
            cls = "sig-strong"
            txt = f"⚡ {sig}"
        elif sig == "BUY":
            cls = "sig-buy"
            txt = f"▲ {sig}"
        else:
            cls = "sig-sell"
            txt = f"▼ {sig}"
        st.markdown(f'<div class="signal-card {cls}">{txt}</div>', unsafe_allow_html=True)
        st.markdown(score_html(score), unsafe_allow_html=True)

        st.markdown("**判定理由**")
        for r in result["reasons"]:
            st.markdown(f"- {r}")

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("H1 勝率", f"{h1z['wr']}%")
        c2.metric("H1 平均損益", f"${h1z['avg_profit']:,}")
        c1.metric("D1 勝率", f"{d1z.get('wr','—')}%" if d1z.get('wr') is not None else "—")
        c2.metric("D1 平均損益", f"${d1z['avg_profit']:,}" if d1z.get('avg_profit') is not None else "—")

    with c_h1:
        st.plotly_chart(rsi_gauge(rsi_h1, symbol, "1H", _H1_ZONE_DATA),
                        use_container_width=True, config={"displayModeBar":False})
        h1v = h1z["verdict"]
        badge_map = {"forbidden":"🔴 禁止ゾーン","caution":"🟡 注意ゾーン",
                     "ok":"🟢 エントリー可","good":"⭐ 優良ゾーン"}
        color_map2 = {"forbidden":"error","caution":"warning","ok":"success","good":"success"}
        getattr(st, color_map2.get(h1v,"info"))(f"H1: {badge_map.get(h1v,'')}")

    with c_d1:
        st.plotly_chart(rsi_gauge(rsi_d1, symbol, "D1", _D1_ZONE_DATA),
                        use_container_width=True, config={"displayModeBar":False})
        d1v = d1z["verdict"]
        getattr(st, color_map2.get(d1v,"info"))(f"D1: {badge_map.get(d1v,'')}")

    with c_hour:
        st.plotly_chart(hour_bar(hour_utc), use_container_width=True,
                        config={"displayModeBar":False})
        h_jst = dt_jst.hour
        if hour_utc in HOUR_FORBIDDEN_UTC:
            st.error(f"⚠️ JST {h_jst}時 — 禁止時間帯")
        elif hour_utc in HOUR_CAUTION_UTC:
            st.warning(f"⚠️ JST {h_jst}時 — 注意時間帯")
        else:
            st.success(f"✅ JST {h_jst}時 — 時間帯OK")

    st.divider()

    # ── 下段: クロスマップ / D1テーブル / 曜日
    c_cross, c_d1tbl, c_dow = st.columns([1.6, 1.2, 0.8])

    with c_cross:
        st.plotly_chart(cross_heatmap(symbol, rsi_h1, rsi_d1),
                        use_container_width=True, config={"displayModeBar":False})
        cv = cross["verdict"]
        cv_msg = {
            "best":      ("success", f"⭐ 最良の組み合わせ — 平均損益 ${cross['avg_profit']:,}"),
            "good":      ("success", f"✅ 良好な組み合わせ — 平均損益 ${cross['avg_profit']:,}"),
            "ok":        ("info",    f"🟢 許容範囲 — 平均損益 ${cross['avg_profit']:,}"),
            "caution":   ("warning", f"🟡 組み合わせ注意 — 平均損益 ${cross['avg_profit']:,}"),
            "forbidden": ("error",   f"🔴 組み合わせNG — 平均損益 ${cross['avg_profit']:,}"),
        }.get(cv, ("info","—"))
        getattr(st, cv_msg[0])(cv_msg[1])

    with c_d1tbl:
        st.markdown(f"**{symbol} 日足 RSIゾーン一覧**")
        st.plotly_chart(d1_zone_table(symbol), use_container_width=True,
                        config={"displayModeBar":False})

    with c_dow:
        st.markdown("**曜日別判定**")
        dow_info = [
            (0,"月","ok",   "+$6.9M"),
            (1,"火","ok",   "+$0.1M"),
            (2,"水","ok",   "+$1.3M"),
            (3,"木","caution","-$4.2M"),
            (4,"金","caution","-$6.4M"),
            (5,"土","forbidden","-$3.9M"),
            (6,"日","ok",   "+$3.6M"),
        ]
        cls_map = {"forbidden":"zone-forbidden","caution":"zone-caution","ok":"zone-ok"}
        for d,name,vd,pnl in dow_info:
            active = " ◀ 今日" if d == dow else ""
            st.markdown(
                f'<span style="display:inline-block;padding:3px 8px;border-radius:5px;font-size:12px;font-weight:600;'
                f'background:{"#3d1212" if vd=="forbidden" else "#3d2800" if vd=="caution" else "#0d2818"};'
                f'color:{"#f85149" if vd=="forbidden" else "#e3b341" if vd=="caution" else "#3fb950"}">'
                f'{name}曜 {pnl}</span>{active}',
                unsafe_allow_html=True,
            )
            st.markdown("")

        st.markdown("---")
        st.markdown("**⛔ 禁止ルール (最重要)**")
        st.error("H1 RSI 40〜55 (BTC/XAU)")
        st.error("H1 RSI 65〜70 (BTC)")
        st.error("D1 RSI 35〜50 (XAU)")
        st.error("BTC: D1>70 × H1 40〜60")
        st.warning("土曜 / JST 6・16・18時")


if __name__ == "__main__":
    main()
