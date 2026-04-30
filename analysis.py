"""
analysis.py  —  RSI(14) 計算・シグナル判定ロジック
データ由来:
  - BTCUSD / XAUUSD の H1・D1 RSI14帯域別勝率・平均損益を
    12,080件のトレード履歴（2020-07〜2026-04）から導出
  - H1 x D1 クロス分析も統合
"""
import numpy as np
import pandas as pd


# ─── 0. yfinance ティッカー対応表 ────────────────────────────

YFINANCE_TICKERS = {
    "BTCUSD": "BTC-USD",
    "XAUUSD": "GC=F",
}


# ─── 1. RSI 計算 ──────────────────────────────────────────────

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def fetch_rsi_yfinance(symbol: str) -> tuple:
    """yfinance から H1・D1 の最新 RSI(14) を取得。(rsi_h1, rsi_d1) を返す。"""
    import yfinance as yf
    ticker = YFINANCE_TICKERS.get(symbol, symbol)

    h1_df = yf.download(ticker, period="30d",  interval="1h", progress=False)
    d1_df = yf.download(ticker, period="6mo",  interval="1d", progress=False)

    def _last_rsi(df: pd.DataFrame) -> float:
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return round(float(calc_rsi(close).dropna().iloc[-1]), 1)

    return _last_rsi(h1_df), _last_rsi(d1_df)


# ─── 2. H1 ゾーン定義 ────────────────────────────────────────

ZONE_BINS   = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100]
ZONE_LABELS = [
    "<20","20-25","25-30","30-35","35-40",
    "40-45","45-50","50-55","55-60",
    "60-65","65-70","70-75","75-80",">80",
]

_H1_ZONE_DATA = {
    "BTCUSD": {
        "<20":   {"wr":0,  "avg_profit":-905,    "buy_wr":None,"sell_wr":0,   "cnt":4,    "verdict":"caution"},
        "20-25": {"wr":42, "avg_profit":-2771,   "buy_wr":47,  "sell_wr":40,  "cnt":62,   "verdict":"caution"},
        "25-30": {"wr":63, "avg_profit":-1403,   "buy_wr":64,  "sell_wr":63,  "cnt":145,  "verdict":"forbidden"},
        "30-35": {"wr":54, "avg_profit":1775,    "buy_wr":59,  "sell_wr":51,  "cnt":220,  "verdict":"ok"},
        "35-40": {"wr":50, "avg_profit":2255,    "buy_wr":49,  "sell_wr":51,  "cnt":373,  "verdict":"ok"},
        "40-45": {"wr":39, "avg_profit":-2789,   "buy_wr":41,  "sell_wr":38,  "cnt":799,  "verdict":"forbidden"},
        "45-50": {"wr":48, "avg_profit":-2941,   "buy_wr":51,  "sell_wr":45,  "cnt":536,  "verdict":"forbidden"},
        "50-55": {"wr":53, "avg_profit":-5875,   "buy_wr":55,  "sell_wr":48,  "cnt":820,  "verdict":"forbidden"},
        "55-60": {"wr":50, "avg_profit":-1009,   "buy_wr":58,  "sell_wr":29,  "cnt":1037, "verdict":"forbidden"},
        "60-65": {"wr":58, "avg_profit":4261,    "buy_wr":62,  "sell_wr":37,  "cnt":667,  "verdict":"good"},
        "65-70": {"wr":44, "avg_profit":-3692,   "buy_wr":47,  "sell_wr":17,  "cnt":629,  "verdict":"forbidden"},
        "70-75": {"wr":54, "avg_profit":6015,    "buy_wr":57,  "sell_wr":32,  "cnt":567,  "verdict":"good"},
        "75-80": {"wr":64, "avg_profit":515,     "buy_wr":64,  "sell_wr":50,  "cnt":254,  "verdict":"ok"},
        ">80":   {"wr":80, "avg_profit":33438,   "buy_wr":80,  "sell_wr":None,"cnt":147,  "verdict":"good"},
    },
    "XAUUSD": {
        "<20":   {"wr":54, "avg_profit":-14847,  "buy_wr":50,  "sell_wr":54,  "cnt":85,   "verdict":"caution"},
        "20-25": {"wr":33, "avg_profit":-15358,  "buy_wr":0,   "sell_wr":33,  "cnt":49,   "verdict":"caution"},
        "25-30": {"wr":48, "avg_profit":-824,    "buy_wr":44,  "sell_wr":50,  "cnt":159,  "verdict":"forbidden"},
        "30-35": {"wr":44, "avg_profit":-5056,   "buy_wr":57,  "sell_wr":37,  "cnt":142,  "verdict":"forbidden"},
        "35-40": {"wr":53, "avg_profit":560,     "buy_wr":38,  "sell_wr":64,  "cnt":216,  "verdict":"ok"},
        "40-45": {"wr":35, "avg_profit":-1649,   "buy_wr":44,  "sell_wr":29,  "cnt":436,  "verdict":"forbidden"},
        "45-50": {"wr":63, "avg_profit":-5695,   "buy_wr":58,  "sell_wr":67,  "cnt":292,  "verdict":"forbidden"},
        "50-55": {"wr":50, "avg_profit":-280,    "buy_wr":50,  "sell_wr":50,  "cnt":580,  "verdict":"forbidden"},
        "55-60": {"wr":52, "avg_profit":260,     "buy_wr":55,  "sell_wr":45,  "cnt":745,  "verdict":"ok"},
        "60-65": {"wr":59, "avg_profit":622,     "buy_wr":65,  "sell_wr":44,  "cnt":825,  "verdict":"good"},
        "65-70": {"wr":60, "avg_profit":784,     "buy_wr":61,  "sell_wr":50,  "cnt":639,  "verdict":"good"},
        "70-75": {"wr":61, "avg_profit":1712,    "buy_wr":65,  "sell_wr":40,  "cnt":415,  "verdict":"good"},
        "75-80": {"wr":67, "avg_profit":984,     "buy_wr":66,  "sell_wr":74,  "cnt":156,  "verdict":"good"},
        ">80":   {"wr":40, "avg_profit":-71,     "buy_wr":47,  "sell_wr":22,  "cnt":744,  "verdict":"forbidden"},
    },
}

# ─── 3. D1 ゾーン定義 ────────────────────────────────────────

_D1_ZONE_DATA = {
    "BTCUSD": {
        "<20":   {"wr":None,"avg_profit":None,  "verdict":"caution"},
        "20-25": {"wr":0,   "avg_profit":-2188, "verdict":"forbidden"},
        "25-30": {"wr":50,  "avg_profit":-141,  "verdict":"caution"},
        "30-35": {"wr":57,  "avg_profit":-2706, "verdict":"forbidden"},
        "35-40": {"wr":52,  "avg_profit":-63,   "verdict":"caution"},
        "40-45": {"wr":39,  "avg_profit":-715,  "verdict":"forbidden"},
        "45-50": {"wr":48,  "avg_profit":6040,  "verdict":"ok"},
        "50-55": {"wr":47,  "avg_profit":774,   "verdict":"ok"},
        "55-60": {"wr":42,  "avg_profit":-850,  "verdict":"forbidden"},
        "60-65": {"wr":56,  "avg_profit":-2229, "verdict":"forbidden"},
        "65-70": {"wr":56,  "avg_profit":443,   "verdict":"ok"},
        "70-75": {"wr":49,  "avg_profit":976,   "verdict":"ok"},
        "75-80": {"wr":53,  "avg_profit":-1830, "verdict":"forbidden"},
        ">80":   {"wr":59,  "avg_profit":3094,  "verdict":"good"},
    },
    "XAUUSD": {
        "<20":   {"wr":None,"avg_profit":None,   "verdict":"caution"},
        "20-25": {"wr":None,"avg_profit":None,   "verdict":"caution"},
        "25-30": {"wr":None,"avg_profit":None,   "verdict":"caution"},
        "30-35": {"wr":56,  "avg_profit":2755,   "verdict":"good"},
        "35-40": {"wr":31,  "avg_profit":-12017, "verdict":"forbidden"},
        "40-45": {"wr":24,  "avg_profit":-3309,  "verdict":"forbidden"},
        "45-50": {"wr":46,  "avg_profit":-1059,  "verdict":"forbidden"},
        "50-55": {"wr":54,  "avg_profit":-5709,  "verdict":"forbidden"},
        "55-60": {"wr":58,  "avg_profit":-564,   "verdict":"caution"},
        "60-65": {"wr":58,  "avg_profit":587,    "verdict":"good"},
        "65-70": {"wr":43,  "avg_profit":-157,   "verdict":"caution"},
        "70-75": {"wr":68,  "avg_profit":875,    "verdict":"good"},
        "75-80": {"wr":60,  "avg_profit":914,    "verdict":"good"},
        ">80":   {"wr":38,  "avg_profit":-1,     "verdict":"forbidden"},
    },
}

# ─── 4. H1 x D1 クロス判定 ───────────────────────────────────

def _cross_bucket(rsi: float) -> str:
    if rsi < 40:  return "<40"
    if rsi < 50:  return "40-50"
    if rsi < 60:  return "50-60"
    if rsi < 70:  return "60-70"
    return ">70"

_CROSS_VERDICT = {
    "BTCUSD": {
        ("<40",  "<40"):   ("caution",  -1060, 60),
        ("<40",  "40-50"): ("forbidden",-296,  30),
        ("40-50","<40"):   ("good",     3967,  50),
        ("40-50","40-50"): ("good",     3591,  50),
        ("40-50","50-60"): ("forbidden",-582,  30),
        ("40-50","60-70"): ("forbidden",-473,  10),
        ("50-60","<40"):   ("ok",       1553,  50),
        ("50-60","40-50"): ("forbidden",-1692, 40),
        ("50-60","50-60"): ("ok",       1997,  50),
        ("50-60","60-70"): ("caution",  -894,  50),
        ("50-60",">70"):   ("forbidden",-9571, 20),
        ("60-70","<40"):   ("ok",       426,   50),
        ("60-70","40-50"): ("forbidden",-4846, 50),
        ("60-70","50-60"): ("forbidden",-4515, 60),
        ("60-70","60-70"): ("good",     1772,  60),
        ("60-70",">70"):   ("caution",  -647,  50),
        (">70",  "40-50"): ("forbidden",-16286,60),
        (">70",  "50-60"): ("forbidden",-8664, 50),
        (">70",  "60-70"): ("good",     6073,  50),
        (">70",  ">70"):   ("best",     13918, 60),
    },
    "XAUUSD": {
        ("<40",  "<40"):   ("forbidden",-8357, 40),
        ("<40",  "40-50"): ("caution",  -867,  50),
        ("<40",  "50-60"): ("forbidden",-2374, 0),
        ("40-50","<40"):   ("caution",  -339,  50),
        ("40-50","40-50"): ("forbidden",-1556, 40),
        ("40-50","50-60"): ("forbidden",-4262, 20),
        ("40-50","60-70"): ("ok",       903,   70),
        ("50-60","<40"):   ("caution",  -677,  70),
        ("50-60","40-50"): ("forbidden",-9093, 50),
        ("50-60","50-60"): ("caution",  -162,  60),
        ("50-60","60-70"): ("forbidden",-1786, 50),
        ("50-60",">70"):   ("forbidden",-2097, 40),
        ("60-70","<40"):   ("ok",       961,   50),
        ("60-70","40-50"): ("ok",       136,   50),
        ("60-70","50-60"): ("ok",       153,   50),
        ("60-70","60-70"): ("caution",  -132,  50),
        ("60-70",">70"):   ("caution",  -339,  40),
        (">70",  "40-50"): ("ok",       8544,  50),
        (">70",  "50-60"): ("good",     1128,  60),
        (">70",  "60-70"): ("good",     1030,  70),
        (">70",  ">70"):   ("ok",       236,   50),
    },
}

# ─── 5. ゾーン取得ヘルパー ───────────────────────────────────

def get_zone_label(rsi: float) -> str:
    for i, (lo, hi) in enumerate(zip(ZONE_BINS[:-1], ZONE_BINS[1:])):
        if lo <= rsi < hi:
            return ZONE_LABELS[i]
    return ">80"

def get_h1_zone(symbol: str, rsi: float) -> dict:
    sym   = symbol if symbol in _H1_ZONE_DATA else "BTCUSD"
    label = get_zone_label(rsi)
    info  = _H1_ZONE_DATA[sym].get(label, {"verdict":"ok","wr":50,"avg_profit":0,"cnt":0})
    return {"zone": label, **info}

def get_d1_zone(symbol: str, rsi: float) -> dict:
    sym   = symbol if symbol in _D1_ZONE_DATA else "BTCUSD"
    label = get_zone_label(rsi)
    info  = _D1_ZONE_DATA[sym].get(label, {"verdict":"ok","wr":50,"avg_profit":0})
    return {"zone": label, **info}

def get_cross_verdict(symbol: str, rsi_h1: float, rsi_d1: float) -> dict:
    sym    = symbol if symbol in _CROSS_VERDICT else "BTCUSD"
    d1_b   = _cross_bucket(rsi_d1)
    h1_b   = _cross_bucket(rsi_h1)
    result = _CROSS_VERDICT[sym].get((d1_b, h1_b))
    if result is None:
        return {"verdict":"ok","avg_profit":0,"wr":50,"d1_bucket":d1_b,"h1_bucket":h1_b}
    return {"verdict":result[0],"avg_profit":result[1],"wr":result[2],
            "d1_bucket":d1_b,"h1_bucket":h1_b}

# ─── 6. 時間帯・曜日 ─────────────────────────────────────────

HOUR_FORBIDDEN_UTC = {9, 16, 21}
HOUR_CAUTION_UTC   = {0, 7, 12}
DOW_FORBIDDEN = {5}
DOW_CAUTION   = {4}

# ─── 7. 統合シグナル判定 ─────────────────────────────────────

def evaluate_signal(
    symbol:    str,
    rsi_h1:    float,
    rsi_d1:    float,
    direction: str,
    hour_utc:  int,
    dow:       int,
) -> dict:
    h1_zone = get_h1_zone(symbol, rsi_h1)
    d1_zone = get_d1_zone(symbol, rsi_d1)
    cross   = get_cross_verdict(symbol, rsi_h1, rsi_d1)

    reasons   = []
    penalties = 0
    bonuses   = 0

    # H1 RSI
    if h1_zone["verdict"] == "forbidden":
        reasons.append(f"🔴 H1 RSI {h1_zone['zone']} は禁止（平均 ${h1_zone['avg_profit']:,}）")
        penalties += 3
    elif h1_zone["verdict"] == "caution":
        reasons.append(f"🟡 H1 RSI {h1_zone['zone']} は注意（勝率 {h1_zone['wr']}%）")
        penalties += 1
    elif h1_zone["verdict"] in ("ok","good"):
        bonuses += 1

    # D1 RSI
    if d1_zone["verdict"] == "forbidden":
        d1_ap = d1_zone.get("avg_profit") or 0
        reasons.append(f"🔴 D1 RSI {d1_zone['zone']} は禁止（平均 ${d1_ap:,}）")
        penalties += 2
    elif d1_zone["verdict"] == "caution":
        reasons.append(f"🟡 D1 RSI {d1_zone['zone']} は注意（勝率 {d1_zone.get('wr','—')}%）")
        penalties += 1
    elif d1_zone["verdict"] in ("ok","good"):
        bonuses += 1

    # クロス
    cv = cross["verdict"]
    if cv == "best":
        reasons.append(f"⭐ H1×D1 最良の組み合わせ（D1:{cross['d1_bucket']} × H1:{cross['h1_bucket']} 平均 ${cross['avg_profit']:,}）")
        bonuses += 2
    elif cv == "good":
        reasons.append(f"✅ H1×D1 良好（D1:{cross['d1_bucket']} × H1:{cross['h1_bucket']} 平均 ${cross['avg_profit']:,}）")
        bonuses += 1
    elif cv == "forbidden":
        reasons.append(f"🔴 H1×D1 組み合わせNG（D1:{cross['d1_bucket']} × H1:{cross['h1_bucket']} 平均 ${cross['avg_profit']:,}）")
        penalties += 2
    elif cv == "caution":
        reasons.append(f"🟡 H1×D1 組み合わせ注意（D1:{cross['d1_bucket']} × H1:{cross['h1_bucket']}）")
        penalties += 1

    # 方向フィルター
    if direction == "buy":
        bwr = h1_zone.get("buy_wr")
        if bwr is not None and bwr < 45:
            reasons.append(f"🟠 H1 Buy勝率 {bwr}% — この帯域のBuyは不利")
            penalties += 1
    else:
        swr = h1_zone.get("sell_wr")
        if swr is not None and swr < 45:
            reasons.append(f"🟠 H1 Sell勝率 {swr}% — この帯域のSellは不利")
            penalties += 1

    # 時間帯
    dow_names = ["月","火","水","木","金","土","日"]
    if hour_utc in HOUR_FORBIDDEN_UTC:
        reasons.append(f"🔴 UTC{hour_utc}時（JST{(hour_utc+9)%24}時）は最悪時間帯")
        penalties += 2
    elif hour_utc in HOUR_CAUTION_UTC:
        reasons.append(f"🟡 UTC{hour_utc}時（JST{(hour_utc+9)%24}時）は注意時間帯")
        penalties += 1

    # 曜日
    if dow in DOW_FORBIDDEN:
        reasons.append(f"🔴 {dow_names[dow]}曜日はエントリー禁止")
        penalties += 2
    elif dow in DOW_CAUTION:
        reasons.append(f"🟡 {dow_names[dow]}曜日は注意")
        penalties += 1

    # スコア
    score = min(100, max(0, 30 + bonuses * 20 - penalties * 15))

    # 総合判定
    if penalties >= 3:
        verdict  = "forbidden"
        signal   = "WAIT"
        strength = None
        score    = max(0, score - 20)
    elif penalties >= 1:
        verdict  = "caution"
        signal   = "WAIT"
        strength = None
        score    = min(score, 50)
    else:
        verdict = "ok"
        if cv == "best" or (h1_zone["verdict"]=="good" and d1_zone["verdict"]=="good"):
            strength = "strong"
            score    = min(100, score + 20)
        elif h1_zone["verdict"] in ("ok","good"):
            strength = "normal"
        else:
            strength = "weak"
        signal = direction.upper()

    if not reasons:
        reasons.append(f"✅ H1 RSI {h1_zone['zone']}・D1 RSI {d1_zone['zone']} ともに有利ゾーン")

    return {
        "signal":   signal,
        "strength": strength,
        "verdict":  verdict,
        "reasons":  reasons,
        "h1_zone":  h1_zone,
        "d1_zone":  d1_zone,
        "cross":    cross,
        "penalties":penalties,
        "bonuses":  bonuses,
        "score":    score,
    }

