"""
analysis.py  —  RSI(14) 計算・シグナル判定ロジック
データ由来:
  - BTCUSD / XAUUSD の1H RSI14帯域別勝率・平均損益を
    12,080件のトレード履歴（2020-07〜2026-04）から導出
"""
import numpy as np
import pandas as pd


# ─── 1. RSI 計算 ─────────────────────────────────────────────

def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder式 RSI(period) を返す。"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ─── 2. バックテスト由来のゾーン定義 ─────────────────────────

# 構造: { symbol: { rsi_zone_label: { "wr", "avg_profit", "buy_wr", "sell_wr", "cnt", "verdict" } } }
# verdict: "forbidden" | "caution" | "ok" | "good"

ZONE_BINS   = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 100]
ZONE_LABELS = [
    "<20", "20-25", "25-30", "30-35", "35-40",
    "40-45", "45-50", "50-55", "55-60",
    "60-65", "65-70", "70-75", "75-80", ">80",
]

_ZONE_DATA = {
    "BTCUSD": {
        "<20":   {"wr": 0,  "avg_profit": -905,    "buy_wr": None, "sell_wr": 0,    "cnt": 4,    "verdict": "caution"},
        "20-25": {"wr": 42, "avg_profit": -2771,   "buy_wr": 47,   "sell_wr": 40,   "cnt": 62,   "verdict": "caution"},
        "25-30": {"wr": 63, "avg_profit": -1403,   "buy_wr": 64,   "sell_wr": 63,   "cnt": 145,  "verdict": "forbidden"},
        "30-35": {"wr": 54, "avg_profit": 1775,    "buy_wr": 59,   "sell_wr": 51,   "cnt": 220,  "verdict": "ok"},
        "35-40": {"wr": 50, "avg_profit": 2255,    "buy_wr": 49,   "sell_wr": 51,   "cnt": 373,  "verdict": "ok"},
        "40-45": {"wr": 39, "avg_profit": -2789,   "buy_wr": 41,   "sell_wr": 38,   "cnt": 799,  "verdict": "forbidden"},
        "45-50": {"wr": 48, "avg_profit": -2941,   "buy_wr": 51,   "sell_wr": 45,   "cnt": 536,  "verdict": "forbidden"},
        "50-55": {"wr": 53, "avg_profit": -5875,   "buy_wr": 55,   "sell_wr": 48,   "cnt": 820,  "verdict": "forbidden"},
        "55-60": {"wr": 50, "avg_profit": -1009,   "buy_wr": 58,   "sell_wr": 29,   "cnt": 1037, "verdict": "forbidden"},
        "60-65": {"wr": 58, "avg_profit": 4261,    "buy_wr": 62,   "sell_wr": 37,   "cnt": 667,  "verdict": "good"},
        "65-70": {"wr": 44, "avg_profit": -3692,   "buy_wr": 47,   "sell_wr": 17,   "cnt": 629,  "verdict": "forbidden"},
        "70-75": {"wr": 54, "avg_profit": 6015,    "buy_wr": 57,   "sell_wr": 32,   "cnt": 567,  "verdict": "good"},
        "75-80": {"wr": 64, "avg_profit": 515,     "buy_wr": 64,   "sell_wr": 50,   "cnt": 254,  "verdict": "ok"},
        ">80":   {"wr": 80, "avg_profit": 33438,   "buy_wr": 80,   "sell_wr": None, "cnt": 147,  "verdict": "good"},
    },
    "XAUUSD": {
        "<20":   {"wr": 54, "avg_profit": -14847,  "buy_wr": 50,   "sell_wr": 54,   "cnt": 85,   "verdict": "caution"},
        "20-25": {"wr": 33, "avg_profit": -15358,  "buy_wr": 0,    "sell_wr": 33,   "cnt": 49,   "verdict": "caution"},
        "25-30": {"wr": 48, "avg_profit": -824,    "buy_wr": 44,   "sell_wr": 50,   "cnt": 159,  "verdict": "forbidden"},
        "30-35": {"wr": 44, "avg_profit": -5056,   "buy_wr": 57,   "sell_wr": 37,   "cnt": 142,  "verdict": "forbidden"},
        "35-40": {"wr": 53, "avg_profit": 560,     "buy_wr": 38,   "sell_wr": 64,   "cnt": 216,  "verdict": "ok"},
        "40-45": {"wr": 35, "avg_profit": -1649,   "buy_wr": 44,   "sell_wr": 29,   "cnt": 436,  "verdict": "forbidden"},
        "45-50": {"wr": 63, "avg_profit": -5695,   "buy_wr": 58,   "sell_wr": 67,   "cnt": 292,  "verdict": "forbidden"},
        "50-55": {"wr": 50, "avg_profit": -280,    "buy_wr": 50,   "sell_wr": 50,   "cnt": 580,  "verdict": "forbidden"},
        "55-60": {"wr": 52, "avg_profit": 260,     "buy_wr": 55,   "sell_wr": 45,   "cnt": 745,  "verdict": "ok"},
        "60-65": {"wr": 59, "avg_profit": 622,     "buy_wr": 65,   "sell_wr": 44,   "cnt": 825,  "verdict": "good"},
        "65-70": {"wr": 60, "avg_profit": 784,     "buy_wr": 61,   "sell_wr": 50,   "cnt": 639,  "verdict": "good"},
        "70-75": {"wr": 61, "avg_profit": 1712,    "buy_wr": 65,   "sell_wr": 40,   "cnt": 415,  "verdict": "good"},
        "75-80": {"wr": 67, "avg_profit": 984,     "buy_wr": 66,   "sell_wr": 74,   "cnt": 156,  "verdict": "good"},
        ">80":   {"wr": 40, "avg_profit": -71,     "buy_wr": 47,   "sell_wr": 22,   "cnt": 744,  "verdict": "forbidden"},
    },
}

# 時間帯NG (UTC, JST = UTC+9)
HOUR_FORBIDDEN_UTC = {9, 16, 21}   # JST 18h→UTC9, 1h→UTC16, 6h→UTC21  ← worst 3
HOUR_CAUTION_UTC   = {0, 7, 12}    # JST 9h, 16h, 21h

# 曜日NG (0=Mon)
DOW_FORBIDDEN = {5}   # 土曜
DOW_CAUTION   = {4}   # 金曜

# ─── 3. ゾーン取得 ───────────────────────────────────────────

def get_zone_label(rsi: float) -> str:
    for i, (lo, hi) in enumerate(zip(ZONE_BINS[:-1], ZONE_BINS[1:])):
        if lo <= rsi < hi:
            return ZONE_LABELS[i]
    return ">80"


def get_zone_info(symbol: str, rsi: float) -> dict:
    """RSI値からゾーン情報を返す。未対応銘柄はBTCUSDのルールを使う。"""
    sym = symbol if symbol in _ZONE_DATA else "BTCUSD"
    label = get_zone_label(rsi)
    info = _ZONE_DATA[sym].get(label, {"verdict": "ok", "wr": 50, "avg_profit": 0, "cnt": 0})
    return {"zone": label, **info}


# ─── 4. 統合シグナル判定 ─────────────────────────────────────

def evaluate_signal(
    symbol: str,
    rsi: float,
    direction: str,      # "buy" | "sell"
    hour_utc: int,
    dow: int,            # 0=Mon
) -> dict:
    """
    RSI・時間帯・曜日を総合してシグナルを返す。

    Returns
    -------
    {
        "signal":   "BUY" | "SELL" | "WAIT",
        "strength": "strong" | "normal" | "weak",
        "verdict":  "ok" | "caution" | "forbidden",
        "reasons":  [...],
        "zone_info": {...},
    }
    """
    zone = get_zone_info(symbol, rsi)
    reasons = []
    penalties = 0

    # ── RSIゾーン評価
    if zone["verdict"] == "forbidden":
        reasons.append(f"🔴 RSI {zone['zone']} は禁止ゾーン（平均損益 ${zone['avg_profit']:,}）")
        penalties += 3
    elif zone["verdict"] == "caution":
        reasons.append(f"🟡 RSI {zone['zone']} は注意ゾーン（勝率 {zone['wr']}%）")
        penalties += 1

    # ── 方向フィルター
    if direction == "buy":
        bwr = zone.get("buy_wr")
        if bwr is not None and bwr < 45:
            reasons.append(f"🟠 Buy勝率 {bwr}% — この帯域でのBuyは不利")
            penalties += 1
    else:
        swr = zone.get("sell_wr")
        if swr is not None and swr < 45:
            reasons.append(f"🟠 Sell勝率 {swr}% — この帯域でのSellは不利")
            penalties += 1

    # ── 時間帯評価
    if hour_utc in HOUR_FORBIDDEN_UTC:
        reasons.append(f"🔴 UTC {hour_utc}時（JST {(hour_utc+9)%24}時）は最悪時間帯")
        penalties += 2
    elif hour_utc in HOUR_CAUTION_UTC:
        reasons.append(f"🟡 UTC {hour_utc}時（JST {(hour_utc+9)%24}時）は注意時間帯")
        penalties += 1

    # ── 曜日評価
    dow_names = ["月", "火", "水", "木", "金", "土", "日"]
    if dow in DOW_FORBIDDEN:
        reasons.append(f"🔴 {dow_names[dow]}曜日はエントリー禁止（合計損益 -$3.85M）")
        penalties += 2
    elif dow in DOW_CAUTION:
        reasons.append(f"🟡 {dow_names[dow]}曜日は注意（合計損益 -$6.38M）")
        penalties += 1

    # ── 総合判定
    if penalties >= 3:
        verdict = "forbidden"
        signal = "WAIT"
        strength = None
    elif penalties >= 1:
        verdict = "caution"
        signal = "WAIT"
        strength = None
    else:
        verdict = "ok"
        ap = zone["avg_profit"]
        wr = zone["wr"]
        if ap > 5000 and wr >= 58:
            strength = "strong"
        elif ap > 0 and wr >= 52:
            strength = "normal"
        else:
            strength = "weak"
        signal = direction.upper()

    if not reasons:
        reasons.append(f"✅ RSI {zone['zone']} は有利ゾーン（勝率 {zone['wr']}%, 平均 ${zone['avg_profit']:,}）")

    return {
        "signal":    signal,
        "strength":  strength,
        "verdict":   verdict,
        "reasons":   reasons,
        "zone_info": zone,
        "penalties": penalties,
    }
