"""
local_analysis.py — MT5 実データでバックテスト検証
------------------------------------------------------
使い方:
  python local_analysis.py --btc BTC_H1.csv --xau XAU_H1.csv

MT5 ヒストリー CSV フォーマット (どちらも自動検出):
  形式A: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>...
  形式B: Date,Time,Open,High,Low,Close,Volume  (カンマ区切り)
  形式C: Datetime,Open,High,Low,Close          (1列の日時)

出力:
  - 銘柄ごとの H1 RSI シグナル件数/日
  - ゾーン別ヒット内訳
  - 合成データとの比較サマリー
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 現在の analysis.py のルールをそのままインポート
from analysis import (
    calc_rsi,
    get_h1_zone, get_d1_zone, get_cross_verdict,
    HOUR_FORBIDDEN_UTC, HOUR_CAUTION_UTC,
    DOW_FORBIDDEN, DOW_CAUTION,
    evaluate_signal,
)

# ─── MT5 CSV ローダー ─────────────────────────────────────────

def load_mt5_csv(path: str) -> pd.DataFrame:
    """
    MT5 エクスポート CSV を読み込み、UTC datetime インデックスの
    OHLC DataFrame を返す。タブ区切り・カンマ区切りを自動判定。
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {path}")

    raw = p.read_text(encoding="utf-8", errors="replace")
    sep = "\t" if "\t" in raw.splitlines()[0] else ","

    df = pd.read_csv(path, sep=sep, header=0, encoding="utf-8", errors="replace")
    df.columns = [c.strip().strip("<>").lower() for c in df.columns]

    # 日時列を統合
    if "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
        )
    elif "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # 先頭列を日時として試みる
        df["datetime"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")

    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    # 列名を正規化
    rename = {}
    for col in df.columns:
        if col in ("open", "o"):         rename[col] = "Open"
        elif col in ("high", "h"):       rename[col] = "High"
        elif col in ("low", "l"):        rename[col] = "Low"
        elif col in ("close", "c"):      rename[col] = "Close"
        elif col in ("volume", "vol", "tickvol"): rename[col] = "Volume"
    df = df.rename(columns=rename)

    required = {"Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"必須列が見つかりません: {missing}\n実際の列: {list(df.columns)}")

    return df[["Open", "High", "Low", "Close"]].astype(float)


# ─── D1 リサンプリング ────────────────────────────────────────

def to_daily(h1_df: pd.DataFrame) -> pd.DataFrame:
    return h1_df.resample("1D").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    ).dropna()


# ─── バックテスト ─────────────────────────────────────────────

def backtest(symbol: str, h1_df: pd.DataFrame, direction: str = "buy") -> pd.DataFrame:
    """
    H1 足を1本ずつ走査し、evaluate_signal が WAIT 以外を返す行を収集する。
    Returns: シグナル行の DataFrame
    """
    d1_df = to_daily(h1_df)
    d1_close = d1_df["Close"]
    h1_close = h1_df["Close"]

    # RSI を事前計算
    rsi_h1_series = calc_rsi(h1_close)
    rsi_d1_series = calc_rsi(d1_close)

    records = []
    for ts, row in h1_df.iterrows():
        if ts not in rsi_h1_series.index:
            continue
        rsi_h1 = rsi_h1_series.loc[ts]
        if np.isnan(rsi_h1):
            continue

        # 対応する D1 RSI を取得（当日の終値 RSI）
        date_key = ts.normalize()
        d1_keys = rsi_d1_series.index[rsi_d1_series.index <= date_key]
        if len(d1_keys) == 0:
            continue
        rsi_d1 = rsi_d1_series.loc[d1_keys[-1]]
        if np.isnan(rsi_d1):
            continue

        hour_utc = ts.hour
        dow = ts.weekday()

        result = evaluate_signal(symbol, rsi_h1, rsi_d1, direction, hour_utc, dow)
        if result["signal"] != "WAIT":
            records.append({
                "datetime": ts,
                "rsi_h1": round(rsi_h1, 1),
                "rsi_d1": round(rsi_d1, 1),
                "h1_zone": result["h1_zone"]["zone"],
                "d1_zone": result["d1_zone"]["zone"],
                "signal": result["signal"],
                "strength": result["strength"],
                "score": result["score"],
                "cross_verdict": result["cross"]["verdict"],
            })

    return pd.DataFrame(records)


# ─── レポート出力 ─────────────────────────────────────────────

def report(symbol: str, signals: pd.DataFrame, h1_df: pd.DataFrame) -> None:
    total_days = max(1, (h1_df.index[-1] - h1_df.index[0]).days)
    n = len(signals)
    per_day = n / total_days

    print(f"\n{'='*55}")
    print(f"  {symbol}  バックテスト結果")
    print(f"{'='*55}")
    print(f"  期間      : {h1_df.index[0].date()} 〜 {h1_df.index[-1].date()}  ({total_days} 日)")
    print(f"  H1 総バー : {len(h1_df):,} 本")
    print(f"  シグナル数: {n:,} 件  ({per_day:.2f} 件/日)")

    if n == 0:
        print("  シグナルなし")
        return

    print(f"\n  ── シグナル強度 ──")
    for s, cnt in signals["strength"].value_counts().items():
        print(f"    {s:10s}: {cnt:4d} 件  ({cnt/n*100:.1f}%)")

    print(f"\n  ── H1 RSI ゾーン内訳 ──")
    for z, cnt in signals["h1_zone"].value_counts().head(8).items():
        print(f"    {z:10s}: {cnt:4d} 件  ({cnt/n*100:.1f}%)")

    print(f"\n  ── D1 RSI ゾーン内訳 ──")
    for z, cnt in signals["d1_zone"].value_counts().head(6).items():
        print(f"    {z:10s}: {cnt:4d} 件  ({cnt/n*100:.1f}%)")

    print(f"\n  ── クロス判定 ──")
    for v, cnt in signals["cross_verdict"].value_counts().items():
        print(f"    {v:10s}: {cnt:4d} 件  ({cnt/n*100:.1f}%)")

    # スコア分布
    bins = [0, 35, 60, 80, 101]
    labels = ["不適(<35)", "要注意(35-60)", "良好(60-80)", "高確度(80+)"]
    signals["score_band"] = pd.cut(signals["score"], bins=bins, labels=labels, right=False)
    print(f"\n  ── スコア帯分布 ──")
    for band, cnt in signals["score_band"].value_counts().sort_index().items():
        print(f"    {band:16s}: {cnt:4d} 件  ({cnt/n*100:.1f}%)")

    print()


# ─── 合成データとの比較 ───────────────────────────────────────

SYNTHETIC = {
    "BTCUSD": {"per_day": 1.47, "mt5_low": 4.4, "mt5_high": 5.9},
    "XAUUSD": {"per_day": 1.47, "mt5_low": 4.4, "mt5_high": 5.9},
}

def compare(symbol: str, actual_per_day: float) -> None:
    ref = SYNTHETIC.get(symbol, {})
    if not ref:
        return
    synth = ref["per_day"]
    lo, hi = ref["mt5_low"], ref["mt5_high"]
    ratio = actual_per_day / synth if synth else 0
    in_range = lo <= actual_per_day <= hi
    status = "✅ 想定範囲内" if in_range else ("⬆️ 想定より多い" if actual_per_day > hi else "⬇️ 想定より少ない")
    print(f"  合成データ比較: {synth:.2f} 件/日 → 実測 {actual_per_day:.2f} 件/日  (×{ratio:.1f})  {status}")
    print(f"  MT5 想定範囲  : {lo}〜{hi} 件/日")


# ─── メイン ──────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MT5 実データ バックテスト")
    parser.add_argument("--btc", metavar="CSV", help="BTCUSD H1 CSV ファイル")
    parser.add_argument("--xau", metavar="CSV", help="XAUUSD H1 CSV ファイル")
    parser.add_argument("--dir", default="buy", choices=["buy", "sell"], help="トレード方向 (default: buy)")
    parser.add_argument("--out", metavar="CSV", help="シグナル一覧を CSV 出力")
    args = parser.parse_args()

    if not args.btc and not args.xau:
        parser.print_help()
        print("\n例: python local_analysis.py --btc BTCUSD_H1.csv --xau XAUUSD_H1.csv")
        sys.exit(1)

    all_signals = []
    tasks = []
    if args.btc:
        tasks.append(("BTCUSD", args.btc))
    if args.xau:
        tasks.append(("XAUUSD", args.xau))

    for symbol, csv_path in tasks:
        print(f"\n[{symbol}] {csv_path} を読み込み中...")
        try:
            h1_df = load_mt5_csv(csv_path)
            print(f"  → {len(h1_df):,} 本  ({h1_df.index[0].date()} 〜 {h1_df.index[-1].date()})")
        except Exception as e:
            print(f"  エラー: {e}")
            continue

        print(f"  RSI 計算・シグナル走査中...")
        signals = backtest(symbol, h1_df, args.dir)
        signals["symbol"] = symbol
        all_signals.append(signals)

        report(symbol, signals, h1_df)
        total_days = max(1, (h1_df.index[-1] - h1_df.index[0]).days)
        compare(symbol, len(signals) / total_days)

    if args.out and all_signals:
        out_df = pd.concat(all_signals, ignore_index=True)
        out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\nシグナル一覧を保存: {args.out}  ({len(out_df)} 件)")


if __name__ == "__main__":
    main()
