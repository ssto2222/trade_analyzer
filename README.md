# 📊 Trade Signal Dashboard

1H RSI(14) を軸にした**エントリーNG判定 & シグナル表示**Webアプリ。

12,080件のトレード履歴（2020-07〜2026-04）を分析し、
RSI帯域・時間帯・曜日ごとの禁止ゾーンをバックテスト由来のルールで可視化します。

---

## 🚀 ローカル起動

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 ファイル構成

```
trade-signal-app/
├── app.py            # Streamlit メインアプリ
├── analysis.py       # RSI計算・シグナル判定ロジック
├── requirements.txt
└── .streamlit/
    └── config.toml   # ダークテーマ設定
```

## 📋 判定ルール概要

### RSIゾーン（1H RSI14）

| 銘柄 | 禁止ゾーン | 優良ゾーン |
|------|----------|----------|
| BTCUSD | 40〜55、65〜70 | 60〜65、70〜75、>80 |
| XAUUSD | <35、40〜50、>80 | 55〜80 |

### 時間帯（JST）

| 判定 | 時間帯 |
|------|--------|
| 🔴 禁止 | 6時・16時・18時 |
| 🟡 注意 | 9時・16時・21時 |

### 曜日

| 判定 | 曜日 |
|------|------|
| 🔴 禁止 | 土曜 |
| 🟡 注意 | 金曜・木曜 |

## ☁️ Streamlit Cloud デプロイ

1. このリポジトリをGitHubにpush
2. [share.streamlit.io](https://share.streamlit.io) でリポジトリを選択
3. Main file: `app.py` を指定してデプロイ

---

> データソース: MT5トレード履歴 CSV（12,080件）  
> 分析期間: 2020-07 〜 2026-04
