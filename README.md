# 🥇 Gold Trading Signal Analyzer
## XAU/USD MA × RSI 売買シグナル分析ダッシュボード

Exness MT5 / yfinance 対応の Streamlit ベース・ゴールドトレーディング分析ツール

---

## 🚀 クイックスタート

```bash
# 1. パッケージインストール
pip install -r requirements.txt

# 2. (Windows + MT5 の場合のみ)
pip install MetaTrader5

# 3. 起動
streamlit run gold_trading_analyzer.py
```

## 📡 データソース

| ソース | OS | 説明 |
|--------|-----|------|
| 🎮 デモデータ | 全OS | シミュレーション価格 (インストール不要) |
| 📊 yfinance | 全OS | Yahoo Finance の金先物 (GC=F) |
| 🔗 Exness MT5 | Windows | MT5ターミナル経由リアルタイム |

## 📈 分析機能

- **移動平均線 (SMA / EMA)**: 短期・長期のクロスオーバー検出
- **RSI**: 買われすぎ・売られすぎゾーン検出
- **ボリンジャーバンド**: 価格の偏差分析
- **MACD**: モメンタム分析
- **シグナル検出**: GC/DC + RSI の複合判定
- **簡易バックテスト**: シグナルベースの損益シミュレーション
- **CSVエクスポート**: データ・シグナルのダウンロード

## ⚙️ Exness MT5 接続手順

1. Exness でアカウント作成
2. MT5 ターミナルをダウンロード・起動
3. `pip install MetaTrader5`
4. サイドバーで「Exness MT5」を選択
5. ログイン情報を入力して接続

## ⚠️ 免責事項

本ツールは学習・参考目的です。投資判断はご自身の責任で行ってください。
