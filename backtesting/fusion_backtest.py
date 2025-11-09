### 📁 NEW FILE: backtesting/fusion_backtest.py

import pandas as pd
from fusion_strategy import fusion_signal
from ml.smc_predictor import predict_smc  # already created earlier
from ml.lstm_predictor import predict_lstm_trend  # you’ll create this below
from sentiment.sentiment_model import analyze_sentiment  # optional

def run_fusion_backtest(df):
    balance = 1000
    trades, wins, losses = 0, 0, 0
    equity_curve = [balance]

    for i in range(60, len(df) - 1):
        sub_df = df.iloc[i-60:i].copy()
        trend_pred = predict_lstm_trend(sub_df)
        smc_pred = predict_smc(sub_df)
        sentiment = "positive"  # ← optional, or call sentiment model

        signal = fusion_signal(trend_pred, smc_pred, sentiment, sub_df)
        if signal:
            trades += 1
            entry = df.iloc[i]["close"]
            next_price = df.iloc[i+1]["close"]
            change = (next_price - entry) / entry

            if change > 0.01:
                balance *= 1 + 0.01
                wins += 1
            elif change < -0.005:
                balance *= 1 - 0.005
                losses += 1
            equity_curve.append(balance)

    win_rate = wins / trades * 100 if trades > 0 else 0
    print(f"[Fusion Backtest] Trades: {trades} | Win Rate: {win_rate:.2f}% | Final Balance: {balance:.2f}")
