### 📁 NEW FILE: utils/trade_notifier.py (🆕 Sends trade notifications to user via console or future add-ons)

import time

def send_trade_notification(signal):
    print("\n[📡 NOTIFICATION] New Trade Signal")
    print(f"Strategy: {signal.get('strategy')}")
    print(f"Prediction: {signal.get('prediction')}")
    print(f"Entry Price: {signal.get('entry')}")
    print(f"Confidence: {signal.get('confidence')}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

# Future: extend this to send to Discord, Email, or Telegram
