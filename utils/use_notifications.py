from app.trade_notifier import send_trade_notification

def notify_if_needed(signal):
    if signal and isinstance(signal, dict):
        send_trade_notification(signal)
        return True