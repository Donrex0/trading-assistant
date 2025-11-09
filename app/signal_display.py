### 📁 NEW FILE: app/signal_display.py (🆕 Used to structure trading signal display rendering)

def format_signal_display(signal):
    return f"""
    **Strategy**: `{signal.get('strategy', '-')}`  
    **Prediction**: `{signal.get('prediction', '-')}`  
    **Entry**: `{signal.get('entry', '-')}`  
    **Confidence**: `{signal.get('confidence', '-')}`
    """

def format_sentiment_display(signal):
    return f"""
    **Sentiment Score**: `{signal.get('sentiment_score', '-')}`  
    **Sentiment Trend**: `{signal.get('sentiment_trend', '-')}`
    """

def format_market_display(signal):
    return f"""
    **Market Trend**: `{signal.get('market_trend', '-')}`  
    **Volume Trend**: `{signal.get('volume_trend', '-')}`  
    **Recent Price Change**: `{signal.get('price_change', '-')}`  
    **Recent Volume Change**: `{signal.get('volume_change', '-')}`
    """
