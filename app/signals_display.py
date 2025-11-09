### 📁 NEW FILE: app/signals_display.py (🆕 Handles group display of signal categories)

import streamlit as st
from app.signal_display import format_signal_display, format_market_display, format_sentiment_display

def show_signals(signals):
    rule_based = [s for s in signals if s.get('strategy') not in ['SMC Pattern Classifier', 'Market Analysis', 'Trend Prediction']]
    ml_signals = [s for s in signals if s.get('strategy') in ['SMC Pattern Classifier', 'Trend Prediction']]
    market_analysis = [s for s in signals if s.get('strategy') == 'Market Analysis']
    sentiment = [s for s in signals if s.get('strategy') == 'Sentiment Analysis']

    if rule_based:
        st.subheader("🎯 Rule-Based Strategies")
        for sig in rule_based:
            st.markdown(format_signal_display(sig))

    if ml_signals:
        st.subheader("🤖 ML Predictions")
        for sig in ml_signals:
            st.markdown(format_signal_display(sig))

    if market_analysis:
        st.subheader("📊 Market Analysis")
        for sig in market_analysis:
            st.markdown(format_market_display(sig))

    if sentiment:
        st.subheader("📰 Sentiment Analysis")
        for sig in sentiment:
            st.markdown(format_sentiment_display(sig))

    if not signals:
        st.info("No signals currently.")
