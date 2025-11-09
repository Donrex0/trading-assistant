import logging
from datetime import datetime
from typing import Optional
import pandas as pd

from strategies.fusion_strategy import FusionStrategy
from ml.lstm_predictor import predict_multi_timeframe
from ml.smc_predictor import predict_smc
from ml.sentiment_model import FinBERTSentimentAnalyzer

# Initialize sentiment analyzer once
sentiment_analyzer = FinBERTSentimentAnalyzer()

logger = logging.getLogger(__name__)

def run_all_strategies(df: pd.DataFrame, symbol: str, sentiment_score: Optional[float] = None) -> dict:
    """
    Run the fusion strategy that combines LSTM, SMC, and sentiment predictions
    
    Args:
        df: DataFrame containing market data
        symbol: Trading symbol (e.g., 'BTCUSDT')
        sentiment_score: Optional pre-calculated sentiment score
        
    Returns:
        Dictionary containing trading signals and analysis results
    """
    try:
        logger.info(f"[🚀] Starting strategy analysis for {symbol}")
        
        # Initialize fusion strategy
        fusion_strategy = FusionStrategy()
        
        # 1. Run LSTM prediction
        logger.info("[🧠] Running LSTM prediction...")
        lstm_result = predict_multi_timeframe(symbol)
        
        # 2. Run SMC prediction
        logger.info("[📊] Running SMC analysis...")
        smc_result = predict_smc(symbol)
        
        # 3. Run sentiment analysis if not provided
        logger.info("[💬] Analyzing market sentiment...")
        if sentiment_score is None:
            sentiment_result = sentiment_analyzer.analyze_sentiment(symbol)
            sentiment_score = sentiment_result.get('score', 0.0)
            sentiment_trend = sentiment_result.get('trend', 'Neutral')
        else:
            sentiment_trend = "Bullish" if sentiment_score > 0 else "Bearish"
            sentiment_result = {
                'sentiment': sentiment_trend,
                'score': sentiment_score,
                'confidence': min(abs(sentiment_score) * 2, 1.0)  # Convert -0.5 to 0.5 -> 1.0 confidence
            }
        
        # 4. Generate fusion signal
        logger.info("[🔀] Generating fusion signal...")
        fusion_signal = fusion_strategy.generate_signals(
            lstm_result, 
            smc_result, 
            sentiment_result
        )
        
        # 5. Prepare signals list for dashboard
        signals = []
        
        # Add LSTM signal
        lstm_trend = "Bullish" if lstm_result.get('final_prediction', 0) == 1 else "Bearish"
        signals.append({
            "symbol": symbol,
            "strategy": "LSTM Prediction",
            "prediction": lstm_trend,
            "confidence": lstm_result.get('final_confidence', 0.0),
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Add SMC signal
        signals.append({
            "symbol": symbol,
            "strategy": "SMC Analysis",
            "prediction": smc_result.get('prediction', 'N'),
            "confidence": smc_result.get('confidence', 0.0),
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Add sentiment signal
        signals.append({
            "symbol": symbol,
            "strategy": "Sentiment Analysis",
            "prediction": sentiment_result.get('sentiment', 'Neutral'),
            "confidence": sentiment_result.get('confidence', 0.0),
            "timestamp": pd.Timestamp.now().isoformat()
        })
        
        # Add fusion signal if available
        if fusion_signal:
            signals.append({
                "symbol": symbol,
                "strategy": "Fusion Strategy",
                "prediction": fusion_signal.get('prediction', 'No signal'),
                "confidence": fusion_signal.get('confidence_score', 0.0),
                "timestamp": pd.Timestamp.now().isoformat()
            })
            logger.info(f"[✅] Generated fusion signal: {fusion_signal.get('prediction', 'No signal')} "
                      f"(Confidence: {fusion_signal.get('confidence_score', 0.0):.2f})")
        else:
            logger.info("[ℹ️] No clear fusion signal generated")
            
        return signals

    except Exception as e:
        error_msg = f"Error running strategies: {str(e)}"
        logger.error(f"[❌] {error_msg}")
        return [{
            'symbol': symbol,
            'signal': 'ERROR',
            'confidence': 0.0,
            'message': error_msg,
            'timestamp': pd.Timestamp.now().isoformat(),
            'best_signal': None,
            'all_signals': [],
            'confidence_summary': {},
            'error': str(e)
        }]
