from typing import Dict, Optional
import pandas as pd
import numpy as np
from ml.smc_predictor import SMCZonePredictor
from ml.sentiment_model import FinBERTSentimentAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FusionStrategy:
    """
    Fusion strategy that combines LSTM predictions, SMC analysis, and sentiment analysis
    to generate high-confidence trading signals.
    """
    
    def __init__(self):
        """Initialize fusion strategy"""
        self.smc_predictor = SMCZonePredictor()
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        self.confidence_thresholds = {
            'lstm': 0.7,
            'smc': 0.8,
            'sentiment': 0.7
        }

    def generate_signals(self, lstm_predictions: Dict[str, any], smc_prediction: Dict[str, any], 
                        sentiment: Dict[str, any]) -> Dict[str, any]:
        """
        Generate trading signals based on multi-timeframe analysis
        
        Args:
            lstm_predictions: LSTM predictions across multiple timeframes
            smc_prediction: SMC zone prediction
            sentiment: Sentiment analysis results
            
        Returns:
            Dictionary containing trading signals and confidence scores
        """
        try:
            # Extract predictions and confidence scores from LSTM
            lstm_confidence = lstm_predictions.get('final_confidence', 0.0)
            lstm_trend = lstm_predictions.get('final_prediction', 'Neutral')
            
            # Extract SMC prediction and confidence
            smc_confidence = smc_prediction.get('confidence', 0.0)
            smc_zone = smc_prediction.get('prediction', 'N')
            
            # Extract sentiment analysis results
            sentiment_confidence = sentiment.get('confidence', 0.0)
            sentiment_label = sentiment.get('sentiment', 'Neutral')
            
            # Log the inputs for debugging
            logger.info(f"[🔍] Generating signal with - LSTM: {lstm_trend} ({lstm_confidence:.2f}), "
                      f"SMC: {smc_zone} ({smc_confidence:.2f}), "
                      f"Sentiment: {sentiment_label} ({sentiment_confidence:.2f})")
            
            # Check if all predictions meet confidence thresholds
            if (lstm_confidence < self.confidence_thresholds['lstm'] or
                smc_confidence < self.confidence_thresholds['smc'] or
                sentiment_confidence < self.confidence_thresholds['sentiment']):
                logger.info("[⚠️] Low confidence predictions - No signal generated")
                return None

            # Generate signal based on combined analysis
            signal = self._generate_fusion_signal(
                lstm_trend, smc_zone, sentiment_label,
                lstm_confidence, smc_confidence, sentiment_confidence
            )
            
            if signal:
                logger.info(f"[✅] Generated fusion signal: {signal['prediction']} "
                          f"(Confidence: {signal['confidence_score']:.2f})")
            else:
                logger.info("[ℹ️] No clear signal generated based on current market conditions")
            
            return signal
            
        except Exception as e:
            logger.error(f"[❌] Error generating fusion signal: {str(e)}")
            return None

    def _generate_fusion_signal(self, lstm_trend: str, smc_zone: str, sentiment: str,
                              lstm_confidence: float, smc_confidence: float, sentiment_confidence: float) -> Optional[Dict[str, any]]:
        """
        Generate final trading signal based on combined analysis
        
        Args:
            lstm_trend: Predicted trend from LSTM ('Bullish' or 'Bearish')
            smc_zone: Predicted SMC zone (e.g., 'BOS', 'BZ', 'SOS', 'SZ')
            sentiment: Market sentiment ('Positive', 'Neutral', 'Negative')
            lstm_confidence: Confidence score for LSTM prediction (0-1)
            smc_confidence: Confidence score for SMC prediction (0-1)
            sentiment_confidence: Confidence score for sentiment analysis (0-1)
            
        Returns:
            Dictionary containing signal details or None if no clear signal
        """
        # Calculate weighted confidence score (weight LSTM and SMC more heavily)
        confidence_score = (lstm_confidence * 0.4 + 
                          smc_confidence * 0.4 + 
                          sentiment_confidence * 0.2)
        
        # Define signal conditions with updated SMC zones and confidence thresholds
        long_conditions = {
            "High": {
                "lstm": "Bullish",
                "smc": ["BOS", "BZ", "BUY"],
                "sentiment": ["Positive"],
                "min_confidence": 0.8
            },
            "Medium": {
                "lstm": "Bullish",
                "smc": ["BOS", "BZ", "BUY", "B"],
                "sentiment": ["Positive", "Neutral"],
                "min_confidence": 0.7
            },
            "Low": {
                "lstm": "Bullish",
                "smc": ["BOS", "BZ", "BUY", "B", "N"],
                "sentiment": ["Positive", "Neutral"],
                "min_confidence": 0.6
            }
        }

        short_conditions = {
            "High": {
                "lstm": "Bearish",
                "smc": ["SOS", "SZ"],
                "sentiment": ["Negative"],
                "min_confidence": 0.8
            },
            "Medium": {
                "lstm": "Bearish",
                "smc": ["SOS", "SZ", "S"],
                "sentiment": ["Negative", "Neutral"],
                "min_confidence": 0.7
            },
            "Low": {
                "lstm": "Bearish",
                "smc": ["SOS", "SZ", "S", "N"],
                "sentiment": ["Negative", "Neutral"],
                "min_confidence": 0.6
            }
        }

        # Check if conditions are met
        for confidence_level, conditions in long_conditions.items():
            if (lstm_trend == conditions["lstm"] and
                smc_zone in conditions["smc"] and
                sentiment in conditions["sentiment"] and
                confidence_score >= conditions["min_confidence"]):
                return {
                    "strategy": "Fusion Strategy",
                    "prediction": "Long Entry",
                    "confidence": confidence_level,
                    "confidence_score": confidence_score,
                    "lstm_trend": lstm_trend,
                    "smc_zone": smc_zone,
                    "sentiment": sentiment,
                    "lstm_confidence": lstm_confidence,
                    "smc_confidence": smc_confidence,
                    "sentiment_confidence": sentiment_confidence
                }

        for confidence_level, conditions in short_conditions.items():
            if (lstm_trend == conditions["lstm"] and
                smc_zone in conditions["smc"] and
                sentiment in conditions["sentiment"] and
                confidence_score >= conditions["min_confidence"]):
                return {
                    "strategy": "Fusion Strategy",
                    "prediction": "Short Entry",
                    "confidence": confidence_level,
                    "confidence_score": confidence_score,
                    "lstm_trend": lstm_trend,
                    "smc_zone": smc_zone,
                    "sentiment": sentiment,
                    "lstm_confidence": lstm_confidence,
                    "smc_confidence": smc_confidence,
                    "sentiment_confidence": sentiment_confidence
                }

        return None
