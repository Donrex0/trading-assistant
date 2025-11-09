"""
SMC (Smart Money Concept) Zone Predictor with batch processing and optimizations.
"""
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor

# Add project root to path if needed
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking for optimal performance
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# Type aliases
Tensor = torch.Tensor
DataFrame = pd.DataFrame

class EnhancedSMCModel(nn.Module):
    """Enhanced SMC model matching the training architecture"""
    def __init__(self, input_size, hidden_size=512, num_layers=4, num_classes=7, dropout=0.12):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_size)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.3)
        )
        
        # Multi-scale feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.2)
            ) for _ in range(3)
        ])
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = hidden_size if i == 0 else hidden_size * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim, hidden_size, 1,
                    batch_first=True, bidirectional=True
                )
            )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout * 0.5,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_projection(x)
        
        # Process through feature extractors
        features = [extractor(x) for extractor in self.feature_extractors]
        x = torch.cat(features, dim=-1)
        
        # Process through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Apply attention
        x, _ = self.attention(x, x, x)
        x = x.mean(dim=1)  # Global average pooling
        
        # Final classification
        return self.classifier(x)

class SMCZonePredictor:
    """Enhanced SMC zone predictor with proper model loading and inference"""
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.sequence_length = 120  # Should match training config
        self.load_model()
    
    def load_model(self):
        """Load the trained model and related artifacts"""
        try:
            model_path = Path("models/bulletproof_smc_98.pt")
            scaler_path = Path("models/smc_scaler.pkl")
            encoder_path = Path("models/smc_label_encoder.pkl")
            
            # Load model
            self.model = EnhancedSMCModel(
                input_size=150,  # Should match training config
                hidden_size=512,
                num_layers=4,
                num_classes=7,
                dropout=0.12
            ).to(self.device)
            
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                logger.info("✅ Successfully loaded SMC model")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load scaler
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Loaded SMC scaler")
            
            # Load label encoder
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info("✅ Loaded SMC label encoder")
                
            
        except Exception as e:
            logger.error(f"[❌] Error loading model components: {str(e)}")
            raise

    def predict(self, symbols: Union[str, List[str]], batch_size: int = 8) -> Union[Dict, List[Dict]]:
        """
        Predict SMC zones for one or multiple symbols with batch processing support.
        
        Args:
            symbols: Single symbol string or list of symbols (e.g., 'BTC/USDT' or ['BTC/USDT', 'ETH/USDT'])
            batch_size: Number of symbols to process in parallel (default: 8)
            
        Returns:
            Dictionary or list of dictionaries with prediction results
        """
        is_single = isinstance(symbols, str)
        symbols = [symbols] if is_single else symbols
        
        try:
            # Process symbols in batches
            results = []
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                batch_results = self._predict_batch(batch_symbols)
                results.extend(batch_results)
            
            return results[0] if is_single else results
            
        except Exception as e:
            logger.error(f"[❌] Error in batch prediction: {str(e)}")
            if is_single:
                return {'error': str(e), 'symbol': symbols[0] if symbols else 'unknown'}
            return [{'error': str(e), 'symbol': sym} for sym in symbols]
    
    def _predict_batch(self, symbols: List[str]) -> List[Dict]:
        """Process a batch of symbols efficiently."""
        if not symbols:
            return []
            
        try:
            # Get data for all symbols in the batch
            batch_data = []
            for symbol in symbols:
                try:
                    # Get latest data for the symbol (implement your data fetching logic here)
                    df = pd.read_csv(get_latest_processed_file())
                    
                    # Engineer features
                    df = self._engineer_features(df)
                    
                    # Prepare input data
                    features = df.iloc[-1][['ma5', 'ma20', 'ma50', 'volatility_10', 'volatility_20',
                                          'momentum_5', 'momentum_10', 'volume_ma', 'volume_ratio',
                                          'price_change', 'price_change_5']].values
                    
                    batch_data.append((symbol, features))
                    
                except Exception as e:
                    logger.error(f"[❌] Error processing {symbol}: {str(e)}")
                    batch_data.append((symbol, None))
            
            # Filter out failed symbols
            valid_data = [(s, f) for s, f in batch_data if f is not None]
            if not valid_data:
                return [{'error': 'No valid data for any symbol', 'symbol': s} for s, _ in batch_data]
            
            # Prepare batch tensor
            valid_symbols, features_list = zip(*valid_data)
            
            # Scale features if scaler is available
            if self.scaler:
                features_list = self.scaler.transform(features_list)
            else:
                logger.warning("[⚠️] Using raw features without scaling")
            
            # Convert to tensor
            x = torch.FloatTensor(features_list).unsqueeze(1).to(self.device)  # Add sequence dimension
            
            # Make predictions in a single forward pass
            with torch.no_grad():
                outputs = self.model(x)
                probabilities = torch.softmax(outputs, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
            
            # Convert to numpy
            predictions_np = predictions.cpu().numpy()
            confidences_np = confidences.cpu().numpy()
            
            # Map predictions to zone names
            results = []
            for i, symbol in enumerate(valid_symbols):
                if self.label_encoder:
                    zone = self.label_encoder.inverse_transform([predictions_np[i]])[0]
                else:
                    zone = f"Zone_{predictions_np[i]}"
                
                results.append({
                    'symbol': symbol,
                    'prediction': int(predictions_np[i]),
                    'zone': zone,
                    'confidence': float(confidences_np[i])
                })
            
            # Add error entries for failed symbols
            failed_symbols = {s for s, f in batch_data if f is None}
            results.extend([{'error': 'Failed to process symbol', 'symbol': s} for s in failed_symbols])
            
            return results
            
        except Exception as e:
            logger.error(f"[❌] Error in batch prediction: {str(e)}")
            return [{'error': str(e), 'symbol': s} for s in symbols]
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimized feature engineering for SMC prediction.
        
        Args:
            df: DataFrame with OHLCV data (must contain: timestamp, open, high, low, close, volume)
            
        Returns:
            DataFrame with additional engineered features
            
        Note:
            This version is optimized for performance while maintaining FP32 precision.
            All operations are vectorized for better performance.
        """
        try:
            # Get prediction
            prediction = self.predict(df)
            
            # Calculate recent price movement
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]
            
            # Calculate volume change
            volume_change = (df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]
            
            return {
                'prediction': prediction,
                'recent_price_change': float(price_change),
                'recent_volume_change': float(volume_change),
                'market_trend': 'up' if price_change > 0 else 'down',
                'volume_trend': 'increasing' if volume_change > 0 else 'decreasing'
            }
            
        except Exception as e:
            logger.error(f"[❌] Error analyzing market: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize predictor
    predictor = SMCZonePredictor()
    
    try:
        # Load latest data
        df = pd.read_csv(get_latest_processed_file(), index_col=0, parse_dates=True)
        
        # Get prediction
        prediction = predictor.predict(df)
        
        # Get market analysis
        analysis = predictor.analyze_market(df)
        
        # Print results
        print("\n[📊] SMC Zone Prediction:")
        print(f"Zone: {prediction['zone']}")
        print(f"Confidence: {prediction['confidence']:.2%}")
        print(f"Price: ${prediction['price']:.2f}")
        print(f"Volume: {int(prediction['volume'])}")
        print(f"Timestamp: {prediction['timestamp']}")
        
        print("\n[📈] Market Analysis:")
        print(f"Recent Price Change: {analysis['recent_price_change']:.2%}")
        print(f"Recent Volume Change: {analysis['recent_volume_change']:.2%}")
        print(f"Market Trend: {analysis['market_trend']}")
        print(f"Volume Trend: {analysis['volume_trend']}")
        
    except Exception as e:
        print(f"[❌] Error in main execution: {str(e)}")

