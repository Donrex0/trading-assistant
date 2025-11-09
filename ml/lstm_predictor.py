# ml/lstm_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import joblib
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration - should match training config
CONFIG = {
    'sequence_length': 180,
    'hidden_size': 768,
    'num_layers': 5,
    'dropout': 0.15,
    'num_classes': 2
}
SEQUENCE_LENGTH = 48

FEATURES = ["open", "high", "low", "close", "volume", "rsi", "macd", "roc", "volatility", "ma", "momentum"]

def preprocess_recent_data(df, sequence_length=SEQUENCE_LENGTH):
    recent_df = df[FEATURES].tail(sequence_length)
    if len(recent_df) < sequence_length:
        raise ValueError("Insufficient data for LSTM prediction")
    x = torch.tensor(recent_df.values, dtype=torch.float32).unsqueeze(0)
    return x

def load_dataframe_for_timeframe(symbol, timeframe):
    file_path = f"data/processed/BTCUSDT_{timeframe}_lstm.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
    df = pd.read_csv(file_path, index_col=0)
    return df

class SuperiorLSTMClassifier(nn.Module):
    """Enhanced LSTM classifier matching the training architecture"""
    def __init__(self, input_size: int, **kwargs):
        super().__init__()
        config = {**CONFIG, **kwargs}
        
        self.input_size = input_size
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Input processing
        self.input_norm = nn.BatchNorm1d(input_size)
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout * 0.3)
        )
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = self.hidden_size if i == 0 else self.hidden_size * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim, 
                    self.hidden_size, 
                    1,  # Single layer per LSTM
                    batch_first=True,
                    bidirectional=True
                )
            )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size * 2,
            num_heads=8,
            dropout=self.dropout * 0.5,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, config['num_classes'])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len, features)
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.input_projection(x)
        
        # Process through LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Apply attention
        x, _ = self.attention(x, x, x)
        x = x.mean(dim=1)  # Global average pooling
        
        # Final classification
        return self.classifier(x)

def predict_single_timeframe(df, model_path):
    input_size = len(FEATURES)
    model = SuperiorLSTMClassifier(input_size)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    x = preprocess_recent_data(df)
    with torch.no_grad():
        output = model(x)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()
    return prediction, confidence

def predict_multi_timeframe(symbol="BTCUSDT"):
    results = []
    for tf in ["15m", "1h", "4h", "1d"]:
        try:
            df = load_dataframe_for_timeframe(symbol, tf)
            model_path = os.path.join("models", "smc_lstm_improved_best.pt")
            pred, conf = predict_single_timeframe(df, model_path)
            results.append({
                "timeframe": tf,
                "prediction": pred,
                "confidence": conf
            })
        except Exception as e:
            print(f"[❌] Error on {tf}: {e}")

    # Final aggregated result (majority vote)
    votes = [r["prediction"] for r in results]
    final_prediction = max(set(votes), key=votes.count) if votes else None

    return {
        "final_prediction": final_prediction,
        "per_timeframe": results
    }

if __name__ == "__main__":
    result = predict_multi_timeframe()
    print("🧠 Final Multi-Timeframe Prediction:", "Bullish" if result["final_prediction"] == 1 else "Bearish")
    for r in result["per_timeframe"]:
        print(f"  ⏱️ {r['timeframe']}: {['Bearish','Bullish'][r['prediction']]} ({r['confidence']:.2f})")
