import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import warnings
import random
from datetime import datetime
from pathlib import Path
from collections import Counter
import logging

# Suppress warnings
warnings.filterwarnings('ignore')

# ENHANCED Configuration for 90%+ SMC accuracy with multi-timeframe alignment
CONFIG = {
    'sequence_length': 120,      # Longer sequences for better pattern recognition
    'hidden_size': 512,          # Larger model capacity  
    'num_layers': 4,             # Deeper network
    'dropout': 0.12,             # Optimized dropout
    'batch_size': 28,            # Optimized batch size
    'num_epochs': 250,           # More epochs
    'target_accuracy': 0.90,     # 90% target
    'learning_rate': 0.0002,     # Optimized learning rate
    'weight_decay': 3e-5,        # Light regularization
    'patience': 35,              # More patience
    'gradient_clip': 0.8,        # Gradient clipping
    'warmup_epochs': 18,         # Learning rate warmup
    'min_lr': 1e-7,              # Minimum learning rate
    'label_smoothing': 0.03,     # Light label smoothing
    'focal_alpha': 2.0,          # Focal loss parameters
    'focal_gamma': 2.5,
    'ensemble_size': 1,          # Single model first
    'confidence_threshold': 0.75, # Confidence filtering
}

# Directories
BASE_DIR = Path(__file__).parent.parent
LABELS_DIR = BASE_DIR / "data" / "labels"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging():
    """Setup enhanced logging"""
    logger = logging.getLogger('EnhancedSMCClassifier')
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

print("🚀 ENHANCED SMC CLASSIFIER WITH MULTI-TIMEFRAME ALIGNMENT")
print("=" * 65)
print(f"🎯 Target: {CONFIG['target_accuracy']:.0%} Accuracy")
print(f"🔧 Device: {DEVICE}")
print(f"⏰ Multi-timeframe: 15m + 1h + 4h alignment")

class EnhancedSMCDataProcessor:
    """Enhanced data processor with multi-timeframe alignment like the LSTM trend classifier"""
    
    @staticmethod
    def engineer_multi_timeframe_features(df):
        """Create comprehensive multi-timeframe features for SMC classification"""
        df = df.copy()
        logger.info(f"🔧 Engineering multi-timeframe SMC features from {df.shape[0]} rows")
        
        # Core price features for BASE (15m) timeframe
        price_cols = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in price_cols):
            try:
                # Enhanced price action features
                df['price_range_pct'] = (df['high'] - df['low']) / df['close']
                df['body_size_pct'] = abs(df['close'] - df['open']) / df['close']
                df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
                df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
                df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
                
                # Explicit 15m timeframe features
                df['15m_open'] = df['open']
                df['15m_high'] = df['high']
                df['15m_low'] = df['low']
                df['15m_close'] = df['close']
                df['15m_price_range_pct'] = df['price_range_pct']
                df['15m_body_size_pct'] = df['body_size_pct']
                df['15m_upper_shadow'] = df['upper_shadow']
                df['15m_lower_shadow'] = df['lower_shadow']
                df['15m_close_position'] = df['close_position']
                
                # Advanced candlestick patterns
                body_size = abs(df['close'] - df['open'])
                total_range = df['high'] - df['low'] + 1e-8
                
                df['is_bullish'] = (df['close'] > df['open']).astype(int)
                df['is_bearish'] = (df['close'] < df['open']).astype(int)
                df['is_doji'] = (body_size / total_range < 0.1).astype(int)
                df['is_hammer'] = ((df['lower_shadow'] > 2 * body_size) & (df['upper_shadow'] < body_size)).astype(int)
                df['is_shooting_star'] = ((df['upper_shadow'] > 2 * body_size) & (df['lower_shadow'] < body_size)).astype(int)
                
                # 15m explicit candlestick patterns
                df['15m_is_bullish'] = df['is_bullish']
                df['15m_is_bearish'] = df['is_bearish']
                df['15m_is_doji'] = df['is_doji']
                df['15m_is_hammer'] = df['is_hammer']
                df['15m_is_shooting_star'] = df['is_shooting_star']
                
                # Price momentum for 15m
                for period in [3, 5, 8, 13, 21]:
                    df[f'price_momentum_{period}'] = df['close'].pct_change(period)
                    df[f'price_acceleration_{period}'] = df[f'price_momentum_{period}'].diff()
                    df[f'15m_price_momentum_{period}'] = df[f'price_momentum_{period}']
                    df[f'15m_price_acceleration_{period}'] = df[f'price_acceleration_{period}']
                
                logger.info("✅ Enhanced 15m price features created")
            except Exception as e:
                logger.warning(f"⚠️ Error creating price features: {e}")
        
        # Technical indicators for 15m
        if 'close' in df.columns:
            try:
                # RSI for multiple periods
                for period in [9, 14, 21]:
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).ewm(span=period).mean()
                    loss = (-delta.where(delta < 0, 0)).ewm(span=period).mean()
                    rs = gain / (loss + 1e-10)
                    df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
                    df[f'15m_rsi_{period}'] = df[f'rsi_{period}']
                    df[f'15m_rsi_oversold_{period}'] = (df[f'rsi_{period}'] < 30).astype(int)
                    df[f'15m_rsi_overbought_{period}'] = (df[f'rsi_{period}'] > 70).astype(int)
                
                # Moving averages
                for period in [7, 14, 21, 50, 100]:
                    df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
                    df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                    df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
                    df[f'15m_sma_{period}'] = df[f'sma_{period}']
                    df[f'15m_ema_{period}'] = df[f'ema_{period}']
                    df[f'15m_price_vs_sma_{period}'] = df[f'price_vs_sma_{period}']
                
                # MACD
                exp1 = df['close'].ewm(span=12).mean()
                exp2 = df['close'].ewm(span=26).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9).mean()
                df['macd_histogram'] = df['macd'] - df['macd_signal']
                df['15m_macd'] = df['macd']
                df['15m_macd_signal'] = df['macd_signal']
                df['15m_macd_histogram'] = df['macd_histogram']
                
                logger.info("✅ Technical indicators created for 15m")
            except Exception as e:
                logger.warning(f"⚠️ Error creating technical indicators: {e}")
        
        # Volume analysis for 15m
        if 'volume' in df.columns:
            try:
                for period in [10, 20, 50]:
                    df[f'volume_sma_{period}'] = df['volume'].rolling(period, min_periods=1).mean()
                    df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_sma_{period}'] + 1e-8)
                    df[f'15m_volume_sma_{period}'] = df[f'volume_sma_{period}']
                    df[f'15m_volume_ratio_{period}'] = df[f'volume_ratio_{period}']
                
                # Volume-Price Trend
                df['vpt'] = (df['volume'] * df['close'].pct_change()).cumsum()
                df['15m_vpt'] = df['vpt']
                
                logger.info("✅ Volume features created for 15m")
            except Exception as e:
                logger.warning(f"⚠️ Error creating volume features: {e}")
        
        # Process higher timeframes (1h, 4h)
        higher_tfs = ["1h", "4h"]
        for tf in higher_tfs:
            tf_close = f'{tf}_close'
            if tf_close in df.columns:
                try:
                    # Higher timeframe analysis
                    df[f'{tf}_returns'] = df[tf_close].pct_change().fillna(0)
                    df[f'{tf}_momentum'] = df[tf_close].diff(3).fillna(0)
                    df[f'{tf}_volatility'] = df[f'{tf}_returns'].rolling(20, min_periods=1).std()
                    
                    if 'close' in df.columns:
                        df[f'{tf}_relative_strength'] = df['close'] / (df[tf_close] + 1e-8)
                        df[f'{tf}_trend_alignment'] = (df['close'] > df[tf_close]).astype(int)
                        df[f'15m_vs_{tf}_alignment'] = (df['close'] > df[tf_close]).astype(int)
                        df[f'15m_vs_{tf}_strength'] = df['close'] / (df[tf_close] + 1e-8)
                    
                    # Higher timeframe RSI
                    if len(df[tf_close].dropna()) > 14:
                        delta = df[tf_close].diff()
                        gain = delta.where(delta > 0, 0).ewm(span=14).mean()
                        loss = (-delta.where(delta < 0, 0)).ewm(span=14).mean()
                        rs = gain / (loss + 1e-10)
                        df[f'{tf}_rsi'] = 100 - (100 / (1 + rs))
                        df[f'{tf}_rsi_oversold'] = (df[f'{tf}_rsi'] < 30).astype(int)
                        df[f'{tf}_rsi_overbought'] = (df[f'{tf}_rsi'] > 70).astype(int)
                    
                    logger.info(f"✅ {tf} timeframe features created")
                except Exception as e:
                    logger.warning(f"⚠️ Error creating {tf} features: {e}")
        
        # Cross-timeframe interaction features
        try:
            if all(f'{tf}_close' in df.columns for tf in higher_tfs) and 'close' in df.columns:
                # Timeframe momentum alignment
                df['tf_momentum_alignment'] = 0
                if '15m_momentum_5' in df.columns:
                    for tf in higher_tfs:
                        if f'{tf}_momentum' in df.columns:
                            same_direction = ((df['15m_momentum_5'] > 0) == (df[f'{tf}_momentum'] > 0)).astype(int)
                            df['tf_momentum_alignment'] += same_direction
                
                # Timeframe RSI convergence
                df['tf_rsi_convergence'] = 0
                if '15m_rsi_14' in df.columns:
                    for tf in higher_tfs:
                        if f'{tf}_rsi' in df.columns:
                            rsi_diff = abs(df['15m_rsi_14'] - df[f'{tf}_rsi'])
                            convergence = (rsi_diff < 15).astype(int)
                            df['tf_rsi_convergence'] += convergence
                
                # Multi-timeframe trend strength
                df['multi_tf_trend_strength'] = 0
                base_sma = df.get('15m_sma_20', df.get('sma_20'))
                if base_sma is not None:
                    for tf in higher_tfs:
                        tf_close = df.get(f'{tf}_close')
                        if tf_close is not None:
                            tf_sma = tf_close.rolling(20, min_periods=1).mean()
                            trend_alignment = (df['close'] > base_sma) == (tf_close > tf_sma)
                            df['multi_tf_trend_strength'] += trend_alignment.astype(int)
                
                logger.info("✅ Cross-timeframe interaction features created")
        except Exception as e:
            logger.warning(f"⚠️ Error creating cross-timeframe features: {e}")
        
        # Enhanced SMC zone features
        if 'smc_zone' in df.columns:
            try:
                # Enhanced zone mapping
                zone_mapping = {
                    'SOS': 0, 'SZ': 1, 'S': 2, 'N': 3, 'B': 4, 'BZ': 5, 'BOS': 6
                }
                df['smc_zone_encoded'] = df['smc_zone'].map(zone_mapping).fillna(3)
                
                # Zone transition analysis
                df['zone_changed'] = (df['smc_zone'] != df['smc_zone'].shift(1)).astype(int)
                df['zone_direction'] = df['smc_zone_encoded'].diff().fillna(0)
                
                # Zone persistence
                zone_groups = df.groupby((df['smc_zone'] != df['smc_zone'].shift()).cumsum())
                df['zone_duration'] = zone_groups.cumcount() + 1
                df['zone_strength'] = df['zone_duration'] / (df['zone_duration'].rolling(50, min_periods=1).max() + 1e-8)
                
                # Previous zone context
                for lag in [1, 2, 3]:
                    df[f'prev_zone_{lag}'] = df['smc_zone_encoded'].shift(lag).fillna(3)
                
                # Zone momentum
                df['zone_momentum'] = df['smc_zone_encoded'].diff(3).fillna(0)
                
                logger.info("✅ Enhanced SMC zone features created")
            except Exception as e:
                logger.warning(f"⚠️ Error creating SMC features: {e}")
        
        logger.info(f"🎯 Multi-timeframe feature engineering completed. Final shape: {df.shape}")
        return df
    
    @staticmethod
    def clean_and_validate_data(df):
        """Clean and validate data for optimal SMC performance"""
        logger.info(f"🧹 Cleaning data with shape: {df.shape}")
        
        df = df.copy()
        
        # Remove infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        # Conservative outlier removal
        for col in numeric_cols:
            if col not in ['smc_zone_encoded', 'timestamp'] and col in df.columns:
                lower_percentile = df[col].quantile(0.001)
                upper_percentile = df[col].quantile(0.999)
                
                if pd.notna(lower_percentile) and pd.notna(upper_percentile):
                    df[col] = df[col].clip(lower_percentile, upper_percentile)
        
        # Smart missing value handling
        df = df.ffill().bfill()
        
        # Fill remaining NaN values
        for col in numeric_cols:
            if col in df.columns and df[col].isna().any():
                if 'volume' in col.lower():
                    df[col] = df[col].fillna(df[col].median())
                elif any(x in col.lower() for x in ['price', 'close', 'open', 'high', 'low']):
                    df[col] = df[col].interpolate().fillna(method='ffill')
                else:
                    df[col] = df[col].fillna(df[col].mean())
        
        # Final safety net
        df = df.fillna(0)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        if df.empty:
            raise ValueError("All data was removed during cleaning")
        
        logger.info(f"✅ Data cleaned. Final shape: {df.shape}")
        return df

class EnhancedSMCDataset(Dataset):
    """Enhanced SMC dataset with multi-timeframe alignment"""
    
    def __init__(self, df, sequence_length=CONFIG['sequence_length'], is_training=True, 
                 scaler=None, features=None):
        self.sequence_length = sequence_length
        self.is_training = is_training
        
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        logger.info(f"🏗️ Creating enhanced SMC dataset from {len(df)} samples")
        
        # Apply enhanced feature engineering
        processor = EnhancedSMCDataProcessor()
        df = processor.engineer_multi_timeframe_features(df)
        df = processor.clean_and_validate_data(df)
        
        if df.empty:
            raise ValueError("DataFrame empty after processing")
        
        # Encode SMC zones
        zone_mapping = {'SOS': 0, 'SZ': 1, 'S': 2, 'N': 3, 'B': 4, 'BZ': 5, 'BOS': 6}
        df['target_encoded'] = df['smc_zone'].map(zone_mapping).fillna(3)
        
        # Feature selection with timeframe priority
        if features is not None:
            self.features = features
            logger.info(f"🔧 Using {len(self.features)} provided features")
        else:
            exclude_cols = ['timestamp', 'smc_zone', 'target_encoded', 'target']
            candidate_features = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
            
            # Prioritize timeframe features
            timeframe_features = []
            for tf in ["15m", "1h", "4h"]:
                tf_features = [f for f in candidate_features if f.startswith(f'{tf}_')]
                timeframe_features.extend(tf_features)
            
            # Add cross-timeframe and other important features
            other_features = [f for f in candidate_features 
                            if not any(f.startswith(f'{tf}_') for tf in ["15m", "1h", "4h"])
                            and f in ['tf_momentum_alignment', 'tf_rsi_convergence', 'multi_tf_trend_strength',
                                    'zone_changed', 'zone_direction', 'zone_duration', 'zone_strength']]
            
            # Combine features with quality filtering
            all_features = timeframe_features + other_features
            high_quality_features = []
            
            for feature in all_features:
                if feature in df.columns:
                    feature_data = df[feature]
                    if (not feature_data.isna().all() and 
                        feature_data.var() > 1e-10 and
                        feature_data.isna().sum() / len(df) < 0.2):
                        high_quality_features.append(feature)
            
            # Feature importance based on target correlation
            if 'target_encoded' in df.columns and len(high_quality_features) > 0:
                feature_importance = []
                for feature in high_quality_features:
                    try:
                        corr = abs(df[feature].corr(df['target_encoded']))
                        feature_importance.append((feature, corr if not pd.isna(corr) else 0))
                    except:
                        feature_importance.append((feature, 0))
                
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                self.features = [f[0] for f in feature_importance[:150]]  # Top 150 features
            else:
                self.features = high_quality_features[:150]
        
        if not self.features:
            raise ValueError("No valid features available")
        
        logger.info(f"🎯 Selected {len(self.features)} enhanced features")
        
        # Count features by timeframe
        tf_counts = {}
        for tf in ["15m", "1h", "4h"]:
            tf_count = len([f for f in self.features if f.startswith(f'{tf}_')])
            tf_counts[tf] = tf_count
        other_count = len([f for f in self.features if not any(f.startswith(f'{tf}_') for tf in ["15m", "1h", "4h"])])
        tf_counts['other'] = other_count
        
        logger.info(f"📈 Feature distribution: {tf_counts}")
        
        # Ensure target exists
        if 'target_encoded' not in df.columns:
            raise ValueError("Target column not found")
        
        df = df.dropna(subset=['target_encoded'])
        if df.empty:
            raise ValueError("No samples with valid targets")
        
        # Enhanced feature scaling
        feature_data = df[self.features].copy().fillna(0)
        
        if scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(feature_data)
        else:
            self.scaler = scaler
            features_scaled = self.scaler.transform(feature_data)
        
        # Create final dataset
        self.data = pd.DataFrame(features_scaled, columns=self.features, index=df.index)
        self.data["target"] = df["target_encoded"].values
        
        # Enhanced confidence calculation
        self.confidence = self._calculate_confidence(df)
        
        # Prepare sequences
        self._prepare_sequences()
        
        logger.info(f"✅ Enhanced SMC dataset created with {len(self.X)} sequences")
    
    def _calculate_confidence(self, df):
        """Calculate confidence scores for SMC predictions"""
        confidence = np.ones(len(df))
        
        # Volume-based confidence
        if 'volume' in df.columns:
            try:
                vol_ma = df['volume'].rolling(20, min_periods=1).mean()
                vol_ratio = df['volume'] / (vol_ma + 1e-8)
                vol_confidence = np.clip(vol_ratio / 2, 0.4, 2.0)
                confidence *= vol_confidence
            except:
                pass
        
        # SMC zone confidence (strong zones get higher confidence)
        if 'smc_zone' in df.columns:
            try:
                zone_confidence = df['smc_zone'].map({
                    'SOS': 2.5, 'BOS': 2.5,   # Strongest signals
                    'SZ': 2.0, 'BZ': 2.0,     # Strong zones
                    'S': 1.6, 'B': 1.6,       # Medium zones
                    'N': 0.8                   # Neutral zone
                }).fillna(1.0)
                confidence *= zone_confidence
            except:
                pass
        
        # Multi-timeframe alignment confidence
        if 'tf_momentum_alignment' in df.columns:
            try:
                tf_confidence = 1 + 0.5 * (df['tf_momentum_alignment'] / 2)  # Normalize by max possible value
                confidence *= tf_confidence
            except:
                pass
        
        # Zone transition confidence (transitions are more predictable)
        if 'zone_changed' in df.columns:
            try:
                transition_confidence = 1 + 0.3 * df['zone_changed']
                confidence *= transition_confidence
            except:
                pass
        
        # Ensure reasonable range
        confidence = np.nan_to_num(confidence, nan=1.0, posinf=2.5, neginf=0.3)
        confidence = np.clip(confidence, 0.3, 2.5)
        
        return confidence
    
    def _prepare_sequences(self):
        """Prepare sequences with enhanced data augmentation"""
        self.X, self.y, self.conf = [], [], []
        
        confidence_array = np.array(self.confidence)
        
        # Track class distribution
        target_counts = Counter(self.data['target'].values)
        max_count = max(target_counts.values())
        min_count = min(target_counts.values())
        
        logger.info(f"📊 SMC zone distribution: {dict(target_counts)}")
        
        # Create sequences
        for i in range(len(self.data) - self.sequence_length):
            try:
                sequence = self.data.iloc[i:i+self.sequence_length][self.features].values
                target = self.data.iloc[i+self.sequence_length]["target"]
                
                conf_idx = i + self.sequence_length
                confidence = confidence_array[conf_idx] if conf_idx < len(confidence_array) else 1.0
                
                if not np.isfinite(sequence).all():
                    continue
                
                self.X.append(sequence)
                self.y.append(target)
                self.conf.append(confidence)
                
                # Data augmentation for minority classes
                if self.is_training and target_counts[target] < max_count * 0.7:
                    # Add augmented sample
                    noise = np.random.normal(0, 0.003, sequence.shape)
                    augmented_sequence = sequence + noise
                    
                    self.X.append(augmented_sequence)
                    self.y.append(target)
                    self.conf.append(confidence * 0.95)
                    
            except Exception as e:
                continue
        
        if len(self.X) == 0:
            raise ValueError("No valid sequences created")
        
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)
        self.conf = torch.tensor(np.array(self.conf), dtype=torch.float32)
        
        final_counts = Counter(self.y.numpy())
        logger.info(f"📈 Final sequence distribution: {dict(final_counts)}")
        logger.info(f"✅ Prepared {len(self.y)} enhanced sequences")
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.conf[idx]

class EnhancedSMCModel(nn.Module):
    """Enhanced SMC model with multi-timeframe attention"""
    
    def __init__(self, input_size, hidden_size=CONFIG['hidden_size'], 
                 num_layers=CONFIG['num_layers'], num_classes=7, dropout=CONFIG['dropout']):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        logger.info(f"🧠 Enhanced SMC Model Architecture:")
        logger.info(f"   📊 Input size: {input_size}")
        logger.info(f"   🏗️ Hidden size: {hidden_size}")
        logger.info(f"   📚 Layers: {num_layers}")
        logger.info(f"   🎯 Classes: {num_classes}")
        
        # Input processing
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
        
        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            input_dim = hidden_size if i == 0 else hidden_size * 2
            self.lstm_layers.append(
                nn.LSTM(
                    input_dim, hidden_size, 1,
                    batch_first=True, dropout=0, bidirectional=True
                )
            )
        
        # Layer normalization for each LSTM layer
        self.lstm_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2) for _ in range(num_layers)
        ])
        
        # Multi-head attention for timeframe alignment
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size * 2,
                num_heads=16,
                dropout=dropout * 0.5,
                batch_first=True
            ) for _ in range(2)
        ])
        
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2) for _ in range(2)
        ])
        
        # SMC-specific feature fusion
        self.smc_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Zone-specific pathways (inspired by SMC structure)
        self.zone_pathway_bull = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        self.zone_pathway_bear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        self.zone_pathway_neutral = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4)
        )
        
        # Final SMC classifier
        self.final_classifier = nn.Sequential(
            nn.Linear(hidden_size // 4 * 3, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout // 2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Auxiliary classifier for regularization
        self.aux_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights
        self.apply(self._enhanced_weight_init)
    
    def _enhanced_weight_init(self, module):
        """Enhanced weight initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        
        # Input processing with normalization
        x_norm = x.transpose(1, 2)  # For batch norm
        x_norm = self.input_norm(x_norm)
        x_norm = x_norm.transpose(1, 2)
        
        # Project to hidden dimension
        x_proj = self.input_projection(x_norm)
        
        # Multi-scale feature extraction
        extracted_features = []
        for extractor in self.feature_extractors:
            feat = extractor(x_proj)
            extracted_features.append(feat)
        
        # Combine extracted features
        x_features = torch.cat(extracted_features, dim=-1)
        
        # Ensure proper dimension for LSTM
        if x_features.size(-1) != self.hidden_size:
            x_features = x_features[:, :, :self.hidden_size]
        
        # Progressive LSTM layers with residual connections
        lstm_out = x_features
        for i, (lstm_layer, norm_layer) in enumerate(zip(self.lstm_layers, self.lstm_norms)):
            lstm_output, _ = lstm_layer(lstm_out)
            lstm_output = norm_layer(lstm_output)
            
            # Residual connection (if dimensions match)
            if i > 0 and lstm_out.size(-1) == lstm_output.size(-1):
                lstm_output = lstm_output + lstm_out
            
            lstm_out = F.dropout(lstm_output, p=self.training * CONFIG['dropout'] * 0.3)
        
        # Multi-layer attention for timeframe alignment
        attended_out = lstm_out
        for attention_layer, attention_norm in zip(self.attention_layers, self.attention_norms):
            attn_output, _ = attention_layer(attended_out, attended_out, attended_out)
            attended_out = attention_norm(attn_output + attended_out)  # Residual connection
        
        # Global feature aggregation
        global_features = torch.mean(attended_out, dim=1)  # Global average pooling
        max_features = torch.max(attended_out, dim=1)[0]   # Global max pooling
        last_features = attended_out[:, -1, :]             # Last timestep
        
        # Combine different aggregations
        combined_features = (global_features + max_features + last_features) / 3
        
        # SMC-specific feature fusion
        fused_features = self.smc_fusion(combined_features)
        
        # SMC zone-specific pathways
        bull_pathway = self.zone_pathway_bull(fused_features)
        bear_pathway = self.zone_pathway_bear(fused_features)
        neutral_pathway = self.zone_pathway_neutral(fused_features)
        
        # Combine zone pathways
        combined_pathways = torch.cat([bull_pathway, bear_pathway, neutral_pathway], dim=-1)
        
        # Final SMC classification
        main_output = self.final_classifier(combined_pathways)
        
        if return_aux:
            # Auxiliary output for training regularization
            aux_output = self.aux_classifier(combined_features)
            return main_output, aux_output
        
        return main_output

class EnhancedSMCFocalLoss(nn.Module):
    """Enhanced focal loss for SMC classification"""
    
    def __init__(self, alpha: float = CONFIG['focal_alpha'], gamma: float = CONFIG['focal_gamma'], 
                 label_smoothing: float = CONFIG['label_smoothing']):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
        # SMC zone-specific class weights
        self.class_weights = torch.tensor([
            2.5,  # SOS - Strong signal
            2.0,  # SZ - Strong zone
            1.5,  # S - Sell zone
            1.0,  # N - Neutral
            1.5,  # B - Buy zone
            2.0,  # BZ - Strong zone
            2.5   # BOS - Strong signal
        ], dtype=torch.float32)
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                confidence: torch.Tensor = None) -> torch.Tensor:
        
        # Move class weights to same device as inputs
        if self.class_weights.device != inputs.device:
            self.class_weights = self.class_weights.to(inputs.device)
        
        # Apply class weights
        weight = self.class_weights[targets]
        
        # Calculate weighted cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        
        # Calculate probabilities and focal weight
        pt = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply class weights and confidence
        weighted_loss = focal_weight * ce_loss * weight
        
        if confidence is not None:
            # Enhanced confidence weighting for SMC
            confidence_weight = 0.4 + 1.1 * confidence  # Scale to [0.4, 1.5]
            weighted_loss = weighted_loss * confidence_weight
        
        return weighted_loss.mean()

class CombinedSMCLoss(nn.Module):
    """Combined loss function for enhanced SMC performance"""
    
    def __init__(self, focal_weight=0.8, ce_weight=0.2, aux_weight=0.1):
        super().__init__()
        self.focal_loss = EnhancedSMCFocalLoss()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.aux_weight = aux_weight
    
    def forward(self, main_outputs, targets, confidence=None, aux_outputs=None):
        # Main losses
        focal = self.focal_loss(main_outputs, targets, confidence)
        ce = self.ce_loss(main_outputs, targets)
        
        total_loss = self.focal_weight * focal + self.ce_weight * ce
        
        # Auxiliary loss for regularization
        if aux_outputs is not None:
            aux_loss = self.ce_loss(aux_outputs, targets)
            total_loss = total_loss + self.aux_weight * aux_loss
        
        return total_loss

def create_enhanced_data_loaders(train_dataset, val_dataset, batch_size=CONFIG['batch_size']):
    """Create enhanced data loaders with advanced sampling for SMC"""
    
    # Enhanced class balancing for SMC zones
    train_targets = train_dataset.y.numpy()
    class_counts = np.bincount(train_targets)
    
    # Calculate enhanced class weights for SMC
    total_samples = len(train_targets)
    class_weights = total_samples / (len(class_counts) * class_counts + 1e-8)
    
    # Apply SMC-specific importance weighting
    smc_importance = {
        0: 2.5,  # SOS - Most important
        1: 2.0,  # SZ
        2: 1.5,  # S
        3: 0.8,  # N - Less important (neutral)
        4: 1.5,  # B
        5: 2.0,  # BZ
        6: 2.5   # BOS - Most important
    }
    
    for i, weight in enumerate(class_weights):
        class_weights[i] = weight * smc_importance.get(i, 1.0)
    
    sample_weights = class_weights[train_targets]
    
    # Enhanced confidence weighting
    confidence_weights = train_dataset.conf.numpy()
    confidence_weights = (confidence_weights - confidence_weights.min()) / (confidence_weights.max() - confidence_weights.min() + 1e-8)
    confidence_weights = 0.5 + 1.0 * confidence_weights  # Scale to [0.5, 1.5]
    
    # Combine weights
    final_weights = sample_weights * confidence_weights
    
    # Create weighted sampler
    weighted_sampler = WeightedRandomSampler(
        weights=final_weights, 
        num_samples=len(final_weights), 
        replacement=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=weighted_sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    logger.info(f"🚀 Enhanced SMC data loaders created")
    logger.info(f"📊 SMC zone distribution: {class_counts}")
    logger.info(f"⚖️ Enhanced class weights: {class_weights}")
    logger.info(f"📈 Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    return train_loader, val_loader

def load_smc_data():
    """Load SMC data with enhanced validation"""
    try:
        labels_path = LABELS_DIR / "BTCUSDT_smc_labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"SMC labels file not found at {labels_path}")
        
        logger.info(f"📂 Loading SMC data from {labels_path}")
        df = pd.read_csv(labels_path)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"✅ Loaded {len(df)} SMC samples")
        
        # Enhanced SMC validation
        if 'smc_zone' not in df.columns:
            raise ValueError("SMC zone column not found in dataset")
        
        zone_dist = df['smc_zone'].value_counts().sort_index()
        logger.info(f"📊 SMC zone distribution: {zone_dist.to_dict()}")
        
        # Check for sufficient samples per zone
        min_samples = zone_dist.min()
        if min_samples < 30:
            logger.warning(f"⚠️ Low sample count for some SMC zones: {min_samples}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error loading SMC data: {str(e)}")
        raise

def train_enhanced_smc_model(df):
    """Enhanced SMC training pipeline for 90%+ accuracy"""
    try:
        logger.info(f"🚀 Starting ENHANCED SMC training pipeline for 90%+ accuracy")
        logger.info(f"📊 Dataset shape: {df.shape}")
        
        # Enhanced data validation
        if df['smc_zone'].nunique() < 3:
            raise ValueError("Need at least 3 SMC zone classes")
        
        # Check zone balance
        zone_counts = df['smc_zone'].value_counts()
        zone_ratio = zone_counts.min() / zone_counts.max()
        logger.info(f"📈 SMC zone balance ratio: {zone_ratio:.3f}")
        
        if zone_ratio < 0.1:
            logger.warning("⚠️ Highly imbalanced SMC zones - applying enhanced balancing")
        
        # Optimized train/validation split (85/15)
        split_idx = int(len(df) * 0.85)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        logger.info(f"🏋️ Train split: {len(train_df)} samples")
        logger.info(f"🎯 Val split: {len(val_df)} samples")
        logger.info(f"📊 Train SMC zones: {train_df['smc_zone'].value_counts().to_dict()}")
        logger.info(f"📊 Val SMC zones: {val_df['smc_zone'].value_counts().to_dict()}")
        
        # Create enhanced datasets
        train_dataset = EnhancedSMCDataset(train_df, is_training=True)
        val_dataset = EnhancedSMCDataset(val_df, is_training=False, scaler=train_dataset.scaler, features=train_dataset.features)
        
        # Create data loaders
        train_loader, val_loader = create_enhanced_data_loaders(train_dataset, val_dataset)
        
        # Initialize enhanced model
        input_size = len(train_dataset.features)
        model = EnhancedSMCModel(input_size=input_size).to(DEVICE)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"🧠 Enhanced SMC model - Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Enhanced optimizer
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=CONFIG['learning_rate'], 
            weight_decay=CONFIG['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=True
        )
        
        # Enhanced learning rate scheduling
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=CONFIG['warmup_epochs']
        )
        
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=40, T_mult=2, eta_min=CONFIG['min_lr']
        )
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[CONFIG['warmup_epochs']]
        )
        
        # Enhanced loss function
        criterion = CombinedSMCLoss(focal_weight=0.8, ce_weight=0.2, aux_weight=0.1)
        
        # Training tracking
        best_val_acc = 0
        best_val_f1 = 0
        epochs_no_improve = 0
        training_history = {
            'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': []
        }
        
        logger.info(f"🎯 TARGET: {CONFIG['target_accuracy']:.1%} SMC validation accuracy")
        logger.info(f"🚀 Starting enhanced SMC training for up to {CONFIG['num_epochs']} epochs...")
        
        for epoch in range(CONFIG['num_epochs']):
            try:
                # Training phase
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_idx, (X_batch, y_batch, conf_batch) in enumerate(train_loader):
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)
                    conf_batch = conf_batch.to(DEVICE)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass with auxiliary output
                    main_outputs, aux_outputs = model(X_batch, return_aux=True)
                    
                    # Combined loss
                    loss = criterion(main_outputs, y_batch, conf_batch, aux_outputs)
                    loss.backward()
                    
                    # Enhanced gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CONFIG['gradient_clip'])
                    
                    optimizer.step()
                    
                    train_loss += loss.item() * X_batch.size(0)
                    _, predicted = torch.max(main_outputs.data, 1)
                    train_total += y_batch.size(0)
                    train_correct += (predicted == y_batch).sum().item()
                
                train_acc = train_correct / train_total
                avg_train_loss = train_loss / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                all_preds = []
                all_targets = []
                
                with torch.no_grad():
                    for X_batch, y_batch, conf_batch in val_loader:
                        X_batch = X_batch.to(DEVICE)
                        y_batch = y_batch.to(DEVICE)
                        conf_batch = conf_batch.to(DEVICE)
                        
                        main_outputs, aux_outputs = model(X_batch, return_aux=True)
                        loss = criterion(main_outputs, y_batch, conf_batch, aux_outputs)
                        
                        val_loss += loss.item() * X_batch.size(0)
                        _, predicted = torch.max(main_outputs.data, 1)
                        
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()
                        
                        all_preds.extend(predicted.cpu().numpy())
                        all_targets.extend(y_batch.cpu().numpy())
                
                val_acc = val_correct / val_total
                val_f1 = f1_score(all_targets, all_preds, average='weighted')
                avg_val_loss = val_loss / val_total
                
                # Update learning rate
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                
                # Update history
                training_history['train_loss'].append(avg_train_loss)
                training_history['train_acc'].append(train_acc)
                training_history['val_loss'].append(avg_val_loss)
                training_history['val_acc'].append(val_acc)
                training_history['val_f1'].append(val_f1)
                training_history['learning_rates'].append(current_lr)
                
                # Progress logging
                if epoch % 5 == 0 or val_acc > best_val_acc:
                    logger.info(f"[Epoch {epoch+1:3d}/{CONFIG['num_epochs']}] "
                               f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                               f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                               f"Val F1: {val_f1:.4f} | LR: {current_lr:.2e}")
                
                # Achievement notifications
                if val_acc >= 0.95:
                    logger.info("🏆 95%+ SMC ACCURACY ACHIEVED! OUTSTANDING! 🎉")
                elif val_acc >= 0.92:
                    logger.info("🔥 92%+ SMC accuracy! Excellent performance!")
                elif val_acc >= 0.90:
                    logger.info("🎯 90%+ SMC ACCURACY ACHIEVED! TARGET MET! ✅")
                elif val_acc >= 0.85:
                    logger.info("📈 85%+ SMC accuracy! Very close to target!")
                elif val_acc >= 0.80:
                    logger.info("✨ 80%+ SMC accuracy! Good progress!")
                
                # Enhanced model saving
                combined_metric = val_acc * 0.8 + val_f1 * 0.2
                best_combined = best_val_acc * 0.8 + best_val_f1 * 0.2
                
                if combined_metric > best_combined:
                    best_val_acc = val_acc
                    best_val_f1 = val_f1
                    epochs_no_improve = 0
                    
                    # Remove all previous model files
                    for old_file in MODEL_DIR.glob("*.pt"):
                        try:
                            os.remove(old_file)
                        except Exception as e:
                            logger.warning(f"Could not remove old model {old_file}: {e}")
                    
                    # Save new best model
                    best_model_path = MODEL_DIR / "best_enhanced_smc_model.pt"
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler': train_dataset.scaler,
                        'features': train_dataset.features,
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'config': CONFIG,
                        'training_history': training_history,
                        'epoch': epoch,
                        'input_size': input_size,
                        'achievement': 'smc_90_plus_target' if val_acc >= 0.90 else 'smc_in_progress',
                        'multi_timeframe_smc': True
                    }, best_model_path)
                    
                    logger.info(f"💾 NEW BEST SMC MODEL SAVED: {best_model_path}")
                    logger.info(f"📊 Model Metrics - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                    
                    # Check if 90% target reached
                    if val_acc >= CONFIG['target_accuracy']:
                        logger.info(f"🎯 90%+ SMC TARGET ACHIEVED! Validation accuracy: {val_acc:.4f}")
                        logger.info("🚀 Enhanced SMC model with multi-timeframe alignment ready!")
                        
                        # Detailed evaluation
                        logger.info("\n" + "="*70)
                        logger.info("🏆 ENHANCED SMC MODEL EVALUATION - 90%+ ACHIEVED!")
                        logger.info(f"✅ SMC Validation Accuracy: {val_acc:.4f}")
                        logger.info(f"✅ SMC Validation F1-Score: {val_f1:.4f}")
                        logger.info(f"✅ Training Epochs: {epoch+1}")
                        logger.info(f"✅ Multi-timeframe Features: {len([f for f in train_dataset.features if any(tf in f for tf in ['15m_', '1h_', '4h_'])])}")
                        
                        # SMC Classification report
                        zone_names = ['SOS', 'SZ', 'S', 'N', 'B', 'BZ', 'BOS']
                        report = classification_report(all_targets, all_preds, 
                                                     target_names=zone_names, 
                                                     digits=4)
                        logger.info(f"📊 SMC Classification Report:\n{report}")
                        
                        # Confusion matrix
                        cm = confusion_matrix(all_targets, all_preds)
                        logger.info(f"🔍 SMC Confusion Matrix:\n{cm}")
                        
                        if val_acc >= 0.95:
                            logger.info("🎊 EXCEPTIONAL: 95%+ SMC accuracy achieved! 🏆")
                        elif val_acc >= 0.92:
                            logger.info("🔥 OUTSTANDING: 92%+ SMC accuracy achieved! 🥇")
                        else:
                            logger.info("🎯 EXCELLENT: 90%+ SMC target achieved! ✅")
                        
                        break
                        
                else:
                    epochs_no_improve += 1
                    
                # Enhanced early stopping
                if epochs_no_improve >= CONFIG['patience']:
                    logger.info(f"⏹️ Early stopping after {CONFIG['patience']} epochs without improvement")
                    logger.info(f"🏅 Best SMC validation accuracy achieved: {best_val_acc:.4f}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error in epoch {epoch+1}: {str(e)}")
                continue
        
        # Final comprehensive evaluation
        logger.info("\n" + "="*80)
        logger.info("🏆 ENHANCED SMC TRAINING COMPLETED WITH MULTI-TIMEFRAME ALIGNMENT")
        logger.info(f"🎯 Target Accuracy: {CONFIG['target_accuracy']:.1%}")
        logger.info(f"🏅 Best Achieved: {best_val_acc:.4f}")
        logger.info(f"🎖️ Best F1-Score: {best_val_f1:.4f}")
        
        if best_val_acc >= CONFIG['target_accuracy']:
            logger.info("✅ SUCCESS: 90%+ SMC accuracy target achieved!")
            logger.info("🚀 Enhanced SMC model ready for production deployment")
        else:
            gap = CONFIG['target_accuracy'] - best_val_acc
            logger.info(f"📈 Close! Gap to SMC target: {gap:.4f}")
            logger.info("💡 Suggestions: Extended training, more data, or ensemble methods")
        
        # Training statistics
        logger.info(f"📊 Enhanced SMC Training Statistics:")
        logger.info(f"   • Total epochs: {len(training_history['val_acc'])}")
        logger.info(f"   • Best epoch: {np.argmax(training_history['val_acc']) + 1}")
        logger.info(f"   • Final learning rate: {training_history['learning_rates'][-1]:.2e}")
        logger.info(f"   • Best combined metric: {best_val_acc * 0.8 + best_val_f1 * 0.2:.4f}")
        logger.info(f"   • Multi-timeframe features used: {len([f for f in train_dataset.features if any(tf in f for tf in ['15m_', '1h_', '4h_'])])}")
        
        return model, train_dataset.scaler, train_dataset.features, best_val_acc, best_val_f1
        
    except Exception as e:
        logger.error(f"❌ Enhanced SMC training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def main():
    """Enhanced main execution function for 90%+ SMC accuracy"""
    try:
        logger.info("🚀 ENHANCED SMC CLASSIFIER WITH MULTI-TIMEFRAME ALIGNMENT FOR 90%+ ACCURACY")
        logger.info("="*85)
        logger.info(f"🎯 Target: {CONFIG['target_accuracy']:.1%} SMC accuracy")
        logger.info(f"🔧 Device: {DEVICE}")
        logger.info(f"⏰ Multi-timeframe: 15m + 1h + 4h alignment")
        logger.info(f"📊 Enhanced SMC Configuration:")
        for key, value in CONFIG.items():
            logger.info(f"   • {key}: {value}")
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # Load and validate SMC data
        df = load_smc_data()
        
        # Enhanced data validation
        if len(df) < CONFIG['sequence_length'] * 15:
            raise ValueError(f"Insufficient SMC data: need at least {CONFIG['sequence_length'] * 15} samples")
        
        # Check SMC zone balance
        zone_dist = df['smc_zone'].value_counts()
        min_zone_ratio = zone_dist.min() / zone_dist.max()
        if min_zone_ratio < 0.05:
            logger.warning(f"⚠️ Severely imbalanced SMC zones: ratio = {min_zone_ratio:.3f}")
            logger.info("📊 Consider collecting more balanced SMC zone data")
        
        # Train the enhanced SMC model
        model, scaler, features, final_accuracy, final_f1 = train_enhanced_smc_model(df)
        
        # Count multi-timeframe features
        timeframe_features = {}
        for tf in ["15m", "1h", "4h"]:
            tf_features = [f for f in features if f.startswith(f'{tf}_')]
            timeframe_features[tf] = len(tf_features)
        other_features = len([f for f in features if not any(f.startswith(f'{tf}_') for tf in ["15m", "1h", "4h"])])
        timeframe_features['other'] = other_features
        
        # Save final production model
        production_model_path = MODEL_DIR / "enhanced_smc_production_90plus.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'features': features,
            'accuracy': final_accuracy,
            'f1_score': final_f1,
            'config': CONFIG,
            'training_date': datetime.now().isoformat(),
            'data_shape': df.shape,
            'smc_zone_distribution': df['smc_zone'].value_counts().to_dict(),
            'model_type': 'EnhancedSMCModel',
            'achievement': 'smc_90_plus_achieved' if final_accuracy >= CONFIG['target_accuracy'] else 'smc_high_performance',
            'feature_count': len(features),
            'timeframe_features': timeframe_features,
            'model_params': sum(p.numel() for p in model.parameters()),
            'multi_timeframe_alignment': True,
            'smc_zones': ['SOS', 'SZ', 'S', 'N', 'B', 'BZ', 'BOS']
        }, production_model_path)
        
        logger.info("\n" + "="*80)
        logger.info("🎉 ENHANCED SMC TRAINING PIPELINE COMPLETED!")
        logger.info(f"📁 Production model saved: {production_model_path}")
        logger.info(f"🏅 Final SMC accuracy: {final_accuracy:.4f}")
        logger.info(f"🎖️ Final SMC F1-score: {final_f1:.4f}")
        logger.info(f"📈 Total features used: {len(features)}")
        logger.info(f"📊 Timeframe feature distribution: {timeframe_features}")
        
        if final_accuracy >= CONFIG['target_accuracy']:
            logger.info("🏆 CONGRATULATIONS! 90%+ SMC accuracy target achieved!")
            logger.info("🚀 Enhanced SMC model ready for live Smart Money Concepts analysis")
            
            if final_accuracy >= 0.95:
                logger.info("🎊 EXCEPTIONAL SMC PERFORMANCE: 95%+ accuracy! 🎊")
            elif final_accuracy >= 0.92:
                logger.info("🔥 OUTSTANDING SMC PERFORMANCE: 92%+ accuracy! 🔥")
            else:
                logger.info("🎯 EXCELLENT SMC PERFORMANCE: 90%+ accuracy achieved! 🎯")
        else:
            logger.info("📊 Enhanced SMC model performance summary:")
            logger.info(f"   • Achieved: {final_accuracy:.1%}")
            logger.info(f"   • Target: {CONFIG['target_accuracy']:.1%}")
            logger.info(f"   • Gap: {(CONFIG['target_accuracy'] - final_accuracy):.1%}")
            logger.info("💡 Consider ensemble training for even higher SMC accuracy")
        
        # SMC zone analysis
        logger.info(f"\n📈 SMC Zone Analysis Summary:")
        logger.info(f"   • SOS/BOS (Strong Signals): High priority zones")
        logger.info(f"   • SZ/BZ (Strong Zones): Important liquidity zones")
        logger.info(f"   • S/B (Basic Zones): Standard support/resistance")
        logger.info(f"   • N (Neutral): Consolidation/ranging areas")
        logger.info(f"   • Multi-timeframe alignment: Enhanced signal strength")
        
        return model, scaler, features, final_accuracy, final_f1
        
    except Exception as e:
        logger.error(f"❌ Enhanced SMC training pipeline failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def load_and_predict_smc(model_path, new_data):
    """Load enhanced SMC model and make predictions"""
    try:
        logger.info(f"📂 Loading enhanced SMC model from {model_path}")
        
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Initialize model
        input_size = checkpoint['input_size']
        model = EnhancedSMCModel(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        # Load scaler and features
        scaler = checkpoint['scaler']
        features = checkpoint['features']
        
        logger.info(f"✅ Enhanced SMC model loaded successfully")
        logger.info(f"📊 Model accuracy: {checkpoint['accuracy']:.4f}")
        logger.info(f"🎯 Using {len(features)} features")
        
        # Check for multi-timeframe features
        timeframe_counts = {}
        for tf in ["15m", "1h", "4h"]:
            tf_count = len([f for f in features if f.startswith(f'{tf}_')])
            timeframe_counts[tf] = tf_count
        logger.info(f"⏰ Timeframe features: {timeframe_counts}")
        
        # Make SMC predictions
        with torch.no_grad():
            # Process new data
            processor = EnhancedSMCDataProcessor()
            processed_data = processor.engineer_multi_timeframe_features(new_data)
            processed_data = processor.clean_and_validate_data(processed_data)
            
            # Scale features
            feature_data = processed_data[features].fillna(0)
            scaled_features = scaler.transform(feature_data)
            
            # Create sequences
            sequences = []
            for i in range(len(scaled_features) - CONFIG['sequence_length']):
                sequence = scaled_features[i:i+CONFIG['sequence_length']]
                sequences.append(sequence)
            
            if sequences:
                X = torch.tensor(np.array(sequences), dtype=torch.float32).to(DEVICE)
                outputs = model(X)
                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Convert predictions back to SMC zone names
                zone_names = ['SOS', 'SZ', 'S', 'N', 'B', 'BZ', 'BOS']
                predicted_zones = [zone_names[pred] for pred in predictions.cpu().numpy()]
                
                return predicted_zones, probabilities.cpu().numpy()
            else:
                logger.warning("⚠️ Not enough data to create sequences for SMC prediction")
                return None, None
        
    except Exception as e:
        logger.error(f"❌ Error loading SMC model or making predictions: {e}")
        raise

def compare_models_performance():
    """Compare performance between original and enhanced SMC models"""
    logger.info("\n" + "="*60)
    logger.info("📊 MODEL PERFORMANCE COMPARISON")
    logger.info("="*60)
    logger.info("Original SMC Model:")
    logger.info("   • Architecture: Basic LSTM")
    logger.info("   • Features: Limited technical indicators")
    logger.info("   • Accuracy: ~56% (as shown in your results)")
    logger.info("   • Timeframe: Single timeframe focus")
    logger.info("")
    logger.info("Enhanced SMC Model:")
    logger.info("   • Architecture: Multi-timeframe LSTM with attention")
    logger.info("   • Features: 150+ engineered features across 15m/1h/4h")
    logger.info("   • Target: 90%+ accuracy")
    logger.info("   • Timeframe: Multi-timeframe alignment")
    logger.info("   • SMC-specific: Zone-aware pathways and loss functions")
    logger.info("   • Enhancements: Focal loss, confidence weighting, data augmentation")

if __name__ == "__main__":
    try:
        # Run the enhanced SMC model training
        final_accuracy = main()
        
        logger.info(f"\n✅ Enhanced SMC model training completed!")
        logger.info(f"🏆 Final accuracy: {final_accuracy[3]:.4f}")
        
        if final_accuracy[3] >= 0.90:
            logger.info("🎯 SUCCESS: 90%+ SMC accuracy achieved!")
            logger.info("🚀 Ready for Smart Money Concepts analysis!")
        else:
            logger.info(f"📊 Achieved {final_accuracy[3]:.1%} SMC accuracy")
            logger.info("💡 Consider additional training or ensemble methods")
        
        # Show comparison
        compare_models_performance()
        
    except Exception as e:
        logger.error(f"❌ Enhanced SMC training error: {str(e)}")
        import traceback
        traceback.print_exc()