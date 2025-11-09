import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from typing import Dict, Optional
from pathlib import Path

# Configuration
SYMBOL = "BTCUSDT"
BASE_TF = "15m"  # Base timeframe
HIGHER_TFS = ["1h", "4h"]  # Higher timeframes to merge

# Configure logging
def setup_logging():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(ch)
    return logger

# Configure logging
logger = setup_logging()

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LABELS_DIR = BASE_DIR / "data" / "labels"

# Ensure labels directory exists
LABELS_DIR.mkdir(parents=True, exist_ok=True)

class SMCLabeler:
    """
    Rule-based labeler for Smart Money Channel (SMC) zones
    
    Uses advanced pattern recognition and technical indicators to identify:
    - BOS (Breakout Start)
    - BZ (Breakout Zone)
    - B (Bullish Zone)
    - N (Neutral Zone)
    - S (Bearish Zone)
    - SZ (Sell Zone)
    - SOS (Sell Out Start)
    """
    
    def __init__(self, window_size: int = 10, consolidation_threshold: float = 0.05):
        """
        Initialize SMC labeler with parameters
        
        Args:
            window_size: Number of periods to look back for SMC analysis
            consolidation_threshold: Threshold for consolidation detection
        """
        if window_size < 5:
            raise ValueError("window_size must be at least 5")
        if consolidation_threshold < 0 or consolidation_threshold > 1:
            raise ValueError("consolidation_threshold must be between 0 and 1")
            
        self.window_size = window_size
        self.consolidation_threshold = consolidation_threshold
        self.zones = {
            'BOS': 'Breakout Start',
            'BZ': 'Breakout Zone', 
            'B': 'Bullish Zone',
            'N': 'Neutral Zone',
            'S': 'Bearish Zone',
            'SZ': 'Sell Zone',
            'SOS': 'Sell Out Start'
        }

    def _safe_numeric_conversion(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Safely convert columns to numeric, handling errors gracefully
        """
        for col in columns:
            if col in df.columns:
                # Convert to numeric, replacing errors with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Replace infinite values with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Fill NaN values with appropriate defaults
                if col in ['open', 'high', 'low', 'close']:
                    # For OHLC, use forward fill then backward fill
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                elif col == 'volume':
                    # For volume, use median or 0
                    df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)
                else:
                    # For other columns, use 0
                    df[col] = df[col].fillna(0)
        
        return df

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for SMC analysis
        """
        try:
            if df is None or len(df) == 0:
                raise ValueError("Input DataFrame cannot be None or empty")
                
            # Make a copy to avoid modifying original
            df = df.copy()
            
            # Ensure numeric conversion for OHLCV columns
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            df = self._safe_numeric_conversion(df, ohlcv_cols)
            
            # Check if we already have RSI (from momentum_rsi column)
            if 'momentum_rsi' in df.columns:
                df['rsi'] = pd.to_numeric(df['momentum_rsi'], errors='coerce').fillna(50)
            else:
                # Calculate RSI if not present
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
                
                # Avoid division by zero
                rs = gain / (loss + 1e-10)
                df['rsi'] = (100 - (100 / (1 + rs))).fillna(50)
            
            # Calculate moving averages with minimum periods
            df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
            
            # Calculate volatility
            df['volatility'] = df['close'].rolling(window=10, min_periods=1).std().fillna(0)
            
            # Calculate momentum  
            df['momentum'] = df['close'].diff(5).fillna(0)
            
            # Use existing MACD if available, otherwise calculate
            if 'trend_macd' in df.columns:
                df['macd'] = pd.to_numeric(df['trend_macd'], errors='coerce').fillna(0)
            else:
                ema12 = df['close'].ewm(span=12).mean()
                ema26 = df['close'].ewm(span=26).mean()
                df['macd'] = (ema12 - ema26).fillna(0)
            
            # Calculate volume indicators
            df['vol_ma10'] = df['volume'].rolling(window=10, min_periods=1).mean()
            df['vol_ratio'] = (df['volume'] / (df['vol_ma10'] + 1e-10)).fillna(1)
            
            # Ensure all calculated indicators are numeric and finite
            indicator_cols = ['rsi', 'ma5', 'ma20', 'volatility', 'momentum', 'macd', 'vol_ma10', 'vol_ratio']
            df = self._safe_numeric_conversion(df, indicator_cols)
            
            # Log the calculation results
            logger.debug(f"Calculated indicators for {len(df)} rows")
            logger.debug(f"RSI range: {df['rsi'].min():.2f} - {df['rsi'].max():.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            raise

    def _detect_consolidation(self, df: pd.DataFrame, idx: int) -> bool:
        """
        Detect consolidation pattern
        """
        if idx < self.window_size:
            return False
            
        try:
            start_idx = max(0, idx - self.window_size)
            window = df.iloc[start_idx:idx]
            
            if len(window) < 3:
                return False
                
            price_range = window['high'].max() - window['low'].min()
            avg_price = window['close'].mean()
            
            # Consolidation if price range is small relative to average price
            return price_range < avg_price * 0.02  # 2% range
            
        except Exception as e:
            logger.debug(f"Error detecting consolidation at index {idx}: {str(e)}")
            return False

    def label_smc_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label SMC zones using pattern recognition
        """
        try:
            if df is None or len(df) == 0:
                raise ValueError("Input DataFrame cannot be None or empty")
                
            # Make a copy
            df = df.copy()
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Validate we have required columns
            required_cols = ['close', 'high', 'low', 'rsi', 'ma5', 'ma20']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Initialize smc_zone column with default 'N'
            df['smc_zone'] = 'N'
            
            # Only process if we have enough data
            if len(df) <= self.window_size:
                logger.warning(f"Not enough data points ({len(df)}) for window size {self.window_size}")
                return df
            
            # Process each row starting from window_size
            zones_assigned = 0
            
            for i in range(self.window_size, len(df)):
                try:
                    # Get current values safely
                    current_close = float(df.iloc[i]['close'])
                    current_rsi = float(df.iloc[i]['rsi'])
                    current_ma5 = float(df.iloc[i]['ma5']) 
                    current_ma20 = float(df.iloc[i]['ma20'])
                    
                    # Get window data
                    start_idx = max(0, i - self.window_size)
                    window = df.iloc[start_idx:i]
                    
                    if len(window) == 0:
                        continue
                        
                    window_high = float(window['high'].max())
                    window_low = float(window['low'].min())
                    prev_close = float(window['close'].iloc[-1]) if len(window) > 0 else current_close
                    
                    # Calculate momentum
                    momentum = current_close - prev_close
                    momentum_pct = momentum / prev_close if prev_close != 0 else 0
                    
                    # Default to neutral
                    zone = 'N'
                    
                    # Strong bullish conditions
                    if (current_close > window_high and 
                        current_rsi > 60 and 
                        momentum_pct > 0.005):  # 0.5% momentum
                        zone = 'BOS'
                    elif (current_close > current_ma20 * 1.01 and 
                          current_rsi > 55 and 
                          momentum_pct > 0.002):  # 1% above MA20, 0.2% momentum
                        zone = 'BZ'
                    elif (current_close > current_ma5 and 
                          current_rsi > 50 and 
                          momentum_pct > 0.001):  # Above MA5, positive momentum
                        zone = 'B'
                    
                    # Strong bearish conditions  
                    elif (current_close < window_low and 
                          current_rsi < 40 and 
                          momentum_pct < -0.005):  # -0.5% momentum
                        zone = 'SOS'
                    elif (current_close < current_ma20 * 0.99 and 
                          current_rsi < 45 and 
                          momentum_pct < -0.002):  # 1% below MA20, negative momentum
                        zone = 'SZ'
                    elif (current_close < current_ma5 and 
                          current_rsi < 50 and 
                          momentum_pct < -0.001):  # Below MA5, negative momentum
                        zone = 'S'
                    
                    # Consolidation (neutral)
                    elif (abs(momentum_pct) < 0.001 and  # Very small movement
                          40 < current_rsi < 60 and  # Neutral RSI
                          self._detect_consolidation(df, i)):
                        zone = 'N'
                    
                    # Assign the zone
                    df.at[df.index[i], 'smc_zone'] = zone
                    if zone != 'N':
                        zones_assigned += 1
                        
                except Exception as e:
                    logger.debug(f"Error processing row {i}: {str(e)}")
                    df.at[df.index[i], 'smc_zone'] = 'N'
                    continue
            
            # Validate results
            zone_counts = df['smc_zone'].value_counts()
            total_labeled = len(df) - self.window_size
            
            logger.info(f"Labeled {total_labeled} data points with SMC zones")
            logger.info(f"Zone distribution: {zone_counts.to_dict()}")
            logger.info(f"Non-neutral zones assigned: {zones_assigned}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error labeling SMC zones: {str(e)}")
            raise

    def label_data(self) -> pd.DataFrame:
        """
        Label SMC zones in all processed files
        """
        try:
            # Get processed files
            processed_files = list(PROCESSED_DIR.glob("BTCUSDT_15m_processed_*.csv"))
            if not processed_files:
                raise FileNotFoundError("No processed files found")
            
            logger.info(f"Found {len(processed_files)} files to process")
            
            all_labeled = []
            successful_files = 0
            
            for file_path in sorted(processed_files):
                try:
                    logger.info(f"Processing: {file_path.name}")
                    
                    # Load data
                    df = pd.read_csv(file_path)
                    
                    if len(df) == 0:
                        logger.warning(f"Empty file: {file_path.name}")
                        continue
                    
                    # Parse timestamp if it exists
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                    
                    # Label the data
                    df_labeled = self.label_smc_zones(df)
                    
                    if len(df_labeled) > 0:
                        all_labeled.append(df_labeled)
                        successful_files += 1
                        logger.info(f"Successfully labeled: {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {str(e)}")
                    continue
            
            if not all_labeled:
                raise ValueError("No files were successfully processed")
            
            # Combine all data
            df_combined = pd.concat(all_labeled, ignore_index=True)
            
            # Save results
            output_file = LABELS_DIR / "BTCUSDT_smc_labels.csv"
            df_combined.to_csv(output_file, index=False)
            logger.info(f"Combined labels saved to: {output_file}")
            
            # Final statistics
            final_zone_counts = df_combined['smc_zone'].value_counts()
            logger.info(f"Final zone distribution: {final_zone_counts.to_dict()}")
            logger.info(f"Successfully processed {successful_files}/{len(processed_files)} files")
            
            return df_combined
            
        except Exception as e:
            logger.error(f"Error in label_data: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize labeler
    labeler = SMCLabeler(window_size=10, consolidation_threshold=0.02)
    
    try:
        # Label data
        df_labeled = labeler.label_data()
        
        # Print summary
        print("\n[📊] SMC Zone Analysis Summary:")
        print(f"Total labeled points: {len(df_labeled)}")
        print(f"Zone distribution:")
        zone_counts = df_labeled['smc_zone'].value_counts()
        for zone, count in zone_counts.items():
            zone_name = labeler.zones.get(zone, 'Unknown')
            percentage = (count / len(df_labeled)) * 100
            print(f"  {zone}: {zone_name} - {count} ({percentage:.1f}%)")
            
        # Check for any issues
        if zone_counts.get('N', 0) == len(df_labeled):
            print("\n[⚠️] Warning: All zones are neutral. Consider adjusting thresholds.")
        elif len(zone_counts) == 1:
            print(f"\n[⚠️] Warning: Only one zone type detected: {list(zone_counts.keys())[0]}")
        else:
            print(f"\n[✅] Successfully detected {len(zone_counts)} different zone types")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        print(f"\n[❌] Error: {str(e)}")
        raise