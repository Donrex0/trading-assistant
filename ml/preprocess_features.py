import os
import pandas as pd
import numpy as np
import ta
from ta.utils import dropna
from ta import add_all_ta_features
from datetime import datetime, timedelta

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SYMBOL = "BTCUSDT"
BASE_TF = "15m"  # Use 15m data as main granularity
HIGHER_TFS = ["1h", "4h"]  # Add context features from higher timeframes
LABEL_TF = "1d"  # Used only for target
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 12, 31)

os.makedirs(PROCESSED_DIR, exist_ok=True)

print(f"[📁] Raw data from: {RAW_DIR}")
print(f"[📁] Processed data saved to: {PROCESSED_DIR}")

def load_data(symbol, date_str, tf):
    filename = f"{symbol}-{tf}-{date_str}.csv"
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"❌ Missing file: {filename}")
        if tf == LABEL_TF:  # If it's the label file
            print(f"⚠️ Skipping {date_str}: no label found")
            return None
        return None
    try:
        df = pd.read_csv(path)
        # Try different timestamp formats
        if "open_time" in df.columns:
            # Try to convert from timestamp (ms)
            try:
                df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            except:
                # If that fails, try to convert from datetime string
                df["timestamp"] = pd.to_datetime(df["open_time"])
        else:
            # Try to convert from timestamp (ms)
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            except:
                # If that fails, try to convert from datetime string
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df
    except Exception as e:
        print(f"⚠️ Error loading {filename}: {e}")
        return None

def add_indicators(df):
    try:
        # Basic preprocessing
        df = df.copy()
        
        # Add basic indicators
        df['roc'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(10).std()
        df['ma'] = df['close'].rolling(5).mean()
        df['momentum'] = df['close'].diff(4)
        
        # Add TA indicators with error handling
        try:
            # Add basic indicators first
            df['trend_psar_up'] = np.where(
                df['high'] > df['high'].shift(1), df['high'], np.nan
            )
            df['trend_psar_down'] = np.where(
                df['low'] < df['low'].shift(1), df['low'], np.nan
            )
            df['trend_psar_up_indicator'] = df['trend_psar_up'].notna().astype(int)
            df['trend_psar_down_indicator'] = df['trend_psar_down'].notna().astype(int)

            # Add more TA indicators
            df['trend_macd'] = ta.trend.MACD(df['close']).macd()
            df['volatility_bbm'] = ta.volatility.BollingerBands(df['close']).bollinger_mavg()
            df['momentum_rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # Clean up any NaN values
            df = df.ffill().fillna(0)
        except Exception as e:
            print(f"⚠️ Warning: TA library error: {e}")
        except Exception as e:
            print(f"⚠️ Warning: TA library error: {e}")
            # Add basic indicators manually if TA fails
            df['trend_psar_up'] = np.where(
                df['high'] > df['high'].shift(1), df['high'], np.nan
            )
            df['trend_psar_down'] = np.where(
                df['low'] < df['low'].shift(1), df['low'], np.nan
            )
            df['trend_psar_up_indicator'] = df['trend_psar_up'].notna().astype(int)
            df['trend_psar_down_indicator'] = df['trend_psar_down'].notna().astype(int)

        # Clean up
        df = df.dropna()
        return df
        
    except Exception as e:
        print(f"❌ Error adding indicators: {str(e)}")
        return None

def get_date_range(start, end):
    while start <= end:
        current_date = start.strftime("%Y-%m-%d")
        # For the last day of the year, use the same day as label date
        if start.month == 12 and start.day == 31:
            label_date = current_date
        else:
            label_date = (start + timedelta(days=1)).strftime("%Y-%m-%d")
        
        yield current_date, label_date
        start += timedelta(days=1)

if __name__ == "__main__":
    all_data = []
    for date_str, label_date in get_date_range(START_DATE, END_DATE):
        print(f"\n📅 Processing {date_str}...")

        base_df = load_data(SYMBOL, date_str, BASE_TF)
        if base_df is None or len(base_df) < 10:
            print(f"⚠️ Skipping {date_str}: base {BASE_TF} data missing")
            continue

        base_df = add_indicators(base_df)
        if base_df is None or base_df.empty:
            print(f"⚠️ Skipping {date_str}: indicators failed on base {BASE_TF}")
            continue

        # Add higher timeframe features
        for tf in HIGHER_TFS:
            tf_df = load_data(SYMBOL, date_str, tf)
            if tf_df is None or len(tf_df) < 5:
                print(f"⚠️ Skipping {date_str}: {tf} data missing")
                continue
            tf_df = tf_df.add_prefix(f"{tf}_")
            # Forward fill to align with 15m base
            base_df = base_df.merge(tf_df, left_index=True, right_index=True, how="left").ffill()

        # Add label data
        # For the last day of the year, use the same day's 1d data
        if date_str == "2023-12-31":
            label_df = load_data(SYMBOL, date_str, LABEL_TF)
        else:
            label_df = load_data(SYMBOL, label_date, LABEL_TF)
        
        if label_df is not None:
            label_df = label_df.add_prefix(f"{LABEL_TF}_")
            base_df = base_df.merge(label_df, left_index=True, right_index=True, how="left")

        if base_df is None or label_df is None:
            continue

        # Add label from next day's 1d close
        label_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        label_df = load_data(SYMBOL, label_date, LABEL_TF)
        if label_df is not None and not label_df.empty:
            next_close = label_df["close"].iloc[0]
            base_df["target"] = (next_close > base_df["close"]).astype(int)
        else:
            print(f"⚠️ Skipping {date_str}: no label found")
            continue

        # Save
        output_path = os.path.join(PROCESSED_DIR, f"{SYMBOL}_15m_processed_{date_str}.csv")
        base_df.to_csv(output_path)
        print(f"✅ Saved {len(base_df)} rows → {output_path}")

        all_data.append(base_df)

    # Combine all
    if all_data:
        full_df = pd.concat(all_data)
        full_out_path = os.path.join(PROCESSED_DIR, f"{SYMBOL}_final_dataset.csv")
        full_df.to_csv(full_out_path)
        print(f"\n✅ Combined full dataset saved: {full_out_path} ({len(full_df)} rows)")
    else:
        print("❌ No valid daily datasets to combine.")
