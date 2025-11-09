import os
import requests
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3.exceptions

BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
TIMEFRAMES = ["15m", "1h", "4h", "1d"]
SYMBOL = "BTCUSDT"  # Using USDT-margined futures
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 1, 1)

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"[INFO] Saving files to: {SAVE_DIR}")
print(f"[INFO] Date range: {START_DATE.strftime('%Y-%m-%d')} to {END_DATE.strftime('%Y-%m-%d')}")
print(f"[INFO] Timeframes: {', '.join(TIMEFRAMES)}")

def get_date_range(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

def download_kline_zip(symbol, interval, date_obj):
    date_str = date_obj.strftime("%Y-%m-%d")
    month_str = date_obj.strftime("%Y-%m")
    year_month_day = date_obj.strftime("%Y-%m-%d")
    
    # Adjust URL structure for Binance data
    if interval == "1d":
        zip_name = f"{symbol}-{interval}-{date_str}.zip"
        url = f"https://data.binance.vision/data/futures/um/daily/klines/{symbol}/{interval}/{zip_name}"
    else:
        zip_name = f"{symbol}-{interval}-{month_str}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{zip_name}"

    save_path = os.path.join(SAVE_DIR, zip_name)
    extract_path = os.path.join(SAVE_DIR, f"{symbol}-{interval}-{year_month_day}.csv")

    # Skip if target day CSV already extracted
    if os.path.exists(extract_path):
        print(f"⏭️ Skipping {zip_name} for {year_month_day} (already exists)")
        return True

    try:
        # Create a session with retry logic and custom headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        retry = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        print(f"📥 Downloading {zip_name} for {year_month_day}...")
        try:
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()  # Raise exception for bad status codes
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        f.flush()  # Ensure data is written immediately

            with zipfile.ZipFile(save_path, 'r') as zip_ref:
                zip_ref.extractall(SAVE_DIR)

            # Filter only the rows of the target day (for <1d timeframes)
            csv_name = zip_name.replace(".zip", ".csv")
            full_csv_path = os.path.join(SAVE_DIR, csv_name)

            if interval != "1d":
                # Read CSV with proper error handling
                try:
                    df = pd.read_csv(full_csv_path, header=None)
                    
                    # Check if we have data
                    if df.empty:
                        print(f"⚠️ Empty CSV file: {csv_name}")
                        return False
                    
                    # Define column names
                    df.columns = [
                        "open_time", "open", "high", "low", "close",
                        "volume", "close_time", "quote_asset_volume",
                        "number_of_trades", "taker_buy_base_volume",
                        "taker_buy_quote_volume", "ignore"
                    ]
                    
                    # Convert open_time to datetime with error handling
                    try:
                        # First, try to convert to numeric (handle any string values)
                        df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
                        # Remove any NaN values that couldn't be converted
                        df = df.dropna(subset=['open_time'])
                        # Convert to datetime
                        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    except Exception as e:
                        print(f"⚠️ Error converting timestamps in {csv_name}: {str(e)}")
                        # Try alternative approach - maybe the timestamps are already in datetime format
                        try:
                            df['open_time'] = pd.to_datetime(df['open_time'])
                        except:
                            print(f"❌ Could not parse timestamps in {csv_name}")
                            return False
                    
                    # Filter for the target date
                    filtered = df[df['open_time'].dt.date == date_obj.date()]
                    
                    if filtered.empty:
                        print(f"⚠️ No data found for {year_month_day} in {csv_name}")
                        return False
                    
                    filtered.to_csv(extract_path, index=False)
                    
                except Exception as e:
                    print(f"⚠️ Error processing CSV {csv_name}: {str(e)}")
                    return False
            else:
                # For 1d timeframe, just rename the file
                os.rename(full_csv_path, extract_path)

            # Clean up original big file
            if os.path.exists(save_path):
                os.remove(save_path)
            if os.path.exists(full_csv_path) and full_csv_path != extract_path:
                os.remove(full_csv_path)

            print(f"✅ Downloaded and saved: {extract_path}")
            return True

        except (requests.exceptions.RequestException, urllib3.exceptions.ProtocolError) as e:
            print(f"❌ Error downloading {zip_name}: {str(e)}")
            return False

    except Exception as e:
        print(f"⚠️ Error processing {zip_name}: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)
        return False

    finally:
        # Add small delay between requests to avoid rate limiting
        time.sleep(1)

if __name__ == "__main__":
    print(f"\n📊 Starting multi-timeframe aligned download for {SYMBOL}")
    failed_downloads = []

    # Try downloading each timeframe in sequence
    for interval in TIMEFRAMES:
        print(f"\n🔄 Processing {interval} timeframe...")
        for current_date in tqdm(list(get_date_range(START_DATE, END_DATE)), desc="📅 Dates"):
            try:
                if not download_kline_zip(SYMBOL, interval, current_date):
                    failed_downloads.append((interval, current_date.strftime("%Y-%m-%d")))
            except Exception as e:
                print(f"❌ Error downloading {interval} {current_date.strftime('%Y-%m-%d')}: {str(e)}")
                failed_downloads.append((interval, current_date.strftime("%Y-%m-%d")))

    if failed_downloads:
        print("\n❌ Some downloads failed:")
        for fail in failed_downloads:
            print("  -", fail)
    else:
        print("\n✅ All downloads completed successfully.")