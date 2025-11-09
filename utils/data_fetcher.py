"""
Data fetching utilities with built-in caching and performance optimizations.
"""
import os
import time
import pandas as pd
import numpy as np
import ccxt
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Local imports
from utils.cache import cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global exchange instance with rate limiting
global_exchange = None

def get_exchange():
    """Get or create a global exchange instance with rate limiting."""
    global global_exchange
    if global_exchange is None:
        global_exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True,
                'recvWindow': 60000,
            },
            'timeout': 30000,  # 30 seconds
        })
    return global_exchange

@cache.memory_cached(ttl=60)  # Cache for 1 minute
def fetch_ohlcv_df(symbol: str, timeframe: str = '1h', limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for a given symbol with caching.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT' or 'BTCUSDT')
        timeframe: Timeframe for the candles (default: '1h')
        limit: Number of candles to fetch (default: 500)
        
    Returns:
        DataFrame with OHLCV data and technical indicators
        
    Raises:
        ValueError: If the trading pair is not supported
        ccxt.ExchangeError: If there's an issue with the exchange API
    """
    try:
        exchange = get_exchange()
        
        # Generate cache key for this specific request
        cache_key = f"ohlcv_{symbol}_{timeframe}_{limit}"
        
        # Try to get from cache first
        cached_data = cache.memory_cache.get(cache_key)
        current_time = time.time()
        
        # Return cached data if not expired
        if cached_data and cached_data[1] > current_time:
            logger.debug(f"Using cached OHLCV data for {symbol} {timeframe}")
            return cached_data[0]
            
        # Convert symbol to exchange format if needed
        original_symbol = symbol
        if '/' not in symbol:
            if symbol.endswith('USDT'):
                symbol = f"{symbol[:-4]}/USDT"
            else:
                symbol = f"{symbol}/USDT"
        
        # Check if the market exists
        exchange.load_markets()
        if symbol not in exchange.markets:
            # Try with USDT suffix if not already tried
            if not original_symbol.endswith('USDT'):
                usdt_symbol = f"{original_symbol}/USDT"
                if usdt_symbol in exchange.markets:
                    symbol = usdt_symbol
                else:
                    raise ValueError(f"Trading pair {symbol} not found. Available pairs include: {', '.join(list(exchange.markets.keys())[:5])}...")
            else:
                raise ValueError(f"Trading pair {symbol} not found. Available pairs include: {', '.join(list(exchange.markets.keys())[:5])}...")
        
        # Fetch fresh data if not in cache or expired
        logger.info(f"Fetching fresh {timeframe} OHLCV data for {symbol}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv or len(ohlcv) == 0:
            raise ValueError(f"No OHLCV data returned for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert string values to float (vectorized for better performance)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Cache the result
        cache.memory_cache[cache_key] = (df, current_time + 60)  # Cache for 1 minute
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}")
        raise

@cache.memory_cached(ttl=300)  # Cache indicator calculations for 5 minutes
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for the given OHLCV data with optimizations.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional technical indicators
    """
    try:
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Vectorized calculations for better performance
        close = df['close'].values
        
        # Calculate moving averages with min_periods=1 to avoid NaN at the beginning
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # Calculate RSI (vectorized implementation)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD (vectorized)
        exp1 = df['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        exp2 = df['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std() * (252 ** 0.5)
        
        # Calculate momentum
        df['momentum'] = df['close'].pct_change(periods=5)
        
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df


def fetch_multiple_pairs(pairs: List[str], timeframe: str = '1h', limit: int = 100, 
                        max_workers: int = 5) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for multiple trading pairs in parallel.
    
    Args:
        pairs: List of trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        timeframe: Timeframe for the candles (default: '1h')
        limit: Number of candles to fetch per pair (default: 100)
        max_workers: Maximum number of parallel workers (default: 5)
        
    Returns:
        Dictionary mapping symbols to their DataFrames
    """
    results = {}
    
    def fetch_single(pair: str) -> Tuple[str, Optional[pd.DataFrame]]:
        try:
            df = fetch_ohlcv_df(pair, timeframe=timeframe, limit=limit)
            return pair, df
        except Exception as e:
            logger.error(f"Error fetching data for {pair}: {str(e)}")
            return pair, None
    
    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all fetch tasks
        future_to_pair = {
            executor.submit(fetch_single, pair): pair 
            for pair in pairs
        }
        
        # Process completed tasks
        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                pair, result = future.result()
                if result is not None:
                    results[pair] = result
                    logger.info(f"Successfully fetched data for {pair}")
            except Exception as e:
                logger.error(f"Error processing {pair}: {str(e)}")
    
    return results

def test_single_pair():
    """Test fetching data for a single trading pair."""
    try:
        start_time = time.time()
        df = fetch_ohlcv_df("BTC/USDT")
        elapsed = time.time() - start_time
        print(f"Fetched {len(df)} candles in {elapsed:.2f} seconds")
        print(df[['close', 'rsi', 'macd']].tail())
        
        # Test cache hit
        start_time = time.time()
        df_cached = fetch_ohlcv_df("BTC/USDT")
        elapsed = time.time() - start_time
        print(f"\nCache hit test: {elapsed:.4f} seconds")
        
    except Exception as e:
        print(f"Error: {str(e)}")


def test_multiple_pairs():
    """Test fetching data for multiple trading pairs in parallel."""
    pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]
    
    print(f"\n{'='*50}")
    print("Testing parallel fetching of multiple pairs...")
    print(f"Pairs: {', '.join(pairs)}")
    
    start_time = time.time()
    results = fetch_multiple_pairs(pairs, max_workers=3)
    elapsed = time.time() - start_time
    
    print(f"\nFetched data for {len(results)}/{len(pairs)} pairs in {elapsed:.2f} seconds")
    for pair, df in results.items():
        print(f"- {pair}: {len(df)} candles, latest close: {df['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test single pair
    test_single_pair()
    
    # Test multiple pairs
    test_multiple_pairs()
