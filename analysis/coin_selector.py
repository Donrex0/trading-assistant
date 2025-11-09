# analysis/coin_selector.py

import requests
import pandas as pd
import time

COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"

def fetch_market_data():
    """Fetch market data from CoinGecko"""
    try:
        params = {
            'vs_currency': 'usd',
            'order': 'volume_desc',
            'per_page': 200,
            'page': 1,
            'sparkline': 'false'
        }
        
        response = requests.get(COINGECKO_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to fetch data from CoinGecko: {str(e)}")
        return []

def filter_top_10_coins(data, min_volume_usdt=10_000_000):
    """Filter top 10 volatile coins with high volume"""
    df = pd.DataFrame(data)
    
    # Filter for volume and price
    df = df[df['total_volume'] > min_volume_usdt]
    df = df[df['current_price'] > 1.0]  # Filter out very low price coins

    # Add volatility metric: (24h high - 24h low) / 24h low
    df['volatility'] = (df['high_24h'] - df['low_24h']) / df['low_24h']
    
    # Sort by volatility
    df = df.sort_values(by='volatility', ascending=False)

    # Pick top 10
    top_10 = df.head(10).copy()

    # Select useful columns
    top_10 = top_10[[
        'symbol', 'current_price', 'price_change_percentage_24h', 'total_volume', 'volatility'
    ]]

    # Rename columns to match previous format
    top_10.columns = [
        'symbol', 'lastPrice', 'priceChangePercent', 'quoteVolume', 'volatility'
    ]

    return top_10

def get_supported_coins():
    """Get list of supported trading pairs from Binance"""
    try:
        import ccxt
        exchange = ccxt.binance()
        exchange.load_markets()
        # Return list of USDT trading pairs
        return [market['symbol'] for market in exchange.markets.values() 
                if market['quote'] == 'USDT' and market['active']]
    except Exception as e:
        print(f"[Error] Failed to fetch supported coins: {e}")
        return []

def get_top_10_coins():
    try:
        # Get supported trading pairs first
        supported_pairs = get_supported_coins()
        
        # Fetch market data
        data = fetch_market_data()
        if data is None or data.empty:
            return pd.DataFrame()
            
        # Filter top coins
        top_coins = filter_top_10_coins(data)
        
        if top_coins.empty:
            return top_coins
            
        # Filter out unsupported pairs
        supported_coins = []
        for _, coin in top_coins.iterrows():
            symbol = coin['symbol']
            usdt_pair = f"{symbol}/USDT"
            if usdt_pair in supported_pairs:
                supported_coins.append(coin)
                
        if not supported_coins:
            print("[Warning] No supported trading pairs found in top coins")
            return pd.DataFrame()
            
        return pd.DataFrame(supported_coins)
        
    except Exception as e:
        print(f"[Error] {e}")
        return pd.DataFrame()
