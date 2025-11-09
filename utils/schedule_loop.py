import schedule
import threading
import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

# Configure logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lazy imports
def _import_ml_dependencies():
    """Lazily import ML dependencies when needed"""
    try:
        from ml.lstm_predictor import predict_multi_timeframe
        from ml.smc_predictor import predict_smc
        from ml.sentiment_model import FinBERTSentimeterAnalyzer
        return predict_multi_timeframe, predict_smc, FinBERTSentimeterAnalyzer
    except ImportError as e:
        logger.warning(f"Could not import ML dependencies: {e}")
        return None, None, None

def _import_strategy_dependencies():
    """Lazily import strategy dependencies when needed"""
    try:
        from strategies.fusion_strategy import FusionStrategy
        from strategies.strategy_runner import run_all_strategies
        return FusionStrategy, run_all_strategies
    except ImportError as e:
        logger.warning(f"Could not import strategy dependencies: {e}")
        return None, None

# Import non-circular dependencies directly
from analysis.coin_selector import get_top_10_coins
from binance.download_and_unzip_binance import download_kline_zip

__all__ = ['start_scheduler', 'top_10_coins', 'smc_analysis_results', 'lstm_predictions', 'sentiment_scores', 'latest_analysis_time']

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to store analysis results
top_10_coins = []
smc_analysis_results = {}
lstm_predictions = {}
sentiment_scores = {}
latest_analysis_time = None

def run_analysis():
    """Run comprehensive analysis cycle across multiple timeframes"""
    global top_10_coins, smc_analysis_results, lstm_predictions, sentiment_scores, latest_analysis_time
    
    # Import ML dependencies only when needed
    predict_multi_timeframe, predict_smc, FinBERTSentimentAnalyzer = _import_ml_dependencies()
    FusionStrategy, run_all_strategies = _import_strategy_dependencies()
    
    if not all([predict_multi_timeframe, predict_smc, FinBERTSentimentAnalyzer, FusionStrategy, run_all_strategies]):
        logger.error("Missing required dependencies for analysis")
        return
    
    try:
        # Get top 10 coins
        top_10_coins = get_top_10_coins()
        if top_10_coins is None or top_10_coins.empty:
            logger.warning("No coins available for analysis")
            return
        
        # Initialize predictors
        sentiment_analyzer = FinBERTSentimentAnalyzer()
        
        # Process each coin
        for index, coin in top_10_coins.iterrows():
            symbol = coin['symbol']
            try:
                # Download and process data for multiple timeframes
                timeframes = ['15m', '1h', '4h', '1d']
                all_dfs = {}
                
                for timeframe in timeframes:
                    try:
                        # Download latest data
                        download_kline_zip(symbol, timeframe, datetime.now().strftime('%Y-%m-%d'))
                    except Exception as e:
                        logger.error(f"Error downloading data for {symbol} {timeframe}: {e}")
                    
                    # Load processed data
                    df_path = f'data/processed/{symbol}_{timeframe}.csv'
                    if os.path.exists(df_path):
                        df = pd.read_csv(df_path, index_col='timestamp', parse_dates=True)
                        all_dfs[timeframe] = df
                    
                # Skip if we don't have enough data
                if not all_dfs:
                    logger.warning(f"[⚠️] Skipping {symbol} - No data available")
                    continue
                
                # Make predictions
                lstm_pred = predict_multi_timeframe(all_dfs)
                smc_pred = predict_smc(symbol)
                
                # Get sentiment analysis
                news_text = get_news_for_symbol(symbol)
                sentiment = sentiment_analyzer.analyze_sentiment(news_text)
                
                # Run fusion strategy
                fusion = FusionStrategy()
                signals = fusion.generate_signals(
                    lstm_pred,
                    smc_pred,
                    sentiment
                )
                
                # Store results
                smc_analysis_results[symbol] = {
                    'smc_prediction': smc_pred,
                    'timestamp': time.time()
                }
                
                lstm_predictions[symbol] = lstm_pred
                sentiment_scores[symbol] = sentiment
                
                # Update progress in UI
                progress = (index + 1) / len(top_10_coins) * 100
                logger.info(f"Analyzing {symbol} ({index+1}/{len(top_10_coins)}) - Progress: {progress}%")
                
                # Small delay between coins to avoid API rate limits
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"[❌] Error processing {symbol}: {str(e)}")
                continue
        
        # Update analysis timestamp
        latest_analysis_time = datetime.now()
        logger.info(f"[✅] Analysis complete. Total coins processed: {len(top_10_coins)}")
        
    except Exception as e:
        logger.error(f"[❌] Error in analysis cycle: {str(e)}")

def start_scheduler():
    """Start the analysis scheduler with staggered timeframes"""
    # Run different timeframes at staggered intervals
    schedule.every(15).minutes.do(run_analysis, timeframes=['15m'])
    schedule.every(1).hour.do(run_analysis, timeframes=['1h'])
    schedule.every(4).hours.do(run_analysis, timeframes=['4h'])
    schedule.every(24).hours.do(run_analysis, timeframes=['1d'])

    def job_runner():
        """Run scheduled jobs continuously"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    # Start the scheduler in a separate thread
    thread = threading.Thread(target=job_runner)
    thread.daemon = True
    thread.start()

# Helper function to fetch coin data
def fetch_coin_data(symbol):
    """Fetch historical data for a coin"""
    try:
        # Fetch data using Binance API
        df = fetch_ohlcv_df(symbol, interval='4h', limit=100)
        
        # Add technical indicators
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma50'] = df['close'].rolling(window=50).mean()
        
        return df
    except Exception as e:
        logger.error(f"[❌] Error fetching data for {symbol}: {str(e)}")
        return None