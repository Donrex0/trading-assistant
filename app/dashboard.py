# app/dashboard.py

import streamlit as st
import pandas as pd
import time
import sys
import os
import logging
import plotly.graph_objects as go

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging first to ensure it's available for all imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add debug info
logger.info(f"Python version: {sys.version}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"Python path: {sys.path}")

# Core functionality imports
import os
import sys
import importlib.util

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import using importlib for better error handling
def import_module_from_path(module_name, path):
    """Import a module from a specific path with detailed error reporting."""
    try:
        # Convert module name to file path
        module_file = module_name.split('.')[-1] + '.py'
        module_path = os.path.join(path, *module_name.split('.')[:-1], module_file)
        
        # Check if the file exists
        if not os.path.exists(module_path):
            # Try alternative path format
            module_path = os.path.join(path, module_name.replace('.', os.sep) + '.py')
            if not os.path.exists(module_path):
                raise ImportError(
                    f"Module file not found. Tried:\n"
                    f"1. {os.path.join(path, *module_name.split('.'), module_file)}\n"
                    f"2. {module_path}"
                )
        
        # Get the module spec
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            raise ImportError(f"Failed to create spec for {module_name} at {module_path}")
        
        # Create and execute the module
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        # Execute the module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            # Clean up and re-raise with more context
            del sys.modules[module_name]
            raise ImportError(
                f"Error executing module {module_name} from {module_path}:\n"
                f"Type: {type(e).__name__}\n"
                f"Message: {str(e)}\n"
                f"Traceback will follow in the main exception handler."
            ) from e
            
        return module
        
    except Exception as e:
        # Add additional context to any errors
        if not isinstance(e, ImportError):
            raise ImportError(
                f"Unexpected error importing {module_name}:\n"
                f"Type: {type(e).__name__}\n"
                f"Message: {str(e)}\n"
                f"Path: {getattr(e, '__file__', 'N/A')}\n"
                f"Working dir: {os.getcwd()}"
            ) from e
        raise

try:
    # Import analysis.coin_selector
    coin_selector_path = os.path.join(project_root, 'analysis')
    if os.path.exists(os.path.join(coin_selector_path, 'coin_selector.py')):
        # Import the module directly using the full path
        coin_selector = import_module_from_path('coin_selector', coin_selector_path)
        fetch_market_data = coin_selector.fetch_market_data
        get_supported_coins = coin_selector.get_supported_coins
    else:
        raise ImportError(
            f"Could not find coin_selector.py in {coin_selector_path}. "
            f"Contents of directory: {os.listdir(coin_selector_path) if os.path.exists(coin_selector_path) else 'Directory not found'}"
        )
    
    # Import utils.data_fetcher
    data_fetcher_path = os.path.join(project_root, 'utils')
    data_fetcher_file = os.path.join(data_fetcher_path, 'data_fetcher.py')
    if os.path.exists(data_fetcher_file):
        # Import the module using importlib
        spec = importlib.util.spec_from_file_location('data_fetcher', data_fetcher_file)
        if spec is None:
            raise ImportError(f"Failed to create spec for data_fetcher from {data_fetcher_file}")
        data_fetcher = importlib.util.module_from_spec(spec)
        sys.modules['data_fetcher'] = data_fetcher
        spec.loader.exec_module(data_fetcher)
        
        # Get the fetch_ohlcv_df function
        if hasattr(data_fetcher, 'fetch_ohlcv_df'):
            fetch_ohlcv_df = data_fetcher.fetch_ohlcv_df
        else:
            raise ImportError(f"fetch_ohlcv_df function not found in {data_fetcher_file}")
    else:
        raise ImportError(
            f"Could not find data_fetcher.py in {data_fetcher_path}. "
            f"Contents of directory: {os.listdir(data_fetcher_path) if os.path.exists(data_fetcher_path) else 'Directory not found'}"
        )
    
except Exception as e:
    import traceback
    print(f"Error during imports: {str(e)}")
    print("Traceback:", traceback.format_exc())
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"sys.path: {sys.path}")
    raise

# Defer imports that might cause circular dependencies
try:
    from utils.schedule_loop import start_scheduler, top_10_coins, smc_analysis_results
except ImportError as e:
    logger.warning(f"Could not import from schedule_loop: {e}")
    start_scheduler = None
    top_10_coins = []
    smc_analysis_results = {}

try:
    from strategies.strategy_runner import run_all_strategies
except ImportError as e:
    logger.warning(f"Could not import strategy_runner: {e}")
    run_all_strategies = None

try:
    from app.signal_display import format_signal_display, format_market_display, format_sentiment_display
    from app.signals_display import show_signals
    from app.trade_notifier import send_trade_notification
except ImportError as e:
    logger.warning(f"Could not import display components: {e}")
    format_signal_display = lambda x: str(x)
    format_market_display = lambda x: str(x)
    format_sentiment_display = lambda x: str(x)
    show_signals = lambda x: None
    send_trade_notification = lambda x: None

st.set_page_config(layout="wide")
st.title("📊 Smart Trading Strategy Dashboard")

# Initialize predictor
st.info("✅ Trading Assistant is ready to make predictions")

def calculate_indicators(df):
    """Calculate technical indicators"""
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['volatility'] = df['close'].rolling(10).std()
    df['momentum'] = df['close'].diff(4)
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    return df

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line, macd - signal_line

def predict_trend(symbol="BTCUSDT"):
    """Predict trend using the predictor module"""
    try:
        # Try to import predict_multi_timeframe if not available
        try:
            from ml.lstm_predictor import predict_multi_timeframe
            result = predict_multi_timeframe(symbol)
            return result
        except ImportError as e:
            logger.warning(f"Could not import predict_multi_timeframe: {e}")
            return {
                'final_prediction': 0,  # Neutral
                'final_confidence': 0.5,
                'per_timeframe': []
            }
    except Exception as e:
        logger.error(f"Prediction error for {symbol}: {str(e)}")
        return {
            'final_prediction': 0,  # Neutral
            'final_confidence': 0.0,
            'per_timeframe': []
        }

# Start the scheduler if not already running
if 'scheduler_started' not in st.session_state:
    try:
        if start_scheduler is not None:
            start_scheduler()
            st.session_state.scheduler_started = True
            logger.info("Scheduler started successfully")
        else:
            logger.warning("Scheduler not available - some features may be limited")
            st.session_state.scheduler_started = False
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        st.session_state.scheduler_started = False

# Create two columns for display
col1, col2 = st.columns([2, 1])

# Initialize empty DataFrame at the beginning
df = pd.DataFrame()

with col1:
    st.header("Top 10 Volatile Coins")
    
    try:
        # Fetch market data
        market_data = fetch_market_data()
        
        if market_data and isinstance(market_data, list) and len(market_data) > 0:
            # Convert list of dicts to DataFrame
            df = pd.DataFrame(market_data)
            
            # Ensure required columns exist
            if all(col in df.columns for col in ['high_24h', 'low_24h', 'current_price', 'price_change_percentage_24h', 'total_volume']):
                # Calculate volatility
                df['volatility'] = (df['high_24h'] - df['low_24h']) / df['low_24h']
                
                # Filter and sort
                df = df[df['total_volume'] > 10000000]  # Filter for volume > $10M
                df = df[df['current_price'] > 1.0]  # Filter out very low price coins
                df = df.sort_values(by='volatility', ascending=False).head(10)
                
                # Define display columns
                display_cols = ['symbol', 'current_price', 'price_change_percentage_24h', 'total_volume', 'volatility']
                
                # Format and display
                format_dict = {
                    'current_price': '{:.4f}',
                    'price_change_percentage_24h': '{:+.2f}%',
                    'total_volume': '{:,.0f}',
                    'volatility': '{:.2%}'
                }
                
                st.dataframe(
                    df[display_cols].style.format(format_dict),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning("Required price data not available in market data")
                df = pd.DataFrame()
        else:
            st.warning("No market data available. Check your internet connection.")
            
    except Exception as e:
        logger.error(f"Error displaying market data: {e}")
        st.error(f"Failed to load market data: {str(e)}")
        df = pd.DataFrame()

    # Only proceed if we have data
    if df is not None and not df.empty:
        # Get list of supported trading pairs from Binance
        try:
            supported_pairs = get_supported_coins()
            if not supported_pairs:
                st.error("Failed to fetch supported trading pairs from Binance")
            else:
                # Show detailed analysis for the first coin only
                try:
                    first_coin = df.iloc[0]
                    symbol = first_coin.get('symbol', '').upper() + 'USDT'
                    
                    st.subheader(f"🔍 {symbol} Detailed Analysis")
                    
                    # Fetch OHLCV data
                    ohlcv_df = fetch_ohlcv_df(symbol, '1h', 100)
                    if not ohlcv_df.empty:
                        df = calculate_indicators(ohlcv_df)
                        
                        # Show technical indicators
                        st.subheader("📊 Technical Indicators")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Price", f"${df['close'].iloc[-1]:.4f}")
                            st.metric("24h Change", f"{first_coin.get('price_change_percentage_24h', 0):.2f}%")
                        
                        with col2:
                            st.metric("24h High", f"${first_coin.get('high_24h', 0):.4f}")
                            st.metric("24h Low", f"${first_coin.get('low_24h', 0):.4f}")
                        
                        with col3:
                            st.metric("24h Volume", f"${first_coin.get('total_volume', 0):,.0f}")
                            st.metric("Volatility", f"{first_coin.get('volatility', 0) * 100:.2f}%")
                        
                        # Show LSTM prediction if available
                        try:
                            prediction = predict_trend(symbol)
                            if prediction and 'final_prediction' in prediction:
                                st.subheader("🔮 Prediction")
                                if prediction['final_prediction'] == 1:
                                    st.success(f"Bullish (Confidence: {prediction.get('final_confidence', 0):.2f})")
                                else:
                                    st.error(f"Bearish (Confidence: {prediction.get('final_confidence', 0):.2f})")
                        except Exception as e:
                            logger.warning(f"Prediction failed: {str(e)}")
                                
                except Exception as e:
                    logger.error(f"Error in detailed analysis: {str(e)}")
                    st.error(f"Failed to load detailed analysis: {str(e)}")
        except Exception as e:
            logger.error(f"Error getting supported pairs: {str(e)}")
            st.error(f"Failed to load supported trading pairs: {str(e)}")
    else:
        st.info("No data available for analysis.")

with col2:
    st.header("SMC Analysis Results")
    
    # Show analysis results
    if smc_analysis_results:
        for symbol, result in smc_analysis_results.items():
            with st.expander(f"{symbol} Analysis"):
                st.json(result['analysis'])
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))
                st.info(f"Last updated: {timestamp}")
    else:
        st.info("No analysis results yet. First analysis will run in a few minutes...")
