import os
from datetime import datetime
import logging
import sys
from pathlib import Path

def get_project_root():
    """Get the absolute path to the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_latest_processed_file():
    """Get the path to the latest processed BTCUSDT data file"""
    project_root = get_project_root()
    processed_dir = os.path.join(project_root, "data", "processed")
    
    # Get all processed files
    files = [f for f in os.listdir(processed_dir) if f.startswith("BTCUSDT_")]
    if not files:
        raise FileNotFoundError("No processed BTCUSDT data files found in processed directory")
    
    # Sort by modification time and get the latest
    files.sort(key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
    latest_file = files[-1]
    return os.path.join(processed_dir, latest_file)

def get_models_dir():
    """Get the absolute path to the models directory, creating it if needed"""
    project_root = get_project_root()
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_model_path(model_name):
    """Get the absolute path to a model file in the models directory"""
    return os.path.join(get_models_dir(), f"{model_name}.pkl")

def get_timestamped_model_path(model_name):
    """Get a timestamped model path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(get_models_dir(), f"{model_name}_{timestamp}.pkl")

def setup_logging():
    """Set up logging configuration"""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
    
    # Set up logging with UTF-8 encoding
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get the logger
    logger = logging.getLogger(__name__)
    
    # Remove emoji from error messages
    def remove_emoji(text):
        return ''.join(c for c in text if c.isprintable())
    
    class NoEmojiFilter(logging.Filter):
        def filter(self, record):
            if record.levelno >= logging.ERROR:
                record.msg = remove_emoji(str(record.msg))
            return True
    
    logger.addFilter(NoEmojiFilter())
    return logger