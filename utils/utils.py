import os
from datetime import datetime
import logging

def get_project_root() -> str:
    """Get the project root directory"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(subdir: str = "") -> str:
    """Get path to data directory"""
    root = get_project_root()
    return os.path.join(root, "data", subdir)

def get_model_path(model_name: str) -> str:
    """Get path to model file"""
    root = get_project_root()
    return os.path.join(root, "models", f"{model_name}.pkl")

def get_latest_processed_file() -> str:
    """Get the latest labeled CSV file"""
    labels_dir = get_data_path("labels")
    files = [f for f in os.listdir(labels_dir) if f.endswith('.csv') and 'labeled' in f]
    if not files:
        raise FileNotFoundError(f"No labeled files found in {labels_dir}")
        
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(labels_dir, x)))
    return os.path.join(labels_dir, latest_file)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_assistant.log')
        ]
    )
    return logging.getLogger(__name__)

def ensure_directory_exists(path: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")

def validate_file_path(file_path: str) -> bool:
    """Validate that a file exists and is readable"""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    
    if not os.access(file_path, os.R_OK):
        logging.error(f"No read permission for file: {file_path}")
        return False
    
    return True
