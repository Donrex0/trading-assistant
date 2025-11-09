# utils/__init__.py
__all__ = ['schedule_loop', 'utils', 'news_parser', 'use_notifications', 'get_model_path', 'get_latest_processed_file']

# Explicitly import and export required functions and variables
from .schedule_loop import (
    start_scheduler,
    top_10_coins,
    smc_analysis_results
)

# Export utils functions
from .utils import get_model_path, get_latest_processed_file
