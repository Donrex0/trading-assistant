"""
Caching utilities for the trading assistant.
Provides in-memory and disk-based caching with TTL (Time To Live) support.
"""
import os
import pickle
import time
import hashlib
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union, Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variable for generic function typing
F = TypeVar('F', bound=Callable[..., Any])

class Cache:
    """Generic caching class with TTL support."""
    
    def __init__(self, cache_dir: str = '.cache', ttl: int = 300):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store disk caches
            ttl: Default time-to-live in seconds for cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = ttl
        self.memory_cache: Dict[str, Tuple[Any, float]] = {}
        
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate a unique cache key for function arguments."""
        # Create a string representation of the arguments
        args_str = f"{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        # Create a hash of the string for a consistent key
        return hashlib.md5(args_str.encode('utf-8')).hexdigest()
    
    def memory_cached(self, ttl: Optional[int] = None) -> Callable[[F], F]:
        """
        Decorator to cache function results in memory.
        
        Args:
            ttl: Time to live in seconds (None for default)
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args, **kwargs):
                cache_key = self._get_cache_key(func.__name__, *args, **kwargs)
                current_time = time.time()
                
                # Check if in cache and not expired
                if cache_key in self.memory_cache:
                    cached_value, expiry = self.memory_cache[cache_key]
                    if expiry > current_time:
                        logger.debug(f"Cache hit for {func.__name__}")
                        return cached_value
                    # Remove expired entry
                    del self.memory_cache[cache_key]
                
                # Call function and cache result
                result = func(*args, **kwargs)
                expiry_time = current_time + (ttl if ttl is not None else self.default_ttl)
                self.memory_cache[cache_key] = (result, expiry_time)
                
                return result
            return wrapper  # type: ignore
        return decorator
    
    def disk_cached(self, ttl: Optional[int] = None, cache_key: Optional[str] = None) -> Callable[[F], F]:
        """
        Decorator to cache function results on disk.
        
        Args:
            ttl: Time to live in seconds (None for default)
            cache_key: Custom cache key (defaults to function name)
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = cache_key or func.__name__
                cache_file = self.cache_dir / f"{key}.pkl"
                current_time = time.time()
                
                # Check if cache file exists and is not expired
                if cache_file.exists():
                    try:
                        with open(cache_file, 'rb') as f:
                            cached_time, result = pickle.load(f)
                            expiry_time = cached_time + (ttl if ttl is not None else self.default_ttl)
                            
                            if expiry_time > current_time:
                                logger.debug(f"Disk cache hit for {key}")
                                return result
                    except (pickle.PickleError, EOFError) as e:
                        logger.warning(f"Error reading cache file {cache_file}: {e}")
                
                # Call function and cache result
                result = func(*args, **kwargs)
                
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump((current_time, result), f)
                except IOError as e:
                    logger.warning(f"Could not write to cache file {cache_file}: {e}")
                
                return result
            return wrapper  # type: ignore
        return decorator
    
    def clear(self, memory: bool = True, disk: bool = True) -> None:
        """Clear the cache."""
        if memory:
            self.memory_cache.clear()
        if disk and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob('*.pkl'):
                try:
                    cache_file.unlink()
                except OSError as e:
                    logger.warning(f"Could not delete cache file {cache_file}: {e}")

# Global cache instance
cache = Cache()

def cached(ttl: Optional[int] = None, disk: bool = False, cache_key: Optional[str] = None) -> Callable[[F], F]:
    """
    Convenience decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds (None for default)
        disk: Whether to use disk cache (default: memory only)
        cache_key: Custom cache key (defaults to function name)
        
    Returns:
        Decorated function with caching
    """
    if disk:
        return cache.disk_cached(ttl=ttl, cache_key=cache_key)
    return cache.memory_cached(ttl=ttl)
