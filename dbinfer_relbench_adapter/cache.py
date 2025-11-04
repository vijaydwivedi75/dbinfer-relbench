"""Caching utilities for DBInfer adapter objects."""

import os
import pickle
from pathlib import Path


def get_cache_dir(cache_dir=None):
    """Get or create the cache directory.

    Args:
        cache_dir: Custom cache directory path. If None, uses ~/.cache/dbinfer_adapters

    Returns:
        Path object pointing to the cache directory
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/dbinfer_adapters")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_cache_path(dataset_name, task_name, cache_dir=None):
    """Generate cache file path for a dataset/task combination.

    Args:
        dataset_name: Name of the DBInfer dataset
        task_name: Name of the task
        cache_dir: Custom cache directory path

    Returns:
        Path object pointing to the cache file
    """
    cache_path = get_cache_dir(cache_dir)
    cache_filename = f"{dataset_name}_{task_name}.pkl"
    return cache_path / cache_filename


def save_to_cache(dataset_name, task_name, dataset_adapter, task_adapter, cache_dir=None):
    """Save adapters to cache.

    Args:
        dataset_name: Name of the DBInfer dataset
        task_name: Name of the task
        dataset_adapter: DBInferDatasetAdapter instance
        task_adapter: DBInferTaskAdapter instance
        cache_dir: Custom cache directory path

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        cache_path = get_cache_path(dataset_name, task_name, cache_dir)
        cache_data = {
            'dataset_adapter': dataset_adapter,
            'task_adapter': task_adapter
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[INFO] Cached adapters to {cache_path}")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to cache adapters: {e}")
        return False


def load_from_cache(dataset_name, task_name, cache_dir=None):
    """Load adapters from cache if available.

    Args:
        dataset_name: Name of the DBInfer dataset
        task_name: Name of the task
        cache_dir: Custom cache directory path

    Returns:
        tuple: (dataset_adapter, task_adapter) or (None, None) if not found
    """
    try:
        cache_path = get_cache_path(dataset_name, task_name, cache_dir)
        if cache_path.exists():
            print(f"[INFO] Loading adapters from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            return cache_data['dataset_adapter'], cache_data['task_adapter']
        else:
            print(f"[INFO] No cache found at {cache_path}")
            return None, None
    except Exception as e:
        print(f"[WARNING] Failed to load from cache: {e}")
        return None, None


def clear_cache(dataset_name=None, task_name=None, cache_dir=None):
    """Clear cache for specific dataset/task or all caches.

    Args:
        dataset_name: If provided with task_name, clears specific cache. If None, clears all.
        task_name: Task name (required if dataset_name is provided)
        cache_dir: Directory for cache files (default: ~/.cache/dbinfer_adapters)
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/dbinfer_adapters")

    cache_path = get_cache_dir(cache_dir)

    if dataset_name is not None and task_name is not None:
        # Clear specific cache
        cache_file = get_cache_path(dataset_name, task_name, cache_dir)
        if cache_file.exists():
            cache_file.unlink()
            print(f"[INFO] Cleared cache for {dataset_name}/{task_name}")
        else:
            print(f"[INFO] No cache found for {dataset_name}/{task_name}")
    else:
        # Clear all caches
        for cache_file in cache_path.glob("*.pkl"):
            cache_file.unlink()
            print(f"[INFO] Cleared cache: {cache_file}")
        print(f"[INFO] All caches cleared")
