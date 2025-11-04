"""Main loader functions for DBInfer datasets and tasks."""

import os
import dbinfer_bench as dbb
from .adapters import DBInferDatasetAdapter, DBInferTaskAdapter
from .cache import load_from_cache, save_to_cache


def load_dbinfer_data(dataset_name, task_name, use_cache=True, force_reload=False,
                      cache_dir=None):
    """Load DBInfer dataset and task with caching support.

    This function provides a convenient interface to load DBInfer datasets and tasks
    with automatic caching to avoid expensive reloading operations. The cached data
    includes both the dataset and task adapters.

    Args:
        dataset_name: Name of the DBInfer dataset (e.g., "diginetica", "imdb")
        task_name: Name of the task (e.g., "ctr", "rating")
        use_cache: Whether to use caching (default: True)
        force_reload: Force reload from source even if cache exists (default: False)
        cache_dir: Directory for cache files (default: ~/.cache/dbinfer_adapters)

    Returns:
        tuple: (dataset_adapter, task_adapter)
            - dataset_adapter: DBInferDatasetAdapter instance
            - task_adapter: DBInferTaskAdapter instance

    Example:
        >>> from dbinfer_relbench_adapter import load_dbinfer_data
        >>> dataset, task = load_dbinfer_data("diginetica", "ctr")
        >>> db = dataset.get_db()
        >>> train_table = task.get_table("train")
    """
    # Set default cache directory if not provided
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/dbinfer_adapters")

    # Try loading from cache first if enabled and not forcing reload
    if use_cache and not force_reload:
        dataset_adapter, task_adapter = load_from_cache(dataset_name, task_name, cache_dir)
        if dataset_adapter is not None and task_adapter is not None:
            print(f"[INFO] Successfully loaded adapters from cache")
            return dataset_adapter, task_adapter

    # Load from source
    try:
        print(f"[INFO] Loading DBInfer dataset from source: {dataset_name}, task: {task_name}")
        dbinfer_dataset = dbb.load_rdb_data(dataset_name)
        print(f"[INFO] Dataset loaded, tables: {list(dbinfer_dataset.tables.keys())}")

        dbinfer_task = dbinfer_dataset.get_task(task_name)
        print(f"[INFO] Task loaded, target table: {dbinfer_task.metadata.target_table}")

        dataset_adapter = DBInferDatasetAdapter(dbinfer_dataset)
        task_adapter = DBInferTaskAdapter(dbinfer_task, dataset_adapter)

        # Validate the adapter
        db = dataset_adapter.get_db()
        table_dict = db.table_dict
        print(f"[INFO] Adapter created, {len(table_dict)} tables with sizes: {[(name, len(table)) for name, table in table_dict.items()][:5]}")

        # Save to cache if enabled
        if use_cache:
            save_to_cache(dataset_name, task_name, dataset_adapter, task_adapter, cache_dir)

        return dataset_adapter, task_adapter

    except Exception as e:
        print(f"[ERROR] Error loading DBInfer data: {e}")
        import traceback
        traceback.print_exc()
        raise
