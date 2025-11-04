"""Example usage of dbinfer-relbench-adapter package."""

import os
from dbinfer_relbench_adapter import load_dbinfer_data, clear_cache


def main():
    """Run example demonstrating the adapter functionality."""
    dataset_name = "diginetica"
    task_name = "ctr"

    # Example with custom cache directory
    custom_cache_dir = os.path.expanduser("~/.cache/relbench_examples")

    print("=" * 60)
    print("Example: DBInfer-RelBench Adapter")
    print("=" * 60)

    # First load (from source, will be cached)
    print("\n1. Loading from source (will be cached)...")
    print("-" * 60)
    dataset_adapter, task_adapter = load_dbinfer_data(
        dataset_name, task_name, cache_dir=custom_cache_dir
    )
    print("✓ Data and task adapters successfully created.")

    # Display dataset info
    db = dataset_adapter.get_db()
    table_dict = db.table_dict
    print(f"\nDataset has {len(table_dict)} tables:")
    for name, table in table_dict.items():
        print(f"  - {name}: {len(table)} rows")

    # Display task info
    print(f"\nTask information:")
    print(f"  - Task type: {task_adapter.task_type}")
    print(f"  - Target column: {task_adapter.target_col}")
    print(f"  - Entity table: {task_adapter.entity_table}")
    print(f"  - Entity column: {task_adapter.entity_col}")
    if task_adapter.num_labels:
        print(f"  - Number of classes: {task_adapter.num_labels}")

    # Get data splits
    train_table = task_adapter.get_table("train")
    val_table = task_adapter.get_table("val")
    test_table = task_adapter.get_table("test")

    print(f"\nData splits:")
    print(f"  - Train: {len(train_table)} samples")
    print(f"  - Validation: {len(val_table)} samples")
    print(f"  - Test: {len(test_table)} samples")

    # Second load (from cache)
    print("\n2. Loading from cache (much faster)...")
    print("-" * 60)
    dataset_adapter2, task_adapter2 = load_dbinfer_data(
        dataset_name, task_name, cache_dir=custom_cache_dir
    )
    print("✓ Data and task adapters successfully loaded from cache.")

    # Force reload
    print("\n3. Force reload (bypassing cache)...")
    print("-" * 60)
    dataset_adapter3, task_adapter3 = load_dbinfer_data(
        dataset_name, task_name, force_reload=True, cache_dir=custom_cache_dir
    )
    print("✓ Data and task adapters successfully reloaded from source.")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
