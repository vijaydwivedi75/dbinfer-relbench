"""Adapter classes to bridge DBInfer datasets and tasks to RelBench interface."""

import dbinfer_bench as dbb
from dbinfer_bench.dataset_meta import DBBColumnDType
from relbench.base import TaskType
import pandas as pd
import numpy as np


class DBInferDatasetAdapter:
    """Adapter for DBInfer datasets to match RelBench Database interface."""

    def __init__(self, dbinfer_dataset):
        print("[INFO] DBInferDatasetAdapter.__init__ called")
        self.dbinfer_dataset = dbinfer_dataset

    def get_db(self):
        print("[INFO] DBInferDatasetAdapter.get_db called")

        class MockTable:
            def __init__(self, table_name, table_data, table_schema, all_tables):
                print(f"[INFO] MockTable.__init__ called for table: {table_name}")
                self.df = pd.DataFrame(table_data)
                print(f"[INFO] DataFrame created for {table_name}, shape: {self.df.shape}")

                # Find primary key or create synthetic one
                self.pkey_col = None
                for col_schema in table_schema.columns:
                    if col_schema.dtype == DBBColumnDType.primary_key:
                        self.pkey_col = col_schema.name
                        break

                if self.pkey_col is None:
                    self.pkey_col = '__synthetic_pk__'
                    self.df[self.pkey_col] = np.arange(len(self.df))
                else:
                    # Ensure primary key is consecutive integers
                    self.df[self.pkey_col] = np.arange(len(self.df))

                # Extract foreign key relationships
                self.fkey_col_to_pkey_table = {}
                for col_schema in table_schema.columns:
                    if col_schema.dtype == DBBColumnDType.foreign_key:
                        try:
                            if hasattr(col_schema, 'link_to') and col_schema.link_to:
                                link_table = col_schema.link_to.split('.')[0]
                                if link_table in all_tables:
                                    self.fkey_col_to_pkey_table[col_schema.name] = link_table
                        except (AttributeError, IndexError):
                            continue

                self.time_col = getattr(table_schema, 'time_column', None)

                # Fix time column format for RelBench compatibility
                if self.time_col and self.time_col in self.df.columns:
                    try:
                        # Convert to datetime if not already
                        if not pd.api.types.is_datetime64_any_dtype(self.df[self.time_col]):
                            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], errors='coerce')
                        # Ensure it's datetime64[ns] format
                        self.df[self.time_col] = self.df[self.time_col].astype('datetime64[ns]')
                    except Exception as e:
                        print(f"Warning: Could not convert time column {self.time_col} to datetime: {e}")
                        self.time_col = None

                # Fix foreign key references safely
                for fkey_col, target_table in self.fkey_col_to_pkey_table.items():
                    if fkey_col in self.df.columns and target_table in all_tables:
                        try:
                            target_size = len(all_tables[target_table])
                            if target_size > 0:
                                # Convert to int, handling NaN
                                fk_values = self.df[fkey_col].fillna(-1).astype(int)

                                # Mark invalid references (out of range or -1) as NaN
                                # Don't clip - just invalidate incorrect references
                                valid_mask = (fk_values >= 0) & (fk_values < target_size)
                                self.df[fkey_col] = fk_values
                                self.df.loc[~valid_mask, fkey_col] = np.nan
                        except (ValueError, TypeError):
                            # If conversion fails, set to NaN
                            self.df[fkey_col] = np.nan

            def __len__(self):
                return len(self.df)

            @property
            def num_nodes(self):
                return len(self.df)

        class MockDB:
            def __init__(self, dataset):
                print("[INFO] MockDB.__init__ called")
                self.dataset = dataset

            @property
            def table_dict(self):
                print("[INFO] MockDB.table_dict property accessed")
                print(f"[INFO] Building table schemas for {len(self.dataset.metadata.tables)} tables")
                table_schemas = {schema.name: schema for schema in self.dataset.metadata.tables}
                print(f"[INFO] Creating MockTable instances for {len(self.dataset.tables)} tables")
                result = {}
                for name, table_data in self.dataset.tables.items():
                    print(f"[INFO] Processing table: {name}")
                    result[name] = MockTable(name, table_data, table_schemas[name], self.dataset.tables)
                print("[INFO] All MockTable instances created")
                return result

        return MockDB(self.dbinfer_dataset)

    @property
    def val_timestamp(self):
        """DBInfer doesn't use timestamps, return None"""
        return None

    @property
    def test_timestamp(self):
        """DBInfer doesn't use timestamps, return None"""
        return None


class DBInferTaskAdapter:
    """Adapter for DBInfer tasks to match RelBench Task interface."""

    def __init__(self, dbinfer_task, dataset_adapter):
        print("[INFO] DBInferTaskAdapter.__init__ called")
        self.dbinfer_task = dbinfer_task
        self.dataset = dataset_adapter

        # Determine task type based on target column cardinality
        target_data = self.dbinfer_task.train_set[self.dbinfer_task.metadata.target_column]
        unique_targets = len(np.unique(target_data))

        if unique_targets == 2:
            self.task_type = TaskType.BINARY_CLASSIFICATION
        elif unique_targets > 2:
            self.task_type = TaskType.MULTICLASS_CLASSIFICATION
        else:
            self.task_type = TaskType.REGRESSION

        self.target_col = self.dbinfer_task.metadata.target_column
        self.entity_table = self.dbinfer_task.metadata.target_table

        # Entity column is the first non-target column (the thing we predict for)
        task_columns = list(self.dbinfer_task.train_set.keys())
        self.entity_col = next((col for col in task_columns if col != self.target_col), task_columns[0])

        # Create entity mapping for consecutive integers
        all_entities = set()
        for split_data in [self.dbinfer_task.train_set, self.dbinfer_task.validation_set, self.dbinfer_task.test_set]:
            if self.entity_col in split_data:
                all_entities.update(pd.Series(split_data[self.entity_col]).dropna().astype(int))

        self.entity_mapping = {int(entity): idx for idx, entity in enumerate(sorted(all_entities))}

        # Create target mapping for classification tasks
        if self.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            all_targets = set()
            for split_data in [self.dbinfer_task.train_set, self.dbinfer_task.validation_set, self.dbinfer_task.test_set]:
                if self.target_col in split_data:
                    target_series = pd.Series(split_data[self.target_col]).dropna()
                    all_targets.update(target_series.astype(str))  # Convert all to strings for consistency

            # Create mapping from original values to consecutive integers
            sorted_targets = sorted(all_targets)
            self.target_mapping = {target: idx for idx, target in enumerate(sorted_targets)}
            print(f"[INFO] Target mapping created: {self.target_mapping}")
        else:
            self.target_mapping = None

        # Store number of labels for multiclass
        if self.task_type == TaskType.MULTICLASS_CLASSIFICATION and self.target_mapping:
            self.num_labels = len(self.target_mapping)
        else:
            self.num_labels = None

        # Add other required attributes for compatibility
        self.timedelta = None  # DBInfer doesn't use time deltas
        self.num_eval_timestamps = 1

        # Add metrics attribute (required by RelBench interface)
        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            from sklearn.metrics import roc_auc_score, accuracy_score
            self.metrics = [roc_auc_score, accuracy_score]
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            from sklearn.metrics import accuracy_score
            self.metrics = [accuracy_score]
        elif self.task_type == TaskType.REGRESSION:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            self.metrics = [mean_absolute_error, mean_squared_error]
        else:
            self.metrics = []

    def get_table(self, split, mask_input_cols=None):
        """Get task table for a specific split (train/val/test)."""
        print(f"[INFO] DBInferTaskAdapter.get_table called for split: {split}")
        if split == "train":
            data = self.dbinfer_task.train_set
        elif split == "val":
            data = self.dbinfer_task.validation_set
        elif split == "test":
            data = self.dbinfer_task.test_set
        else:
            raise ValueError(f"Unknown split: {split}")
        print(f"[INFO] Data retrieved for {split}, keys: {list(data.keys()) if hasattr(data, 'keys') else 'no keys'}")

        df = pd.DataFrame(data)

        # Map entity IDs to consecutive integers
        if self.entity_col in df.columns:
            df[self.entity_col] = df[self.entity_col].astype(int).map(self.entity_mapping)

        # Map target values for classification tasks
        if self.target_mapping and self.target_col in df.columns:
            # Convert targets to strings and map to integers
            df[self.target_col] = df[self.target_col].astype(str).map(self.target_mapping)

        class MockTable:
            def __init__(self, df, target_col, time_col=None):
                print(f"[INFO] Task MockTable.__init__ called, df shape: {df.shape}, target_col: {target_col}")
                self.df = df
                self.target_col = target_col
                self.time_col = time_col

            def __len__(self):
                return len(self.df)

        result = MockTable(df, self.target_col, getattr(self.dbinfer_task.metadata, 'time_column', None))
        print(f"[INFO] Returning MockTable for {split}")
        return result

    def evaluate(self, pred, table=None):
        """Evaluate predictions using appropriate metrics for the task type."""
        print(f"[INFO] DBInferTaskAdapter.evaluate called")

        if table is None:
            # Default to test table when no table is provided (RelBench convention)
            table = self.get_table("test")

        true_labels = table.df[self.target_col].values

        # Ensure prediction and label lengths match
        if len(pred) != len(true_labels):
            print(f"[INFO] Length mismatch: pred={len(pred)}, labels={len(true_labels)}")
            # Truncate to minimum length
            min_len = min(len(pred), len(true_labels))
            pred = pred[:min_len]
            true_labels = true_labels[:min_len]

        if self.task_type == TaskType.BINARY_CLASSIFICATION:
            from sklearn.metrics import roc_auc_score, accuracy_score
            # Handle both probability and logit outputs
            if pred.min() >= 0 and pred.max() <= 1:
                # Already probabilities
                prob_pred = pred
            else:
                # Convert logits to probabilities
                prob_pred = 1 / (1 + np.exp(-pred))
            return {
                "roc_auc": roc_auc_score(true_labels, prob_pred),
                "accuracy": accuracy_score(true_labels, prob_pred > 0.5)
            }
        elif self.task_type == TaskType.MULTICLASS_CLASSIFICATION:
            from sklearn.metrics import accuracy_score, f1_score
            pred_classes = pred.argmax(axis=1) if pred.ndim > 1 else pred
            return {
                "accuracy": accuracy_score(true_labels, pred_classes),
                "multiclass_f1": f1_score(true_labels, pred_classes, average='macro')
            }
        elif self.task_type == TaskType.REGRESSION:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            return {
                "mae": mean_absolute_error(true_labels, pred),
                "mse": mean_squared_error(true_labels, pred)
            }
        else:
            return {"accuracy": 0.0}  # Fallback

    def filter_dangling_entities(self, table):
        """Filter out dangling entities - no-op for DBInfer since entities are pre-validated."""
        return table

    def get_db(self):
        """Get database object - delegate to dataset adapter."""
        return self.dataset.get_db()
