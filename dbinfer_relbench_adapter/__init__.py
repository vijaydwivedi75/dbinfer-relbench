"""DBInfer-RelBench Adapter Package.

This package provides adapters to bridge DBInfer datasets and tasks to the RelBench interface,
with built-in caching support to handle version compatibility issues and improve loading times.
"""

import warnings

# Check if DGL is installed and show helpful message if not
try:
    import dgl
except ImportError:
    warnings.warn(
        "\n" + "="*70 + "\n"
        "DGL is not installed or not compatible with your PyTorch version.\n\n"
        "To install DGL for PyTorch 2.3.0, run:\n"
        "  pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html\n\n"
        "Or use the automated installation script from the repository:\n"
        "  bash install.sh\n"
        + "="*70,
        ImportWarning,
        stacklevel=2
    )

from .adapters import DBInferDatasetAdapter, DBInferTaskAdapter
from .loader import load_dbinfer_data
from .cache import clear_cache, get_cache_dir, get_cache_path

__version__ = "0.1.3"

__all__ = [
    "DBInferDatasetAdapter",
    "DBInferTaskAdapter",
    "load_dbinfer_data",
    "clear_cache",
    "get_cache_dir",
    "get_cache_path",
]
