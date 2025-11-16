import sys
from dbinfer_relbench_adapter import load_dbinfer_data

CACHE_DIR = sys.argv[1] if len(sys.argv) > 1 else "./dbinfer_data_cache"

datasets = [
    "avs",
    # "mag",
    # "diginetica",
    # "retailrocket",
    # "seznam",
    # "amazon",
    # "stackexchange",
    # "outbrain-small",
]

tasks = {
    "avs": ["repeater"],
    # "mag": ["cite", "venue"],
    # "diginetica": ["ctr", "purchase"],
    # "retailrocket": ["cvr"],
    # "seznam": ["charge", "prepay"],
    # "amazon": ["rating", "purchase", "churn"],
    # "stackexchange": ["churn", "upvote"],
    # "outbrain-small": ["ctr"],
}

for dataset in datasets:
    for task in tasks[dataset]:
        load_dbinfer_data(dataset, task, cache_dir=CACHE_DIR)