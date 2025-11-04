# DBInfer-RelBench Adapter

Adapter to use [DBInfer datasets](https://github.com/awslabs/multi-table-benchmark) with the RelBench interface.

## Installation

```bash
pip install dbinfer-relbench-adapter
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html
```

## Example

```python
from dbinfer_relbench_adapter import load_dbinfer_data

# Load dataset and task
dataset, task = load_dbinfer_data("diginetica", "ctr")


# This interface is compatible with RelBench.
# Example usage (full dataset and task list below)

dataset = 'diginetica'
task = 'ctr'
```

```diff
- dataset: Dataset = get_dataset(dataset, download=True) # for relbench data
- task: EntityTask = get_task(dataset, task, download=True) # for relbench task
+ dataset, task = load_dbinfer_data(dataset, task) # use this line instead
```

## Full list of supported datasets and tasks

See the full list of datasets and their data card in the accompanying paper.

| Dataset name  | Task names                          |
|:-------------:|:------------------------------------|
|`avs`          |`repeater`                           |
|`mag`          |`cite`, `venue`                      |
|`diginetica`   |`ctr`, `purchase`                    |
|`retailrocket` |`cvr`                                |
|`seznam`       |`charge`, `prepay`                   |
|`amazon`       |`rating`, `purchase`, `churn`        |
|`stackexchange`|`churn`, `upvote`                    |
|`outbrain-small`     |`ctr`                                |

```
@article{dbinfer,
  title={4DBInfer: A 4D Benchmarking Toolbox for Graph-Centric Predictive Modeling on Relational DBs},
  author={Wang, Minjie and Gan, Quan and Wipf, David and Cai, Zhenkun and Li, Ning and Tang, Jianheng and Zhang, Yanlin and Zhang, Zizhao and Mao, Zunyao and Song, Yakun and Wang, Yanbo and Li, Jiahang and Zhang, Han and Yang, Guang and Qin, Xiao and Lei, Chuan and Zhang, Muhan and Zhang, Weinan and Faloutsos, Christos and Zhang, Zheng},
  journal={arXiv preprint arXiv:2404.18209},
  year={2024}
}
```

## License

MIT

## Acknowledgements

This adapter package was made possible thanks to [Yangyi Shen](https://www.linkedin.com/in/yangyi-shen-232514264/) and [Claude Code](https://www.claude.ai/).
