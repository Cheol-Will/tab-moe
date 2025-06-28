import json
from pathlib import Path

import pandas as pd



def get_dataset_name(dataset_path: str) -> str:
    """
    >>> get_dataset_name('data/california')
    'california'
    >>> get_dataset_name('data/regression-num-large-0-year')
    'year'
    """
    name = dataset_path.removeprefix('data/')
    return (
        name.split('-', 4)[-1]  # The "why" benchmark.
        if name.startswith(('classif-', 'regression-'))
        else name
    )



def main():
    model_list = [
        'tabm', 
        'tabm-piecewiselinear', 
        'moe-sparse', 
        'moe-sparse-piecewiselinear', 
        'moe-sparse-shared', 
        'moe-sparse-shared-piecewiselinear',
        'moe-mini-sparse',
        'moe-mini-sparse-piecewiselinear',
        # 'moe-mini-sparse-shared',
        'moe-mini-sparse-shared-piecewiselinear',
    ]
    # for model in model_list:
    #     print_hyperparameters(model)


if __name__ == "__main__":
    main()