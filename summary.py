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

model_list = [
    'tabm', 
    'tabm-piecewiselinear', 
    'moe-sparse-shared', 
    'moe-sparse-shared-piecewiselinear',
    'moe-mlp'
]

for model in model_list:
        
    # model = 'moe-mlp' 

    # Load all training runs.
    df = pd.json_normalize([
        json.loads(x.read_text())
        for x in Path('exp').glob(f'{model}/**/0-evaluation/*/report.json')
    ])
    df['Dataset'] = df['config.data.path'].map(get_dataset_name)

    # Aggregate the results over the random seeds.
    print(model, '-'*50)
    print(df.groupby('Dataset')['metrics.test.score'].agg(['mean', 'std']))


model_list = [
    'moe-sparse-shared', 
    'moe-sparse-shared-piecewiselinear',
    'moe-mlp'
]

for model in model_list:
    df = pd.json_normalize([
        json.loads(x.read_text())
        for x in Path('exp').glob(f'{model}/**/0-tuning/report.json')
    ])
    df['Dataset'] = df['best.config.data.path'].map(get_dataset_name)
    df.index = df['Dataset']

    hyperparameters = [
        'k',
        'n_blocks',
        'd_block',
        'dropout',
        'moe_ratio',
        'num_experts',
    ]
    cols = [f'best.config.model.backbone.{hp}' for hp in hyperparameters]
    print(model, '-'*50)
    print(df[cols])


