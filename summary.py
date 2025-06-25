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

model = 'moe-mlp'  # Or any other model from the exp/ directory.

# Load all training runs.
df = pd.json_normalize([
    json.loads(x.read_text())
    for x in Path('exp').glob(f'{model}/**/0-evaluation/*/report.json')
])
df['Dataset'] = df['config.data.path'].map(get_dataset_name)

# Aggregate the results over the random seeds.
print(df.groupby('Dataset')['metrics.test.score'].agg(['mean', 'std']))