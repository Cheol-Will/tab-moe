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

def print_hyperparameters(model):

    # Load all training runs.
    df = pd.json_normalize([
        json.loads(x.read_text())
        for x in Path('exp').glob(f'{model}/**/0-evaluation/*/report.json')
    ])
    print(df.shape)  # (1290, 181)
    df['Dataset'] = df['config.data.path'].map(get_dataset_name)

    # The hyperparameters.
    hyperparameters = [
        'config.model.backbone.n_blocks',
        'config.model.backbone.d_block',
        'config.model.backbone.dropout',
        'config.optimizer.lr',
        'config.optimizer.weight_decay',
    ]
    if 'moe' in model:
        hyperparameters += [
            'config.model.backbone.k',
            'config.model.backbone.moe_ratio',
            'config.model.backbone.num_experts',
        ]
    else:
        hyperparameters += [
            'config.model.k',
        ]

   
    # When summarizing hyperparameters (but not metrics),
    # it is enough to keep only one seed per dataset.
    dfh = df.loc[df['config.seed'] == 0, ['Dataset', *hyperparameters]]

    # Add additional "hyperparameters".
    dfh['has_dropout'] = (dfh['config.model.backbone.dropout'] > 0).astype(float)
    dfh['has_weight_decay'] = (dfh['config.optimizer.weight_decay'] > 0).astype(float)

    # Some datasets have multiple splits, so they must be aggregated first.
    dfh = dfh.groupby('Dataset').mean()

    # Finally, compute the statistics.
    # NOTE: it is important to take all statistics into account, especially the quantiles,
    # not only the mean value, because the latter is not robust to outliers.
    print(model)
    print(dfh)
    print()



def main():
    model_list = [
        'tabm', 
        'tabm-piecewiselinear', 
        'moe-sparse-shared', 
        'moe-sparse-shared-piecewiselinear',
        'moe-mini-sparse-shared-piecewiselinear',
        'moe-sparse',
    ]
    for model in model_list:
        print_hyperparameters(model)


if __name__ == "__main__":
    main()