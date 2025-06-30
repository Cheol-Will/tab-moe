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

def print_metrics(model):
    # Load all training runs.
    df = pd.json_normalize([
        json.loads(x.read_text())
        for x in Path('exp').glob(f'{model}/**/0-evaluation/*/report.json')
    ])
    df['Dataset'] = df['config.data.path'].map(get_dataset_name)

    # Aggregate the results over the random seeds.
    print(model, '-'*50)
    print(df.groupby('Dataset')['metrics.test.score'].agg(['mean', 'std']))
    print()

    df.to_csv(f"exp/{model}/metrics.csv")


def print_hyperparameters(model):

    # Load all training runs.
    df = pd.json_normalize([
        json.loads(x.read_text())
        for x in Path('exp').glob(f'{model}/**/0-evaluation/*/report.json')
    ])
    # print(df.shape)  # (1290, 181)
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
    dfh.loc['AVERAGE'] = dfh.mean(axis=0)

    # Finally, compute the statistics.
    # NOTE: it is important to take all statistics into account, especially the quantiles,
    # not only the mean value, because the latter is not robust to outliers.
    print(model)
    print(dfh)
    print()

    dfh.to_csv(f"exp/{model}/hyperparameters.csv")


def print_tuning_time(model):
    df = pd.json_normalize([
        json.loads(x.read_text())
        for x in Path('exp').glob(f'{model}/**/0-tuning/report.json')
    ])
    df['Dataset'] = df['best.config.data.path'].map(get_dataset_name)
    df.index = df['Dataset']
    df = df.sort_index()
    idx = [
        "adult", "black-friday", "california", "churn", "covtype2", "diamond", 
        "higgs-small", "house", "microsoft", "otto",  
    ]
    print(model, '-'*50)
    print(df.loc[idx, "time"])
    print()
    df = df.loc[idx, "time"]
    df.to_csv(f"seraching_time_{model}.csv")

def main():
    model_list = [
        'tabm', 
        'moe-sparse', 
        'moe-sparse-shared', 
        'moe-mini-sparse',
        # 'moe-mini-sparse-shared',
        'tabm-piecewiselinear', 
        'moe-sparse-piecewiselinear', 
        'moe-sparse-shared-piecewiselinear',
        'moe-mini-sparse-piecewiselinear',
        'moe-mini-sparse-shared-piecewiselinear',
    ]

    for model in model_list:
        print_metrics(model)

 
    for model in model_list:
        print_hyperparameters(model)

if __name__ == "__main__":
    main()