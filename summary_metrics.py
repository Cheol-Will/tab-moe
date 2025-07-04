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
    df_agg = df.groupby('Dataset')['metrics.test.score'].agg(['mean', 'std'])
    print(df_agg)
    print()


def summary_metrics_table(model_list, data_list, output_path="output/metrics.csv", is_print = False, is_save = False):
    """
    Aggregate performance scores from multiple models into a single table and save as CSV.
    """
    all_results = {}

    for model in model_list:
        json_files = list(Path('exp').glob(f'{model}/**/0-evaluation/*/report.json'))
        if not json_files:
            continue

        df = pd.json_normalize([
            json.loads(x.read_text())
            for x in json_files
        ])
        df['Dataset'] = df['config.data.path'].map(get_dataset_name)
        df_agg = df.groupby('Dataset')['metrics.test.score'].mean()
        df_agg = df_agg.reindex(data_list)  # make sure rows are aligned
        all_results[model] = df_agg

    df_all = pd.DataFrame(all_results)
    df_all = df_all.T
    if is_print:
        print(df_all)
    if is_save:
        df_all.to_csv(output_path, float_format="%.4f")
        print(f"Saved all model metrics to {output_path}")
    
    return df_all

def summary_hyperparameters_one_model(model, is_print = False, is_save = False):

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
    if is_print: 
        print(model)
        print(dfh)
        print()

    if is_save:
        dfh.to_csv(f"exp/{model}/hyperparameters.csv")        


def summary_hyperparameters(model_list, output_path="output/average_hyperparameters.csv", is_print = False, is_save = False):
    """
    Extract and aggregate 'AVERAGE' hyperparameters from multiple models into a single table.
    Save the result as a CSV with 4 decimal places.
    """
    records = []

    for model in model_list:
        json_files = list(Path('exp').glob(f'{model}/**/0-evaluation/*/report.json'))
        if not json_files:
            continue

        df = pd.json_normalize([
            json.loads(x.read_text())
            for x in json_files
        ])
        df['Dataset'] = df['config.data.path'].map(get_dataset_name)

        # Define base hyperparameters
        hyperparameters = [
            'config.model.backbone.n_blocks',
            'config.model.backbone.d_block',
            'config.model.backbone.dropout',
            'config.optimizer.lr',
            'config.optimizer.weight_decay',
        ]

        if 'moe' in model:
            optional_params = [
                'config.model.backbone.k',
                'config.model.backbone.moe_ratio',
                'config.model.backbone.num_experts',
            ]
        elif 'tabrm' in model:
            optional_params = [
                'config.model.k',
            ]            
        else:
            optional_params = ['config.model.k']

        # Include only those that exist
        for p in optional_params:
            if p in df.columns:
                hyperparameters.append(p)

        selected_cols = ['Dataset'] + [p for p in hyperparameters if p in df.columns]
        dfh = df.loc[df['config.seed'] == 0, selected_cols]

        # Derived binary indicators
        if 'config.model.backbone.dropout' in dfh.columns:
            dfh['has_dropout'] = (dfh['config.model.backbone.dropout'] > 0).astype(float)
        if 'config.optimizer.weight_decay' in dfh.columns:
            dfh['has_weight_decay'] = (dfh['config.optimizer.weight_decay'] > 0).astype(float)

        dfh = dfh.groupby('Dataset').mean()
        if is_print:
            print(dfh)
        
        average_row = dfh.mean(axis=0)
        average_row.name = model  # Use model name as row index

        records.append(average_row)

    if is_save: 
        # Combine all into a single DataFrame
        df_all_avg = pd.DataFrame(records)
        df_all_avg.index.name = "Model"

        # Save to CSV with 4 decimal precision
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_all_avg.to_csv(output_path, float_format="%.4f")
        print(f"Saved average hyperparameters of all models to {output_path}")

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


def save_rank_csv(model: str, data_list: list[str]) -> pd.DataFrame:
    """
    Add our model's metrics to paper metrics table and calculate rank statistics.
    """

    paper = pd.read_csv("output/paper_metrics.csv")
    df = summary_metrics_table([model], data_list, is_print=False, is_save=False)
    model_cols = [c for c in paper.columns if c not in ("dataset", "direction")]
    summary_t = df.T.reset_index().rename(columns={"Dataset": "dataset"})
    merged = paper.merge(summary_t, on="dataset", how="left")
    merged = merged[merged["dataset"].isin(data_list)]
    merged.drop(columns=["TabPFN"], axis=1, inplace=True)
    # merged.dropna(axis=0, inplace=True)

    # preprocessing
    merged.loc[merged["dataset"] == "house", model] /= 10000
    model_cols = [c for c in merged.columns if c not in ("dataset", "direction")]
    merged[model_cols] = merged[model_cols].abs()

    rank_df = pd.DataFrame(index=merged["dataset"], columns=model_cols, dtype=float)

    n_models = len(model_cols)
    for i, row in merged.iterrows():
        ds = row["dataset"]
        asc = (row["direction"] == "lower_is_better")  # True if regression else False
        ranks = row[model_cols].rank(ascending=asc, method="min")
        # filNaN 
        ranks = ranks.fillna(n_models)
        rank_df.loc[ds] = ranks

    avg_rank = rank_df.mean(axis=0)
    avg_rank = avg_rank.sort_values()
    out_file = f"output/metrics_merged_{model}.csv"
    merged.to_csv(out_file, index=False, float_format="%.4f")
    avg_rank.to_csv(f"output/avg_rank_{model}.csv")
    print(f"\n▶ Saved merged metrics to `{out_file}`")
    print(merged)
    print(rank_df)
    print(avg_rank)
    return merged


def save_ranks_csv(model: list[str], data_list: list[str], file_name: str = None) -> pd.DataFrame:
    """
    Add our model's metrics to paper metrics table and calculate rank statistics.
    """
    if model is not None:
            
        paper = pd.read_csv("output/paper_metrics.csv")
        df = summary_metrics_table(model, data_list, is_print=False, is_save=False)
        
        model_cols = [c for c in paper.columns if c not in ("dataset", "direction")]
        summary_t = df.T.reset_index().rename(columns={"Dataset": "dataset"})
        
        merged = paper.merge(summary_t, on="dataset", how="left")
        merged = merged[merged["dataset"].isin(data_list)]
        merged.drop(columns=["TabPFN"], axis=1, inplace=True)
        # merged.dropna(axis=0, inplace=True)

        # preprocessing
        merged.loc[merged["dataset"] == "house", model] /= 10000

    else:
        paper = pd.read_csv("output/paper_metrics.csv")
        merged = paper
    model_cols = [c for c in merged.columns if c not in ("dataset", "direction")]
    merged[model_cols] = merged[model_cols].abs()
    print(merged)
    rank_df = pd.DataFrame(index=merged["dataset"], columns=model_cols, dtype=float)

    n_models = len(model_cols)
    for i, row in merged.iterrows():
        ds = row["dataset"]
        asc = (row["direction"] == "lower_is_better")  # True if regression else False
        ranks = row[model_cols].rank(ascending=asc, method="min")
        # filNaN 
        ranks = ranks.fillna(n_models)
        rank_df.loc[ds] = ranks

    if file_name is None:
        from datetime import datetime
        file_name = datetime.now()
    
    avg_rank = rank_df.mean(axis=0)
    avg_rank = avg_rank.sort_values().round(3)
    out_file = f"output/metrics_merged_{file_name}.csv"
    merged.to_csv(out_file, index=False, float_format="%.4f")
    avg_rank.to_csv(f"output/avg_rank_{file_name}.csv")
    print(f"\n▶ Saved merged metrics to `{file_name}`")

    print(merged)
    print(rank_df)
    print(avg_rank)
    return merged



def main():
    model_list = [
        'mlp-piecewiselinear',
        'tabm-piecewiselinear',  
        'tabm-mini-piecewiselinear',  # reported as best in TabM paper 
        # 'moe-sparse', 
        # 'moe-sparse-shared', 
        # 'moe-mini-sparse',
        # 'moe-mini-sparse-shared',
        'moe-sparse-piecewiselinear', 
        # 'moe-sparse-shared-piecewiselinear',
        # 'moe-mini-sparse-piecewiselinear',
        'moe-mini-sparse-shared-piecewiselinear',
        'tabrm-piecewiselinear', # Retrieval + Shared MLP
        'tabrmv2-piecewiselinear', # Retrieval + TabM (Batch ensemble)
        # 'tabrmv2-mini-piecewiselinear' # Retrieval + TabM-mini (Packed Batch ensemble)
    ]
    data_list = [
        "adult", 
        "black-friday", 
        "california", 
        "churn", 
        "covtype2", 
        "diamond", 
        "higgs-small", 
        "house", 
        "microsoft", 
        "otto",  
    ] 

    # Report-view of performance table
    summary_metrics_table(model_list, data_list, output_path="output/metrics.csv", is_print=True, is_save=False)
    summary_hyperparameters(model_list, output_path="output/average_hyperparameters.csv", is_print=False, is_save=False)
    # 

    model_list = [
         'moe-sparse-piecewiselinear', 
        # 'moe-sparse-shared-piecewiselinear',
        # 'moe-mini-sparse-piecewiselinear',
        'moe-mini-sparse-shared-piecewiselinear',
        'tabrm-piecewiselinear', # Retrieval + Shared MLP
        'tabrmv2-piecewiselinear', # Retrieval + TabM (Batch ensemble)
        # 'tabrmv2-mini-piecewiselinear' # Retrieval + TabM-mini (Packed Batch ensemble)
    ]
    # for model in model_list:
    #     save_rank_csv(model, data_list)

    data_list = [
        "adult", 
        "black-friday", 
        "california", 
        "churn", 
        # "covtype2", 
        "diamond", 
        "higgs-small", 
        # "house", 
        # "microsoft", 
        # "otto",  
    ] 
    save_ranks_csv(model=None, data_list=None, file_name="paper_avg_rank")

    # save_ranks_csv(model_list, data_list)

if __name__ == "__main__":
    main()