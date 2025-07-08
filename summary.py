import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def get_dataset_name(dataset_path: str) -> str:
    name = dataset_path.removeprefix('data/')
    return (
        name.split('-', 4)[-1]
        if name.startswith(('classif-', 'regression-'))
        else name
    )

def load_reports(model: str, report_type: str = 'evaluation') -> pd.DataFrame:
    pattern = f'{model}/**/0-{report_type}/*/report.json'
    json_files = list(Path('exp').glob(pattern))
    if not json_files:
        return pd.DataFrame()
    
    df = pd.json_normalize([
        json.loads(x.read_text()) for x in json_files
    ])
    df['Dataset'] = df.get('config.data.path', df.get('best.config.data.path')).map(get_dataset_name)
    return df

def print_metrics(model):
    df = load_reports(model)
    if df.empty:
        print(f"No data found for {model}")
        return

    df_agg = df.groupby('Dataset')['metrics.test.score'].agg(['mean', 'std'])
    print(model, '-' * 50)
    print(df_agg, '\n')

def summary_metrics_table(model_list, data_list, output_path="output/metrics.csv", is_print=False, is_save=False):
    all_results = {}
    for model in model_list:
        df = load_reports(model)
        if df.empty:
            continue

        df_agg = df.groupby('Dataset')['metrics.test.score'].mean().reindex(data_list)
        all_results[model] = df_agg

    df_all = pd.DataFrame(all_results).T
    if is_print:
        print(df_all)
    if is_save:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_all.to_csv(output_path, float_format="%.4f")
        print(f"Saved all model metrics to {output_path}")
    return df_all

def extract_hyperparameters(df, model):
    base_params = [
        'config.model.backbone.n_blocks',
        'config.model.backbone.d_block',
        'config.model.backbone.dropout',
        'config.optimizer.lr',
        'config.optimizer.weight_decay'
    ]

    moe_params = [
        'config.model.backbone.k',
        'config.model.backbone.moe_ratio',
        'config.model.backbone.num_experts'
    ]

    default_k = ['config.model.k']
    params = base_params + (moe_params if 'moe' in model else default_k)
    selected = ['Dataset'] + [p for p in params if p in df.columns]

    dfh = df[df['config.seed'] == 0][selected].copy()
    if 'config.model.backbone.dropout' in dfh.columns:
        dfh['has_dropout'] = (dfh['config.model.backbone.dropout'] > 0).astype(float)
    if 'config.optimizer.weight_decay' in dfh.columns:
        dfh['has_weight_decay'] = (dfh['config.optimizer.weight_decay'] > 0).astype(float)

    return dfh.groupby('Dataset').mean()

def summary_hyperparameters(model_list, output_path="output/average_hyperparameters.csv", is_print=False, is_save=False):
    records = []
    for model in model_list:
        df = load_reports(model)
        if df.empty:
            continue

        dfh = extract_hyperparameters(df, model)
        if is_print:
            print(model)
            print(dfh)
            print()

        avg_row = dfh.mean().rename(model)
        records.append(avg_row)

    if is_save and records:
        df_all_avg = pd.DataFrame(records)
        df_all_avg.index.name = "Model"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_all_avg.to_csv(output_path, float_format="%.4f")
        print(f"Saved average hyperparameters of all models to {output_path}")

def print_tuning_time(model):
    df = load_reports(model, report_type='tuning')
    if df.empty:
        print(f"No tuning data found for {model}")
        return
    
    df['time'] = pd.to_timedelta(df['time'])
    total = df.groupby('Dataset')['time'].sum().sort_values()
    print(model, '-' * 50)
    print(total, '\n')
    print("Total:", total.sum())
    total.to_csv(f"searching_time_{model}.csv", header=["total_time"])

def calculate_ranks(merged: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    if model_cols is None:
        model_cols = merged.columns
        model_cols = model_cols.drop(["dataset", "direction"])
    rank_df = pd.DataFrame(index=merged["dataset"], columns=model_cols, dtype=float)
    n_models = len(model_cols)
    for _, row in merged.iterrows():
        asc = row["direction"] == "lower_is_better"
        ranks = row[model_cols].rank(ascending=asc, method="min").fillna(n_models)
        rank_df.loc[row["dataset"]] = ranks
    return rank_df

def merge_and_calculate_rank(model: str, benchmark_model : list[str] = None, data_list: list[str] = None, is_save: bool = False, is_print: bool = False, file_name: str = None) -> pd.DataFrame:
    paper = pd.read_csv("output/paper_metrics.csv")
    model = [model] if isinstance(model, str) else model
    df = summary_metrics_table(model, data_list)
    summary_t = df.T.reset_index().rename(columns={"Dataset": "dataset"})
    merged = paper.merge(summary_t, on="dataset", how="left")
    if data_list is not None:
        merged = merged[merged["dataset"].isin(data_list)]
    merged.drop(columns=["TabPFN"], inplace=True)

    if "house" in merged["dataset"].values:
        merged.loc[merged["dataset"] == "house", model] /= 10000

    model_cols = [c for c in merged.columns if c not in ("dataset", "direction")]
    if benchmark_model is not None:
        model_cols = model + benchmark_model
    # print(merged.columns)
    merged = merged[['dataset', 'direction'] + model_cols]
    merged[model_cols] = merged[model_cols].abs().round(4)

    rank_df = calculate_ranks(merged, model_cols)
    avg_rank = rank_df.mean(axis=0).sort_values()

    if is_save:
        file_name = file_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        merged.to_csv(f"output/metrics_merged_{file_name}.csv", index=False, float_format="%.4f")
        avg_rank.to_csv(f"output/avg_rank_{file_name}.csv")
        print(f"\nSaved merged metrics to output/metrics_merged_{file_name}.csv")
    if is_print:
        print(merged)
        print(rank_df)
        print(avg_rank)
    return merged


def main():
    model_list = [
        # 'moe-sparse-piecewiselinear', 
        # 'moe-sparse-shared-piecewiselinear',
        # 'moe-mini-sparse-piecewiselinear',
        # 'moe-mini-sparse-shared-piecewiselinear',
        # 'tabrm-piecewiselinear', # Retrieval + Shared MLP
        'tabrmv2-piecewiselinear', # Retrieval + TabM (Batch ensemble)
        # 'tabrmv2-periodic', # Retrieval + TabM (Batch ensemble)
        'tabrmv2-mini-periodic', # Retrieval + TabM-mini (Mini ensemble)
        # 'tabrmv2-mini-piecewiselinear', # Retrieval + TabM-mini (Packed Batch ensemble)
        # 'tabrmv3-cs-periodic',
        # 'tabrmv3-mini-cs-periodic',
        # 'tabrmv3-shared-cs-periodic'
        # 'tabrmoev3-periodic',
        # current -------------------------
        'tabrmv3-mini-periodic',        
        'tabrmoev3-periodic',
        'tabrmv4-mini-periodic',        
        'tabrmv4-shared-periodic',
        'tabrmoev4-periodic',
        # 'tabrmv4-moe-periodic',
    ]
    benchmark_model = [
        'MLP',
        'ResNet',
        'Trompt',
        'MLP-Mixer',
        'Excel-plugins',
        'SAINT',
        'FT-T',
        'XGBoost',
        'LightGBM',
        'CatBoost',
        'TabR',
        'TabR-periodic',
        'MNCA',
        'MNCA-periodic',
        'TabM',
        'TabMmini-piecewiselinear',
    ]

    # merged = merge_and_calculate_rank(model=model_list, benchmark_model=benchmark_model, data_list=None, is_save=False, is_print=False)
    # filtered = merged[merged['tabrmv4-mini-periodic'].notna()]
    
    target_model_list = [
        'moe-sparse-piecewiselinear', 
        'moe-mini-sparse-shared-piecewiselinear',
        'tabrmv2-piecewiselinear',        
        'tabrmv3-mini-periodic',        
        'tabrmoev3-periodic',
        'tabrmv4-mini-periodic',        
        'tabrmv4-shared-periodic',
        'tabrmoev4-periodic',
        'tabrmoev4-drop-periodic'
    ]    
    for target_model in target_model_list:
        continue
    # filter_model = 'tabrmv4-shared-periodic'
        merged = merge_and_calculate_rank(model=[target_model], benchmark_model=None, data_list=None, is_save=False, is_print=False)
        # print(merged)
        filtered = merged[merged[target_model].notna()]
        filtered_rank =  calculate_ranks(merged=filtered, model_cols=None)
        avg_rank = filtered_rank.mean(axis=0).sort_values()
        print(target_model)
        print(filtered)
        print(avg_rank)
        print()
    merged = merge_and_calculate_rank(model=target_model_list, benchmark_model=None, data_list=None, is_save=False, is_print=False)
    print(merged)
    # filtered = merged[merged['tabrmoev3-periodic'].notna()]
    
    #
    #  filtered = merged[merged["tabrmv2-mini-periodic"].notna()]
    # filtered.to_csv("output/Temp_123.csv")
    # print(filtered.loc[:, ["dataset", "direction", "tabrmv2-periodic", "CatBoost", "TabMmini-piecewiselinear"]])


if __name__ == "__main__":
    main()