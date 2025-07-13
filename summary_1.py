import json
import pandas as pd
import numpy as np
from pathlib import Path

def get_dataset_name(dataset_path: str) -> str:
    name = dataset_path.removeprefix('data/')
    return (
        name.split('-', 4)[-1]
        if name.startswith(('classif-', 'regression-'))
        else name
    )

def load_benchmark(json_path: str) -> pd.DataFrame:
    with open(json_path, 'r') as f:
        raw = json.load(f)

    records = []
    for dataset, info in raw.items():
        direction = info['direction']
        for m in info['models']:
            records.append({
                'dataset': dataset,
                'direction': direction,
                'method':  m['method'],
                'mean':    m.get('single_model_mean'),
                'std':     m.get('single_model_std')
            })
    df = pd.DataFrame.from_records(records)
    df['dataset'] = df['dataset'].str.replace(' ', '_', regex=False) # replace ' ' into '_'
    
    return df

def load_target_single(model: str, report_type: str = 'evaluation') -> pd.DataFrame:
    pattern = f'{model}/**/0-{report_type}/*/report.json'
    json_files = list(Path('exp').glob(pattern))
    if not json_files:
        return pd.DataFrame()
    
    df = pd.json_normalize([
        json.loads(x.read_text()) for x in json_files
    ])
    
    df['dataset'] = df.get('config.data.path', df.get('best.config.data.path')).map(get_dataset_name)
    df['metrics.test.score'] = df['metrics.test.score'].abs()
    df.loc[df['dataset'] == 'house', 'metrics.test.score'] /= 10000 
    
    df_agg = df.groupby('dataset')['metrics.test.score'].agg(['mean', 'std']).reset_index()
    df_agg['method'] = model

    return df_agg


def pivot_target_single(df: pd.DataFrame, direction_map: dict[str,str]) -> pd.DataFrame:
    df['direction'] = df['dataset'].map(direction_map)
    def adjust(row):
        if pd.isna(row['mean']) or pd.isna(row['std']):
            return np.nan
        if row['direction']=='higher_is_better':
            return row['mean'] - row['std']
        else:
            return -(row['mean'] + row['std'])

    df['adj_score'] = df.apply(adjust, axis=1)
    pivot = df.pivot(index='dataset', columns='method', values='adj_score').reset_index()
    dirs = df[['dataset','direction']].drop_duplicates()
    pivot = pivot.merge(dirs, on='dataset', how='left')
    cols = ['dataset','direction'] + [c for c in pivot.columns if c not in ('dataset','direction')]
    return pivot[cols]

def merge_and_rank(
    bench_long: pd.DataFrame,
    tgt_long: pd.DataFrame,
    direction_map: dict[str,str],
    bench_models: list[str] = None,           
) -> pd.DataFrame:
    if tgt_long is not None:
        tgt = tgt_long.copy()
        tgt['direction'] = tgt['dataset'].map(direction_map)

        cols = ['dataset','direction','method','mean','std']
        combined = pd.concat([bench_long[cols], tgt[cols]], ignore_index=True)

        if bench_models is not None:
            tgt_models = tgt['method'].unique().tolist()
            allowed_models = set(bench_models) | set(tgt_models)
            combined = combined[combined['method'].isin(allowed_models)]

        allowed_datasets = tgt['dataset'].unique().tolist()
        combined = combined[combined['dataset'].isin(allowed_datasets)]
    else:
        combined = bench_long
        if bench_models is not None:
            allowed_models = set(bench_models) 
            combined = combined[combined['method'].isin(allowed_models)]
        cols = ['dataset','direction','method','mean','std']
        combined = combined[cols]


    def rank_group(g):
        direction = g['direction'].iat[0]
        ascending = (direction == 'lower_is_better')
        sg = g.sort_values('mean', ascending=ascending).reset_index(drop=True)

        ranks = {}
        ref_mean, ref_std = sg.loc[0, ['mean','std']]
        curr_rank = 1
        ranks[sg.loc[0,'method']] = curr_rank

        for i in range(1, len(sg)):
            mname = sg.loc[i,'method']
            mmean = sg.loc[i,'mean']
            mstd  = sg.loc[i,'std']

            if direction == 'higher_is_better':
                same_tier = (mmean >= ref_mean - ref_std)
            else:
                same_tier = (mmean <= ref_mean + ref_std)

            if same_tier:
                ranks[mname] = curr_rank
            else:
                curr_rank += 1
                ref_mean, ref_std = mmean, mstd
                ranks[mname] = curr_rank

        return pd.Series(ranks)
    rank_long = (
        combined
        .groupby('dataset', sort=False)
        .apply(rank_group)
        .reset_index()
        .rename(columns={'level_1':'method', 0:'rank'})
    )
    high_ds = [ds for ds, dir_val in direction_map.items() if dir_val == 'higher_is_better']
    low_ds  = [ds for ds, dir_val in direction_map.items() if dir_val == 'lower_is_better']

    high_rank = rank_long[rank_long['dataset'].isin(high_ds)].reset_index(drop=True)
    low_rank  = rank_long[rank_long['dataset'].isin(low_ds)].reset_index(drop=True)


    return rank_long, high_rank, low_rank


def pivot_rank(rank_long: pd.DataFrame) -> pd.DataFrame:
    """
    Transform a long-form rank DataFrame (dataset, method, rank)
    into a wide-form matrix with:
      - index = method (rows)
      - columns = dataset
      - values = rank

    Missing ranks will be NaN.
    """
    # pivot
    rank_wide = rank_long.pivot(
        index='method',
        columns='dataset',
        values='rank'
    )
    # optional: sort methods and datasets
    rank_wide = rank_wide.sort_index(axis=0).sort_index(axis=1)
    return rank_wide


def pivot_mean_std(
    bench_long: pd.DataFrame,
    tgt: pd.DataFrame,
    bench_models: list[str] = None,
    use_std: bool = True,
) -> pd.DataFrame:
    """
    format string : mean_std
    """

    cols = ['dataset','method','mean','std']
    combined = pd.concat([bench_long[cols], tgt[cols]], ignore_index=True)

    if bench_models is not None:
        tgt_models = tgt['method'].unique().tolist()
        allowed_models = set(bench_models) | set(tgt_models)
        combined = combined[combined['method'].isin(allowed_models)]

    allowed_datasets = tgt['dataset'].unique().tolist()
    combined = combined[combined['dataset'].isin(allowed_datasets)]

    def fmt(row):
        if not use_std:
            if pd.isna(row['mean']):
                return ""
            else:
                return f"{row['mean']:.4f}"
        else:
            if pd.isna(row['std']):
                if pd.isna(row['mean']):
                    return ""
                else:
                    return f"{row['mean']:.4f}"
            else:
                return f"{row['mean']:.4f}±{row['std']:.4f}"

    combined['mean_std'] = combined.apply(fmt, axis=1)

    pivot = combined.pivot(
        index='method',
        columns='dataset',
        values='mean_std'
    )

    pivot.columns = [c.replace(" ", "_") for c in pivot.columns]

    allowed_datasets = pivot.columns

    return pivot


def print_directions_one_line(direction_map, datasets):
    # "churn: higher_is_better, adult: higher_is_better, …"
    pairs = [f"{ds}: {direction_map.get(ds.replace('_', ' '), 'unknown')[:-10]}" for ds in datasets]
    print(", ".join(pairs))


if __name__ == "__main__":

    target_models = [
        # 'tabrmv2-piecewiselinear',        
        # 'tabrmv2-mini-periodic', # Retrieval + TabM-mini (Mini ensemble)
        # 'tabrmv3-mini-periodic',        
        # 'tabrmoev3-periodic',
        # 'tabrmv4-mini-periodic',        
        # 'tabrmv4-shared-periodic',
        # 'tabrmoev4-periodic',
        # 'tabrmoev4-drop-periodic',
        # 'tabr-pln-periodic',
        # 'reproduced-tabr-periodic',
    ]
    # tgt = load_target_single('rep-tabr-periodic')
    # tgt = load_target_single('tabr-pln-multihead-periodic')
    # tgt = load_target_single('retransformer-periodic')
    
    with open("output/paper_metrics.json", "r") as f:
        raw = json.load(f)

    direction_map = {
        dataset: info["direction"]
        for dataset, info in raw.items()
    }


    bench_models = [
        'MLP', 'Excel-plugins', 'SAINT', 'FT-T',
        'T2G', 'MNCA', 'TabR', 'MLP-piecewiselinear',
        'LightGBM', 'XGBoost', 'CatBoost',
        'MNCA-periodic', 'TabR-periodic',
        'TabM', 'TabMmini-piecewiselinear',
    ]
    # model = 'retransformer-periodic'
    # model = 'tabr-pln-multihead-periodic'
    # model = 'tabr-pln-periodic'
    # model = 'rep-tabr-periodic'
    # model = "tabm-rankp-piecewiselinear"
    # model = "tabpln-mini-piecewiselinear"
    model = "taba-piecewiselinear"

    tgt = load_target_single(model)
    bench = load_benchmark("output/paper_metrics.json")

    print(tgt)
    print(bench)
    ranks, clf_rank, reg_rank = merge_and_rank(bench, tgt, direction_map, bench_models)
    # rank_list = [clf_rank, reg_rank, ranks]
    rank_list = [ranks]
    for rank in rank_list:
        print(rank)
        ranks_pivot = pivot_rank(rank)
        avg_ranked = ranks_pivot.mean(axis=1).sort_values()
        mean_std_table = pivot_mean_std(bench, tgt, bench_models, use_std=False)

        print(ranks_pivot)
        print(mean_std_table)

        print("\nDataset directions:")
        print_directions_one_line(direction_map, mean_std_table.columns)
        print()

        print(avg_ranked)
        print()
        
        mean_std_table.to_csv(f"output/metrics_for_ppt_250711_{model}.csv") 
        avg_ranked.to_csv(f"output/avg_ranks_for_ppt_250711_{model}.csv") 

    # tabm_bench, _, _ = merge_and_rank(bench, None, direction_map, bench_models)
    # ranks_pivot = pivot_rank(tabm_bench)
    # avg_ranked = ranks_pivot.mean(axis=1).sort_values()
    # print(avg_ranked)