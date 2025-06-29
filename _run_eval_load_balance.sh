# The config is stored at exp/<any/path>/0-evaluation/0.toml
# 'moe-sparse', 
# 'moe-sparse-piecewiselinear', 
# 'moe-sparse-shared', 
# 'moe-sparse-shared-piecewiselinear',
# 'moe-mini-sparse',
# 'moe-mini-sparse-piecewiselinear',
# # 'moe-mini-sparse-shared', -> No Result
# 'moe-mini-sparse-shared-piecewiselinear',


# model="moe-sparse-shared-piecewiselinear"
# model="moe-sparse-piecewiselinear"
model="moe-mini-sparse-shared-piecewiselinear"
data_list=("adult" "black-friday" "california" "diamond" "higgs-small" "house" "otto" "covtype2" "microsoft" "churn")
for data in "${data_list[@]}"
do
    mkdir -p exp/${model}/${data}/0-load-balance/0/ 

    cp exp/${model}/${data}/0-evaluation/0.toml exp/${model}/${data}/0-load-balance/
    cp exp/${model}/${data}/0-evaluation/0/checkpoint.pt exp/${model}/${data}/0-load-balance/0/

    python bin/evaluate_load_balance.py exp/${model}/${data}/0-load-balance --function "bin.model_load_balance.main"
done