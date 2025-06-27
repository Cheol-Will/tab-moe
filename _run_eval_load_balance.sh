# The config is stored at exp/<any/path>/0-evaluation/0.toml

model="moe-sparse-shared-piecewiselinear"

mkdir -p exp/${model}/churn/0-load-balance/0/ 

cp exp/${model}/churn/0-evaluation/0.toml exp/${model}/churn/0-load-balance/
cp exp/${model}/churn/0-evaluation/0/checkpoint.pt exp/${model}/churn/0-load-balance/0/

python bin/evaluate_load_balance.py exp/${model}/churn/0-load-balance --function "bin.model_load_balance.main"