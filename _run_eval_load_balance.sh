# The config is stored at exp/<any/path>/0-evaluation/0.toml

model="moe-sparse-shared-piecewiselinear"

python bin/evaluate_load_balance.py exp/${model}/churn/0-evaluation --function "bin.model_load_balance.main"