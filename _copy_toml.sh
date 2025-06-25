#!/usr/bin/env bash
src_type="mlp"
arch_type="moe-mlp"
data_list=("adult" "black-friday")
# data_list=("california" "churn" "covtype2" "diamond" "higgs-small" "house" "microsoft" "otto" "tabred" "why")
# data_list=("adult" "black-friday" "california" "churn" "covtype2" "diamond" "higgs-small" "house" "microsoft" "otto" "tabred" "why")

for dataset in "${data_list[@]}"
do
    echo "Processing dataset: ${dataset}"
    mkdir -p "exp/${arch_type}/${dataset}"
    cp "exp/${src_type}/${dataset}/0-tuning.toml" "exp/${arch_type}/${dataset}/"
done