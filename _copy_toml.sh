#!/usr/bin/env bash
# src_type="tabm"
src_type="moe-sparse-shared-piecewiselinear"
dest_type="moe-mini-sparse-shared-piecewiselinear"
# data_list=("adult" "black-friday")
# data_list=("california" "churn" "covtype2" "diamond" "higgs-small" "house" "microsoft" "otto" "tabred" "why")
data_list=("adult" "black-friday" "california" "churn" "covtype2" "diamond" "higgs-small" "house" "microsoft" "otto")

for dataset in "${data_list[@]}"
do
    echo "Processing dataset: ${dataset}"
    mkdir -p "exp/${dest_type}/${dataset}"
    cp "exp/${src_type}/${dataset}/0-tuning.toml" "exp/${dest_type}/${dataset}/"
done