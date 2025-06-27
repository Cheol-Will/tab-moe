#!/usr/bin/env bash

arch_type="moe-mlp"
# data_list=("california")
# data_list=("california" "churn" "covtype2" "diamond" "higgs-small" "house" "microsoft" "adult" "black-friday")
# data_list=("adult" "black-friday" "california" "churn" "diamond" "higgs-small" "house" "microsoft" "otto"  "covtype2")
data_list=("diamond" "higgs-small" "house" "microsoft" "adult" "black-friday")

for data in "${data_list[@]}"
do
    python bin/go.py exp/${arch_type}/${data}/0-tuning --continue
done