#!/usr/bin/env bash

# arch_type="moe-sparse-shared"
arch_type="moe-sparse-piecewiselinear"

data_list=("adult" "black-friday" "california" "diamond" "higgs-small" "house" "otto" "covtype2" "microsoft" "churn")

for data in "${data_list[@]}"
do
    python bin/go.py exp/${arch_type}/${data}/0-tuning --continue
done