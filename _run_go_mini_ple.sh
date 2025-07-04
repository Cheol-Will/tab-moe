#!/usr/bin/env bash

# covtype2 takes 19 hours
arch_type="moe-mini-sparse-shared-piecewiselinear"
data_list=("adult" "black-friday" "california" "churn" "diamond" "higgs-small" "house" "otto" "microsoft" "covtype2")
# data="adult"

for data in "${data_list[@]}"
do
    python bin/go.py exp/${arch_type}/${data}/0-tuning --continue
done