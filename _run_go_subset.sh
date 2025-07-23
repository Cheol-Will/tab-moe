#!/usr/bin/env bash

arch_list=(
  "qtabformer-query-4-key-k-value-ky-mqa-moh" # GPU 0
  # "qtabformer-query-4-key-k-value-ky-mqa" # GPU 1
  # "qtabformer-query-4-key-k-value-ky-mqa-d4" # GPU 2
)

data_list=(
  "churn"
  "tabred/sberbank-housing"
  "why/regression-num-medium-0-medical_charges"
  "why/regression-cat-medium-0-OnlineNewsPopularity/"
  "tabred/ecom-offers"
  "why/classif-num-medium-0-credit"
)

for arch_type in "${arch_list[@]}"
do
  for data in "${data_list[@]}"
  do
    python bin/go.py exp/${arch_type}/${data}/0-tuning --continue | tee exp/${arch_type}/${data}/0-tuning.log
  done
done