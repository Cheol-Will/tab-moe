#!/usr/bin/env bash

arch_list=(
  "reformer-d3-h4-m96"
)

data_list=(
  # "churn"
  "why/regression-num-medium-0-medical_charges"
  # "tabred/sberbank-housing"
  # "why/regression-cat-medium-0-OnlineNewsPopularity/"
  # "tabred/ecom-offers"
  # "why/classif-num-medium-0-credit"
)

for arch_type in "${arch_list[@]}"
do
  for data in "${data_list[@]}"
  do
    python bin/go.py exp/${arch_type}/${data}/0-tuning --continue | tee exp/${arch_type}/${data}/0-tuning.log
  done
done