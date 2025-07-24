#!/usr/bin/env bash

arch_list=(
  # "qtabformer-query-4-key-k-value-ky-mqa-moh" # GPU 0
  # "qtabformer-query-4-key-k-value-ky-mqa" # GPU 1
  # "qtabformer-query-4-key-k-value-ky-mqa-d4" # GPU 2
  # "qraugmlp-key-k-value-ky-m96" # GPU 0
  # "qraugmlp-key-k-value-ky-m96" # GPU 1

  # "qraugmlp-key-k-value-qky-m32" # GPU 0
  # "qraugmlp-key-k-value-ky-m32" # GPU 2
  # "qtab-naive-cossim-cl" # GPU 0
  # "qraugresnet-key-k-value-ky-m32"

  # "qraugresnet-key-k-value-qky-m32" # GPU 2

  "qtabformer-query-1-key-k-value-ky-mqa-moh" # GPU 0
  # "qtabformer-query-4-key-k-value-ky-mqa-moh" # GPU 1, 2 

)

data_list=(
  # regression
  "tabred/sberbank-housing"
  "why/regression-num-medium-0-medical_charges" # GPU 1
  "why/regression-cat-medium-0-OnlineNewsPopularity/" # GPU 2

  ## classification
  "churn"
  "tabred/ecom-offers" # GPU 0
  "why/classif-num-medium-0-credit"

)

for model in "${arch_list[@]}"
do
  for data in "${data_list[@]}"
  do
    python bin/go.py exp/${model}/${data}/0-tuning --continue | tee exp/${model}/${data}/0-tuning.log
  done
done