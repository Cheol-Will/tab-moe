#!/usr/bin/env bash

arch_list=(
  # "qreformer-d3-h4-m128-mqa"
  # "qreformer-d1-h4-m32"
  # "qreformer-d0-m96"
  # "qreformer-deubg-d1-h1-m32"
  # "qreformer-deubg-d3-h4-m32"
  # "qtab-naive-t1"


###############################
  # GPU0
  # "qtab-naive-sdp-t02"
  # "qtab-naive-sdp-t001"
###############################
  # GPU0

  # "qreformer-deubg-d3-h4-m32"
  # "qreformer-deubg-d3-h4-m96-mqa"
  # "qreformer-deubg-d3-h4-m128-mqa"


###############################
  # GPU2
  # "qtab-naive-l2-t1"
  # "qtab-naive-l2-t02"
  # "qtab-naive-sdp-t02"
  # "qtab-naive-l2-t001"
  # "qtab-naive-cossim-t01"
###############################
  # "qtabformer-key-cossim-t01"
  # "qtabformer-key-cossim-t001"

###############################
  # "qtabformer-key-y-cossim-t01"
  # "qtabformer-key-y-cossim-t001"
###############################
  # "qtab-naive-cossim-t02"
  # "qtabformer-key-cossim-t002"
  # "qtabformer-key-y-cossim-t002"
###############################


  # "qtabformerv3-key-key-value-y-cossim-t01"


  # "qtabformerv3-key-key-value-key-y-cossim-t01"

  # "qtabformerv3-key-ky-value-ky-cossim-t01"
# 
  # "qtabformerv4-key-ky-value-ky-cossim-t01"

  # "qtabformer-key-k-value-ky-cossim-t01"
  # "retransformer-periodic"
  # "qtab-naive-cossim"
  # "qtabformer-key-k-value-ky-cossim"
  # "qtabformer-key-ky-value-ky-cossim"
  # "qtab-naive-cossim"

  # "qtabformer-query-1-key-k-value-ky-mha"
  # "qtabformer-query-4-key-k-value-ky-mha"
  ##################################################
  # "qtabformer-query-1-key-k-value-ky-mqa" # gpu
  # "qtabformer-query-4-key-k-value-ky-mqa" # 


  # "qtabformer-query-8-key-k-value-ky-mqa" # GPU0
  # "qtabformer-query-8-key-k-value-ky-mha" #GPU1
  # "qtabformer-query-16-key-k-value-ky-mqa" # GPU 2




  # "qtabformer-query-4-key-k-value-ky-mha-8" # GPU0

  # "qtabformer-query-4-key-k-value-ky-mqa-8" # GPU 1
  
  
  # "qtabformer-query-4-key-k-value-ky-mha-moh" # GPU 1
  # "qtabformer-query-4-key-k-value-ky-mqa-moh" # GPU 2


  # "qtabformer-query-1-key-k-value-ky-mha-4"
  # "qtabformer-query-4-key-k-value-ky-mha-4-moh"

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