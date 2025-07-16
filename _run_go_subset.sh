#!/usr/bin/env bash
# arch_type="tabr-pln-periodic"
# arch_type="retransformer-periodic"
# arch_type="tabr-pln-multihead-periodic"

# arch_type="rep-tabr-periodic"

# arch_type="tabpln-mini-piecewiselinear"
# arch_type="taba-piecewiselinear"
# arch_type="taba-k128-piecewiselinear"
# arch_type="taba-piecewiselinear"
# arch_type="taba-moe-piecewiselinear"
# arch_type="reformer-moh-plr"
# arch_type="reformer-d1-m64"
# arch_type="reformer-d1-m32"
arch_type="reformer-d1-m32-aux"

# data_list=("churn" "house" "adult" "california" "diamond" "otto" "higgs-small" "black-friday" "microsoft" "covtype2")
data_list=(
  "churn"
  "tabred/sberbank-housing"
  "why/regression-cat-medium-0-OnlineNewsPopularity/"
  "tabred/ecom-offers"
  "why/classif-num-medium-0-credit"
  "why/regression-num-medium-0-medical_charges"
  "house"
  ########################################################
  # "why/classif-num-medium-0-bank-marketing"
  # "why/classif-num-medium-0-kdd_ipums_la_97-small"
  # "why/classif-cat-medium-0-KDDCup09_upselling"
  ########################################################
  # "why/classif-num-medium-0-MagicTelescope"
  # "why/regression-num-medium-0-fifa"
  # "adult"
  # "why/regression-num-large-0-year"
  # "why/regression-cat-medium-0-house_sales"
  # "tabred/maps-routing"
  # "california"
  # "tabred/cooking-time"
  # "why/regression-cat-medium-0-Brazilian_houses"
  # "why/regression-num-medium-0-elevators"
)

for data in "${data_list[@]}"
do
    python bin/go.py exp/${arch_type}/${data}/0-tuning --continue | tee exp/${arch_type}/${data}/0-tuning.log 
done
