#!/usr/bin/env bash

model="moe"
data="adult"

mkdir -p "exp/reproduce/${model}/${data}"

cp "exp/${model}/${data}/0-tuning.toml" "exp/reproduce/${model}/${data}/"

python bin/tune.py "exp/reproduce/${model}/${data}/0-tuning.toml" --continue