#!/usr/bin/env bash

arch_type="moe-mlp"
data="adult"

python bin/tune.py "exp/reproduce/${arch_type}/${data}/0-tuning.toml" --continue