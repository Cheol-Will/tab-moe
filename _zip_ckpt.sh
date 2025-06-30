#!/bin/bash

OUTFILE="checkpoint_eval0_only.tar.gz"

models=("moe-mini-sparse-shared-piecewiselinear" "moe-sparse-piecewiselinear")

TMP_LIST=$(mktemp)

for model in "${models[@]}"; do
    find "exp/$model" -type f -path "*/0-evaluation/0/checkpoint.pt" >> "$TMP_LIST"
done

tar -czf "$OUTFILE" -T "$TMP_LIST"

rm "$TMP_LIST"

echo "Done: $OUTFILE created with only 0-evaluation/0/checkpoint.pt files."
