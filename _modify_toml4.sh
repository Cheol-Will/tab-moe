#!/usr/bin/env bash
set -euo pipefail

# DEST_ROOT="exp/rep-tabr-periodic/why"
DEST_ROOT="exp/rep-tabr-periodic/tabred"

find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  # Replace n_trials = 100 -> n_trials = 25
  if grep -q '^n_trials *= *100' "$file"; then
    sed -i 's/^\(n_trials *= *\)100$/\125/' "$file"
    echo "Replaced n_trials in: $file"
  fi

  # Remove n_startup_trials = 20
  if grep -Eq '^\[sampler\]|^n_startup_trials *= *20' "$file"; then
    sed -i '/^\[sampler\]/d; /^n_startup_trials *= *20/d' "$file"
    echo "Removed sampler block in: $file"
  fi


  # Remove share_training_batches = false
  if grep -q '^share_training_batches *= *false' "$file"; then
    sed -i '/^share_training_batches *= *false/d' "$file"
    echo "Removed share_training_batches in: $file"
  fi


  


done


