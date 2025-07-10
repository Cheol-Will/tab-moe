#!/usr/bin/env bash
set -euo pipefail

# DEST_ROOT="exp/rep-tabr-periodic/why"
DEST_ROOT="exp/tabr-pln-periodic"

find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  # Remove share_training_batches = false
  if grep -q '^share_training_batches *= *false' "$file"; then
    sed -i '/^share_training_batches *= *false/d' "$file"
    echo "Removed share_training_batches in: $file"
  fi
done