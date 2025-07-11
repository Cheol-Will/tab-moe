#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="exp/retransformer-aux-periodic"

find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do

  # [space.model] add p
  if grep -q '^\[space\.model\]' "$file" && ! grep -q '^p *= *\[' "$file"; then
    sed -i '/^\[space\.model\]/a \
aux_loss_weight = [\
    "_tune_",\
    "loguniform",\
    0.01,\
    1.0,\
]' "$file"
    echo "Inserted p parameter under [space.model] in: $file"
  fi

done
