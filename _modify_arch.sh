#!/usr/bin/env bash
set -euo pipefail

# DEST_ROOT="exp/rep-ensemble"
DEST_ROOT="exp/tabm-rankp-piecewiselinear"

find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  # Replace arch_type = "tabm" with "tabm-rankone"
  if grep -q '^arch_type *= *"tabm-rankone"' "$file"; then
    sed -i 's/^arch_type *= *"tabm-rankone"/arch_type = "tabm-rankp"/' "$file"
    echo "Replaced arch_type in: $file"
  fi

  # add p=
done
