#!/usr/bin/env bash
set -euo pipefail

# Check the destination root!!
# DEST_ROOT="exp/mlp-debug2"
# DEST_ROOT="exp/tabrmoev4-drop-periodic"
DEST_ROOT="exp/tabr-pln-periodic"


# find o-tuning.toml and replace below
# arch_type = "tabr" -> arch_type = "tabr-pln"
# k = 1 -> k = 32

find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  # Replace arch_type = "tabr" -> "tabr-pln"
  if grep -q '^arch_type *= *"tabr"' "$file"; then
    sed -i 's/^\(arch_type *= *\)"tabr"/\1"tabr-pln"/' "$file"
    echo "Replaced arch_type in: $file"
  fi

  # Replace k = 1 -> k = 32
  if grep -q '^k *= *1$' "$file"; then
    sed -i 's/^\(k *= *\)1$/\132/' "$file"
    echo "Replaced k in: $file"
  fi
done
