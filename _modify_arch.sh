#!/usr/bin/env bash
set -euo pipefail

# DEST_ROOT="exp/rep-ensemble"
# DEST_ROOT="exp/taba-piecewiselinear"
DEST_ROOT="exp/taba-moe-piecewiselinear"

# dest_type="taba-k128-piecewiselinear"


find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  # Replace arch_type = "tabm" with "tabm-rankone"
  if grep -q '^arch_type *= *"taba"' "$file"; then
    sed -i 's/^arch_type *= *"taba"/arch_type = "taba-moe"/' "$file"
    echo "Replaced arch_type in: $file"
  fi

#   if grep -q '^k *= *32$' "$file"; then
#     sed -i 's/^k *= *32$/k = 128/' "$file"
#     echo "Replaced k value in: $file"
#   fi

#   if grep -q '^n_blocks = \[' "$file"; then
#     sed -i '/^n_blocks = \[/,/^]/c\
# n_blocks = [\
#     "_tune_",\
#     "int",\
#     1,\
#     10,\
# ]' "$file"
#     echo "Replaced n_blocks block in: $file"
#   fi


  # add p=
done
