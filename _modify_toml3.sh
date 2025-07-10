#!/usr/bin/env bash
set -euo pipefail

# DEST_ROOT="exp/rep-tabr-periodic/why"
DEST_ROOT="exp/rep-tabr-periodic/tabred"

find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  # 2) replace d_main
  if grep -q '^d_main = \[' "$file"; then
    sed -i '/^d_main = \[/,/^]/c\
d_main = [\
    "_tune_",\
    "int-power-of-two",\
    7,\
    10,\
]' "$file"
    echo "Replaced d_main block in: $file"
  fi
done
