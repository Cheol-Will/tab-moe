#!/usr/bin/env bash
set -euo pipefail

DEST_ROOT="exp/rep-tabr-periodic/why"

REPLACEMENT_BLOCK=$(cat << 'EOF'
[space.model.num_embeddings]
type = "PeriodicEmbeddings"
n_frequencies = [
    "_tune_",
    "int",
    8,
    96,
]
frequency_init_scale = [
    "_tune_",
    "loguniform",
    0.01,
    100.0,
]
d_embedding = [
    "_tune_",
    "int",
    4,
    64,
]
lite = true
EOF
)

# 1) Find all 0-tuning.toml files
echo "Searching for 0-tuning.toml under ${DEST_ROOT}..."
mapfile -d $'\0' files < <(find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0)

if [ ${#files[@]} -eq 0 ]; then
  echo "ERROR: No files found in ${DEST_ROOT}" >&2
  exit 1
fi

echo "Found ${#files[@]} file(s):"
printf '  %s\n' "${files[@]}"

# 2) Process each file
for file in "${files[@]}"; do
  echo "Updating $file"
  tmp="${file}.tmp"

  # Use awk to:
  # 1. Detect [space.model.num_embeddings] block
  # 2. Remove that block
  # 3. Replace with new content
  awk -v replacement="$REPLACEMENT_BLOCK" '
    BEGIN { in_block = 0 }
    /^\[space\.model\.num_embeddings\]/ {
      print replacement
      in_block = 1
      next
    }
    /^\[/ && in_block {
      in_block = 0
    }
    !in_block
  ' "$file" > "$tmp"

  mv "$tmp" "$file"
done

echo "All files have been updated."
