#!/usr/bin/env bash
set -euo pipefail

# Check the destination root!!
DEST_ROOT="exp/mlp-debug2"

# 1) Define replacement block
BLOCK=$(cat << 'EOF'
[space.model]
arch_type = "tabrm"
sample_rate = [
    "_tune_",
    "uniform",
    0.05,
    0.6,
]
k = [
    "_tune_",
    "int",
    32,
    128,
    16
]

[space.model.backbone]
n_blocks = [
    "_tune_",
    "int",
    1,
    4,
]
d_block = [
    "_tune_",
    "int",
    64,
    512,
    16,
]
dropout = [
    "_tune_",
    "?uniform",
    0.0,
    0.0,
    0.5,
]

[space.model.num_embeddings]
type = "PiecewiseLinearEmbeddingsV2"
d_embedding = [
    "_tune_",
    "int",
    8,
    32,
    4,
]

[space.bins]
n_bins = [
    "_tune_",
    "int",
    2,
    128,
]
EOF
)

# 2) Find all 0-tuning.toml files
echo "Searching for 0-tuning.toml under ${DEST_ROOT}..."
mapfile -d $'\0' files < <(find "${DEST_ROOT}" -type f -name "0-tuning.toml" -print0)

if [ ${#files[@]} -eq 0 ]; then
  echo "ERROR: No files found in ${DEST_ROOT}" >&2
  exit 1
fi

echo "Found ${#files[@]} file(s):"
printf '  %s\n' "${files[@]}"

# 3) Process each file
for file in "${files[@]}"; do
  echo "Updating $file"
  tmp="${file}.tmp"

  # keep everything up to (but not including) [space.model]
  sed '/^\[space\.model\]/,$d' "$file" > "$tmp"
  # append the replacement block
  printf '%s\n' "$BLOCK" >> "$tmp"
  # overwrite original
  mv "$tmp" "$file"
done

echo "All files have been updated."
