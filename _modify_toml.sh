#!/usr/bin/env bash
set -euo pipefail

# Check the destination root!!
# DEST_ROOT="exp/mlp-debug2"
DEST_ROOT="exp/tabrmv2-mini-periodic"


# 1) Define replacement block
BLOCK=$(cat << 'EOF'

[space.model]
arch_type = "tabrmv2-mini"
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
    5,
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
type = "PeriodicEmbeddings"
n_frequencies = [
    "_tune_",
    "int",
    16,
    96,
    4,
]
d_embedding = [
    "_tune_",
    "int",
    16,
    32,
    4,
]
frequency_init_scale = [
    "_tune_",
    "loguniform",
    0.01,
    10.0,
]
lite = false

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
