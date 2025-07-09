#!/usr/bin/env bash
set -euo pipefail

# Check the destination root!!
# DEST_ROOT="exp/mlp-debug2"
# DEST_ROOT="exp/tabrmoev4-drop-periodic"
DEST_ROOT="exp/reproduced-tabr-periodic"


# delete the line with graident_clipping_norm 
find "${DEST_ROOT}" -type f -name "0-tuning.toml" -exec sed -i '/^gradient_clipping_norm/d' {} +



# 1) Define replacement block
BLOCK=$(cat << 'EOF'
[space.optimizer]
type = "AdamW"
lr = [
    "_tune_",
    "loguniform",
    3e-05,
    0.001,
]
weight_decay = [
    "_tune_",
    "?loguniform",
    0.0,
    1e-06,
    0.0001,
]

[space.model]
arch_type = "tabr"
k = 1
context_size = 96
share_training_batches = false
d_main = [
    "_tune_",
    "int",
    96,
    384,
]
context_dropout = [
    "_tune_",
    "?uniform",
    0.0,
    0.0,
    0.6,
]
d_multiplier = 2.0
encoder_n_blocks = [
    "_tune_",
    "int",
    0,
    1,
]
predictor_n_blocks = [
    "_tune_",
    "int",
    1,
    2,
]
mixer_normalization = "auto"
dropout0 = [
    "_tune_",
    "?uniform",
    0.0,
    0.0,
    0.6,
]
dropout1 = 0.0
normalization = "LayerNorm"
activation = "ReLU"

[space.model.num_embeddings]
type = "PeriodicEmbeddings"
n_frequencies = [
    "_tune_",
    "int",
    16,
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
    16,
    64,
]
lite = true
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
#   sed '/^\[space\.model\]/,$d' "$file" > "$tmp"
  sed '/^\[space\.optimizer\]/,$d' "$file" > "$tmp"
  # append the replacement block
  printf '%s\n' "$BLOCK" >> "$tmp"
  # overwrite original
  mv "$tmp" "$file"
done

echo "All files have been updated."
