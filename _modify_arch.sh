#!/usr/bin/env bash
set -euo pipefail

# Uncomment one of the DEST_ROOTs below:
# DEST_ROOT="exp/rep-ensemble"
# DEST_ROOT="exp/taba-piecewiselinear"
# DEST_ROOT="reformer-periodic"
# DEST_ROOT="reformer-d1-m32"
DEST_ROOT="reformer-d1-m32-aux"
# DEST_ROOT="reformer-d1-m64"

find "exp/${DEST_ROOT}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  echo "Processing $file …"

  # 1) Replace function entry
  if grep -q '^function *= *"bin\.model\.main"' "$file"; then
    sed -i 's|^function *= *"bin\.model\.main"|function = "bin.reformer.main"|' "$file"
    echo "   Updated function to bin.reformer.main"
  fi

  # 2) Remove old tuning entries under [space.model]
  sed -i '/^\[space\.model\]/,/^\[/ {
    /^k *=/d
    /^context_size *=/d
    /^context_dropout *=/,/^\]/d
    /^encoder_n_blocks *=/,/^\]/d
    /^predictor_n_blocks *=/,/^\]/d
  }' "$file"
  echo "   Removed old model hyperparams"

  # 3) Ensure or insert new parameters under [space.model]
  declare -A params=(
    [momentum]=0.999
    [queue_ratio]=64
    [context_size]=32
    [use_aux_loss]=true
    [multi_output_head]=false
    [encoder_n_blocks]=1
    [predictor_n_blocks]=1
  )

  for name in "${!params[@]}"; do
    value=${params[$name]}
    if grep -q "^[[:space:]]*${name}[[:space:]]*=" "$file"; then
      # replace existing
      sed -i "s|^[[:space:]]*${name}[[:space:]]*=.*|${name} = ${value}|" "$file"
      echo "   Replaced $name = $value"
    else
      # insert after [space.model]
      sed -i "/^\[space\.model\]/a ${name} = ${value}" "$file"
      echo "   Inserted $name = $value"
    fi
  done

done
