#!/usr/bin/env bash
set -euo pipefail

# Uncomment one of the DEST_ROOTs below:
# DEST_ROOT="reformer-d3-h4-m32"
# DEST_ROOT="reformer-d3-h4-m32-aux"
# DEST_ROOT="reformer-d3-h4-m32-mqa"

# DEST_ROOT="reformer-d3-h4-m32-mqa-adapter"
# DEST_ROOT="reformer-d3-h4-m32-adapter"

dest_type="reformer-d1-h4-m32"
# dest_type="reformer-d3-h4-m96-mqa"


find "exp/${dest_type}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  echo "Processing $file …"

  # 1) Replace function entry
  if grep -q '^function *= *"bin\.model\.main"' "$file"; then
    sed -i 's|^function *= *"bin\.model\.main"|function = "bin.reformer.main"|' "$file"
    echo "   Updated function to bin.reformer.main"
  fi

  # 2) Remove old tuning entries under [space.model] (single‐line deletions only)
  sed -i '/^\[space\.model\]/,/^\[/ {
    /^k *=/d
    /^context_size *=/d
    /^context_dropout *=/d
    /^encoder_n_blocks *=/d
    /^predictor_n_blocks *=/d
  }' "$file"
  echo "   Removed old model hyperparams"

  # 3) Ensure or insert new parameters under [space.model]
  declare -A params=(
    [momentum]=0.999
    [queue_ratio]=64
    [context_size]=32
    [use_aux_loss]=false
    [use_adapter]=false
    [multi_output_head]=false
    [encoder_n_blocks]=1
    [predictor_n_blocks]=1
    [num_heads]=4
    [predictor_type]="\"mqa\""
    [k]=1
  )
  for name in "${!params[@]}"; do
    value=${params[$name]}
    if grep -q "^[[:space:]]*${name}[[:space:]]*=" "$file"; then
      sed -i "s|^[[:space:]]*${name}[[:space:]]*=.*|${name} = ${value}|" "$file"
      echo "   Replaced $name = $value"
    else
      sed -i "/^\[space\.model\]/a ${name} = ${value}" "$file"
      echo "   Inserted $name = ${value}"
    fi
  done

  # 4) Hard‑code d_main mapping via sed – in‐block substitutions
  sed -E -i "/^[[:space:]]*d_main *= *\\[/,/^[[:space:]]*\\]/ {
    s/\"int\"/\"int-power-of-two\"/;
    s/^([[:space:]]*)16,/\1 4,/;
    s/^([[:space:]]*)96,/\1 6,/;
    s/^([[:space:]]*)384,/\1 9,/;
  }" "$file"
  echo "   Updated d_main mapping"

done
