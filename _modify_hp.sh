#!/usr/bin/env bash
set -euo pipefail

# dest_type="qtabformer-query-4-key-k-value-ky-mqa-d4"
# dest_type="qraugmlp-query-key-k-value-ky"
# dest_type="qraugmlp-key-k-value-ky-m32"
# dest_type="qtab-naive-cossim-cl"
# dest_type="qraugresnet-key-k-value-ky-m32"
# dest_type="qraugresnet-key-k-value-qky-m32"


dest_type="qtabformer-query-1-key-k-value-ky-mqa-moh"

find "exp/${dest_type}" -type f -name "0-tuning.toml" -print0 | while IFS= read -r -d '' file; do
  echo "Processing $file"

  # Replace function
  # if grep -q '^function *= *"[^"]*"' "$file"; then
  #   sed -i 's|^function *= *"[^"]*"|function = "bin.qr_aug_mlp.main"|' "$file"
  #   echo "   Updated function to bin.qr_aug_mlp.main"
  # fi

  # # # Remove single lines under [space.model] 
  # sed -i '/^\[space\.model\]/,/^\[/ {
  #   /^k *=/d
  #   /^distance_metric *=/d
  #   /^temperature *=/d
  #   /^context_size *=/d
  #   /^context_dropout *=/d
  #   /^encoder_n_blocks *=/d
  #   /^use_aux_loss *=/d
  #   /^use_adapter *=/d
  #   /^num_heads *=/d
  #   /^predictor_n_blocks *=/d
  #   /^predictor_type *=/d
  #   /^k *=/d
  #   /^multi_output_head *=/d
  #   /^arch_type *=/d
  #   /^activation *=/d
  #   /^normalization *=/d
  # }' "$file"
  # echo "   Removed old model hyperparams"

  # # 3) Ensure or insert new parameters under [space.model]
  declare -A params=(
    # [context_size]=32      
    # [momentum]=0.999
    # [queue_ratio]=64
    # [multi_output_head]=false
    # [encoder_n_blocks]=1
    # [predictor_n_blocks]=1
    # [num_heads]=4
    # [query_expansion_ratio]=4
    # [attention_type]="\"mqa\""
    # [use_key_as_value]=true
    # [use_multi_output_head]=false
    # [use_mlp_head]=false
    # [distance_metric]="\"cossim\""
    # [use_label_encoder]=true
    # [k]=1
    # [temperature]=0.1
    [query_expansion_ratio]=1
    # [use_key_as_value]=false
    # [use_qk_as_value]=true
    # [use_skip_connection]=true
  )
  for name in "${!params[@]}"; do
    value=${params[$name]}
    if grep -q "^[[:space:]]*${name}[[:space:]]*=" "$file"; then
      sed -i "s|^[[:space:]]*${name}[[:space:]]*=.*|${name} = ${value}|" "$file"
      echo "   Replac $name = $value"
    else
      sed -i "/^\[space\.model\]/a ${name} = ${value}" "$file"
      echo "   Insert $name = ${value}"
    fi
  done



  # sed -i "/^\[space\.model\]/a temperature = [\n  \"_tune_\",\n  \"loguniform\",\n  0.01,\n 1,\n]" "$file"



  # # delete predictor_n_blocks
  
  # if grep -q '^[[:space:]]*predictor_n_blocks[[:space:]]*=' "$file"; then
  #   sed -i '/^[[:space:]]*predictor_n_blocks[[:space:]]*=\s*\[/,/^[[:space:]]*]/d' "$file"
  #   # sed -i '/^[[:space:]]*predictor_n_blocks[[:space:]]*=/d' "$file"
  #   echo "   Remov predictor_n_blocks line"
  # fi
  # sed -i "/^\[space\.model\]/a predictor_n_blocks = [\n  \"_tune_\",\n  \"int\",\n  1,\n 4,\n]" "$file"
  


  #   # # delete num_heads
  # if grep -q '^[[:space:]]*num_heads[[:space:]]*=' "$file"; then
  #   sed -i '/^[[:space:]]*num_heads[[:space:]]*=/d' "$file"
  #   echo "   Remov num_heads line"
  # fi
  # sed -i "/^\[space\.model\]/a num_heads = [\n  \"_tune_\",\n  \"int\",\n  4,\n 8,\n 4,\n]" "$file"
  
  
  # add temperature as tuning parameter
  # sed -i "/^\[space\.model\]/a temperature = [\n  \"_tune_\",\n  \"categorical\",\n  [0.01, 0.05, 0.1, 0.15, 0.2]\n]" "$file"
  # echo "   Insert temperature tuning block"

  # if grep -q '^[[:space:]]*dropout1[[:space:]]*=' "$file"; then
  #   sed -i '/^[[:space:]]*dropout1[[:space:]]*=/d' "$file"
  #   echo "   Remove dropout1 line"
  # fi

  # sed -i "/^\[space\.model\]/a dropout1 = [\n  \"_tune_\",\n  \"?uniform\",\n  0.0,\n 0.0,\n  0.6,\n]" "$file"
  # echo "   Insert dropout1 tuning block"

  # if grep -q '^[[:space:]]*temperature[[:space:]]*=\s*\[\s*$' "$file"; then
  #   sed -i '/^[[:space:]]*temperature[[:space:]]*=\s*\[\s*$/,/^[[:space:]]*]/d' "$file"
  #   echo "   Removed temperature tuning block"
  # fi
  # sed -i "/^\[space\.model\]/a temperature = [\n  \"_tune_\",\n  \"loguniform\",\n  0.01,\n 1,\n]" "$file"
  # sed -i "/^\[space\.model\]/a temperature_c = [\n  \"_tune_\",\n  \"loguniform\",\n  0.01,\n 1,\n]" "$file"
  # sed -i "/^\[space\.model\]/a contrastive_loss_weight = [\n  \"_tune_\",\n   \"categorical\",\n  [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8]\n]" "$file"


  # # 4) d_main
  # sed -E -i "/^[[:space:]]*d_main *= *\\[/,/^[[:space:]]*\\]/ {
  #   s/\"int\"/\"int-power-of-two\"/;
  #   s/^([[:space:]]*)16,/\1 4,/;
  #   s/^([[:space:]]*)96,/\1 6,/;
  #   s/^([[:space:]]*)384,/\1 9,/;
  # }" "$file"
  # echo "   Updated d_main mapping"


  #########################################################
  # config for label bins

#   sed -i '/^\[space\.label_bins\]/,/^\[/d' "$file"
#   echo "   Removed old [space.label_bins] section"


#   cat << 'EOF' >> "$file"

# [space.label_bins]
# n_bins = [
#     "_tune_",
#     "int",
#     2,
#     10,
# ]
# EOF
#   echo "   Appended [space.bins] section"  
done
