#!/usr/bin/env bash
set -euo pipefail

# src_type="retransformer-periodic"
src_type="reformer-d3-h4-m32"
# src_type="reformer-d1-m32-aux"
# src_type="reformer-d1-m64"

# src_type="reformer-d3-h4-m32-mqa"
# dest_type="reformer-d3-h4-m32-mqa"
# dest_type="reformer-d3-h4-m32-adapter"
# dest_type="reformer-d3-h4-m32-mqa-adapter"
# dest_type="reformer-d3-h4-m96"
# dest_type="reformer-d3-h4-m64-mqa"
dest_type="reformer-d3-h4-m96-mqa"


# 1) Copy top-level tuning files if they don't already exist
data_list=("adult" "black-friday" "california" "churn" "covtype2" "diamond" "higgs-small" "house" "microsoft" "otto")
for dataset in "${data_list[@]}"; do
    echo "Processing dataset: ${dataset}"
    src="exp/${src_type}/${dataset}/0-tuning.toml"
    dest_dir="exp/${dest_type}/${dataset}"
    dest="${dest_dir}/0-tuning.toml"

    mkdir -p "${dest_dir}"

    if [[ ! -f "${src}" ]]; then
        echo "Warning: source file not found for ${dataset}" >&2
        continue
    fi

    if [[ -f "${dest}" ]]; then
        echo "Skipping ${dataset}: destination already exists"
    else
        echo "Copying from ${src} to ${dest}"
        cp "${src}" "${dest}"
    fi
done

# 2) Copy tuning files under tabred or why if they don't already exist
data_list=("tabred" "why")
for dataset in "${data_list[@]}"; do
    echo "Processing dataset: ${dataset}"
    root="exp/${src_type}/${dataset}"

    for sub in "${root}"/*; do
        [[ -d "${sub}" ]] || continue
        name=$(basename "${sub}")
        src="${sub}/0-tuning.toml"
        dest_dir="exp/${dest_type}/${dataset}/${name}"
        dest="${dest_dir}/0-tuning.toml"

        mkdir -p "${dest_dir}"

        if [[ ! -f "${src}" ]]; then
            echo "Warning: source file not found in ${dataset}/${name}" >&2
            continue
        fi

        if [[ -f "${dest}" ]]; then
            echo "Skipping ${dataset}/${name}: destination already exists"
        else
            echo "Copying ${src} → ${dest}"
            cp "${src}" "${dest}"
        fi
    done
done

echo "All done"
