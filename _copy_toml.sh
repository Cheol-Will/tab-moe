#!/usr/bin/env bash
set -euo pipefail


# src_type="tabrm-piecewiselinear"
# src_type="tabr-pln-periodic"
# src_type="tabm-piecewiselinear"
# src_type="tabm-rankone-piecewiselinear"
# src_type="retransformer-periodic"
src_type="tabm-piecewiselinear"
# src_type="tabm-mini-piecewiselinear"
# dest_type="rep-tabr-periodic"
# ="mlp"
# src_type="tabrmv2-mini-periodic"
# src_type="mlp"
# dest_type="tabrmv2-mini-piecewiselinear"
# dest_type="tabrmv2-mini-periodic"
# dest_type="tabrmoev3-periodic"
# dest_type="tabr-pln-periodic"
# dest_type="rep-tabr-periodic"
# dest_type="tabr-pln-periodic"
# dest_type="tabr-pln-multihead-periodic"
# dest_type="tabm-rankp-piecewiselinear"

# dest_type="retransformer-aux-periodic"

dest_type="taba-piecewiselinear"

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
