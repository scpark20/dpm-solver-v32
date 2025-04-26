#!/usr/bin/env bash

DEVICES="0,1,2,3,4,5,6,7"
CONFIG="imagenet256_guided.yml"

# scale: 8.0 → 2.0,  steps: 20 → 5
for scale in 8.0 6.0 4.0 2.0; do
    SCALE_DIR="/data/guided-diffusion/scale/rbf_ecp_marginal_sep${scale}"

    for steps in 20 15 12 10 8 6 5; do
        # 2) rbf_ecp_marginal order=2
        CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py \
            --order=2 \
            --config="${CONFIG}" \
            --exp="rbf_ecp_marginal_sep_${steps}_scale${scale}_order2" \
            --scale_dir="${SCALE_DIR}" \
            --timesteps="${steps}" \
            --sample_type="rbf_ecp_marginal_sep" \
            --scale="${scale}" \
            --lower_order_final

        # 3) rbf_ecp_marginal order=3
        CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py \
            --order=3 \
            --config="${CONFIG}" \
            --exp="rbf_ecp_marginal_sep_${steps}_scale${scale}_order3" \
            --scale_dir="${SCALE_DIR}" \
            --timesteps="${steps}" \
            --sample_type="rbf_ecp_marginal_sep" \
            --scale="${scale}" \
            --lower_order_final
    done
done
