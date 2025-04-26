#!/usr/bin/env bash

DEVICES="0,1,2,3,4,5,6,7"
CONFIG="imagenet256_guided.yml"
scale="2.0"
SCALE_DIR="/data/guided-diffusion/scale/rbf_ecp_marginal_sep2.0"

for steps in 5 6 8 10 12 15 20
do
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
