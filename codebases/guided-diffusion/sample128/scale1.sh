DEVICES="0"
CONFIG="imagenet256_guided.yml"
scale="1.0"
SCALE_DIR="/data/guided-diffusion/scale/rbf_ecp_marginal"

# 1) unipc order=2
for steps in 10 20
do
    CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py \
        --order=2 \
        --config="${CONFIG}" \
        --exp="unipc_${steps}_scale${scale}_order2" \
        --scale_dir="${SCALE_DIR}" \
        --timesteps="${steps}" \
        --sample_type="unipc" \
        --scale="${scale}" \
        --lower_order_final
done

# 2) rbf_ecp_marginal order=2
for steps in 10 20
do
    CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py \
        --order=2 \
        --config="${CONFIG}" \
        --exp="rbf_ecp_marginal_${steps}_scale${scale}_order2" \
        --scale_dir="${SCALE_DIR}" \
        --timesteps="${steps}" \
        --sample_type="rbf_ecp_marginal" \
        --scale="${scale}" \
        --lower_order_final
done

# 3) rbf_ecp_marginal order=3
for steps in 10 20
do
    CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py \
        --order=3 \
        --config="${CONFIG}" \
        --exp="rbf_ecp_marginal_${steps}_scale${scale}_order3" \
        --scale_dir="${SCALE_DIR}" \
        --timesteps="${steps}" \
        --sample_type="rbf_ecp_marginal" \
        --scale="${scale}" \
        --lower_order_final
done
