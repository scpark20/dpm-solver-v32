DEVICES="0,1,2,3,4,5,6,7"
#DEVICES="0"
CONFIG="imagenet256_guided.yml"

for scale in 2.0 4.0 6.0 8.0; do
    DC_DIR="/data/guided-diffusion/dc/imagenet256/scale${scale}"

    for steps in 5 6 8 10 12 15 20 25; do
        # 2) rbf_ecp_marginal order=2
        CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py \
            --order=3 \
            --config="${CONFIG}" \
            --exp="dcsolver_order3_${steps}_scale${scale}" \
            --dc_dir="${DC_DIR}" \
            --timesteps="${steps}" \
            --sample_type="dcsolver" \
            --scale="${scale}" \
            --lower_order_final

    done
done
