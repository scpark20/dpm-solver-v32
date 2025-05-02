CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"
SCALE=2.0
ORDER=4
SCALE_DIR="/data/edm/scale/rbf_ecp_marginal${SCALE}"

for steps in 5 6 8 10 12 15 20 25 30 35 40; do

CUDA_VISIBLE_DEVICES="0" python sample.py --sample_folder="rbf_ecp_marginal_"$SCALE"_"$ORDER"_"$steps --order=$ORDER --ckp_path=$CKPT_PATH --method=rbf_ecp_marginal --steps=$steps --skip_type=logSNR --scale_dir=$SCALE_DIR

done