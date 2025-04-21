CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"
SCALE=3.0
ORDER=3
SCALE_DIR="/data/edm/scale/rbf_ecp_marginal${SCALE}"

for steps in 5 6 8 10 12 15 20 25 30 35 40; do

python sample.py --sample_folder="rbf_ecp_marginal_"$steps"_"$SCALE"_"$ORDER --order=$ORDER --ckp_path=$CKPT_PATH --method=rbf_ecp_marginal --steps=$steps --skip_type=logSNR --scale_dir=$SCALE_DIR

done