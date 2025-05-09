CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"

for steps in 200; do

python sample_hist.py --sample_folder="dpm_solver++_"$steps --ckp_path=$CKPT_PATH --method=dpm_solver++ --steps=$steps --skip_type=logSNR

done