CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"

for steps in 5 6 8 10 12 15 20 25 30 35 40; do

CUDA_VISIBLE_DEVICES="0" python sample.py --sample_folder="dpm_solver++_"$steps --ckp_path=$CKPT_PATH --method=dpm_solver++ --steps=$steps --skip_type=logSNR

done