CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"
ORDER=3
DC_DIR="/data/edm/dc"

for steps in 20 25 35; do

CUDA_VISIBLE_DEVICES="0" python sample.py --sample_folder="dcsolver_"$ORDER"_"$steps --order=$ORDER --ckp_path=$CKPT_PATH --method=dcsolver --steps=$steps --skip_type=logSNR --dc_dir=$DC_DIR

done