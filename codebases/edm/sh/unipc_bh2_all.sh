CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"

for steps in 40 35 30 25 20 15 12 10 8 6 5; do

CUDA_VISIBLE_DEVICES="1" python sample.py --sample_folder="uni_pc_bh2_"$steps --unipc_variant=bh2 --ckp_path=$CKPT_PATH --method=uni_pc --steps=$steps --skip_type=logSNR

done