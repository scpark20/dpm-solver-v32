CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"

for steps in 5 6 8 10 12 15 20 25 30 35 40; do

python sample.py --sample_folder="uni_pc_bh1_"$steps --unipc_variant=bh1 --ckp_path=$CKPT_PATH --method=uni_pc --steps=$steps --skip_type=logSNR

done