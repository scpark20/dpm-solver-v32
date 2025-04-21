CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"

for steps in 200; do

if [ $steps -lt 10 ]; then
STATS_DIR="statistics/edm-cifar10-32x32-uncond-vp/0.002_80.0_1200_1024"
else
STATS_DIR="statistics/edm-cifar10-32x32-uncond-vp/0.002_80.0_120_4096"
fi

python sample_test.py --sample_folder="dpm_solver++_"$steps --ckp_path=$CKPT_PATH --method=dpm_solver++ --steps=$steps --skip_type=logSNR

done