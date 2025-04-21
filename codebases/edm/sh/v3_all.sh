CKPT_PATH="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"

for steps in 5 6 8 10 12 15 20 25 30 35 40; do

if [ $steps -lt 10 ]; then
STATS_DIR="statistics/edm-cifar10-32x32-uncond-vp/0.002_80.0_1200_1024"
else
STATS_DIR="statistics/edm-cifar10-32x32-uncond-vp/0.002_80.0_120_4096"
fi

python sample.py --sample_folder="dpm_solver_v3_"$steps --statistics_dir=$STATS_DIR --ckp_path=$CKPT_PATH --method=dpm_solver_v3 --steps=$steps --skip_type=logSNR

done