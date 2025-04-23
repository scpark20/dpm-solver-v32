CKPT_PATH="/data/checkpoints/cifar10_ddpmpp_deep_continuous/checkpoint_8.pth"
CONFIG="configs/vp/cifar10_ddpmpp_deep_continuous.py"
METHOD="rbf_ecp_marginal_same"
SCALE_DIR="/data/score_sde/scale/"${METHOD}
for steps in 5 6 8 10 12 15 20; do

if [ $steps -le 10 ]; then
    EPS="1e-3"
    STATS_DIR="statistics/cifar10_ddpmpp_deep_continuous/0.001_1200_4096"
    if [ $steps -le 8 ]; then
        if [ $steps -le 5 ]; then
            p_pseudo="True"
            lower_order_final="False"
            use_corrector="True"
        else
            p_pseudo="False"
            lower_order_final="True"
            use_corrector="True"
        fi
    else
        p_pseudo="False"
        lower_order_final="True"
        use_corrector="False"
    fi
else
    STATS_DIR="statistics/cifar10_ddpmpp_deep_continuous/0.0001_1200_4096"
    EPS="1e-4"
    p_pseudo="False"
    lower_order_final="True"
    use_corrector="True"
fi

CUDA_VISIBLE_DEVICES='6' python sample.py --config=$CONFIG --ckp_path=$CKPT_PATH --scale_dir=$SCALE_DIR --sample_folder=${METHOD}"_"$steps --config.sampling.method=${METHOD} --config.sampling.steps=$steps --config.sampling.eps=$EPS
done