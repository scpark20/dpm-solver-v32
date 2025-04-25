CKPT_PATH="/data/checkpoints/cifar10_ddpmpp_deep_continuous/checkpoint_8.pth"
CONFIG="configs/vp/cifar10_ddpmpp_deep_continuous.py"
SCALE_DIR="/data/score_sde/scale/rbf_ecp_marginal_xt"
PAIR_NPZ='/data/score_sde/outputs/checkpoint_8/euler_1000/samples_0.npz'
for steps in 5 6 8 10 12 15; do

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

python xt_matching.py --config=$CONFIG --pair_npz=$PAIR_NPZ --scale_dir=$SCALE_DIR --ckp_path=$CKPT_PATH --config.sampling.method=rbf_ecp_marginal_xt --config.sampling.steps=$steps --config.sampling.eps=$EPS
done