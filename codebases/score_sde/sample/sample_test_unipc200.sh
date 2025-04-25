CKPT_PATH="/data/checkpoints/cifar10_ddpmpp_deep_continuous/checkpoint_8.pth"
CONFIG="configs/vp/cifar10_ddpmpp_deep_continuous.py"
for steps in 200; do

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
python sample_test.py --config=$CONFIG --return_hist=True --ckp_path=$CKPT_PATH --sample_folder="UniPC_bh1_"$steps --config.sampling.method=uni_pc --config.sampling.steps=$steps --config.sampling.variant=bh1 --config.sampling.eps=$EPS
done