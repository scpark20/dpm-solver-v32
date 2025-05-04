CONFIG="imagenet256_guided_for_dc.yml"
STATS_DIR="statistics/imagenet256_guided/500_1024"

for steps in 1000; do
    for sampleMethod in 'dpmsolver++'; do
        for scale in 2.0 4.0 6.0 8.0; do
            CUDA_VISIBLE_DEVICES='0' python sample_raw.py --config=$CONFIG \
                                 --exp=${sampleMethod}_${steps}_scale${scale} \
                                 --timesteps=$steps \
                                 --sample_type=$sampleMethod \
                                 --order=1 \
                                 --scale=$scale \
                                 --lower_order_final
        done
    done
done
