CONFIG="imagenet256_guided_raw256.yml"
STATS_DIR="statistics/imagenet256_guided/500_1024"

for steps in 200; do
    for sampleMethod in 'unipc'; do
        for scale in 1.0 2.0 4.0 6.0 8.0; do
            python sample_raw.py --config=$CONFIG \
                                 --exp=${sampleMethod}_${steps}_scale${scale} \
                                 --timesteps=$steps \
                                 --sample_type=$sampleMethod \
                                 --scale=$scale \
                                 --lower_order_final
        done
    done
done
