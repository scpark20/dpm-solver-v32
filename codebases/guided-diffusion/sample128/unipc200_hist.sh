CONFIG="imagenet128_guided_raw256.yml"

for steps in 200; do
    for sampleMethod in 'unipc'; do
        for scale in 2.0 4.0 6.0 8.0; do
            python sample_raw.py --config=$CONFIG \
                                 --exp=${sampleMethod}_${steps}_scale${scale}_hist \
                                 --timesteps=$steps \
                                 --sample_type=$sampleMethod \
                                 --scale=$scale \
                                 --lower_order_final
        done
    done
done
