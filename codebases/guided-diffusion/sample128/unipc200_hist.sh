DEVICES="0,1,2,3,4,5,6,7"
CONFIG="imagenet128_guided_raw128.yml"

for steps in 200; do
    for sampleMethod in 'unipc'; do
        for scale in 2.0 4.0 6.0 8.0; do
            CUDA_VISIBLE_DEVICES="${DEVICES}" python sample_raw.py --config=$CONFIG \
                                 --exp=${sampleMethod}_${steps}_scale${scale}_hist1024 \
                                 --timesteps=$steps \
                                 --sample_type=$sampleMethod \
                                 --scale=$scale \
                                 --lower_order_final
        done
    done
done
