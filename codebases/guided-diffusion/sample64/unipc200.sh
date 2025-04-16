CONFIG="imagenet64_raw128.yml"

for steps in 200; do
    for sampleMethod in 'unipc'; do
        python sample_raw.py --config=$CONFIG \
                                --exp=${sampleMethod}_${steps}_scale${scale} \
                                --timesteps=$steps \
                                --sample_type=$sampleMethod \
                                --lower_order_final
    done
done
