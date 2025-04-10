CONFIG="imagenet256_guided.yml"
scale="0.0"
STATS_DIR="statistics/imagenet256_guided/500_1024"

for steps in 200; do
for sampleMethod in 'unipc'; do

python sample_raw.py --config=$CONFIG --exp=$sampleMethod"_"$steps"_scale"$scale --timesteps=$steps --sample_type=$sampleMethod --scale=$scale --lower_order_final

done
done