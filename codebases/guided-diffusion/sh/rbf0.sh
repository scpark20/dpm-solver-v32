DEVICES="0"
CONFIG="imagenet256_guided.yml"
scale="2.0"
STATS_DIR="statistics/imagenet256_guided/500_1024"

for steps in 5 20; do
for sampleMethod in 'rbf'; do

CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py --config=$CONFIG --exp=$sampleMethod"_"$steps"_scale"$scale --statistics_dir=$STATS_DIR --timesteps=$steps --sample_type=$sampleMethod --scale=$scale --lower_order_final

done
done