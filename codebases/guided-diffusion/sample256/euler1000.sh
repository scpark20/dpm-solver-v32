DEVICES="0"
CONFIG="imagenet256_guided.yml"
scale="0.0"

for steps in 1000; do
for sampleMethod in 'lagrange'; do

CUDA_VISIBLE_DEVICES="${DEVICES}" python sample_raw.py --config=$CONFIG --order=1 --exp=$sampleMethod"_"$steps"_scale"$scale --timesteps=$steps --sample_type=$sampleMethod --scale=$scale --lower_order_final

done
done