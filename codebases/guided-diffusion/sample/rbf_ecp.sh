DEVICES="0"
CONFIG="imagenet256_guided.yml"
scale="2.0"
order='2'
SCALE_DIR="/data/guided-diffusion/scale/rbf_ecp"

for sampleMethod in 'rbf_ecp'; do
for steps in 5 10 15 20 25 30 35 40; do

CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py --order=$order --config=$CONFIG --exp=$sampleMethod"_"$steps"_scale"$scale --scale_dir=$SCALE_DIR --timesteps=$steps --sample_type=$sampleMethod --scale=$scale --lower_order_final

done
done