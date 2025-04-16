DEVICES="2"
CONFIG="imagenet256_guided.yml"
scale="2.0"
order='3'
SCALE_DIR="/data/guided-diffusion/scale/rbf_ecp_marginal"

for sampleMethod in 'rbf_ecp_marginal'; do
for steps in 10 20; do

CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py --order=$order --config=$CONFIG --exp=$sampleMethod"_"$steps"_scale"$scale --scale_dir=$SCALE_DIR --timesteps=$steps --sample_type=$sampleMethod --scale=$scale --lower_order_final

done
done