DEVICES="0"
CONFIG="imagenet256_guided.yml"
scale="2.0"
variant='bh2'
predictor_order='2'
corrector_order='3'
SCALE_DIR="/data/guided-diffusion/scale/unipc_rbf"

for sampleMethod in 'unipc_rbf'; do
for steps in 5 10 15 20 25 30 35 40; do

CUDA_VISIBLE_DEVICES="${DEVICES}" python sample.py --variant=$variant --predictor_order=$predictor_order --corrector_order=$corrector_order --config=$CONFIG --exp=$sampleMethod"_"$steps"_scale"$scale --scale_dir=$SCALE_DIR --timesteps=$steps --sample_type=$sampleMethod --scale=$scale --lower_order_final

done
done