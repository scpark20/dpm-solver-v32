#!/usr/bin/env bash

prompts='prompts/prompts.txt'
model='sd-v1-4'
scale='3.5'

config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
H=512
W=512
C=4
f=8
STATS_DIR="statistics/sd-v1-4/${scale}_250_1024"
SCALE_DIR="/data/ldm/scale${scale}"
sampleMethod='rbf'

for order in 2 3
do
  for steps in 5 6 8 10 12 15 20
  do
    CUDA_VISIBLE_DEVICES='1' python txt2img_latent.py --fixed_code \
      --from-file "$prompts" \
      --steps "$steps" \
      --statistics_dir "$STATS_DIR" \
      --outdir "outputs/${model}/${sampleMethod}_order${order}_steps${steps}_scale${scale}" \
      --method "$sampleMethod" \
      --order "$order" \
      --scale "$scale" \
      --scale_dir "$SCALE_DIR" \
      --config "$config" \
      --ckpt "$ckpt" \
      --H "$H" \
      --W "$W" \
      --C "$C" \
      --f "$f"
  
  done
done