#!/usr/bin/env bash

prompts='prompts/prompts.txt'
model='sd-v1-4'
scale='1.5'

config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
H=512
W=512
C=4
f=8

for steps in 1000
do
  for sampleMethod in 'rbf_euler'
  do
    CUDA_VISIBLE_DEVICES='0' python txt2img_latent.py --fixed_code \
      --from-file "$prompts" \
      --steps "$steps" \
      --statistics_dir "$STATS_DIR" \
      --outdir "outputs/${model}/${sampleMethod}_steps${steps}_scale${scale}" \
      --method "$sampleMethod" \
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
