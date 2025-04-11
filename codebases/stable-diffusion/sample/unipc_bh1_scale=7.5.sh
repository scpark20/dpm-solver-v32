#!/usr/bin/env bash

prompts='prompts/prompts.txt'
model='sd-v1-4'
scale='7.5'

config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
H=512
W=512
C=4
f=8
STATS_DIR="statistics/sd-v1-4/${scale}_250_1024"

for steps in 5 10 15 20 25 30 35 40
do
  for sampleMethod in 'uni_pc_bh1'
  do
    python txt2img_latent.py --fixed_code \
      --from-file "$prompts" \
      --steps "$steps" \
      --statistics_dir "$STATS_DIR" \
      --outdir "outputs/${model}/${sampleMethod}_steps${steps}_scale${scale}" \
      --method "$sampleMethod" \
      --scale "$scale" \
      --config "$config" \
      --ckpt "$ckpt" \
      --H "$H" \
      --W "$W" \
      --C "$C" \
      --f "$f"
  done
done
