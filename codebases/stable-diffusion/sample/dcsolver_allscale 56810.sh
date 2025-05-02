#!/usr/bin/env bash

prompts='prompts/prompts.txt'
model='sd-v1-4'

config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
H=512
W=512
C=4
f=8
sampleMethod='dcsolver'

for scale in 1.5 3.5 5.5 7.5 9.5
do
  DC_DIR="/data/ldm/dc${scale}"

  for order in 3
  do
    for steps in 5 6 8 10
    do
      CUDA_VISIBLE_DEVICES='0' python txt2img_latent.py --fixed_code \
        --from-file "$prompts" \
        --steps "$steps" \
        --outdir "outputs/${model}/${sampleMethod}_order${order}_steps${steps}_scale${scale}" \
        --method "$sampleMethod" \
        --order "$order" \
        --scale "$scale" \
        --dc_dir "$DC_DIR" \
        --config "$config" \
        --ckpt "$ckpt" \
        --H "$H" \
        --W "$W" \
        --C "$C" \
        --f "$f"
    done
  done
done
