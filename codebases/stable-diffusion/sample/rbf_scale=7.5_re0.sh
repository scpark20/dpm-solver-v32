#!/usr/bin/env bash
# run_sd_latent.sh
# 사용 예)  ./run_sd_latent.sh 1 12        # GPU 1, steps 12
#         ./run_sd_latent.sh              # GPU 0, steps 5

# ───────────────── 기본값 및 인자 처리 ─────────────────
CUDA_NUM=${1:-0}     # 첫 번째 인자: GPU 번호 (기본 0)
STEPS=${2:-5}        # 두 번째 인자: sampling steps (기본 5)

# 고정 파라미터
prompts='prompts/prompts.txt'
model='sd-v1-4'
scale='7.5'

config="configs/stable-diffusion/v1-inference.yaml"
ckpt="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
H=512; W=512; C=4; f=8
STATS_DIR="statistics/sd-v1-4/${scale}_250_1024"
SCALE_DIR="/data/ldm/scale${scale}_re"
sampleMethod='rbf'

# ───────────────── 실행 ─────────────────
order=4              # 필요하면 추가 인자로 빼도 됨

CUDA_VISIBLE_DEVICES="$CUDA_NUM" \
python txt2img_latent.py --fixed_code \
  --from-file "$prompts" \
  --steps  "$STEPS" \
  --statistics_dir "$STATS_DIR" \
  --outdir "outputs/${model}/${sampleMethod}_order${order}_steps${STEPS}_scale${scale}" \
  --method "$sampleMethod" \
  --order  "$order" \
  --scale  "$scale" \
  --scale_dir "$SCALE_DIR" \
  --config "$config" \
  --ckpt   "$ckpt" \
  --H "$H" --W "$W" --C "$C" --f "$f"
