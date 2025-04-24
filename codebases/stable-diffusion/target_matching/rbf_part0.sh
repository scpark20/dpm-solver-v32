#!/usr/bin/env bash
# ─────────────────────────────────────────────
# run_sample_rbf.sh
#   • 여러 NFE × ORDER 조합을 일괄 실행
#   • 인자
#       1) GPU 번호   (기본 0)
#       2) N          (기본 128)
#       3) M          (기본 6)
#       4) K          (기본 20)
#       5) SCALE      (기본 5.5)
# ─────────────────────────────────────────────
set -e

CUDA=${1:-0}
N=${2:-1024}
M=${3:-6}
K=${4:-16}
SCALE=${5:-7.5}

export CUDA_VISIBLE_DEVICES="$CUDA"

# OFFSET을 인자로 받지 않고 K * CUDA로 자동 계산
OFFSET=$(( K * CUDA ))

# 실험할 파라미터 목록
NFE_ARR=(5 6 8 10 12 15 20)
ORDER_ARR=(4)

for ORDER in "${ORDER_ARR[@]}"; do
  for NFE in "${NFE_ARR[@]}"; do
    echo ">>> GPU=$CUDA | ORDER=$ORDER | NFE=$NFE | N=$N | M=$M | OFFSET=$OFFSET | K=$K | SCALE=$SCALE"
    python -m target_matching.rbf_part \
      --config configs/stable-diffusion/v1-inference.yaml \
      --ckpt   models/ldm/stable-diffusion-v1/sd-v1-4.ckpt \
      --H 512 --W 512 --C 4 --f 8 \
      --N "$N" --M "$M" --offset "$OFFSET" --K "$K" \
      --scale "$SCALE" --order "$ORDER" --nfe "$NFE"
  done
done
