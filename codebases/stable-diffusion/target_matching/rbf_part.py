#!/usr/bin/env python
# sample_rbf.py
# ─────────────────────────────────────────────────────────────
# Stable-Diffusion latent-space sampling with RBFSampler
#   • 모든 주요 하이퍼파라미터(N, M, OFFSET, K, SCALE, ORDER, NFE)를
#     커맨드라인 인자로 조정 가능
#   • 기본값은 기존 스크립트에서 쓰던 값으로 설정
# ─────────────────────────────────────────────────────────────

import os, sys, argparse, random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import autocast
from tqdm import tqdm

# ────────────────────────────── 1. 인자 파서
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Latent-space target-matching sampling with RBFSampler"
    )

    # (1) Stable-Diffusion 모델 로딩용 설정
    p.add_argument("--config", default="configs/stable-diffusion/v1-inference.yaml")
    p.add_argument("--ckpt",   default="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt")
    p.add_argument("--H", type=int, default=512)          # VAE 해상도(높이)
    p.add_argument("--W", type=int, default=512)          # VAE 해상도(너비)
    p.add_argument("--C", type=int, default=4)            # latent channels
    p.add_argument("--f", type=int, default=8)            # down-scaling factor
    p.add_argument("--precision", default="autocast",
                   choices=["autocast", "full"])

    # (2) 샘플링 하이퍼파라미터(요청사항)
    p.add_argument("--N",      type=int,   default=128, help="총 샘플 개수")
    p.add_argument("--M",      type=int,   default=6,   help="미니-배치 크기")
    p.add_argument("--offset", type=int,   default=0,   help="파일 번호 오프셋")
    p.add_argument("--K",      type=int,   default=20,  help="반복 루프 횟수")
    p.add_argument("--scale",  type=float, default=5.5, help="CFG 스케일")
    p.add_argument("--order",  type=int,   default=2,   choices=[1,2,3,4,5,6,7],
                   help="RBF 차수 (order)")
    p.add_argument("--nfe",    type=int,   default=5,
                   help="function evaluations (NFE)")

    # (3) 데이터/저장 경로
    p.add_argument("--archive_root", default="/data/archive/sd-v1-4",
                   help="원본 .pt 아카이브 최상위 폴더")
    p.add_argument("--save_root",    default="/data/ldm",
                   help="결과 저장 루트 디렉터리")

    return p

# ────────────────────────────── 2. 헬퍼: Stable-Diffusion 로드
def load_sd_model(config_path: str, ckpt_path: str, device: torch.device):
    from txt2img_latent import load_model_from_config
    cfg = OmegaConf.load(config_path)
    model = load_model_from_config(cfg, ckpt_path)
    return model.to(device)

# ────────────────────────────── 3. 메인
def main():
    parser = build_parser()
    opt    = parser.parse_args()

    # ── 3-1. 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 3-2. Stable-Diffusion 로드
    model = load_sd_model(opt.config, opt.ckpt, device)
    model.eval()
    print("[Init]  Stable-Diffusion 로드 완료")

    # ── 3-3. Sampler 선택
    from ldm.models.diffusion.rbf     import RBFSampler
    from ldm.models.diffusion.uni_pc  import UniPCSampler
    sampler = RBFSampler(model)   # 필요시 UniPCSampler(model)
    print("[Init]  Sampler 준비 완료")

    # ── 3-4. 학습 데이터(.pt) 읽기
    pt_dir   = f"{opt.archive_root}/dpm_solver++_steps200_scale{opt.scale}"
    pt_files = sorted([str(Path(pt_dir) / f)
                       for f in os.listdir(pt_dir) if f.endswith(".pt")])
    if len(pt_files) == 0:
        sys.exit(f"❌ .pt 파일을 찾을 수 없습니다 → {pt_dir}")

    prompts_list, x_T_list, x_0_list = [], [], []
    for i in range(1000):  # 최대 1000개만
        data = torch.load(pt_files[i])
        prompts_list.append(data["text"])
        x_T_list.append(data["latent"])
        x_0_list.append(data["image"])

    prompts_list = np.ravel(prompts_list)[: opt.N]
    x_T_list     = torch.cat(x_T_list, dim=0)[: opt.N]
    x_0_list     = torch.cat(x_0_list, dim=0)[: opt.N]
    print(f"[Data]  prompts={len(prompts_list)}  x_T={x_T_list.shape}  x_0={x_0_list.shape}")

    # ── 3-5. 저장 디렉터리
    scale_dir = f"{opt.save_root}/scale{opt.scale}_re"
    os.makedirs(scale_dir, exist_ok=True)

    # ── 3-6. Sampling 루프
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    for number in range(opt.K):
        # (i) 미니-배치 샘플링
        idx       = np.random.randint(0, opt.N, size=(opt.M,))
        prompts   = list(prompts_list[idx])
        x_T_batch = x_T_list[idx].to(device)
        x_0_batch = x_0_list[idx].to(device)

        with torch.no_grad(), precision_scope("cuda"), model.ema_scope():
            # unconditional / conditional embedding
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning([""] * len(prompts))
            c  = model.get_learned_conditioning(prompts)

            # (ii) RBF Target-Matching
            samples, _ = sampler.target_matching(
                S=opt.nfe,
                shape=(4, 64, 64),
                conditioning=c,
                batch_size=len(prompts),
                verbose=False,
                unconditional_guidance_scale=opt.scale,
                unconditional_conditioning=uc,
                eta=0.0,
                x_T=x_T_batch,
                x_0=x_0_batch,
                order=opt.order,
                number=opt.offset + number,
                scale_dir=scale_dir,
            )
        print(f"[{number+1:02d}/{opt.K}]  완료")

    print("✓ 모든 샘플링 완료!")

# ──────────────────────────────
if __name__ == "__main__":
    main()
