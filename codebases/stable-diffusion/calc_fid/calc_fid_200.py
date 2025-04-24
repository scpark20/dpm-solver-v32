import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ───────────────────────── Stable-Diffusion Init. ─────────────────────────
from txt2img_latent import get_parser, load_model_from_config, chunk

parser = get_parser()
args_list = [
    "--config", "configs/stable-diffusion/v1-inference.yaml",
    "--ckpt",   "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
    "--H", "512", "--W", "512", "--C", "4", "--f", "8",
]
opt = parser.parse_args(args_list)

import torch
from omegaconf import OmegaConf
config = OmegaConf.load(opt.config)
model  = load_model_from_config(config, opt.ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = model.to(device).eval()

# ───────────────────────── Inception-V3 Init. (FID) ───────────────────────
from pytorch_fid.inception import InceptionV3
import torchvision.transforms as T

DIMS       = 2048
BLOCK_IDX  = InceptionV3.BLOCK_INDEX_BY_DIM[DIMS]
inception = InceptionV3([BLOCK_IDX],
                        resize_input=True,
                        normalize_input=True).to(device).eval()

@torch.no_grad()
def get_inception_feats(img_tensor):            # (B,3,H,W) in [0,1]
    feats = inception(img_tensor)[0]            # (B,2048,1,1)
    return feats.squeeze(-1).squeeze(-1)        # (B,2048)

# ───────────────────────── Latent → PIL 편의 함수 ─────────────────────────
from PIL import Image
def decode_to_pil(latents):                     # latents: (B,C,H,W)
    x = model.decode_first_stage(latents.to(device))
    x = torch.clamp((x + 1.0) / 2.0, 0, 1)      # [0,1]
    x = (x * 255).byte().cpu()
    return [Image.fromarray(t.permute(1,2,0).numpy()) for t in x]

# ───────────────────────── 특징 추출 루틴 (CLIP→FID) ───────────────────────
from tqdm import tqdm
def calc_fid_feats(pt_path, save_path, batch=16):
    os.makedirs(save_path, exist_ok=True)
    files = [os.path.join(pt_path, f) for f in os.listdir(pt_path) if f.endswith('.pt')]

    for file in tqdm(files, desc=f'FID ↦ {os.path.basename(pt_path)}'):
        dst = os.path.join(save_path, os.path.basename(file))
        if os.path.exists(dst):
            continue

        data    = torch.load(file, map_location='cpu')   # {"image": latents, ...}
        latents = data['image']                          # (N,C,H,W)
        feats_all = []

        # 배치 분할 (GPU VRAM 절약)
        for s in range(0, latents.shape[0], batch):
            imgs  = decode_to_pil(latents[s:s+batch])
            tensor= torch.stack([T.functional.to_tensor(im) for im in imgs]).to(device)
            feats = get_inception_feats(tensor)          # (b,2048)
            feats_all.append(feats.cpu())
            del imgs, tensor, feats
            torch.cuda.empty_cache()

        feats_all = torch.cat(feats_all, dim=0)          # (N,2048)
        torch.save({"inception_feats": feats_all}, dst)

# ───────────────────────── 실행 ─────────────────────────
root_dir   = '/data/archive/sd-v1-4'
model_name = 'dpm_solver++'
steps      = 200

for scale in [1.5, 3.5, 5.5, 7.5, 9.5]:
    src = os.path.join(root_dir, f"{model_name}_steps{steps}_scale{scale}")
    calc_fid_feats(src, src + '_fid')
