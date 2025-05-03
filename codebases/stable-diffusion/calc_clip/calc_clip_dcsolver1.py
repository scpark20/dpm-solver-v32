import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''Stable Diffusion Init.'''
from txt2img_latent import get_parser, load_model_from_config, chunk

parser = get_parser()

# 2) args_list 정의 (원하는 인자들을 문자열 리스트로)
args_list = [
    "--config", "configs/stable-diffusion/v1-inference.yaml",
    "--ckpt", "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
    "--H", "512",
    "--W", "512",
    "--C", "4",
    "--f", "8",
]

# 3) parse_args() 실행
opt = parser.parse_args(args_list)

import torch
from omegaconf import OmegaConf

config = OmegaConf.load(f"{opt.config}")
model = load_model_from_config(config, f"{opt.ckpt}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.eval()

''' CLIP Model Init.'''
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

'''Decode and PIL'''
import numpy as np
import os
import torch
from einops import rearrange
import matplotlib.pyplot as plt
from PIL import Image

# 1) decode() 안에서 PIL Image까지 바로 꺼내도록 살짝 손봐두면 편함
def decode_to_pil(image):
    with torch.no_grad():
        x = model.decode_first_stage(image.to(device))
    x = torch.clamp((x + 1.0) / 2.0, 0, 1)          # [0,1]
    x = (x * 255).byte().cpu()                      # uint8
    # BCHW  ->  list[HWC]
    imgs = [Image.fromarray(t.permute(1,2,0).numpy()) for t in x]
    return imgs

''' Define Calc Function'''
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

def calc_cosim(pt_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    files = [os.path.join(pt_path, f) for f in os.listdir(pt_path) if '.pt' in f]

    for file in tqdm(files):
        basename = os.path.basename(file)
        save_file = os.path.join(save_path, basename)
        if os.path.exists(save_file):
            continue

        data = torch.load(file)
        texts = clip.tokenize(data['text']).to(device)
        images = torch.stack([preprocess(image) for image in decode_to_pil(data['image'])]).to(device)
        
        with torch.no_grad():
            image_features = clip_model.encode_image(images)
            text_features = clip_model.encode_text(texts)
            cosim = F.cosine_similarity(image_features, text_features)
        torch.save({"cosim": cosim.cpu(),
                    "text_features": text_features.cpu(),
                    "image_features": image_features.cpu(),
                    }, save_file)


''' Run '''
root_dir = '/data/ldm/outputs/sd-v1-4'
model_names = ['dcsolver_order3']
for scale in [1.5, 3.5, 5.5, 7.5, 9.5]:
    for model_name in model_names:
        for steps in [5, 15]:
            path = os.path.join(root_dir, f"{model_name}_steps{steps}_scale{scale}")
            if os.path.exists(path):
                try:
                    calc_cosim(path, path + '_clip')
                except Exception as e:
                    # 오류 메시지만 출력하고 다음 항목으로 계속 진행
                    print(f"[ERROR] calc_cosim 실패: {path} → {e}")