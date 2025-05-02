import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
print('done')

from ldm.models.diffusion.dcsolver import DCSampler

sampler = DCSampler(model)
print('done')

N = 10
M = 10
K = 1
#for SCALE in [1.5, 3.5, 5.5, 7.5, 9.5]:
for SCALE in [1.5, 3.5, 5.5, 7.5, 9.5]:
    os.makedirs(f'/data/ldm/dc{SCALE}', exist_ok=True)

    import numpy as np

    pt_dir = f'/data/ldm/outputs/sd-v1-4/dpm_solver++_steps1000_scale{SCALE}'
    pt_files = [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if '.pt' in f]
    prompts_list = []
    traj_list = []
    timesteps = None
    for i in range(2):
        data = torch.load(pt_files[i])
        prompts_list.append(data['text'])
        traj_list.append(data['traj'])
        timesteps = data['timesteps']

    prompts_list = np.ravel(prompts_list)[:N]
    traj_list = torch.cat(traj_list, dim=1)[:, :N]
    print(len(prompts_list), traj_list.shape, timesteps.shape)

    from torch import autocast
    from contextlib import nullcontext

    for ORDER in [3]:
        for NFE in [5, 6, 8, 10, 12, 15, 20]:
            for number in range(K):
                index = np.random.randint(0, N, size=(M,))
                prompts = list(prompts_list[index])
                traj = traj_list[:, index].to(device)
                print(prompts)
                precision_scope = autocast if opt.precision == "autocast" else nullcontext
                with precision_scope("cuda"):
                    with model.ema_scope():
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(len(prompts) * [""])
                        c = model.get_learned_conditioning(prompts)
                        samples, _ = sampler.target_matching(
                            (traj.to(device), timesteps),
                            S=NFE,
                            shape=(4, 64, 64),
                            conditioning=c,
                            batch_size=len(prompts),
                            verbose=False,
                            unconditional_guidance_scale=SCALE,
                            unconditional_conditioning=uc,
                            eta=0,
                            order=ORDER,
                            number=number,
                            dc_dir=f'/data/ldm/dc{SCALE}'
                        )
