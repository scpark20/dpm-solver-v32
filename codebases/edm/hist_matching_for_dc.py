import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sigma_min=0.002
sigma_max=80
device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix the seed for z = sde.prior_sampling(shape).to(device) in deterministic sampling
torch.manual_seed(10)

N = 10

dc_dir = f"/data/edm/dc/"
root_dir = 'samples/edm-cifar10-32x32-uncond-vp/dpm_solver++_200'
ckp_path="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"

os.makedirs(dc_dir, exist_ok=True)

noises = []
samples = []
file = os.path.join(root_dir, f'samples_0.npz')
data = np.load(file)
traj = torch.tensor(data['hist'], device=device)[:, :N]
timesteps = torch.tensor(data['timesteps'])

import pickle
from samplers.utils import NoiseScheduleEDM, model_wrapper

# Load network.

print(f'Loading network from "{ckp_path}"...')
with open(ckp_path, "rb") as f:
    net = pickle.load(f)["ema"].to(device)

ns = NoiseScheduleEDM()
from samplers.dc_solver import DCSolver
dcsolver = DCSolver(ns, dc_dir=dc_dir)
def dc_sampler(model_fn, traj, timesteps, steps, order, skip_type='logSNR'):
    with torch.no_grad():
        dcsolver.ref_xs = traj
        dcsolver.ref_ts = timesteps
        pred = dcsolver.search_dc(
            model_fn,
            None,
            steps=steps,
            t_start=sigma_max,
            t_end=sigma_min,
            order=order,
            skip_type=skip_type,
            method='multistep'
        )
        return pred

sampling_fn = dc_sampler

import torch.nn.functional as F

for steps in [5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40]:
    for order in [3]:
        noise_pred_fn = model_wrapper(net, ns, None)
        pred = sampling_fn(noise_pred_fn, traj, timesteps, steps, order)
        print('Recon. Loss :', F.mse_loss(pred, traj[-1]))