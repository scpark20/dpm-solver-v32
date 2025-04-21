import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sigma_min=0.002
sigma_max=80
device = "cuda" if torch.cuda.is_available() else "cpu"
# Fix the seed for z = sde.prior_sampling(shape).to(device) in deterministic sampling
torch.manual_seed(10)

N = 128
M = 128
K = 1
log_scale_max = 3.0
log_scale_min = -log_scale_max

scale_dir = f"/data/edm/scale/rbf_ecp_marginal{log_scale_max}"
root_dir = 'samples/edm-cifar10-32x32-uncond-vp/dpm_solver++_200'
ckp_path="/data/checkpoints/edm-cifar10-32x32-uncond-vp.pkl"

os.makedirs(scale_dir, exist_ok=True)

noises = []
samples = []
for i in range(8):
    file = os.path.join(root_dir, f'samples_{i}.npz')
    data = np.load(file)
    print(data['noises'].shape, data['samples'].shape)
    noises.append(data['noises'])
    samples.append(data['samples'])

noises = torch.tensor(np.concatenate(noises, axis=0))[:N]
samples = torch.tensor(np.concatenate(samples, axis=0))[:N]
print(noises.shape, samples.shape)

import pickle
from samplers.utils import NoiseScheduleEDM, model_wrapper

# Load network.

print(f'Loading network from "{ckp_path}"...')
with open(ckp_path, "rb") as f:
    net = pickle.load(f)["ema"].to(device)

ns = NoiseScheduleEDM()

from samplers.rbf_ecp_marginal import RBFSolverECPMarginal
rbf = RBFSolverECPMarginal(ns, algorithm_type="data_prediction", scale_dir=scale_dir, log_scale_max=log_scale_max, log_scale_min=log_scale_min)
def rbf_sampler(model_fn, z, x, steps, order, number, skip_type='logSNR'):
    with torch.no_grad():
        pred = rbf.sample_by_target_matching(
            model_fn,
            z,
            x,
            steps=steps,
            t_start=sigma_max,
            t_end=sigma_min,
            order=order,
            skip_type=skip_type,
            number=number
        )
        return pred

sampling_fn = rbf_sampler

import torch.nn.functional as F

for steps in [5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40]:
    for order in [3, 4]:
        for number in range(K):
            indexes = np.random.randint(0, len(noises), size=(M,))
            zs = noises[indexes].to(device)
            xs = samples[indexes].to(device)
            zs = zs.to(torch.float64) * sigma_max
            noise_pred_fn = model_wrapper(net, ns, None)
            pred = sampling_fn(noise_pred_fn, zs, xs, steps, order, number)
            print('Recon. Loss :', F.mse_loss(pred, xs))