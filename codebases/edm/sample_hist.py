# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
import logging
import os
import torch
import io
import time
import numpy as np


from torchvision.utils import make_grid, save_image
from samplers.dpm_solver import DPM_Solver
from samplers.uni_pc import UniPC
from samplers.rbf import RBFSolverGLQ10LagTime as RBF
from samplers.rbf_unipc import RBFUniPC
from samplers.dpm_solver_v3 import DPM_Solver_v3
from samplers.heun import Heun
from samplers.utils import NoiseScheduleEDM, model_wrapper
import functools
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("ckp_path", None, "Checkpoint path.")
flags.DEFINE_string("statistics_dir", None, "Statistics path for DPM-Solver-v3.")
flags.DEFINE_string("method", None, "Method: heun/dpm_solver++/uni_pc/dpm_solver_v3")
flags.DEFINE_string("eval_folder", "samples", "The folder name for storing evaluation results")
flags.DEFINE_string("sample_folder", "sample", "The folder name for storing samples")
flags.DEFINE_string("unipc_variant", "bh1", "UniPC variant: bh1/bh2")
flags.DEFINE_integer("steps", default=10, help="Number of sampling steps")
flags.DEFINE_integer("order", default=3, help="Order for sampling")
flags.DEFINE_boolean("denoise_to_zero", default=False, help="Denoise at the last step")
flags.DEFINE_string("skip_type", "logSNR", "The timestep schedule for sampling")
flags.mark_flags_as_required(["ckp_path", "method"])


def main(argv):
    sample(
        FLAGS.ckp_path,
        FLAGS.statistics_dir,
        FLAGS.eval_folder,
        FLAGS.sample_folder,
        FLAGS.method,
        FLAGS.steps,
        FLAGS.order,
        FLAGS.skip_type,
        FLAGS.denoise_to_zero,
        FLAGS.unipc_variant,
    )


def sample(
    ckp_path,
    statistics_dir,
    eval_folder,
    sample_dir,
    method,
    steps,
    order,
    skip_type,
    denoise_to_zero,
    unipc_variant,
    batch_size=16,
    num_samples=128,
    sigma_min=0.002,
    sigma_max=80,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Fix the seed for z = sde.prior_sampling(shape).to(device) in deterministic sampling
    torch.manual_seed(10)

    eval_dir = os.path.join(eval_folder, ckp_path.split("/")[-1].split(".")[-2])
    os.makedirs(eval_dir, exist_ok=True)

    # Load network.
    print(f'Loading network from "{ckp_path}"...')
    with open(ckp_path, "rb") as f:
        net = pickle.load(f)["ema"].to(device)

    ns = NoiseScheduleEDM()

    if method == "heun":
        heun = Heun(ns)

        def heun_sampler(model_fn, z):
            with torch.no_grad():
                x = heun.sample(
                    model_fn,
                    z,
                    steps=steps,
                    t_start=sigma_max,
                    t_end=sigma_min,
                    skip_type=skip_type,
                )
                return x, steps

        sampling_fn = heun_sampler

    elif method == "dpm_solver++":
        dpm_solver = DPM_Solver(ns, algorithm_type="dpmsolver++", correcting_x0_fn=None)

        def dpm_solver_sampler(model_fn, z):
            with torch.no_grad():
                x, intermediates, timesteps = dpm_solver.sample(
                    model_fn,
                    z,
                    steps=steps - 1 if denoise_to_zero else steps,
                    t_start=sigma_max,
                    t_end=sigma_min,
                    order=order,
                    skip_type=skip_type,
                    lower_order_final=True,
                    denoise_to_zero=denoise_to_zero,
                    return_intermediate=True
                )
                return x, intermediates, timesteps, steps

        sampling_fn = dpm_solver_sampler
    elif method == "uni_pc":
        uni_pc = UniPC(ns, algorithm_type="data_prediction", correcting_x0_fn=None, variant=unipc_variant)

        def uni_pc_sampler(model_fn, z):
            with torch.no_grad():
                x = uni_pc.sample(
                    model_fn,
                    z,
                    steps=steps - 1 if denoise_to_zero else steps,
                    t_start=sigma_max,
                    t_end=sigma_min,
                    order=order,
                    skip_type=skip_type,
                    lower_order_final=True,
                    denoise_to_zero=denoise_to_zero,
                )
                return x, steps

        sampling_fn = uni_pc_sampler

    elif method == "rbf":
        rbf = RBF(ns, algorithm_type="data_prediction", correcting_x0_fn=None)

        def rbf_sampler(model_fn, z):
            with torch.no_grad():
                x = rbf.sample(
                    model_fn,
                    z,
                    steps=steps - 1 if denoise_to_zero else steps,
                    t_start=sigma_max,
                    t_end=sigma_min,
                    order=order,
                    skip_type=skip_type,
                    lower_order_final=True,
                    denoise_to_zero=denoise_to_zero,
                )
                return x, steps

        sampling_fn = rbf_sampler

    elif method == "rbf_unipc":
        rbf_unipc = RBFUniPC(ns, algorithm_type="data_prediction", correcting_x0_fn=None, variant=unipc_variant)

        def rbf_unipc_sampler(model_fn, z):
            with torch.no_grad():
                x = rbf_unipc.sample(
                    model_fn,
                    z,
                    steps=steps - 1 if denoise_to_zero else steps,
                    t_start=sigma_max,
                    t_end=sigma_min,
                    order=order,
                    skip_type=skip_type,
                    lower_order_final=True,
                    denoise_to_zero=denoise_to_zero,
                )
                return x, steps

        sampling_fn = rbf_unipc_sampler

    elif method == "dpm_solver_v3":
        assert statistics_dir is not None, "No appropriate statistics found."
        print("Use statistics", statistics_dir)

        dpm_solver_v3 = DPM_Solver_v3(
            statistics_dir,
            ns,
            steps=steps,
            t_start=sigma_max,
            t_end=sigma_min,
            skip_type=skip_type,
            device=device,
        )

        def dpm_solver_v3_sampler(model_fn, z):
            with torch.no_grad():
                x = dpm_solver_v3.sample(
                    model_fn,
                    z,
                    order=order,
                    p_pseudo=steps <= 5,
                    use_corrector=steps <= 6,
                    c_pseudo=False,
                    lower_order_final=True,
                )
            return x, steps

        sampling_fn = dpm_solver_v3_sampler
    else:
        assert False, f"Method {method} not supported."

    # Directory to save samples. Different for each host to avoid writing conflicts
    this_sample_dir = os.path.join(eval_dir, sample_dir)
    os.makedirs(this_sample_dir, exist_ok=True)
    logging.info(this_sample_dir)
    num_sampling_rounds = num_samples // batch_size + 1
    for r in range(num_sampling_rounds):
        latents = torch.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution]).to(device)
        noises = latents
        latents = latents.to(torch.float64) * sigma_max
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim)[torch.randint(net.label_dim, size=[batch_size])].to(device)
        noise_pred_fn = model_wrapper(net, ns, class_labels)
        samples_raw, traj, timesteps, n = sampling_fn(noise_pred_fn, latents)
        logging.info("sampling -- ckpt: %s, round: %d (NFE %d)" % (ckp_path.split("/")[-1].split(".")[-2], r, n))
        np.savez_compressed(os.path.join(this_sample_dir, f"samples_{r}.npz"),
                            noises=noises.cpu(),
                            samples=samples_raw.cpu(),
                            hist=torch.stack(traj, dim=0).cpu(),
                            timesteps=timesteps.cpu(),
                            )

if __name__ == "__main__":
    app.run(main)
