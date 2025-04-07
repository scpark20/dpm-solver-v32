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

"""Training and evaluation"""

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import torch
import io
import time
import numpy as np

# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from torchvision.utils import make_grid, save_image
from utils import restore_checkpoint
from models.utils import get_noise_fn
from samplers.dpm_solver_v3 import DPM_Solver_v3
from samplers.utils import NoiseScheduleVP
import functools

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("ckp_path", None, "Checkpoint path.")
flags.DEFINE_string("scale_dir", None, "dir to save scale")
flags.DEFINE_string("pair_npz", None, "pair npz to read")
flags.mark_flags_as_required(["ckp_path", "config"])

def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def main(argv):
    sample(FLAGS.config, FLAGS.ckp_path, FLAGS.pair_npz, FLAGS.scale_dir)

def sample(config, ckp_path, pair_npz, scale_dir):
    # Fix the seed for z = sde.prior_sampling(shape).to(device) in deterministic sampling
    torch.manual_seed(config.seed)
    
    # Create data normalizer and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Initialize model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
    state = dict(model=score_model, ema=ema, step=0)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unsupported.")

    sampling_shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)

    state = restore_checkpoint(ckp_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, target_matching=True, scale_dir=scale_dir)
    sampling_fn = functools.partial(sampling_fn, score_model)
    data = np.load(pair_npz)
    x = torch.tensor(data['prior'], device=config.device)
    target = torch.tensor(data['hist'], device=config.device)
    samples_raw, n = sampling_fn(x, target)

if __name__ == "__main__":
    app.run(main)
