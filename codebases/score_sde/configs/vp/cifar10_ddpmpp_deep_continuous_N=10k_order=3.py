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

# Lint as: python3
"""Training NCSNv3 on CIFAR-10 with continuous sigmas."""

from configs.default_cifar10_configs import get_default_configs


def get_config():
    config = get_default_configs()
    config.eval.num_samples = 10000
    
    # training
    training = config.training
    training.sde = "vpsde"
    training.continuous = True
    training.reduce_mean = True
    training.n_iters = 950001

    # sampling
    sampling = config.sampling

    # sampling.method = 'pc'
    # sampling.predictor = 'euler_maruyama'
    # sampling.corrector = 'none'

    # sampling.method = 'ode'
    # sampling.eps = 1e-4
    # sampling.noise_removal = False
    # sampling.rk45_rtol = 1e-5
    # sampling.rk45_atol = 1e-5

    # sampling.method = "dpm_solver"
    sampling.dpm_solver_method = "multistep"
    sampling.dpm_solver_algorithm_type = "dpmsolver++"
    sampling.rtol = 0.05

    # sampling.method = "uni_pc"
    sampling.uni_pc_method = "multistep"
    sampling.uni_pc_algorithm_type = "data_prediction"
    sampling.variant = "bh1"

    # dpm_solver and uni_pc
    sampling.thresholding = False
    sampling.noise_removal = False

    sampling.method = "dpm_solver_v3"
    sampling.eps = 1e-3
    sampling.order = 3
    sampling.steps = 1000
    sampling.skip_type = "logSNR"
    sampling.predictor_pseudo = False
    sampling.use_corrector = True
    sampling.corrector_pseudo = False
    sampling.lower_order_final = True
    sampling.degenerated = False

    # data
    data = config.data
    data.centered = True

    # model
    model = config.model
    model.name = "ncsnpp"
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 8
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "none"
    model.progressive_input = "none"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.init_scale = 0.0
    model.embedding_type = "positional"
    model.fourier_scale = 16
    model.conv_size = 3

    return config
