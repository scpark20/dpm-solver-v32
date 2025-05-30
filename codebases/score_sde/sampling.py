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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
import functools

import torch
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn, get_noise_fn
from scipy import integrate
import sde_lib
from models import utils as mutils
from samplers.utils import NoiseScheduleVP
from samplers.dpm_solver import DPM_Solver
from samplers.uni_pc import UniPC
from samplers.rbf import RBFSolverGLQ10LagTime
from samplers.rbf4 import RBFSolver4
from samplers.rbf_spd import RBFSolverSPD
from samplers.rbf_spd_plag_tm import RBFSolverSPDPlagTM
from samplers.rbf_spd_clag_tm import RBFSolverSPDClagTM
from samplers.rbf_spd_plag_xt import RBFSolverSPDPlagXt
from samplers.rbf_spd_clag_xt import RBFSolverSPDClagXt
from samplers.rbf_spd_ptm_cxt import RBFSolverSPDPTMCXT
from samplers.rbf_spd_pxt_ctm import RBFSolverSPDPXTCTM
from samplers.rbf_spd_ecp import RBFSolverSPDECP
from samplers.rbf_spd_xt import RBFSolverSPDXt
from samplers.rbf_spd_xt_nonlower import RBFSolverSPDXtNL
from samplers.rbf_spd_xt4 import RBFSolverSPDXt4
from samplers.rbf_spd_xt5 import RBFSolverSPDXt5
from samplers.rbf_spd_const import RBFSolverSPDConst
from samplers.rbf_xt import RBFSolverXt
from samplers.rbf_x0 import RBFSolverX0
from samplers.rbf_gram import RBFSolverGram
from samplers.rbf_gram_lag import RBFSolverGramLag
from samplers.rbf_spd_gram_lag import RBFSolverSPDGramLag
from samplers.rbf_plag_ctarget import RBFSolverPLagCTarget
from samplers.rbf_ptarget_clag import RBFSolverPTargetCLag
from samplers.rbf_inception_lag import RBFSolverInceptionLag
from samplers.rbf_100 import RBFSolver100
from samplers.rbf_const import RBFSolverConst
from samplers.rbf_const_grid import RBFSolverConstGrid
from samplers.rbf_ecp import RBFSolverECP
from samplers.rbf_marginal_gram import RBFSolverMarginalGram
from samplers.rbf_marginal_inception import RBFSolverMarginalInception
from samplers.rbf_marginal import RBFSolverMarginal
from samplers.rbf_ecp_marginal import RBFSolverECPMarginal
from samplers.rbf_ecp_marginal_lower import RBFSolverECPMarginalLower
from samplers.rbf_ecp_marginal_lower1 import RBFSolverECPMarginalLower1
from samplers.rbf_ecp_marginal_sep import RBFSolverECPMarginalSep
from samplers.rbf_ecp_marginal_xt import RBFSolverECPMarginalXt
from samplers.rbf_ecp_marginal_same import RBFSolverECPMarginalSame
from samplers.rbf_ecp_marginal_spd import RBFSolverECPMarginalSPD
from samplers.rbf_ecp_marginal_lagp import RBFSolverECPMarginalLagP
from samplers.rbf_ecp_marginal_lagc import RBFSolverECPMarginalLagC
from samplers.dc_solver import DCSolver

from samplers.rbf_ecp_same import RBFSolverECPSame
from samplers.rbf_ecp_same4 import RBFSolverECPSame4
from samplers.rbf_ecp_same5 import RBFSolverECPSame5
from samplers.rbf_dual import RBFDual
from samplers.rbf_unipc import RBFUniPC
from samplers.lagrange_solver import LagrangeSolver
from samplers.lagrange_solver_mix import LagrangeSolverMix
from samplers.rbf_mix import RBFSolverMix
from samplers.rbf_mix_ecp_same import RBFSolverMixECPSame

_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f"Already registered model with name: {local_name}")
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, target_matching=False, return_prior=False, return_hist=False, scale_dir=None, dc_dir=None):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == "ode":
        sampling_fn = get_ode_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            denoise=config.sampling.noise_removal,
            eps=config.sampling.eps,
            rtol=config.sampling.rk45_rtol,
            atol=config.sampling.rk45_atol,
            device=config.device,
        )
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == "pc":
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(
            sde=sde,
            shape=shape,
            predictor=predictor,
            corrector=corrector,
            inverse_scaler=inverse_scaler,
            snr=config.sampling.snr,
            n_steps=config.sampling.n_steps_each,
            probability_flow=config.sampling.probability_flow,
            continuous=config.training.continuous,
            denoise=config.sampling.noise_removal,
            eps=config.sampling.eps,
            device=config.device,
        )
    elif sampler_name.lower() == "dpm_solver":
        sampling_fn = get_dpm_solver_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.dpm_solver_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.dpm_solver_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            return_hist=return_hist
        )
    elif sampler_name.lower() == "uni_pc":
        sampling_fn = get_uni_pc_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            return_hist=return_hist
        )
    elif sampler_name.lower() == "rbf":
        sampling_fn = get_rbf_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )

    elif sampler_name.lower() == "rbf4":
        sampling_fn = get_rbf4_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )

    elif sampler_name.lower() == "rbf_spd_ptm_cxt":
        sampling_fn = get_rbf_spd_ptm_cxt_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_spd_pxt_ctm":
        sampling_fn = get_rbf_spd_pxt_ctm_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_spd_plag_tm":
        sampling_fn = get_rbf_spd_plag_tm_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_spd_plag_xt":
        sampling_fn = get_rbf_spd_plag_xt_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_spd_clag_tm":
        sampling_fn = get_rbf_spd_clag_tm_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_spd_clag_xt":
        sampling_fn = get_rbf_spd_clag_xt_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_spd":
        sampling_fn = get_rbf_spd_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_spd_ecp":
        sampling_fn = get_rbf_spd_ecp_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_spd_xt":
        sampling_fn = get_rbf_spd_xt_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_spd_xt_nonlower":
        sampling_fn = get_rbf_spd_xt_nonlower_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_spd_xt4":
        sampling_fn = get_rbf_spd_xt4_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )            

    elif sampler_name.lower() == "rbf_spd_xt5":
        sampling_fn = get_rbf_spd_xt5_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )                

    elif sampler_name.lower() == "rbf_spd_const":
        sampling_fn = get_rbf_spd_const_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_xt":
        sampling_fn = get_rbf_xt_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_x0":
        sampling_fn = get_rbf_x0_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_plag_ctarget":
        sampling_fn = get_rbf_plag_ctarget_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )

    elif sampler_name.lower() == "rbf_ptarget_clag":
        sampling_fn = get_rbf_ptarget_clag_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_gram":
        sampling_fn = get_rbf_gram_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_gram_lag":
        sampling_fn = get_rbf_gram_lag_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_spd_gram_lag":
        sampling_fn = get_rbf_spd_gram_lag_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_inception_lag":
        sampling_fn = get_rbf_inception_lag_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_100":
        sampling_fn = get_rbf_100_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )

    elif sampler_name.lower() == "rbf_const":
        sampling_fn = get_rbf_const_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_const_grid":
        sampling_fn = get_rbf_const_grid_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        ) 

    elif sampler_name.lower() == "rbf_marginal_gram":
        sampling_fn = get_rbf_marginal_gram_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )               

    elif sampler_name.lower() == "rbf_marginal_inception":
        sampling_fn = get_rbf_marginal_inception_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )                   

    elif sampler_name.lower() == "rbf_ecp":
        sampling_fn = get_rbf_ecp_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )

    elif sampler_name.lower() == "dcsolver":
        sampling_fn = get_dc_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            dc_dir=dc_dir,
        )        

    elif sampler_name.lower() == "rbf_ecp_marginal":
        sampling_fn = get_rbf_ecp_marginal_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
            return_hist=return_hist
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_lower":
        sampling_fn = get_rbf_ecp_marginal_lower_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_lower1":
        sampling_fn = get_rbf_ecp_marginal_lower1_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_sep":
        sampling_fn = get_rbf_ecp_marginal_sep_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_xt":
        sampling_fn = get_rbf_ecp_marginal_xt_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
            return_hist=return_hist
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_lagp":
        sampling_fn = get_rbf_ecp_marginal_lagp_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_lagc":
        sampling_fn = get_rbf_ecp_marginal_lagc_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_spd":
        sampling_fn = get_rbf_ecp_marginal_spd_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_to1":
        sampling_fn = get_rbf_ecp_marginal_to1_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_to3":
        sampling_fn = get_rbf_ecp_marginal_to3_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal_same":
        sampling_fn = get_rbf_ecp_marginal_same_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_marginal":
        sampling_fn = get_rbf_marginal_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_ecp_marginal4":
        sampling_fn = get_rbf_ecp_marginal4_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_ecp_marginal5":
        sampling_fn = get_rbf_ecp_marginal5_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )      

    elif sampler_name.lower() == "rbf_ecp_marginal6":
        sampling_fn = get_rbf_ecp_marginal6_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )                  

    elif sampler_name.lower() == "rbf_ecp_same":
        sampling_fn = get_rbf_ecp_same_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )        

    elif sampler_name.lower() == "rbf_ecp_same4":
        sampling_fn = get_rbf_ecp_same4_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )            

    elif sampler_name.lower() == "rbf_ecp_same5":
        sampling_fn = get_rbf_ecp_same5_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )            

    elif sampler_name.lower() == "rbf_mix":
        sampling_fn = get_rbf_mix_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )

    elif sampler_name.lower() == "rbf_mix_ecp_same":
        sampling_fn = get_rbf_mix_ecp_same_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    

    elif sampler_name.lower() == "rbf_dual":
        sampling_fn = get_rbf_dual_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            target_matching=target_matching,
            scale_dir=scale_dir,
        )    
    elif sampler_name.lower() == "rbf_unipc":
        sampling_fn = get_rbf_unipc_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
        )
    elif sampler_name.lower() == "lagrange":
        sampling_fn = get_lagrange_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            return_prior=return_prior,
        )        
    elif sampler_name.lower() == "lagrange_mix":
        sampling_fn = get_lagrange_mix_sampler(
            sde=sde,
            shape=shape,
            inverse_scaler=inverse_scaler,
            steps=config.sampling.steps,
            eps=config.sampling.eps,
            skip_type=config.sampling.skip_type,
            method=config.sampling.uni_pc_method,
            order=config.sampling.order,
            denoise=config.sampling.noise_removal,
            algorithm_type=config.sampling.uni_pc_algorithm_type,
            thresholding=config.sampling.thresholding,
            rtol=config.sampling.rtol,
            variant=config.sampling.variant,
            lower_order_final=config.sampling.lower_order_final,
            device=config.device,
            return_prior=return_prior,
        )            
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name="euler_maruyama")
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1.0 / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name="reverse_diffusion")
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_predictor(name="ancestral_sampling")
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma**2 - adjacent_sigma**2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma**2 * (sigma**2 - adjacent_sigma**2)) / (sigma**2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1.0 - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name="none")
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x


@register_corrector(name="langevin")
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name="ald")
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

    We include this corrector only for completeness. It was not directly used in our paper.
    """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if (
            not isinstance(sde, sde_lib.VPSDE)
            and not isinstance(sde, sde_lib.VESDE)
            and not isinstance(sde, sde_lib.subVPSDE)
        ):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean


@register_corrector(name="none")
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


def get_pc_sampler(
    sde,
    shape,
    predictor,
    corrector,
    inverse_scaler,
    snr,
    n_steps=1,
    probability_flow=False,
    continuous=False,
    denoise=True,
    eps=1e-3,
    device="cuda",
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(
        shared_predictor_update_fn,
        sde=sde,
        predictor=predictor,
        probability_flow=probability_flow,
        continuous=continuous,
    )
    corrector_update_fn = functools.partial(
        shared_corrector_update_fn, sde=sde, corrector=corrector, continuous=continuous, snr=snr, n_steps=n_steps
    )

    def pc_sampler(model):
        """The PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, model=model)
                x, x_mean = predictor_update_fn(x, vec_t, model=model)

            return inverse_scaler(x_mean if denoise else x), sde.N * (n_steps + 1)

    return pc_sampler


def get_ode_sampler(
    sde, shape, inverse_scaler, denoise=False, rtol=1e-5, atol=1e-5, method="RK45", eps=1e-3, device="cuda"
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      inverse_scaler: The inverse data normalizer.
      denoise: If `True`, add one-step denoising to final samples.
      rtol: A `float` number. The relative tolerance level of the ODE solver.
      atol: A `float` number. The absolute tolerance level of the ODE solver.
      method: A `str`. The algorithm used for the black-box ODE solver.
        See the documentation of `scipy.integrate.solve_ivp`.
      eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (sde.T, eps), to_flattened_numpy(x), rtol=rtol, atol=atol, method=method
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler


def get_dpm_solver_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="singlestep",
    order=3,
    denoise=False,
    algorithm_type="dpmsolver",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    lower_order_final=True,
    device="cuda",
    return_hist=False
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def dpm_solver_sampler(model):
        """The DPM-Solver sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            dpm_solver = DPM_Solver(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if return_hist:
                x, intermediates, timesteps = dpm_solver.sample(
                x,
                steps=steps - 1 if denoise else steps,
                t_start=sde.T,
                t_end=eps,
                order=order,
                skip_type=skip_type,
                method=method,
                denoise_to_zero=denoise,
                atol=atol,
                rtol=rtol,
                lower_order_final=lower_order_final,
                return_intermediate=return_hist
                )
                return inverse_scaler(x), intermediates, timesteps, steps
            else:
                x = dpm_solver.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    denoise_to_zero=denoise,
                    atol=atol,
                    rtol=rtol,
                    lower_order_final=lower_order_final,
                )
                return inverse_scaler(x), steps

    return dpm_solver_sampler


def get_uni_pc_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    return_hist=False
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def uni_pc_sampler(model):
        """The UniPC sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            uni_pc = UniPC(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                variant=variant,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            x = uni_pc.sample(
                x,
                steps=steps - 1 if denoise else steps,
                t_start=sde.T,
                t_end=eps,
                order=order,
                skip_type=skip_type,
                method=method,
                denoise_to_zero=denoise,
                atol=atol,
                rtol=rtol,
                lower_order_final=lower_order_final,
                return_intermediate=return_hist
            )
            if return_hist:
                x, timesteps, hist = x
                return inverse_scaler(x), timesteps, hist, steps
            return inverse_scaler(x), steps

    return uni_pc_sampler

def get_rbf_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverGLQ10LagTime(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf4_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolver4(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_plag_tm_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDPlagTM(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_ptm_cxt_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None, xt=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDPTMCXT(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    xt=xt,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_pxt_ctm_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None, xt=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDPXTCTM(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    xt=xt,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_plag_xt_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDPlagXt(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_clag_tm_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDClagTM(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_clag_xt_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDClagXt(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPD(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_ecp_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDECP(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_xt_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDXt(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_xt_nonlower_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDXtNL(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_xt4_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDXt4(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_xt5_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDXt5(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_spd_const_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverSPDConst(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_xt_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverXt(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_x0_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverX0(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_plag_ctarget_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverPLagCTarget(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_ptarget_clag_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverPTargetCLag(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_gram_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_gram_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_gram = RBFSolverGram(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_gram.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_gram.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_gram_sampler

def get_rbf_gram_lag_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_gram_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_gram = RBFSolverGramLag(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_gram.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_gram.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_gram_sampler

def get_rbf_spd_gram_lag_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_gram_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_gram = RBFSolverSPDGramLag(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_gram.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_gram.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_gram_sampler

def get_rbf_inception_lag_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_inception_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverInceptionLag(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_inception_sampler

def get_rbf_100_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_100_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolver100(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_100_sampler

def get_rbf_const_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_const_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverConst(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_const_sampler

def get_rbf_const_grid_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_const_grid_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverConstGrid(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_const_grid_sampler

def get_rbf_ecp_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECP(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_marginal_gram_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_marginal_gram_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverMarginalGram(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_marginal_gram_sampler

def get_rbf_marginal_inception_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverMarginalInception(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_sampler

def get_rbf_ecp_marginal_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
    return_hist=False
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginal(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                    return_intermediate=return_hist
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            if return_hist:
                x, timesteps, hist = x
                return inverse_scaler(x), timesteps, hist, steps
            else:
                return inverse_scaler(x), steps

            

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_lower_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginalLower(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_lower1_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginalLower1(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_dc_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    dc_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def dc_sampler(model, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            dcsolver = DCSolver(
                noise_pred_fn,
                ns,
                dc_dir=dc_dir
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = dcsolver.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                dcsolver.ref_xs = target[0]
                dcsolver.ref_ts = target[1]
                x = dcsolver.search_dc(
                    target[0][0],
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return dc_sampler

def get_rbf_ecp_marginal_sep_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginalSep(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_xt_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
    return_hist=False
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            if not target_matching:
                rbf_ecp = RBFSolverECPMarginal(
                    noise_pred_fn,
                    ns,
                    algorithm_type=algorithm_type,
                    correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                    scale_dir=scale_dir,
                )
                # Initial sample
                x = sde.prior_sampling(shape).to(device)    
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                    return_intermediate=return_hist
                )
            else:
                rbf_ecp = RBFSolverECPMarginalXt(
                    noise_pred_fn,
                    ns,
                    algorithm_type=algorithm_type,
                    correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                    scale_dir=scale_dir,
                )
                # Initial sample
                x = sde.prior_sampling(shape).to(device)    
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            if return_hist:
                x, timesteps, hist = x
                return inverse_scaler(x), timesteps, hist, steps
            else:
                return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_lagp_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginalLagP(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_lagc_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginalLagC(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_spd_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginalSPD(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_to1_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginal(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
                log_scale_max=1.0,
                log_scale_min=-1.0
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_to3_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginal(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
                log_scale_max=3.0,
                log_scale_min=-3.0,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal_same_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginalSame(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_marginal_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverMarginal(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler    

def get_rbf_ecp_marginal4_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginal(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=4,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=4,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal5_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginal(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=5,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=5,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_marginal6_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp = RBFSolverECPMarginal(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=6,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=6,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_sampler

def get_rbf_ecp_same_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_same_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp_same = RBFSolverECPSame(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp_same.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp_same.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_same_sampler

def get_rbf_ecp_same4_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_same4_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp_same = RBFSolverECPSame4(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp_same.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp_same.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_same4_sampler

def get_rbf_ecp_same5_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_ecp_same5_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf_ecp_same = RBFSolverECPSame5(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf_ecp_same.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf_ecp_same.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_ecp_same5_sampler

def get_rbf_mix_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_mix_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverMix(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_mix_sampler

def get_rbf_mix_ecp_same_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_mix_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFSolverMixECPSame(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target=target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_mix_sampler


def get_rbf_dual_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    target_matching=False,
    scale_dir=None,
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_dual_sampler(model, prior=None, target=None):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFDual(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                scale_dir=scale_dir,
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            if not target_matching:
                x = rbf.sample(
                    x,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            else:
                x = rbf.sample_by_target_matching(
                    prior,
                    target,
                    steps=steps - 1 if denoise else steps,
                    t_start=sde.T,
                    t_end=eps,
                    order=order,
                    skip_type=skip_type,
                    method=method,
                    lower_order_final=lower_order_final,
                )
            return inverse_scaler(x), steps

    return rbf_dual_sampler

def get_lagrange_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    return_prior=False
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def lagrange_sampler(model):
        """The Lagrange sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            lagrange = LagrangeSolver(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
            )
            # Initial sample
            prior = sde.prior_sampling(shape).to(device)
            x, hist, xt = lagrange.sample(
                prior,
                steps=steps - 1 if denoise else steps,
                t_start=sde.T,
                t_end=eps,
                order=order,
                skip_type=skip_type,
                method=method,
                lower_order_final=lower_order_final,
            )
            if return_prior:
                return inverse_scaler(x), steps, prior, x, hist, xt
            else:
                return inverse_scaler(x), steps

    return lagrange_sampler

def get_lagrange_mix_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
    return_prior=False
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def lagrange_mix_sampler(model):
        """The Lagrange sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            lagrange = LagrangeSolverMix(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
            )
            # Initial sample
            prior = sde.prior_sampling(shape).to(device)
            x = lagrange.sample(
                prior,
                steps=steps - 1 if denoise else steps,
                t_start=sde.T,
                t_end=eps,
                order=order,
                skip_type=skip_type,
                method=method,
                lower_order_final=lower_order_final,
            )
            if return_prior:
                return x, steps, prior
            else:
                return inverse_scaler(x), steps

    return lagrange_mix_sampler

def get_rbf_unipc_sampler(
    sde,
    shape,
    inverse_scaler,
    steps=10,
    eps=1e-3,
    skip_type="logSNR",
    method="multistep",
    order=3,
    denoise=False,
    algorithm_type="data_prediction",
    thresholding=False,
    rtol=0.05,
    atol=0.0078,
    variant="bh1",
    lower_order_final=True,
    device="cuda",
):
    """Create a UniPC sampler.
    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      variant: [bh1, bh2, vary_coeff], decide which variant of UniPC is used
      device: PyTorch device.
    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    ns = NoiseScheduleVP("linear", continuous_beta_0=sde.beta_0, continuous_beta_1=sde.beta_1)

    def rbf_sampler(model):
        """The RBF sampler funciton.
        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            noise_pred_fn = get_noise_fn(sde, model, train=False, continuous=True)
            rbf = RBFUniPC(
                noise_pred_fn,
                ns,
                algorithm_type=algorithm_type,
                correcting_x0_fn="dynamic_thresholding" if thresholding else None,
                variant=variant
            )
            # Initial sample
            x = sde.prior_sampling(shape).to(device)
            x = rbf.sample(
                x,
                steps=steps - 1 if denoise else steps,
                t_start=sde.T,
                t_end=eps,
                order=order,
                skip_type=skip_type,
                method=method,
                lower_order_final=lower_order_final,
            )
            return inverse_scaler(x), steps

    return rbf_sampler
