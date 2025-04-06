import os
import torch
import torch.nn.functional as F
import numpy as np
from .utils import expand_dims
import math

# Mix Prediction
class RBFSolverMix:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type=None, # deprecated
            correcting_x0_fn=None,
            thresholding_max_val=1.,
            dynamic_thresholding_ratio=0.995,
            scale_dir=None,
            log_scale_min=-2.0,
            log_scale_max=2.0,
            log_scale_num=100
    ):
        self.model = lambda x, t: model_fn(x, t.expand((x.shape[0])))
        self.noise_schedule = noise_schedule
        
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.scale_dir = scale_dir
        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max
        self.log_scale_num = log_scale_num
        

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return noise, x0

        
    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.
        """
        if skip_type == 'logSNR':
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == 'time_uniform':
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == 'time_quadratic':
            t_order = 2
            t = torch.linspace(t_T**(1. / t_order), t_0**(1. / t_order), N + 1).pow(t_order).to(device)
            return t
        else:
            raise ValueError("Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type))

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)
    
    def get_kernel_matrix(self, lambdas, beta):
        # (p, 1)
        lambdas = lambdas[:, None]
        # (p, p)
        K = torch.exp(-beta**2 * (lambdas - lambdas.T) ** 2)
        return K

    def get_integral_vector(self, lambda_s, lambda_t, lambdas, beta, tau):
        from scipy.integrate import quad
        if beta == 0.0:
            if tau == 0.0:
                return (lambda_t - lambda_s) * torch.ones_like(lambdas)
            else:
                return (torch.exp(tau*lambda_t) - torch.exp(tau*lambda_s))/tau * torch.ones_like(lambdas)

        device = lambda_s.device
        lambda_s = lambda_s.cpu()
        lambda_t = lambda_t.cpu()
        lambdas = lambdas.cpu()

        vals = []
        for lambda_u in lambdas:    
            def integrand(lmbd):
                return np.exp(tau*lmbd - beta**2 * (lmbd - lambda_u)**2)
            val, _ = quad(integrand, lambda_s, lambda_t)
            vals.append(val)

        return torch.tensor(vals, device=device)
            
    def get_coefficients(self, lambda_s, lambda_t, lambdas, beta_noise, beta_data, tau):
        lambda_s = lambda_s.to(torch.float64)
        lambda_t = lambda_t.to(torch.float64)
        lambdas = lambdas.to(torch.float64)
        p = len(lambdas)

        '''Noise Coefficients'''
        # (p+1,)
        integral1 = self.get_integral_vector(lambda_s, lambda_t, lambdas, beta_noise, tau-1.0)
        integral2 = self.get_integral_vector(lambda_s, lambda_t, lambdas[:1], 0, tau-1.0)
        integral_aug = torch.cat([integral1, integral2], dim=0)
        # (p+1, p+1)
        kernel = self.get_kernel_matrix(lambdas, beta_noise)
        eye = torch.eye(p+1, device=kernel.device).to(torch.float64)
        kernel_aug = 1 - eye
        kernel_aug[:p, :p] = kernel
        # (p,)
        noise_coeffs = torch.linalg.solve(kernel_aug, integral_aug)
        noise_coeffs = noise_coeffs[:p]

        '''Data Coefficients'''
        # (p+1,)
        integral1 = self.get_integral_vector(lambda_s, lambda_t, lambdas, beta_data, tau)
        integral2 = self.get_integral_vector(lambda_s, lambda_t, lambdas[:1], 0, tau)
        integral_aug = torch.cat([integral1, integral2], dim=0)
        # (p+1, p+1)
        kernel = self.get_kernel_matrix(lambdas, beta_data)
        eye = torch.eye(p+1, device=kernel.device).to(torch.float64)
        kernel_aug = 1 - eye
        kernel_aug[:p, :p] = kernel
        # (p,)
        data_coeffs = torch.linalg.solve(kernel_aug, integral_aug)
        data_coeffs = data_coeffs[:p]

        return noise_coeffs.float(), data_coeffs.float()

    def get_lag_kernel_matrix(self, lambdas):
        return torch.vander(lambdas, N=len(lambdas), increasing=True)

    def ell(self, a, k, lam0, lam1):
        from math import exp, factorial
        
        if a == 0:
            return (lam1**(k+1) - lam0**(k+1)) / (k+1)
        
        F = lambda x: exp(a*x)*sum(
            ((-1)**m * factorial(k) / (factorial(k-m)*a**(m+1)))*x**(k-m)
            for m in range(k+1)
        )
        
        return F(lam1) - F(lam0)

    def get_lag_integral_vector(self, a, lambda_s, lambda_t, lambdas):
        vector = [self.ell(a, k, lambda_s, lambda_t) for k in range(len(lambdas))]
        return torch.tensor(vector, device=lambdas.device)

    def get_lag_coefficients(self, lambda_s, lambda_t, lambdas, tau):
        lambda_s = lambda_s.to(torch.float64)
        lambda_t = lambda_t.to(torch.float64)
        lambdas = lambdas.to(torch.float64)
        
        # (p,)
        noise_integral = self.get_lag_integral_vector(tau-1, lambda_s, lambda_t, lambdas)
        data_integral = self.get_lag_integral_vector(tau, lambda_s, lambda_t, lambdas)
        
        # (p, p)
        kernel = self.get_lag_kernel_matrix(lambdas)
        kernel_inv = torch.linalg.inv(kernel)
        
        # (p,)
        noise_coefficients = kernel_inv.T @ noise_integral
        data_coefficients = kernel_inv.T @ data_integral
        return noise_coefficients.float(), data_coefficients.float()
    
    def get_next_sample(self, sample, i, hist, signal_rates, noise_rates, lambdas, tau,
                        p, log_s_noise, log_s_data, corrector=False):
        
        h = lambdas[i+1] - lambdas[i]
        beta_noise = float(1 / (np.exp(log_s_noise)*h))
        beta_data = float(1 / (np.exp(log_s_data)*h))

        # for predictor, (λ_i, λ_i-1, ..., λ_i-p+1), shape : (p,),
        # for corrector, (λ_i+1, λ_i, ..., λ_i-p+1), shape : (p+1,)
        lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])

        # for predictor, (c_i, c_i-1, ..., c_i-p+1), shape : (p,),
        # for corrector, (c_i+1, c_i, ..., c_i-p+1), shape : (p+1,)
        noise_coeffs, data_coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array, beta_noise, beta_data, tau)
        noise_lag_coeffs, data_lag_coeffs = self.get_lag_coefficients(lambdas[i], lambdas[i+1], lambda_array, tau)
        if log_s_noise >= self.log_scale_max:
            noise_coeffs = noise_lag_coeffs
        if log_s_data >= self.log_scale_max:
            data_coeffs = data_lag_coeffs
        
        # for predictor, (ε_i, ε_i-1, ..., ε_i-p+1), shape : (p,),
        # for corrector, (ε_i+1, λ_i, ..., ε_i-p+1), shape : (p+1,)
        hist = hist[i-p+1:i+(2 if corrector else 1)][::-1]
        hist_sum = sum([(tau-1)*nc*h[0] + tau*dc*h[1] for nc, dc, h in zip(noise_coeffs, data_coeffs, hist)])
        
        sample_coeff = (signal_rates[i+1]/signal_rates[i])**(1-tau) * (noise_rates[i+1]/noise_rates[i])**tau
        hist_coeff = signal_rates[i+1]**(1-tau) * noise_rates[i+1]**tau
        next_sample = sample_coeff*sample + hist_coeff*hist_sum
        return next_sample
    
    def get_loss_by_target_matching(self, target, i, hist, lambdas, tau, p, log_s, loss_type='noise', corrector=False):
        if loss_type == 'noise':
            beta_noise = float(1 / (np.exp(log_s) * abs(lambdas[i+1] - lambdas[i])))
            lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])
            noise_coeffs, _ = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array, beta_noise, 1, tau)
            hist = hist[i-p+1:i+(2 if corrector else 1)][::-1]
            noise_sum = sum([nc*h[0] for nc, h in zip(noise_coeffs, hist)])
            integral_noise = self.get_integral_vector(lambdas[i], lambdas[i+1], lambdas[:1], 0, tau-1.0)
            noise_loss = F.mse_loss(noise_sum/integral_noise, target)
            return noise_loss
        else:
            beta_data = float(1 / (np.exp(log_s) * abs(lambdas[i+1] - lambdas[i])))
            lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])
            _, data_coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array, 1, beta_data, tau)
            hist = hist[i-p+1:i+(2 if corrector else 1)][::-1]
            data_sum  = sum([dc*h[1] for dc, h in zip(data_coeffs, hist)])
            integral_data = self.get_integral_vector(lambdas[i], lambdas[i+1], lambdas[:1], 0, tau)
            data_loss = F.mse_loss(data_sum/integral_data, target)
            return data_loss

    def sample_by_target_matching(self, x, target,
                                  steps, t_start, t_end, order=3, skip_type='logSNR',
                                  method='data_prediction', lower_order_final=True):
        print('def sample_by_target_matching start!!!')
        noise_target = x
        data_target = target
        print(f"x.shape: {x.shape}, target.shape: {target.shape}, steps: {steps}, order: {order}, skip_type: {skip_type}, lower_order_final: {lower_order_final}")

        # 샘플링할 시간 범위 설정 (t_0, t_T)
        # diffusion 모델의 경우 t=1(혹은 T)에서 x는 가우시안 노이즈 상태라고 가정.
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start

        # 텐서가 올라갈 디바이스 설정
        device = x.device
        
        # 샘플링 과정에서 gradient 계산은 하지 않으므로 no_grad()
        with torch.no_grad():

            # 실제로 사용할 time step array를 구한다.
            # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            lambdas = torch.tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps], device=device)
            signal_rates = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
            noise_rates = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)

            log_scales = np.linspace(self.log_scale_min, self.log_scale_max, self.log_scale_num)
            optimal_log_scales_p_noise = []
            optimal_log_scales_p_data = []
            optimal_log_scales_c_noise = []
            optimal_log_scales_c_data = []
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            
            pred_losses_noise_list = []
            pred_losses_data_list = []
            corr_losses_noise_list = []
            corr_losses_data_list = []
            taus = np.linspace(0, 1, steps)
            for i in range(0, steps):
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)
                    
                # ===predictor===
                pred_losses_noise = []
                pred_losses_data  = []
                for log_scale in log_scales:
                    noise_loss = self.get_loss_by_target_matching(noise_target, i, hist, lambdas, taus[i], p, log_scale, loss_type='noise', corrector=False)
                    data_loss  = self.get_loss_by_target_matching(data_target,  i, hist, lambdas, taus[i], p, log_scale, loss_type='data',  corrector=False)
                    pred_losses_noise.append(noise_loss.detach().item())
                    pred_losses_data.append(data_loss.detach().item())
                pred_losses_noise_list.append(pred_losses_noise)
                pred_losses_data_list.append(pred_losses_data)

                optimal_log_scale_noise = log_scales[np.stack(pred_losses_noise).argmin()]
                optimal_log_scales_p_noise.append(optimal_log_scale_noise)
                optimal_log_scale_data = log_scales[np.stack(pred_losses_data).argmin()]
                optimal_log_scales_p_data.append(optimal_log_scale_data)
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas, taus[i], p, optimal_log_scale_noise, optimal_log_scale_data, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                corr_losses_noise = []
                corr_losses_data  = []
                for log_scale in log_scales:
                    noise_loss = self.get_loss_by_target_matching(noise_target, i, hist, lambdas, taus[i], p, log_scale, loss_type='noise', corrector=True)
                    data_loss  = self.get_loss_by_target_matching(data_target,  i, hist, lambdas, taus[i], p, log_scale, loss_type='data',  corrector=True)
                    corr_losses_noise.append(noise_loss.detach().item())
                    corr_losses_data.append(data_loss.detach().item())
                corr_losses_noise_list.append(corr_losses_noise)
                corr_losses_data_list.append(corr_losses_data)

                optimal_log_scale_noise = log_scales[np.stack(corr_losses_noise).argmin()]
                optimal_log_scales_c_noise.append(optimal_log_scale_noise)
                optimal_log_scale_data = log_scales[np.stack(corr_losses_data).argmin()]
                optimal_log_scales_c_data.append(optimal_log_scale_data)
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas, taus[i], p, optimal_log_scale_noise, optimal_log_scale_data, corrector=True)
                x = x_corr

        optimal_log_scales_p_noise = np.array(optimal_log_scales_p_noise)
        optimal_log_scales_p_data = np.array(optimal_log_scales_p_data)
        optimal_log_scales_c_noise = np.array(optimal_log_scales_c_noise + [0.0])
        optimal_log_scales_c_data = np.array(optimal_log_scales_c_data + [0.0])
        optimal_log_scales = np.stack([optimal_log_scales_p_noise,
                                       optimal_log_scales_p_data,
                                       optimal_log_scales_c_noise,
                                       optimal_log_scales_c_data], axis=0)
        
        if self.scale_dir is not None:
            save_file = os.path.join(self.scale_dir, f'NFE={steps},p={order}.npz')
            np.savez(save_file,
                     optimal_log_scales=optimal_log_scales,
                     pred_losses_noise_list=pred_losses_noise_list,
                     pred_losses_data_list=pred_losses_data_list,
                     corr_losses_noise_list=corr_losses_noise_list,
                     corr_losses_data_list=corr_losses_data_list)
            print(save_file, ' saved!')

        # 최종적으로 x를 반환
        return x

    def load_optimal_log_scales(self, steps, order):
        try:
            load_file = os.path.join(self.scale_dir, f'NFE={steps},p={order}.npz')
            log_scales = np.load(load_file)['optimal_log_scales']
        except:
            return None
        print(load_file, 'loaded!')
        return log_scales
    
    def sample(self, x, steps,
               t_start=None, t_end=None,
               order=3, skip_type='logSNR', method='data_prediction',
               lower_order_final=True,
               log_scale_p=2.0,
               log_scale_c=0.0,
              ):
        # log_scale : predictor, corrector 모든 step에 적용할 log_scale, log_scales가 load안되면 log_scale로 작동
        # log_scales : predictor, corrector, step별로 적용할 log_scale array, shape : (2, NFE)

        log_scales = self.load_optimal_log_scales(steps, order)

        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        
        # 텐서가 올라갈 디바이스 설정
        device = x.device

        # 샘플링 과정에서 gradient 계산은 하지 않으므로 no_grad()
        with torch.no_grad():

            # 실제로 사용할 time step array를 구한다.
            # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            lambdas = torch.tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps], device=device)
            signal_rates = torch.tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps], device=device)
            noise_rates = torch.tensor([self.noise_schedule.marginal_std(t) for t in timesteps], device=device)
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            taus = np.linspace(0, 1, steps)
            for i in range(0, steps):
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)

                # ===predictor===
                log_s_noise = log_scale_p if log_scales is None else log_scales[0, i]
                log_s_data = log_scale_p if log_scales is None else log_scales[1, i]
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas, taus[i],
                                              p=p, log_s_noise=log_s_noise, log_s_data=log_s_data, corrector=False)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                log_s_noise = log_scale_c if log_scales is None else log_scales[2, i]
                log_s_data = log_scale_c if log_scales is None else log_scales[3, i]
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas, taus[i],
                                              p=p, log_s_noise=log_s_noise, log_s_data=log_s_data, corrector=True)
                x = x_corr
        # 최종적으로 x를 반환
        return x