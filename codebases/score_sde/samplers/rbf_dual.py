import os
import torch
import torch.nn.functional as F
import numpy as np
from .utils import expand_dims
import math

# dual matching
class RBFDual:
    def __init__(
            self,
            model_fn,
            noise_schedule,
            algorithm_type="data_prediction",
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
        assert algorithm_type in ["data_prediction", "noise_prediction"]

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.predict_x0 = algorithm_type == "data_prediction"
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

    def model_fn(self, x, t, dual=False):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """

        if dual:
            noise = self.noise_prediction_fn(x, t)
            alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
            x0 = (x - sigma_t * noise) / alpha_t
            if self.correcting_x0_fn is not None:
                x0 = self.correcting_x0_fn(x0)
            return noise, x0

        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

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

    def get_integral_vector(self, lambda_s, lambda_t, lambdas, beta):
        
        # Handle the zero-beta case.
        if beta == 0:
            if self.predict_x0:
                return (torch.exp(lambda_t) - torch.exp(lambda_s)) * torch.ones_like(lambdas)
            else:    
                return (torch.exp(-lambda_s) - torch.exp(-lambda_t)) * torch.ones_like(lambdas)

        h = lambda_t - lambda_s
        s = 1/(beta*h)
        
        # closed-form
        def log_erf_diff(a, b):
            return torch.log(torch.erfc(b)) + torch.log(1.0-torch.exp(torch.log(torch.erfc(a)) - torch.log(torch.erfc(b))))
        
        if self.predict_x0:
            r_u = (lambdas - lambda_s) / h
            log_prefactor = lambda_t + torch.log(h) + ((s*h)**2/4 + h*(r_u-1)) + torch.log(0.5*np.sqrt(np.pi)*s)
            upper = (r_u + s**2*h/2)/s
            lower = (r_u + s**2*h/2 - 1)/s
            result = torch.exp(log_prefactor + log_erf_diff(upper, lower))
            return result.float()
        else:
            r_u = (lambdas - lambda_s) / h
            log_prefactor = -lambda_t + torch.log(h) + ((s*h)**2/4 + h*(1-r_u)) + torch.log(0.5*np.sqrt(np.pi)*s)
            upper = (1 - r_u + s**2*h/2)/s
            lower = (-r_u + s**2*h/2)/s
            result = torch.exp(log_prefactor + log_erf_diff(upper, lower))
            return result.float()

    def get_coefficients(self, lambda_s, lambda_t, lambdas, beta):
        lambda_s = lambda_s.to(torch.float64)
        lambda_t = lambda_t.to(torch.float64)
        lambdas = lambdas.to(torch.float64)
        beta = beta.to(torch.float64)

        p = len(lambdas)
        # (p,)
        integral1 = self.get_integral_vector(lambda_s, lambda_t, lambdas, beta)
        #print('integral1 :', lambda_s, beta, integral1)
        # (1,)
        integral2 = self.get_integral_vector(lambda_s, lambda_t, lambdas[:1], 0)
        
        # (p+1,)
        integral_aug = torch.cat([integral1, integral2], dim=0)

        # (p, p)
        kernel = self.get_kernel_matrix(lambdas, beta)
        eye = torch.eye(p+1, device=kernel.device).to(torch.float64)
        kernel_aug = 1 - eye
        kernel_aug[:p, :p] = kernel
        # (p,)
        coefficients = torch.linalg.solve(kernel_aug, integral_aug)
        coefficients = coefficients[:p]  # (p+1,) 중 앞 p개만 슬라이싱
        return coefficients.float()

    def get_lag_kernel_matrix(self, lambdas):
        return torch.vander(lambdas, N=len(lambdas), increasing=True)

    def get_lag_integral(self, a: float, b: float, k: int) -> float:
        if k < 0 or not float(k).is_integer():
            raise ValueError("k must be a non-negative integer.")

        k = int(k)  # 확실하게 int 변환
        k_factorial = math.factorial(k)

        def F(x: float) -> float:
            # F(λ) = -k! * exp(-λ) * Σ_{m=0}^k [λ^m / m!]
            poly_sum = 0.0
            for m in range(k+1):
                poly_sum += (x**m) / math.factorial(m)

            return -k_factorial * math.exp(-x) * poly_sum
        
        def G(x: float) -> float:
            # G(λ) = (-1)^k * k! * exp(λ) * Σ_{m=0}^k [(-λ)^m / m!]
            poly_sum = 0.0
            for m in range(k+1):
                poly_sum += ((-x)**m) / math.factorial(m)

            return (-1)**k * k_factorial * math.exp(x) * poly_sum

        if self.predict_x0:
            return G(b) - G(a)
        else:
            return F(b) - F(a)

    def get_lag_integral_vector(self, lambda_s, lambda_t, lambdas):
        vector = [self.get_lag_integral(lambda_s, lambda_t, k) for k in range(len(lambdas))]
        return torch.tensor(vector, device=lambdas.device)

    def get_lag_coefficients(self, lambda_s, lambda_t, lambdas):
        # (p,)
        integral = self.get_lag_integral_vector(lambda_s, lambda_t, lambdas)
        # (p, p)
        kernel = self.get_lag_kernel_matrix(lambdas)
        kernel_inv = torch.linalg.inv(kernel)
        # (p,)
        coefficients = kernel_inv.T @ integral
        return coefficients
    
    def get_next_sample(self, sample, i, hist, signal_rates, noise_rates, lambdas, p, beta, corrector=False, lagrange=False):
        '''
        sample : (b, c, h, w), tensor
        i : current sampling step, scalar
        hist : [ε_0, ε_1, ...] or [x_0, x_1, ...], tensor list
        signal_rates : [α_0, α_1, ...], tensor list
        lambdas : [λ_0, λ_1, ...], scalar list
        beta : beta of RBF kernel
        corrector : True or False
        '''
        
        # for predictor, (λ_i, λ_i-1, ..., λ_i-p+1), shape : (p,),
        # for corrector, (λ_i+1, λ_i, ..., λ_i-p+1), shape : (p+1,)
        lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])

        # for predictor, (c_i, c_i-1, ..., c_i-p+1), shape : (p,),
        # for corrector, (c_i+1, c_i, ..., c_i-p+1), shape : (p+1,)
        if lagrange:
            #print('Lagrange!!!')
            coeffs = self.get_lag_coefficients(lambdas[i], lambdas[i+1], lambda_array)
        else:
            coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array, beta)

        # for predictor, (ε_i, ε_i-1, ..., ε_i-p+1), shape : (p,),
        # for corrector, (ε_i+1, λ_i, ..., ε_i-p+1), shape : (p+1,)
        datas = hist[i-p+1:i+(2 if corrector else 1)][::-1]
        
        data_sum = sum([coeff * data for coeff, data in zip(coeffs, datas)])
        if self.predict_x0:
            next_sample = noise_rates[i+1]/noise_rates[i]*sample + noise_rates[i+1]*data_sum
        else:
            next_sample = signal_rates[i+1]/signal_rates[i]*sample - signal_rates[i+1]*data_sum
        return next_sample

    def get_loss_by_target_matching(self, x, i, noise_target, data_target, hist, signal_rates, noise_rates, log_scale, lambdas, p, corrector=False):
        beta = 1 / (np.exp(log_scale) * abs(lambdas[i+1] - lambdas[i]))
        lambda_array = torch.flip(lambdas[i-p+1:i+(2 if corrector else 1)], dims=[0])
        coeffs = self.get_coefficients(lambdas[i], lambdas[i+1], lambda_array, beta)
        
        datas = hist[i-p+1:i+(2 if corrector else 1)][::-1]
        data_sum = sum([coeff * data for coeff, data in zip(coeffs, datas)])

        integral = (torch.exp(lambdas[i+1]) - torch.exp(lambdas[i]))
        pred = data_sum / integral
        data_loss = F.mse_loss(data_target, pred)

        a = (signal_rates[i+1]/signal_rates[i] - noise_rates[i+1]/noise_rates[i])*x
        integral = (torch.exp(-lambdas[i]) - torch.exp(-lambdas[i+1]))
        pred = (a - noise_rates[i+1]*data_sum) / (signal_rates[i+1]*integral)
        noise_loss = F.mse_loss(noise_target, pred)

        return data_loss, noise_loss

    def sample_by_target_matching(self, noise_target, data_target,
                                  steps, t_start, t_end, order=3, skip_type='logSNR',
                                  method='data_prediction', lower_order_final=True):
        print('def sample_by_target_matching start!!!')
        print(f"noise_target.shape: {noise_target.shape}, data_target.shape: {data_target.shape}, steps: {steps}, order: {order}, skip_type: {skip_type}, lower_order_final: {lower_order_final}")
        # 샘플링할 시간 범위 설정 (t_0, t_T)
        # diffusion 모델의 경우 t=1(혹은 T)에서 x는 가우시안 노이즈 상태라고 가정.
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start

        # 텐서가 올라갈 디바이스 설정
        x = noise_target
        device = x.device
        
        # 샘플링 과정에서 gradient 계산은 하지 않으므로 no_grad()
        with torch.no_grad():

            # 실제로 사용할 time step array를 구한다.
            # timesteps는 길이가 steps+1인 1-D 텐서: [t_T, ..., t_0]
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
            lambdas = torch.Tensor([self.noise_schedule.marginal_lambda(t) for t in timesteps])
            signal_rates = torch.Tensor([self.noise_schedule.marginal_alpha(t) for t in timesteps])
            noise_rates = torch.Tensor([self.noise_schedule.marginal_std(t) for t in timesteps])

            log_scales = np.linspace(self.log_scale_min, self.log_scale_max, self.log_scale_num)
            optimal_log_scales_p = []
            optimal_log_scales_c = []
            
            hist = [None for _ in range(steps)]
            hist[0] = self.model_fn(x, timesteps[0])   # model(x,t) 평가값을 저장
            
            pred_losses_list = []
            pred_data_losses_list = []
            pred_noise_losses_list = []

            corr_losses_list = []
            corr_data_losses_list = []
            corr_noise_losses_list = []

            r = 0.999
            for i in range(0, steps):
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)
                    
                # ===predictor===
                pred_losses = []
                pred_data_losses = []
                pred_noise_losses = []
                for log_scale in log_scales:
                    data_loss, noise_loss = self.get_loss_by_target_matching(x, i, noise_target, data_target, hist, signal_rates, noise_rates, log_scale, lambdas, p, corrector=False)
                    loss = data_loss * r + noise_loss * (1-r)
                    pred_losses.append(loss.detach().item())
                    pred_data_losses.append(data_loss.detach().item())
                    pred_noise_losses.append(noise_loss.detach().item())
                    
                pred_losses_list.append(np.stack(pred_losses))
                pred_data_losses_list.append(np.stack(pred_data_losses))
                pred_noise_losses_list.append(np.stack(pred_noise_losses))

                argmin = np.stack(pred_losses).argmin()
                optimal_log_scale = log_scales[argmin]
                optimal_log_scales_p.append(optimal_log_scale)
                beta = 1 / (np.exp(optimal_log_scale) * abs(lambdas[i+1] - lambdas[i]))
                lagrange = (optimal_log_scale >= self.log_scale_max)
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                            p=p, beta=beta, corrector=False, lagrange=lagrange)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                corr_losses = []
                corr_data_losses = []
                corr_noise_losses = []
                for log_scale in log_scales:
                    data_loss, noise_loss = self.get_loss_by_target_matching(x, i, noise_target, data_target, hist, signal_rates, noise_rates, log_scale, lambdas, p, corrector=True)
                    loss = data_loss * r + noise_loss * (1-r)
                    corr_losses.append(loss.detach().item())
                    corr_data_losses.append(data_loss.detach().item())
                    corr_noise_losses.append(noise_loss.detach().item())

                corr_losses_list.append(np.stack(corr_losses))
                corr_data_losses_list.append(np.stack(corr_data_losses))
                corr_noise_losses_list.append(np.stack(corr_noise_losses))

                argmin = np.stack(corr_losses).argmin()
                optimal_log_scale = log_scales[argmin]
                optimal_log_scales_c.append(optimal_log_scale)
                beta = 1 / (np.exp(optimal_log_scale) * abs(lambdas[i+1] - lambdas[i]))
                lagrange = (optimal_log_scale >= self.log_scale_max)
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                            p=p, beta=beta, corrector=True, lagrange=lagrange)
                x = x_corr
            
        optimal_log_scales_p = np.array(optimal_log_scales_p)
        optimal_log_scales_c = np.array(optimal_log_scales_c + [0.0])
        optimal_log_scales = np.stack([optimal_log_scales_p, optimal_log_scales_c], axis=0)

        if self.scale_dir is not None:
            save_file = os.path.join(self.scale_dir, f'NFE={steps},p={order}.npz')

            np.savez(save_file,
                     optimal_log_scales=optimal_log_scales,
                     pred_losses_list=pred_losses_list,
                     pred_data_losses_list=pred_data_losses_list,
                     pred_noise_losses_list=pred_noise_losses_list,
                     corr_losses_list=corr_losses_list,
                     corr_data_losses_list=corr_data_losses_list,
                     corr_noise_losses_list=corr_noise_losses_list,
                     )
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
        if log_scales is None:
            raise ValueError(f"Failed to load optimal log scales for steps={steps} and order={order}.")
        
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
            
            for i in range(0, steps):
                if lower_order_final:
                    p = min(i+1, steps - i, order)
                else:
                    p = min(i+1, order)

                # ===predictor===
                s = log_scale_p if log_scales is None else log_scales[0, i]
                lagrange = (s >= self.log_scale_max)
                beta = 1 / (np.exp(s) * abs(lambdas[i+1] - lambdas[i]))
                x_pred = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, beta=beta, corrector=False, lagrange=lagrange)
                
                if i == steps - 1:
                    x = x_pred
                    break
                
                # predictor로 구한 x_pred를 이용해서 model_fn 평가
                hist[i+1] = self.model_fn(x_pred, timesteps[i+1])
                
                # ===corrector===
                s = log_scale_c if log_scales is None else log_scales[1, i]
                lagrange = (s >= self.log_scale_max)
                beta = 1 / (np.exp(s) * abs(lambdas[i+1] - lambdas[i]))
                x_corr = self.get_next_sample(x, i, hist, signal_rates, noise_rates, lambdas,
                                              p=p, beta=beta, corrector=True, lagrange=lagrange)
                x = x_corr
        # 최종적으로 x를 반환
        return x