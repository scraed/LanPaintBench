from diffusers import DDPMPipeline, ImagePipelineOutput, DDIMPipeline
from diffusers.utils import make_image_grid
import os
from diffusers.utils.torch_utils import randn_tensor
import copy
import torch
from torch.func import vmap, grad, grad_and_value
from typing import Union, Optional, Any, List, Tuple
import numpy as np
from utils_math import *
from torch import autograd
from torch.utils.checkpoint import checkpoint as grad_ckpt
from tqdm import tqdm
import math
from functools import partial

# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构

same_seeds(0)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = arr.to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
def noise_like(shape, device, repeat=False):
    def repeat_noise():
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )

    def noise():
        return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


class GeneralPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)
        # if self.unet.scheduler is not None:
        self.unet.scheduler = self.scheduler
    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:

        self.unet.to(self.device)
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
    
        image = self.sample_loop(batch_size, generator)
        return self.to_output(image, output_type, return_dict)
        
    def to_output(self, image, output_type="pil", return_dict=True):
        image = image.cpu().numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
    def init_image(self, image, batch_size, generator):
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator, dtype=self.unet.dtype)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.unet.dtype)
        image = image * self.scheduler.init_noise_sigma
        image = image.to(self.device)
        return image
    def sample_loop(self, batch_size, generator):
        image = self.init_image(None, batch_size, generator)
        for t in self.progress_bar(self.scheduler.timesteps):
            # 1. predict noise model_output
            
            image_rescale = self.scheduler.scale_model_input(image, t )
            #scale_factor = torch.mean( image * image_rescale ) / torch.mean( image_rescale ** 2 )

            model_output = self.unet.cal_eps(image_rescale, t)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image).prev_sample
        return image

class ConditionPipeline(GeneralPipeline):
    def __init__(self, unet, scheduler):
        super().__init__(unet, scheduler)
    @torch.no_grad()
    def __call__(
        self,
        y,
        mask,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        batch_size = y.shape[0]
        # set step values
        self.unet.to(self.device)
        
        self.scheduler.set_timesteps(num_inference_steps)
    

        image = self.sample_loop(y, mask, generator)
        return self.to_output(image, output_type, return_dict)
    def sample_loop(self, y, mask, generator):
        image = self.init_image(None, len(y), generator).to(self.device)
        # print('euler scheduler',self.scheduler.timesteps)
        abt = 1/(1+self.scheduler.sigmas**2)
        # print('step size',0.02 * (1 - abt) ** 0.5)
        # print('-log sigma',-torch.log(abt))
        for t, sigma in self.progress_bar( zip( self.scheduler.timesteps, self.scheduler.sigmas[:-1] ) ):
            # 1. predict noise model_output
            abt = 1/( 1+sigma**2 )

            y_t = y * abt ** 0.5 + torch.randn_like(y) * (1 - abt) ** 0.5 
            
            x_t = self.scheduler.scale_model_input(image, t ).to(y.device)
            scale_factor = torch.mean( image * x_t ) / torch.mean( x_t ** 2 )

            x_t = x_t * (1 - mask) + y_t * mask

            B, C = x_t.shape[:2]
            model_output = self.unet.cal_eps(x_t, t)

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, x_t * scale_factor).prev_sample
        return image*(1-mask) + y*mask



class LangevinPipeline(ConditionPipeline):
    def __init__(self, unet, scheduler, step_size = 0.1, n_steps = 10, m = 1, alpha = 0.):
        super().__init__(unet, scheduler)
        self.step_size = step_size
        self.n_steps = n_steps
        self.alpha = alpha
        self.m = m
    def forward_diffuse(self, y):
        # Forward diffusion of motif to get y at each step:
        # For simplicity, we use the scheduler's sigmas and same formula as in ConditionPipeline:
        # abt = 1/(1+sigma^2), so forward diffuse: y_t = sqrt(abt)*y + sqrt(1 - abt)*noise
        y_forward_diffusion = []
        device = y.device
        abts = ( 1/(1+self.scheduler.sigmas**2) ).flip(0)
        y_t = y
        for abt, abt_next in zip(abts[:-1], abts[1:]):
            noise = torch.randn_like(y)
            y_t = y_t * (abt_next/abt)**0.5 + noise * ((1 - (abt_next/abt))**0.5)
            y_forward_diffusion.append(y_t.to(device))
        y_forward_diffusion.reverse()
        return y_forward_diffusion

    def sample_loop(self, y, mask, generator):
        image = self.init_image(None, len(y), generator).to(self.device)
        y_forward_diffusion = self.forward_diffuse(y)
        unet = self.unet
        self.latent_image = y
        def compress_tensor(mask):
            if torch.all( mask - mask[0:1] == 0 ):
                mask = mask[0:1] # if mask is the same for all images, use the first one
            if len(mask.shape) > 1:
                if torch.all( mask - mask[:,0:1] == 0 ):
                    mask = mask[:,0:1] # if mask is the same for all images, use the first one
            return mask
        mask = compress_tensor(mask)
        for t, sigma, y_t in self.progress_bar( zip( self.scheduler.timesteps, self.scheduler.sigmas[:-1], y_forward_diffusion ) ):
            # 1. predict noise model_output
            abt = 1/( 1+sigma**2 )
            x_t = self.scheduler.scale_model_input(image, t ).to(y.device)
            scale_factor = torch.mean( image * x_t ) / torch.mean( x_t ** 2 )

            x_t = x_t * (1 - mask) + y_t * mask  ####去掉replace
            step_size = self.step_size * (1 - abt) ** 0
            #print("stepsize", step_size)

            current_times = (sigma, abt)
            
            #print("sigma:",sigma, "sigma_mid:",sigma_mid)


            args = None
            for i in range(self.n_steps):
                score_func = partial( self.score_model, y = y, mask = mask, abt = abt, t = t )
                # detect 
                x_t, args = self.langevin_dynamics(x_t, score_func , mask, self.step_size, current_times, sigma_x = self.sigma_x(abt), sigma_y = self.sigma_y(abt), args = args)  
            B,C = x_t.shape[:2]
            model_output = self.unet.cal_eps(x_t, t)
            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, x_t * scale_factor).prev_sample

        return image*(1-mask) + y*mask
    def mid_times(self, current_times, step_size):
        sigma, abt = current_times
        tt = torch.log(1+sigma**2)
        tt_mid = torch.max( tt - step_size, tt*0 )
        sigma_mid = (torch.exp(tt_mid) - 1) ** 0.5
        abt_mid = 1/(1+sigma_mid**2)
        return sigma_mid, abt_mid

    def score_model(self, x_t, y, mask, abt, t):
        # the score function for the Langevin dynamics
        B,C = x_t.shape[:2]
        e_t = self.unet.cal_eps(x_t, t)
        score_x = -e_t/(1 - abt)**0.5 
        score_y = - ( x_t - abt**0.5 * y ) /(1 - abt)
        return score_x * (1 - mask) + score_y * mask
    def sigma_y(self, abt):
        return 0*(1-abt)**self.m # y is computed by forward, no need to activate y iteration
    def sigma_x(self, abt):
        return (1 - abt + abt * self.alpha)**self.m
    def langevin_dynamics(self, x_t, score, mask, step_size, current_times, sigma_x = 1, sigma_y = 0, args = None):
        
        sigma, abt = current_times


        score_model = score(x_t)
        eps_model = -score_model * (1 - abt)**0.5 # eps = -sqrt(1-abt)*score 

        x = x_t + sigma_x * step_size * score_model + torch.randn_like(x_t) * (2*step_size*sigma_x)**0.5 
        x = x * (1 - mask)
        y = x_t + sigma_y * step_size * score_model + torch.randn_like(x_t) * (2*step_size*sigma_y)**0.5
        y = y * mask
        x_t = x + y
        return x_t, None



class LangevinCharaPipeline(LangevinPipeline):
    def __init__(self, unet, scheduler, step_size = 0.1, n_steps = 10, m = 1, alpha = 0., chara_lamb = 10, chara_beta = 1):
        super().__init__(unet, scheduler, step_size, n_steps, m, alpha)
        self.chara_lamb = chara_lamb
        self.chara_beta = chara_beta
    def score_model(self, x_t, y, mask, abt, t):
        # the score function for the Langevin dynamics
        lamb = self.chara_lamb
        beta = self.chara_beta * (1-abt)**0.5
        B,C = x_t.shape[:2]
        e_t = self.unet.cal_eps(x_t, t)
        score_x = -e_t/(1 - abt)**0.5 
        score_y = (- (1 + lamb) * ( x_t - abt**0.5 * y ) /(1 - abt)**0.5 + lamb *  e_t) /(1 - abt)**0.5
        return score_x * (1 - mask) + score_y * mask
    def sigma_y(self, abt):
        return (1-abt)**self.m
    def sigma_x(self, abt):
        return (1 - abt + abt * self.alpha)**self.m



class ULDCharaPipeline_Sampler(LangevinCharaPipeline):
    def __init__(self, unet, scheduler, step_size = 0.1, n_steps = 10, friction = 2, m = 1, alpha = 0, chara_lamb = 10, chara_beta = 2):
        super().__init__(unet, scheduler, step_size, n_steps, m, alpha, chara_lamb, chara_beta=1.)
        self.friction = friction**2 #step_size ~ 0.15 
        # self.chara_lamb=0
    def prepare_step_size(self, current_times, step_size, sigma_x, sigma_y):
        # -------------------------------------------------------------------------
        # Unpack current times parameters (sigma and abt)
        sigma, abt = current_times

        dtx = self.step_size * sigma_x 
        dty = self.step_size * sigma_y        

        # -------------------------------------------------------------------------
        # Define friction parameter Gamma_hat for each branch.
        # Using dtx**0 provides a tensor of the proper device/dtype.

        A_x_T = 1
        A_y_T = 1 + self.chara_lamb
        A_x = A_x_T / ( 1 - abt + abt * self.alpha ) 
        A_y = A_y_T / ( 1 - abt )


        Gamma_x = self.friction*A_x #* (1 - A_x_T + A_x)
        Gamma_y = self.friction*A_y #* (1 - A_y_T + A_y)

        D_x = (2 * (1 + sigma**2) )**0.5
        D_y = (2 * (1 + sigma**2) )**0.5

        return sigma, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y
    def x0_evalutation(self, x_t, score, sigma, abt, args):
        score_model = score(x_t / (1 + sigma**2)**0.5  )
        eps_model = -score_model * (1 - abt) ** 0.5

        x0 = x_t - sigma * eps_model
        return x0
    def langevin_dynamics(self, x_t, score, mask, step_size, current_times, sigma_x=1, sigma_y=0, args=None):
        
        sigma, abt = current_times[0].to(x_t.device), current_times[1].to(x_t.device)
        x_t = x_t * (1 + sigma**2)**0.5
        # prepare the step size and time parameters
        with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
            step_sizes = self.prepare_step_size((sigma, abt), step_size, sigma_x, sigma_y)
            sigma, dtx, dty, Gamma_x, Gamma_y, A_x, A_y, D_x, D_y = step_sizes
        if torch.mean(dtx) <= 0.:
            return x_t, args

        A = A_x * (1-mask) + A_y * mask
        D = D_x * (1-mask) + D_y * mask
        dt = dtx * (1-mask) + dty * mask
        Gamma = Gamma_x * (1-mask) + Gamma_y * mask
        def Coef_C(x_t):
            x0 = self.x0_evalutation(x_t, score, sigma, abt, args)
            C = ( x0 - x_t ) /  (1-abt) + A * x_t
            return C
        def advance_time(x_t, v, dt, Gamma, A, C, D):
            dtype = x_t.dtype
            with torch.autocast(device_type=x_t.device.type, dtype=torch.float32):
                osc = StochasticHarmonicOscillator(Gamma, A, C, D )
                x_t, v = osc.dynamics(x_t, v, dt )
            x_t = x_t.to(dtype)
            v = v.to(dtype)
            return x_t, v
        if args is None:
            v = None
            C = Coef_C(x_t)
            x_t, v = advance_time(x_t, v, dt, Gamma, A, C, D)
        else:
            v, C = args

            x_t, v = advance_time(x_t, v, dt/2, Gamma, A, C, D)

            C_new = Coef_C(x_t)
            v = v + Gamma**0.5 * ( C_new - C) *dt

            x_t, v = advance_time(x_t, v, dt/2, Gamma, A, C, D)

            C = C_new

        x_t = x_t / (1 + sigma**2)**0.5    
        return x_t, (v, C)

class ULDPipeline_Sampler(ULDCharaPipeline_Sampler):
    def __init__(self, unet, scheduler, step_size = 0.1, n_steps = 10, friction = 2, m = 1, alpha = 0.25, chara_beta=1.):
        super().__init__(unet, scheduler, step_size = step_size, n_steps = n_steps, friction = friction, m = m, alpha = alpha, chara_lamb = 0., chara_beta = chara_beta)

