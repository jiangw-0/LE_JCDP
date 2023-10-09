import pdb
from math import exp
import torch
from utils import *
import tqdm
import pytorch_ssim
import matplotlib.pyplot as plt
import lpips


def diff_purify(x, diffusion, max_iter, mode, config, cond_fc, model_kwargs, args):
    # From noisy initialized image to purified image
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    x_adv = transform_raw_to_diff(x).to(config.device.diff_device)

    if config.purification.guide_mode == 'LPIPS':
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn = loss_fn.to(config.device.diff_device) 

    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)
            x_advt = diffusion.diffuse_t_steps(x_adv, t)

            x_adv_t = x_advt
            # scale = exp(config.purification.guide_exp_a * t / config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base
            if config.purification.guide_mode == 'MSE': 
                selected = -1 * F.mse_loss(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'LPIPS':
                dist = loss_fn(x_in, x_adv_t)
                selected = -1 * dist.mean()
                #print('LPIPS:', selected)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'SSIM':
                selected = pytorch_ssim.ssim(x_in, x_adv_t)
                scale = diffusion.compute_scale(x_in,t, config.attack.ptb*2/255. / 3. / config.purification.guide_scale)
            elif config.purification.guide_mode == 'CONSTANT': 
                scale = config.purification.guide_scale
            gradient = torch.autograd.grad(selected.sum(), x_in)[0]
            #print(f"{config.purification.guide_mode}_gradient: {gradient[0][0][0]}")
            print(f"{config.purification.guide_mode}_scale: {scale}")
            condfn = gradient * scale
            
            return condfn

    

    with torch.no_grad():
        images = []
        xt_reverse = x_adv
        for i in range(max_iter): 
            xt = diffusion.diffuse_t_steps(xt_reverse, config.purification.purify_step)
            xt_reverse = diffusion.denoise(
                xt.shape[0], 
                n_steps=config.purification.purify_step, 
                x=xt.to(config.device.diff_device),
                curr_step=config.purification.purify_step, 
                # progress_bar=tqdm.tqdm,
                cond_fn = cond_fn if config.purification.cond == True else None,  
                cond_fc = cond_fc if config.purification.condcc == True or config.purification.condc == True else None, 
                is_condcc = True if config.purification.condcc == True else False,
                model_kwargs = model_kwargs
            )

            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            images.append(x_pur)

    return images
