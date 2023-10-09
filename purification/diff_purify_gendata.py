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
    images_list = []
    transform_raw_to_diff = raw_to_diff(config.structure.dataset)
    transform_diff_to_raw = diff_to_raw(config.structure.dataset)
    #x_adv = transform_raw_to_diff(x).to(config.device.diff_device)


    label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
              5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

    if config.purification.guide_mode == 'LPIPS':
        loss_fn = lpips.LPIPS(net='alex')
        loss_fn = loss_fn.to(config.device.diff_device) 
    

    #无条件时##
    # if args.isx36:
    #     with torch.no_grad():
    #         images = []
    #         xt_reverse = x_adv
    #         x36 = diffusion.diffuse_t_steps(xt_reverse, 36)
    #         x36_reverse = diffusion.denoise(
    #             x36.shape[0], 
    #             n_steps=36, 
    #             x=x36.to(config.device.diff_device), 
    #             curr_step=36, 
    #             # progress_bar=tqdm.tqdm,
    #             cond_fn = None
    #         )


    def cond_fn(x_reverse_t, t):
        """
        Calculate the grad of guided condition.
        """
        with torch.enable_grad():
            x_in = x_reverse_t.detach().requires_grad_(True)
            x_advt = diffusion.diffuse_t_steps(x_adv, t)

            #dist36 = loss_fn(x_in, x36_reverse)
            #torch.Size([100, 1, 1, 1])
            #tensor(0.0303, device='cuda:0', grad_fn=<MeanBackward0>)
            #distadv = loss_fn(x_in, x_advt)
            #torch.Size([100, 1, 1, 1])
            #tensor(0.0135, device='cuda:0', grad_fn=<MeanBackward0>)

            #use x_adv or x_reverse_36
            if args.isx36:
                x_adv_t = x36_reverse
            else:
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
            #print(f"{config.purification.guide_mode}_-s*g: {condfn[0][0][0]}")
            return condfn

    

    with torch.no_grad():
        images = []
        #xt_reverse = x_adv
        for i in range(max_iter): #4
            # # method 1: save every step pic
            # images = []
            # xt = diffusion.diffuse_t_steps(x, config.purification.purify_step)
            # for j in range(config.purification.purify_step):
            #     xt = diffusion.denoise(xt.shape[0], n_steps=1, x=xt.to(config.device.diff_device), curr_step=(config.purification.purify_step-j), progress_bar=tqdm.tqdm)
            #     x_pur_t = xt.clone().detach()
            #     x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)
            #     images.append(x_pur)
            # images_list.append(images)
            
            # method 2: save final step pic
            
            #xt = diffusion.diffuse_t_steps(xt_reverse, config.purification.purify_step)
            #config.purification.purify_step = 36
             
            #from noise 
            xT = None

            xt_reverse = diffusion.denoise(
                config.structure.bsize,#xt.shape[0], batch_size
                n_steps=config.purification.purify_step, 
                #x=xt.to(config.device.diff_device),
                x=xT,
                curr_step=config.purification.purify_step, 
                # progress_bar=tqdm.tqdm,
                cond_fn = cond_fn if config.purification.cond == True else None,  #距离
                cond_fc = cond_fc if config.purification.condcc == True or config.purification.condc == True else None,  #距离2
                # cond_fc = cond_fc if config.purification.condc == True else None, #标签
                is_condcc = True if config.purification.condcc == True else False,
                model_kwargs = model_kwargs
            )

            x_pur_t = xt_reverse.clone().detach()
            x_pur = torch.clamp(transform_diff_to_raw(x_pur_t), 0.0, 1.0)

            # for m in range(20):
            #     x_uint = x_pur[m].mul(255).byte()
            #     x_uint = x_uint.cpu().numpy().transpose(1, 2, 0)#.astype('uint8')

            #     save_path = os.path.join(args.log,  'reverse'+ str(config.purification.purify_step))

            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path,exist_ok=True)
                
            #     #cifar10
            #     save_name = os.path.join(save_path, str(m)+'_'+str(model_kwargs["y"][m].cpu().numpy())+'_'+str(label_dict[model_kwargs["y"][m].cpu().numpy().tolist()])+'_'+str(i)+'.jpg')
                
            #     #cifar100
            #     # save_name = os.path.join(save_path, str(m)+'_'+str(model_kwargs["y"][m].cpu().numpy())+'_'+str(i)+'.jpg')
                
            #     plt.imsave(save_name, x_uint)

            images.append(x_pur)

    return images
