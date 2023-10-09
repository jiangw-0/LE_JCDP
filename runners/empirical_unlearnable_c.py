
import pdb
import torch
from torchvision.utils import save_image
import torchvision.utils as vutils
from distutils.command.config import config
from utils import *
from clf_models.networks import * 

from datetime import datetime
import tqdm
import pandas as pd
import torchvision
from purification.diff_purify import *
from guided_diffusion import dist_util
from pytorch_diffusion.diffusion import Diffusion
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import pylab
import os

__all__ = ['Empirical_cond']


class Empirical_cond():
    def __init__(self, args, parser, config):
        #self.args = parser if config.purification.condc else args
        self.args = parser if config.purification.cond or config.purification.condc else args
        self.config = config
        self.config.attack.if_attack = True
        

    def run(self, log_progress):
        # Normalize on classifiers 
        transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)

        # Output log file configuration
        if self.args.debug is not True:
            sys.stdout = log_progress

        # Import dataset
        start_time = datetime.now()
        print("[{}] Start importing dataset {}".format(str(datetime.now()), self.config.structure.dataset))
        
        
        testLoader = importData(dataset=self.config.structure.dataset, perturbtype = self.args.perturb_type, perturb_path= self.args.perturb_path,
                                train=True, shuffle=False, bsize=self.config.structure.bsize )

        print("[{}] Finished importing dataset {}".format(str(datetime.now()), self.config.structure.dataset))

        # Import networks
        start_time = datetime.now()

        # import Diff Pretrained network
        print("[{}] Start importing Diff Pretrained network".format(str(datetime.now())))
        model_name = 'ema_cifar10'
        diffusion = Diffusion.from_pretrained(model_name, self.args.dpm, device=self.config.device.diff_device)

        df_columns = ["Epoch", "nData", "att_time", "pur_time", "clf_time", \
                      "std_acc", "att_acc", "pur_acc_l", "pur_acc_s", "pur_acc_o", \
                      "pur_acc_list_l", "pur_acc_list_s", "pur_acc_list_o", "count_att", "count_diff"]
        if self.config.purification.purify_natural:
            df_columns.append("nat_pur_acc_l")
            df_columns.append("nat_pur_acc_s")
            df_columns.append("nat_pur_acc_o")
            df_columns.append("nat_pur_acc_list_l")
            df_columns.append("nat_pur_acc_list_s")
            df_columns.append("nat_pur_acc_list_o")
        df = pd.DataFrame(columns=df_columns)

        if self.config.purification.guide_mode == 'LPIPS' or self.config.purification.guide_mode2 == 'LPIPS' :
            loss_fn = lpips.LPIPS(net='alex')
            loss_fn = loss_fn.to(self.config.device.diff_device) 

        def cond_fcc(x_reverse_t, t): 
            """
            Calculate the grad of guided condition.
            """
            with torch.enable_grad():
                x_in = x_reverse_t.detach().requires_grad_(True)
                x_advt = diffusion.diffuse_t_steps(x_adv, t)
                x_adv_t = x_advt
                # scale = exp(config.purification.guide_exp_a * t / config.purification.purify_step+config.purification.guide_exp_b) + config.purification.guide_scale_base
                if self.config.purification.guide_mode2 == 'MSE': 
                    selected = -1 * F.mse_loss(x_in, x_adv_t)
                    scale = diffusion.compute_scale(x_in,t, self.config.attack.ptb*2/255. / 3. / self.config.purification.guide_scale)
                elif self.config.purification.guide_mode2 == 'LPIPS':
                    dist = loss_fn(x_in, x_adv_t)
                    selected = -1 * dist.mean()
                    scale = diffusion.compute_scale(x_in,t, self.config.attack.ptb*2/255. / 3. / self.config.purification.guide_scale)
                elif self.config.purification.guide_mode2 == 'SSIM':
                    selected = pytorch_ssim.ssim(x_in, x_adv_t)
                    scale = diffusion.compute_scale(x_in,t, self.config.attack.ptb*2/255. / 3. / self.config.purification.guide_scale)
                elif self.config.purification.guide_mode2 == 'CONSTANT': 
                    scale = self.config.purification.guide_scale
                gradient = torch.autograd.grad(selected.sum(), x_in)[0]
                #print(f"{config.purification.guide_mode}_gradient: {gradient[0][0][0]}")
                print(f"{self.config.purification.guide_mode2}_scale: {scale}")
                condfn = gradient * scale
                return condfn

        x_learnable_list = []
        
        for i, (x,y) in enumerate(tqdm.tqdm(testLoader)):
            
            start_time = datetime.now() 
            print("[{}] Epoch {}".format(str(datetime.now()), i))
            if self.config.structure.dataset == 'REMCIFAR10' or self.config.structure.dataset == 'REMSVHN':
                x = x.add(0.5).to(self.config.device.diff_device)
            else:
                x = x.to(self.config.device.diff_device)   
            y = y.to(self.config.device.diff_device).long()

            transform_raw_to_diff = raw_to_diff(self.config.structure.dataset)
            x_adv = transform_raw_to_diff(x).to(self.config.device.diff_device)

            model_kwargs = {}
            model_kwargs["y"] = y
            
            # purify natural image
            x_nat_pur_list_list = []
            y_label_list = []
            start_time = datetime.now()
            print("[{}] Epoch {}:\tBegin purifying {} natural images".format(str(datetime.now()), i, i))

            for j in range(self.config.purification.path_number):  # 0
                
                x_nat_pur_list = diff_purify(
                    x, diffusion,
                    self.config.purification.max_iter,
                    mode="purification",
                    config=self.config,
                    cond_fc=cond_fcc,
                    model_kwargs=model_kwargs,
                    args = self.args
                )
                # return image
                x_nat_pur_list_list.append(x_nat_pur_list)
            

            x_learnable = x_nat_pur_list[-1]
            x_learnable_list.append(x_learnable)
            

            purify_natural_time = elapsed_seconds(start_time, datetime.now())
            # print("[{}] Epoch {}:\t{:.2f} seconds to purify {} natural images".format(str(datetime.now()), i, purify_natural_time, self.config.structure.bsize))
            print("[{}] Epoch {}:\t{:.2f} seconds to purify {} natural images".format(str(datetime.now()), i,
                                                                                      purify_natural_time, i))

        x_learnable_list = torch.cat(x_learnable_list, 0)
        img_learnable = x_learnable_list.mul(255).byte()
        img_learnable = img_learnable.cpu().numpy().transpose(0, 2, 3, 1)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        save_name = os.path.join(self.save_path, 'learnable'+'_'+str(self.config.purification.join_mode)+'.npy')
        np.save(save_name , img_learnable)



