import numpy as np
import pdb
import torch
from pytorch_diffusion.model import Model
import os
from pytorch_diffusion.ema import EMAHelper
def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,) #1000,
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a).float().to(device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,)+(1,)*(len(x_shape)-1))
    return out


def diffusion_step(x, t, *,
                   noise=None,
                   sqrt_alphas,
                   sqrt_one_minus_alphas):
    """
    Sample from q(x_t | x_{t-1}) (eq. (2))
    """
    if noise is None:
        noise = torch.randn_like(x)
    assert noise.shape == x.shape
    return (
        extract(sqrt_alphas, t, x.shape) * x +
        extract(sqrt_one_minus_alphas, t, x.shape) * noise
    )


def denoising_step(x, t, *,
                   model,
                   logvar,
                   sqrt_recip_alphas_cumprod,
                   sqrt_recipm1_alphas_cumprod,
                   posterior_mean_coef1,
                   posterior_mean_coef2,
                   return_pred_xstart=False,
                   cond_num_1 = None,
                   cond_num_2 = None
                   ):
    """
    Sample from p(x_{t-1} | x_t)
    """
    # instead of using eq. (11) directly, follow original implementation which,
    # equivalently, predicts x_0 and uses it to compute mean of the posterior
    # 1. predict eps via model
    model_output = model(x, t)
    # 2. predict clipped x_0
    # (follows from x_t=sqrt_alpha_cumprod*x_0 + sqrt_one_minus_alpha*eps)
    pred_xstart = (extract(sqrt_recip_alphas_cumprod, t, x.shape)*x -
                   extract(sqrt_recipm1_alphas_cumprod, t, x.shape)*model_output)
    pred_xstart = torch.clamp(pred_xstart, -1, 1)
    # 3. compute mean of q(x_{t-1} | x_t, x_0) (eq. (6))
    mean = (extract(posterior_mean_coef1, t, x.shape)*pred_xstart +
            extract(posterior_mean_coef2, t, x.shape)*x)

    logvar = extract(logvar, t, x.shape)
    
    a = 1.0
    b = 1.0
    if cond_num_2 is not None or  cond_num_1 is not None:
        print('trade: a / b = ', a, b)
    
    if cond_num_2 is not None:   
        sgradient = cond_num_2.float()
        mean_sum = torch.exp(logvar)*sgradient
        mean = mean.float() + a * mean_sum
        
        print('Label/d2_s*d*g:', mean_sum[0][0][0][0])
        print('Label/d2_mean:', mean[0][0][0][0])
    if cond_num_1 is not None:   
        mean_sum1 = torch.exp(logvar)*cond_num_1.float()
        mean = mean.float() + b * mean_sum1
        print('cond/d1_-s*d*g:', mean_sum1[0][0][0][0])
        print('cond/d1_mean:', mean[0][0][0][0])


    # sample - return mean for t==0
    noise = torch.randn_like(x)
    mask = 1-(t==0).float()
    mask = mask.reshape((x.shape[0],)+(1,)*(len(x.shape)-1))
    sample = mean + mask*torch.exp(0.5*logvar)*noise
    sample = sample.float()
    if return_pred_xstart:
        return sample, pred_xstart
    return sample


class Diffusion(object):
    def __init__(self, diffusion_config, model_config, device=None):
        self.init_diffusion_parameters(**diffusion_config)
        self.model = Model(**model_config)
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.model.to(self.device)


    def init_diffusion_parameters(self, **config):
        self.model_var_type = config.get("model_var_type", "fixedsmall")
        betas=get_beta_schedule(
            beta_schedule=config['beta_schedule'],
            beta_start=config['beta_start'],
            beta_end=config['beta_end'],
            num_diffusion_timesteps=config['num_diffusion_timesteps'] #1000
        )

        self.num_timesteps = betas.shape[0]


        alphas = 1.0-betas
        alphas_cumprod = np.cumprod(alphas, axis=0) # \bar{alpha}_t
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1]) # \bar{alpha}_t-1
        posterior_variance = betas*(1.0-alphas_cumprod_prev) / (1.0-alphas_cumprod) # \tilde{beta}_t
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        # they are all numpy arrays of shape (T,)
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_recip_alphas_cumprod = sqrt_recip_alphas_cumprod
        self.sqrt_recipm1_alphas_cumprod = sqrt_recipm1_alphas_cumprod
        self.posterior_mean_coef1 = posterior_mean_coef1
        self.posterior_mean_coef2 = posterior_mean_coef2
        self.sqrt_alphas = np.sqrt(alphas)
        self.sqrt_one_minus_alphas = np.sqrt(1. - alphas)

        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))
        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))


    @classmethod
    def from_pretrained(cls, name, dpm, device=None):
        cifar10_cfg = {
            "resolution": 32,  #32
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1,2,2,2),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.1,
        }
        lsun_cfg = {
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1,1,2,2,4,4),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.0,
        }

        model_config_map = {
            "cifar10": cifar10_cfg,
            "lsun_bedroom": lsun_cfg,
            "lsun_cat": lsun_cfg,
            "lsun_church": lsun_cfg,
        }

        diffusion_config = {
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "num_diffusion_timesteps": 1000,
        }
        model_var_type_map = {
            "cifar10": "fixedlarge",
            "cifar100": "fixedlarge",
            "lsun_bedroom": "fixedsmall",
            "lsun_cat": "fixedsmall",
            "lsun_church": "fixedsmall",
        }
        ema = name.startswith("ema_")
        basename = name[len("ema_"):] if ema else name
        diffusion_config["model_var_type"] = model_var_type_map[basename]

        print("Instantiating")
        diffusion = cls(diffusion_config, model_config_map[basename], device)
        diffusion.model.to(diffusion.device)
        ###pre-trained####
        
        ckptpath = dpm
        ckpt = torch.load(ckptpath)
        diffusion.model = torch.nn.DataParallel(diffusion.model)
        diffusion.model.load_state_dict(ckpt[0], strict=True)
        print("Loading checkpoint {}".format(ckptpath))
        ema_helper = EMAHelper(mu=0.9999)
        ema_helper.register(diffusion.model)
        ema_helper.load_state_dict(ckpt[-1])
        ema_helper.ema(diffusion.model)
        

        diffusion.model.eval()
        print("Moved model to {}".format(diffusion.device))
        return diffusion


    def denoise(self, n, n_steps=None, x=None, curr_step=None,
                progress_bar=lambda i, total=None: i,
                callback=lambda x, i, x0=None: None, cond_fn=None, cond_fc=None, is_condcc=None, model_kwargs=None):
        # n is the batchsize
        with torch.no_grad():
            if curr_step is None:
                curr_step = self.num_timesteps  #betas.shape[0]

            assert curr_step > 0, curr_step

            if n_steps is None or curr_step-n_steps < 0:
                n_steps = curr_step

            if x is None:
                x = torch.randn(n, 3, 32, 32)
                x = x.to(self.device)

            for i in progress_bar(reversed(range(curr_step-n_steps, curr_step)), total=n_steps):
                

                t = (torch.ones(n)*i).to(self.device)
                
                if cond_fn is not None:  #距离1
                    cond_num_1 = cond_fn(x, i)
                else:
                    cond_num_1 = None

                if cond_fc is not None:
                    if is_condcc:
                        cond_num_2 = cond_fc(x, i)    #距离2
                    else:
                        cond_num_2 = cond_fc(x, t, **model_kwargs)  #标签
                    
                else:
                    cond_num_2 = None

                com = False
                #MSE36-11,LPIPS10-0   
                if com:
                    print('purify mode:', com)
                    x, x0 = denoising_step(x,
                                            t=t,
                                            model=self.model,
                                            logvar=self.logvar,
                                            sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                            posterior_mean_coef1=self.posterior_mean_coef1,
                                            posterior_mean_coef2=self.posterior_mean_coef2,
                                            return_pred_xstart=True,
                                            cond_num_1 = cond_num_1,
                                            cond_num_2 = cond_num_2
                                            #cond_num_2 = None
                                            )
                else:
                    if i > 10 :
                        x, x0 = denoising_step(x,
                                            t=t,
                                            model=self.model,
                                            logvar=self.logvar,
                                            sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                            posterior_mean_coef1=self.posterior_mean_coef1,
                                            posterior_mean_coef2=self.posterior_mean_coef2,
                                            return_pred_xstart=True,
                                            cond_num_1 = cond_num_1,    #距离
                                            #cond_num_1 = None,
                                            #cond_num_2 = cond_num_2
                                            cond_num_2 = None
                                            )
                    
                    if i < 11:
                        x, x0 = denoising_step(x,
                                            t=t,
                                            model=self.model,
                                            logvar=self.logvar,
                                            sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                            sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                            posterior_mean_coef1=self.posterior_mean_coef1,
                                            posterior_mean_coef2=self.posterior_mean_coef2,
                                            return_pred_xstart=True,
                                            #cond_num_1 = cond_num_1,
                                            cond_num_1 = None,
                                            cond_num_2 = cond_num_2   #Label/距离2
                                            
                                            )
                callback(x, i, x0=x0)

            return x


    def diffuse(self, n, n_steps=None, x=None, curr_step=None,
                progress_bar=lambda i, total=None: i,
                callback=lambda x, i: None):
        with torch.no_grad():
            if curr_step is None:
                curr_step = 0

            assert curr_step < self.num_timesteps, curr_step

            if n_steps is None or curr_step+n_steps > self.num_timesteps:
                n_steps = self.num_timesteps-curr_step

            assert x is not None

            for i in progress_bar(range(curr_step, curr_step+n_steps), total=n_steps):
                t = (torch.ones(n)*i).to(self.device)
                x = diffusion_step(x,
                                   t=t,
                                   sqrt_alphas=self.sqrt_alphas,
                                   sqrt_one_minus_alphas=self.sqrt_one_minus_alphas)
                callback(x, i+1)

            return x

    def diffuse_t_steps(self, x0, t):
        # x is a torch tensor of shape (B,C,H,W)
        # t is a interger range from 0 to T-1
        alpha_bar = self.alphas_cumprod[t]
        xt = np.sqrt(alpha_bar) * x0 + np.sqrt(1-alpha_bar) * torch.randn_like(x0)
        return xt

    def compute_scale(self,x, t, m):
        alpha_bar = self.alphas_cumprod[t]
        return np.sqrt(1-alpha_bar) / (m*np.sqrt(alpha_bar))

    @staticmethod
    def torch2hwcuint8(x, clip=False):
        if clip:
            x = torch.clamp(x, -1, 1)
        x = x.detach().cpu()
        x = x.permute(0,2,3,1)
        x = (x+1.0)*127.5
        x = x.numpy().astype(np.uint8)
        return x

    @staticmethod
    def save(x, format_string, start_idx=0):
        import os, PIL.Image
        os.makedirs(os.path.split(format_string)[0], exist_ok=True)
        x = Diffusion.torch2hwcuint8(x)
        for i in range(x.shape[0]):
            PIL.Image.fromarray(x[i]).save(format_string.format(start_idx+i))



if __name__ == "__main__":
    import sys, tqdm
    name = sys.argv[1] if len(sys.argv)>1 else "cifar10"
    bs = int(sys.argv[2]) if len(sys.argv)>2 else 1
    nb = int(sys.argv[3]) if len(sys.argv)>3 else 1
    diffusion = Diffusion.from_pretrained(name)
    for ib in tqdm.tqdm(range(nb), desc="Batch"):
        x = diffusion.denoise(bs, progress_bar=tqdm.tqdm)
        idx = ib*bs
        diffusion.save(x, "results/"+name+"/{:06}.png", start_idx=idx)
