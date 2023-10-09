### packages
# software packages

import string
from typing import TextIO
import torch

import numpy as np
import traceback
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from PIL import Image
import yaml
import logging
import time
import sys
import pdb
import shutil
from sklearn.manifold import TSNE
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
from runners import *



### main.py
# (1) Import configs (replace argparse. Too many argparse flags now)
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # Dataset and save logs
    parser.add_argument('--log', default='logsGen2', help='Output path, including images and logs')

    parser.add_argument('--config', type=str, default='cifar10_Un.yml', help='Path for saving running related data.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp_mode', type=str, default='Full', help='Available: [Full, Partial, One]')
    parser.add_argument('--runner', type=str, default='Empirical_Un',help='Available: [Empirical, Empirical_Un, Empirical_cond, Certified, Deploy]')
    
    # Arguments not to be touched
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    # parser.add_argument('--CIFARC_CLASS', type=int, default=-1)
    # parser.add_argument('--CIFARC_SEV', type=int, default=1)
    parser.add_argument('--perturb_type', type=str, default='EM_c', help='Available: EM_s, EM_c')
    parser.add_argument('--filter', type=str, default='gaussian',
                        choices=['averaging', 'gaussian', 'median', 'bilateral'],
                        help='select the low pass filter; only works in [mix] mode')
    parser.add_argument('--dpm', type=str, default='./model', help='Path for dpm model.')
    parser.add_argument('--perturb_path', type=str, default='./model', help='Path for dpm model.')
    parser.add_argument('--debug', type=bool, default=False,
                        help='select the low pass filter; only works in [mix] mode')
    
                        

    args = parser.parse_args()
    run_id = str(os.getpid())
    run_time = time.strftime('%Y-%b-%d-%H-%M-%S')
    # args.doc = '_'.join([args.doc, run_id, run_time])
    

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)
    

    #define the folder name
    if new_config.purification.joincond == True:    #joint-cond
        args.log = os.path.join(args.log,"{}_{}_COND:{}".format(
            new_config.structure.dataset,    #"CIFAR10-Un"
            str(new_config.attack.attack_method),   ##"clf_pgd"
            new_config.purification.join_mode   #MSE_LPIPS
            ),
            "step_{}_iter_{}_path_{}_per={}_{}_ML{}_scale".format(
            new_config.purification.purify_step,   #36
            new_config.purification.max_iter,      #4  /1
            new_config.purification.path_number,   #1
            new_config.attack.ptb,                 #8.0
            f'{new_config.purification.guide_scale}+{new_config.purification.guide_scale_base}',
            new_config.purification.cond
            ))  #60000+0
    if new_config.purification.cond == True and new_config.purification.joincond == False:  #cond 1 guide
        args.log = os.path.join(args.log,"{}_{}_COND:{}".format(
            new_config.structure.dataset,    #"CIFAR10-Un"
            str(new_config.attack.attack_method),   ##"clf_pgd"
            new_config.purification.guide_mode   #SSIM
            ),
            "step_{}_iter_{}_path_{}_per={}_{}".format(
            new_config.purification.purify_step,   #36
            new_config.purification.max_iter,      #4  /1
            new_config.purification.path_number,   #1
            new_config.attack.ptb,                 #8.0
            f'{new_config.purification.guide_scale}+{new_config.purification.guide_scale_base}'
            ))  #70000+0
    
    elif new_config.purification.condcc == False and new_config.purification.cond == False and  new_config.purification.joincond == False: 
        
        args.log = os.path.join(args.log,"{}_{}".format(
            new_config.structure.dataset, #"CIFAR10-Un"
            str(new_config.attack.attack_method) ##"clf_pgd"
            ),
            "step_{}_iter_{}_path_{}_per={}".format(
            new_config.purification.purify_step,  #36
            new_config.purification.max_iter, #4
            new_config.purification.path_number, #1
            new_config.attack.ptb #8.0 
            ))
    
    if not os.path.exists(args.log):
        os.makedirs(args.log,exist_ok=True)


    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def create_argparser(): 
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=100,
        use_ddim=False,
        model_path='',
        log = 'logsde',
        config = 'cifar10_Un.yml',
        seed = '1234',
        exp_mode = 'Full',
        runner = 'Empirical_cond',
        verbose = 'info',
        perturb_type='',
        perturb_path='',
        dpm = ' ',
        debug=True
        )

    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def outputargs(a, file):
    my_dict = vars(a)
    with open(file, "w") as f:
        for key, value in my_dict.items():
            print("{:<10}{}".format(key + ":", value), file=f)
            print("{:<10}{}".format(key + ":", value))


def main(): 
    args, config = parse_args_and_config()
    log_progress = open(os.path.join(args.log, f"log_progress_{config.device.rank}"), "w")
    if args.debug is not True:
        sys.stdout = log_progress
    logging.info("Config =")
    print(">" * 80)

    parser = None
    
    if config.purification.cond:
        parser = create_argparser().parse_args()
        vars(parser).update(vars(args))
        #add_dict_to_argparser(pargs, my_dicta)
        outputargs(parser, os.path.join(args.log, "outputparser.txt"))
    else:
        outputargs(args, os.path.join(args.log, "outputargs.txt"))

    print("<" * 80)
    outputargs(config, os.path.join(args.log, "outputconfig.txt"))
    print("<" * 80)

    
    try:
        if config.purification.cond:
            par = parser
        else:
            par = None
            #runner = eval(args.runner)(args, parser, config)
        
        runner = eval(args.runner)(args, par, config)
        print('args.runner:', args.runner)
        #args.runner = 'Empirical_Un'
        #runners/empirical_unlearnable.py
        runner.run(log_progress)
    except:
        logging.error(traceback.format_exc())

    log_progress.close()

    return 0

if __name__ == '__main__':
    sys.exit(main())