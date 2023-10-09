import torch
import pickle
import collections
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import sys
path_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(path_root)



if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def importData(dataset, train, shuffle, bsize, perturbtype='E_c', perturb_path=''):
    '''
    dataset: datasets (CIFAR10, CIFAR100, SVHN)
    train: True if training set, False if test set
    shuffle: Whether to shuffle or not
    bsize: minibatch size
    '''
    # Set transform
    dataset_list = [ "CIFAR10-Un","CIFAR100-Un"]
    perturb_type = perturbtype
    perturb_tensor_filepath = perturb_path
    if dataset not in dataset_list:
        sys.exit("Non-handled dataset")

    if dataset=="CIFAR10-Un":
        transform = transforms.Compose([transforms.ToTensor()])
        #path = os.path.join(path_root, "datasets", "CIFAR10")
        path = '/home/jiang/home2/data/cifar-10-python'
        
        dataset = CIFAR10(path, train=train, download=False, transform=transform)
        print(dataset.data.shape)  
        
        perturb_tensor_filepath = '/home/jiang/home2/UUEE/Unlearnable-Examples-main/results/CIFAR10/resnet18/perturbation_samplewise.pt'
    
        print(perturb_tensor_filepath)

        perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        
       
        perturb_tensor = perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        #samplewise or classwise 
        print('perturb type:', perturb_type )
        if perturb_type == 'EM_s':
            dataset.data = dataset.data + perturb_tensor
            dataset.data = np.clip(dataset.data, a_min=0, a_max=255)
            dataset.data = dataset.data.astype(np.uint8)
        
    dataloader = DataLoader(dataset, batch_size=bsize, shuffle=shuffle, num_workers=4)
        
    return dataloader
    