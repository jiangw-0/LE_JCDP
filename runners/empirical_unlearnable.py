# figure out how diffusion steps diverse
import pdb

from torchvision.utils import save_image
from distutils.command.config import config
from utils import *
from clf_models.networks import *
from datetime import datetime
import tqdm
import pandas as pd
import torchvision
from purification.diff_purify import *
#from purification.diff_purify_saveimages import *
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
import os


__all__ = ['Empirical_Un']

class Empirical_Un():
    def __init__(self,args, parser, config):
        self.args = parser if config.purification.condc else args
        self.config = config
        self.config.attack.if_attack = True
        self.save_path = os.path.join(self.args.log, str(self.config.purification.join_mode))
    def run(self, log_progress):
        # Normalize on classifiers 
        transform_raw_to_clf = raw_to_clf(self.config.structure.dataset)

        # Output log file configuration
        
        sys.stdout = log_progress
        # log_output = open(os.path.join(self.args.log, "log_output"), "w")

        # Import dataset
        start_time = datetime.now()
        print("[{}] Start importing dataset {}".format(str(datetime.now()), self.config.structure.dataset))
        if self.config.structure.dataset == 'CIFAR10-C':
            self.config.attack.if_attack = False
            testLoader_list = importData(dataset=self.config.structure.dataset, train=False, shuffle=False, bsize=self.config.structure.bsize)
            testLoader = testLoader_list[self.config.structure.CIFARC_CLASS-1][self.config.structure.CIFARC_SEV-1]
        elif self.config.structure.dataset == 'TinyImageNet':
            data_transforms = transforms.Compose([
                    transforms.ToTensor(),
            ])
            path_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
            path_root = os.path.join(path_root, "datasets")
            image_datasets = TINYIMAGENET(path_root,train=False,transform=data_transforms)
            testLoader = torch.utils.data.DataLoader(image_datasets, batch_size=self.config.structure.bsize, shuffle=True, num_workers=64)
        elif self.config.structure.dataset == 'ImageNet' and self.config.attack.attack_method == 'bpda_strong':
            testLoader = importData(dataset=self.config.structure.dataset, train=False, shuffle=True, bsize=self.config.structure.bsize)
        elif self.config.structure.dataset == 'ImageNet-C':
            self.config.attack.if_attack = False
            testLoader = importData(
                dataset=self.config.structure.dataset, train=False, shuffle=False, bsize=self.config.structure.bsize,
                distortion_name=self.config.structure.distortion_name, severity=self.config.structure.severity )
        # elif self.config.structure.dataset == 'ImageNet' and self.config.structure.run_samples <= 10000:
        #     testLoader = importData(dataset=self.config.structure.dataset, train=False, shuffle=True, bsize=self.config.structure.bsize)
        else:
            #dataset=self.config.structure.dataset="CIFAR10-Un"
            testLoader = importData(dataset=self.config.structure.dataset, train=True, shuffle=False,perturbtype = self.args.perturb_type, bsize=self.config.structure.bsize)
            #print(dataset.data.shape)  5000
        print("[{}] Finished importing dataset {}".format(str(datetime.now()), self.config.structure.dataset))


        # Import classifier networks
        start_time = datetime.now()
        print("[{}] Start importing network".format(str(datetime.now())))

        if self.config.structure.dataset in ["ImageNet","ImageNet-C"]:
            if self.config.structure.classifier == 'ResNet152':
                network_clf = torchvision.models.resnet152(pretrained=True).to(self.config.device.clf_device)
            elif self.config.structure.classifier == 'ResNet50':
                network_clf = torchvision.models.resnet50(pretrained=True).to(self.config.device.clf_device)
            network_clf.eval()
        elif self.config.structure.clf_log not in ["cifar10_carmon", "cifar10_wu", "cifar10_zhang"]:
            network_clf = eval(self.config.structure.classifier)().to(self.config.device.clf_device)
            #classifier: "Wide_ResNet"  clf_device: "cuda:0"
            network_clf = torch.nn.DataParallel(network_clf)

        if self.config.structure.dataset in ["CIFAR10", "CIFAR10-C", "CIFAR100"]: # CIFAR10 setting, trained by WideResNet
            states_att = torch.load(os.path.join('clf_models/run/logs', self.config.structure.clf_log, '{}.t7'.format(self.config.classification.checkpoint)), map_location=self.config.device.clf_device) # Temporary t7 setting
            network_clf = states_att['net'].to(self.config.device.clf_device)
        # elif self.config.structure.dataset in ["ImageNet"]:   # Get network_clf from loaded network

        #import Diff Pretrained network
        print("[{}] Start importing Diff Pretrained network".format(str(datetime.now())))

        if self.config.structure.dataset in ["LSPCIFAR10", "REMCIFAR10","ImageNet100-UnLSP","CIFAR10-Un","CIFAR100","CIFAR100-Un"]:
            model_name = 'ema_cifar10'
            diffusion = Diffusion.from_pretrained(model_name, device=self.config.device.diff_device)

        elif self.config.structure.dataset in ["ImageNet","ImageNet-C"]:
            print("creating model and diffusion...")
            model, diffusion = create_model_and_diffusion(
                **args_to_dict(self.config.net, model_and_diffusion_defaults().keys())
            )
            model.load_state_dict(
                torch.load(self.config.net.model_path, map_location="cpu")
            )
            model.to(self.config.device.clf_device)
            if self.config.net.use_fp16:
                model.convert_to_fp16()
            model.eval() 

        df_columns = ["Epoch", "nData", "att_time", "pur_time", "clf_time", \
                        "std_acc", "att_acc", "pur_acc_l", "pur_acc_s", "pur_acc_o", \
                        "pur_acc_list_l", "pur_acc_list_s", "pur_acc_list_o","count_att" ,"count_diff"]
        if self.config.purification.purify_natural:
            df_columns.append("nat_pur_acc_l")
            df_columns.append("nat_pur_acc_s")
            df_columns.append("nat_pur_acc_o")
            df_columns.append("nat_pur_acc_list_l")
            df_columns.append("nat_pur_acc_list_s")
            df_columns.append("nat_pur_acc_list_o")
        df = pd.DataFrame(columns=df_columns)


        label_dict = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
              5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
        def num_to_label(y_tensor):
            labels = []
            y_label = y_tensor.cpu().numpy().tolist()
            for l in y_label:
                label = label_dict[l]
                labels.append(label)
            return labels

        # Run
        x_learnable_list = []
        #len(testLoader) = 500
        for i, (x,y) in enumerate(tqdm.tqdm(testLoader)):

            # if i > 0:
            #     break
            # if i<self.config.structure.start_epoch:
            #     continue

            start_time = datetime.now()
            print("[{}] Epoch {}".format(str(datetime.now()), i))
            x = preprocess(x, self.config.structure.dataset) #only for preprocess cifar10-c
            x = x.float().to(self.config.device.diff_device)
            y = y.to(self.config.device.diff_device).long()
            model_kwargs = {}
            model_kwargs["y"] = y

            ##########visualize
            for m in range(10):
                x_uint = x[m].mul(255).byte()
                x_uint = x_uint.cpu().numpy().transpose(1, 2, 0)#.astype('uint8')
                save_path = os.path.join(self.args.log, 'x0'+ str(self.config.purification.purify_step))
                if not os.path.exists(save_path):
                    os.makedirs(save_path,exist_ok=True)
                save_name = os.path.join(save_path, str(m)+'_'+str(y[m].cpu().numpy())+'.jpg')
                plt.imsave(save_name, x_uint)


            # purify natural image
            x_nat_pur_list_list = []
            start_time = datetime.now()
            print("[{}] Epoch {}:\tBegin purifying {} natural images".format(str(datetime.now()), i, i))
            for j in range(self.config.purification.path_number):  # 0,1
                if self.config.structure.dataset in ["LSPCIFAR10","REMCIFAR10","ImageNet100-UnLSP","CIFAR10-Un","CIFAR10-C","CIFAR100","CIFAR100-Un",]:
                    x_nat_pur_list = diff_purify(
                        x, diffusion,
                        self.config.purification.max_iter,
                        mode="purification",
                        config=self.config,
                        cond_fc = None,
                        model_kwargs=model_kwargs,
                        args=self.args
                        )
                elif self.config.structure.dataset in ["ImageNet"]:
                    x_nat_pur_list = purify_imagenet(x, diffusion, model,
                        self.config.purification.max_iter,
                        mode="purification",
                        config=self.config)
                x_nat_pur_list_list.append(x_nat_pur_list)
            x_learnable = x_nat_pur_list[-1]
            

            if i < 1:
                for m in range(25):
                    x_uint = x_learnable[m].mul(255).byte()
                    x_uint = x_uint.cpu().numpy().transpose(1, 2, 0)#.astype('uint8')

                    save_path = os.path.join(self.args.log,  'reverse'+ str(self.config.purification.purify_step))
                    if not os.path.exists(save_path):
                        os.makedirs(save_path,exist_ok=True)

                    #cifar 10
                    #save_name = os.path.join(save_path, str(m)+'_'+str(y[m].cpu().numpy())+'_'+ str(label_dict[y[m].cpu().numpy().tolist()])+'.jpg')

                    #cifar100
                    save_name = os.path.join(save_path, str(m)+'_'+str(y[m].cpu().numpy())+'_'+str(i)+'.jpg')
                    
                    plt.imsave(save_name, x_uint)

            x_learnable_list.append(x_learnable)
            purify_natural_time = elapsed_seconds(start_time, datetime.now())
            #print("[{}] Epoch {}:\t{:.2f} seconds to purify {} natural images".format(str(datetime.now()), i, purify_natural_time, self.config.structure.bsize))
            print("[{}] Epoch {}:\t{:.2f} seconds to purify {} natural images".format(str(datetime.now()), i,purify_natural_time, i))


        x_learnable_list = torch.cat(x_learnable_list, 0)
        img_learnable = x_learnable_list.mul(255).byte()
        img_learnable = img_learnable.cpu().numpy().transpose(0, 2, 3, 1)
        # x_uint = img_learnable[0]
        # plt.imshow(x_uint)
        # plt.show()



        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path,exist_ok=True)
        #np.save(self.save_path+'learnable_cifar10.npy', img_learnable)
        #np.save(self.save_path + 'learnable_cifar10MSET4.npy', img_learnable)
        save_name = os.path.join(self.save_path, 'learnable'+'.npy')
        np.save(save_name , img_learnable)
