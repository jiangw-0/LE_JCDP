# Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples

This repo contains the official PyTorch implementation of ["Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples"](https://arxiv.org/abs/2305.09241) (ACM MM 2023), by Wan Jiang*, [Yunfeng Diao*](http://faculty.hfut.edu.cn/diaoyunfeng/en/index.htm), [He Wang](https://drhewang.com/), Jianxin Sun, Meng Wang and Richang Hong (*co-primary authors).

![overview.png](https://github.com/jiangw-0/LE_JCDP/blob/main/imgs/overview.png)

# Dependencies

Below is the key environment under which the code was developed, not necessarily the minimal requirements:

1.  Python 3.10
2.  Pytorch 2.0.1
3.  Cuda 11.8


# Warning

The code has not been exhaustively tested. You need to run it at your own risk. The author will try to actively maintain it and fix reported bugs but this can be delayed.

# Experiments in the paper

We provide an example of JCDP on CIFAR-10 poisons generated by [EM](https://github.com/HanxunH/Unlearnable-Examples#generate-noise-for-unlearnable-examples).

## Unlearnable Examples

Prepare poisoned images as `.pt` files in folder `unlearnable_exs/`. 
Here are the download links for our generated the Unlearnable Examples following [EM](https://github.com/HanxunH/Unlearnable-Examples#generate-noise-for-unlearnable-examples):

*   [sample-wise](https://drive.google.com/drive/folders/1Cr9U5AwoA0LW36kdFpE5IdH1erFfwfJ-)
*   [class-wise](https://drive.google.com/drive/folders/1Ax2GzzKMgX_GvlhT8etZDOiHgUqBIgu5)

## Download pre-trained models

We have released checkpoints for the main models in the paper.
Here are the download links for each model checkpoint:

*   [fine-tuning ](https://drive.google.com/drive/folders/1_h76h7sVIoxyGi2OCuF7eL7-KWYMq1gq)
*   [from scratch](https://drive.google.com/drive/folders/1b52GoGGyEQHHec2ThOQw196_ehpSSxD9)

Download the relevant model checkpoints into a folder called `models/`.
You can train other types of diffusion models as well, which works here as well.

## Generate  the corresponding Learnable Examples by JCDP

run main\_Un.py to generate Learnable Examples in `data/`.

```python
python main_Un.py --config cifar10_Un.yml \
                  --runner Empirical_cond \
                  --dpm models/xxx.pth \
                  --perturb_path unlearnable_exs/xxx.pt \
                  --log data 
```
- `config` is the path to the config file. `eg. cifar10_Un.yml`. Our prescribed config files are provided in `configs/`.  
- `runner` is the path to the runner file. `eg. Empirical_cond`.
- `dpm` is the path for dpm model.  `eg. models/ckpt_10000.pth`. 
- `perturb_path` is the path for unlearnable examples.  `eg. unlearnable_exs/resnet18_perturbation_samplewise.pt`. 
- `log` is the ouitput path, including images and logs.  `eg. data`. 

# Citation(Bibtex)

If you find this code to be useful for your research, please consider citing.

```
@article{jiang2023unlearnable,
title={Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples},
author={Jiang, Wan and Diao, Yunfeng and Wang, He and Sun, Jianxin and Wang, Meng and Hong, Richang},
journal={arXiv preprint arXiv:2305.09241},year={2023}
```

# Contact

Please email <jiangw000@mail.hfut.edu.cn> for further questions.

# Acknowledgment

Diffusion Models Beat GANS on Image Synthesis：https://github.com/openai/guided-diffusion


