"""
Test for GANpatch
使用训练好的GAN网络生成对抗性补丁，并且将补丁放入场景中。定量化测试我们所提方法的性能
控制生成的补丁，图案与周围环境能够融合起来
ver2，直接生成N个补丁，已知补丁和场景直接手动进行apply。
保存的pt文件，尺寸为torch.Size([3. 64. 64])

2025.6.24
生成指定数量的尺寸为[32, 64]的对抗补丁，并保存可视化结果
"""
import argparse
from tqdm import tqdm
import os
import stat
import shutil
import random
import time
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from shapely.geometry import Polygon

from ensemble_tools.detection_model import init_detector
from mmdet.apis import inference_detector
from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier, TotalVariation  # 这里选择load_data_1中的函数
from ensemble_tools.utils0 import find_patch_positions_v2, find_patch_positions_v3, zero_out_bounding_boxes_v2, weights_init, convert_to_le90
import ensemble_tools.GANmodels.dcgan as dcgan
from ensemble_tools.patch_dataloader import ForestPatch, ScenePatch

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description='GAN-patch quantitative testing settings')
parser.add_argument('--work_dir', default=None, help='folder to output files')
parser.add_argument('--generate_patch_number', default=200, type=int, help="total generate patch number")
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument('--seed', default='2025', type=int, help='choose seed')
parser.add_argument('--netG', default='', type=str, help='choose generator checkpoint path')
# config for fake patch
# parser.add_argument('--patch_size', default=64, type=int, help='size of patch in digital scene of 640*384')

args = parser.parse_args()

# check
args_dict = vars(args)
print('Experimental settings:\n', args_dict)


### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
# patch setting
patch_size = [32, 64]
patch_h, patch_w = patch_size[0], patch_size[1]
netG_path = args.netG
patch_number = args.generate_patch_number

# test setting
device = args.device

# GAN setting
ngpu = 1
nz = 100
ngf = 64
nc = 3
G_extra_layers = 0


### -----------------------------------------------------------    Initialize    ---------------------------------------------------------------------- ###
# set random seed 
Seed = args.seed  # 37564 7777
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

# create working files
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

    os.chmod(args.work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
else:
    shutil.rmtree(args.work_dir)
    print('--------------File exists! Existing file with the same name has been removed!--------------')

subfolders = ['fake_patch', 'patch_pt']  # fake_patch用于存储生成的50xPatch_per_Image个补丁
for subfolder in subfolders:
    subfolder_path = os.path.join(args.work_dir, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        os.chmod(subfolder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)


### -----------------------------------------------------------   preparing   ---------------------------------------------------------------------- ###
# prepare generator
netG = dcgan.DCGAN_G_Rect_2(patch_size, nz, nc, ngf, ngpu, G_extra_layers)
netG.apply(weights_init)
if netG_path != '':  # load checkpoint if needed
    netG.load_state_dict(torch.load(netG_path))
print(netG)

if device == "cuda:0":
    netG.cuda()


### ---------------------------------------------------------- Generating perturbed scenes -------------------------------------------------------------------- ###
torch.cuda.empty_cache()

# generate patches
patches_seed = torch.FloatTensor(patch_number, nz, 1, 1).normal_(0, 1).cuda()  # 一次生成1张补丁
patches = netG(patches_seed).data  # generated_patch: Tensor, torch.Size([patch_number, 3, patch_size, patch_size])
patches.data = patches.data.mul(0.5).add(0.5)   # shape: torch.Size([patch_number, 3, 64, 64])

# save pt file and visualize results
patches_copy = patches.clone()
for patch_index in range(patch_number):
    # save visualize results
    single_patch_squeeze = patches_copy[patch_index, :, :, :]   # torch.Size([3, 64, 64])
    single_patch_np = single_patch_squeeze.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), BGR
    single_patch_np = (np.ascontiguousarray(single_patch_np) * 255).astype(np.uint8)
    perturbed_image_save_path = os.path.join(args.work_dir, 'fake_patch', f'patch_{patch_index}.jpg')
    cv2.imwrite(perturbed_image_save_path, single_patch_np)
    # save pt file
    saving_patch = patches_copy[patch_index, :, :, :]
    save_pt_patch = os.path.join(args.work_dir, 'patch_pt', f'patch_{patch_index}.pt')
    torch.save(saving_patch, save_pt_patch)





