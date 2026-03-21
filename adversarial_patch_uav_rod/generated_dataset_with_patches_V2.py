"""
生成包含patches的训练集场景图片。每张图上在bounding box以外的地方，放N个某种花纹的补丁
训练集场景图片位置：/home/yyx/Adversarial/datasets/UAV-ROD/train_640x360/images
log 2025.1.14
放置一个DSAP补丁的代码，放置矩形补丁。
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
import torchvision.transforms as transforms

from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier
from ensemble_tools.utils0 import find_patch_positions_v3, convert_to_le90
import ensemble_tools.GANmodels.dcgan as dcgan

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description='GAN-patch training settings')
parser.add_argument('--work_dir', default=None, help='folder to save images with patches')
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument('--seed', default='2025', type=int, help='choose seed')
# config for adv patch optimization
parser.add_argument('--gpu', default=1, type=int, help='number of GPUs to use')
# config for fake patch
# parser.add_argument('--patch_ratio', default=1.0, type=float, help='patch zooming ratio. default=0.6')
# parser.add_argument('--patch_ratio_range', default=0.1, type=float, help='patch zooming ratio. default=0.1. meaning: [patch_ratio-patch_ratio_range, patch_ratio+patch_ratio_range')
# # config for training process
parser.add_argument('--patch_per_image', default=4, type=int, help='number of patch on each scene during training')
# transformation setting
parser.add_argument('--min_contrast', default=0.8, type=float, help='minimum contrast ratio boarder in transformation')
parser.add_argument('--max_contrast', default=1.2, type=float, help='maximum contrast ratio boarder in transformation')
parser.add_argument('--min_brightness', default=-0.1, type=float, help='minimum brightness boarder in transformation')
parser.add_argument('--max_brightness', default=0.1, type=float, help='maximum brightness boarder in transformation')
parser.add_argument('--noise_factor', default=0.1, type=float, help='noise factor')
# detector
parser.add_argument('--detector', default='retinanet_o', help='detector')

args = parser.parse_args()

# check
args_dict = vars(args)
print('Experimental settings:\n', args_dict)

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
if args.detector == 'retinanet_o':
    netG_path = '/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_DSAP_rect_patch/multidetectors/DSAP_retinanet_uavrod/checkpoints/netG_epoch1200.pth'
elif args.detector == 'faster_rcnn_o':
    netG_path = '/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_DSAP_rect_patch/multidetectors/DSAP_faster_rcnn_uavrod/checkpoints/netG_epoch1200.pth'
elif args.detector == 'gliding_vertex':
    netG_path = '/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_DSAP_rect_patch/multidetectors/DSAP_gliding_vertex_uavrod/checkpoints/netG_epoch1200.pth'
elif args.detector == 'oriented_rcnn':
    netG_path = '/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_DSAP_rect_patch/multidetectors/DSAP_oriented_rcnn_uavrod/checkpoints/netG_epoch1200.pth'
elif args.detector == 'roi_trans':
    netG_path = '/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_DSAP_rect_patch/multidetectors/DSAP_roitrans_uavrod/checkpoints/netG_epoch1200.pth'
elif args.detector == 's2anet':
    netG_path = '/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_DSAP_rect_patch/multidetectors/DSAP_s2anet_uavrod/checkpoints/netG_epoch1200.pth'
else:
    raise Exception('Wrong target Detector!')

# patch setting
patch_size = [32, 64]
image_BatchSize = 1  # 一次iteration中使用多少张scene图片。在这里确定为batch size=1
Patch_per_Image = args.patch_per_image  # N
# GeneratePatch_batchsize = image_BatchSize * Patch_per_Image  # BN
patch_ratio = 1.0
patch_ratio_range = 0.0 # 这里在创造数据集的时候不对补丁进行长宽比上的放缩

# training setting
device = args.device
n_epochs = 1    # 这里n_epochs定死为1，不要修改
img_size = (360, 640)

# transformation setting
mini_contrast = args.min_contrast
maxi_contrast = args.max_contrast
mini_brightness = args.min_brightness
maxi_brightness = args.max_brightness
noise_fact = args.noise_factor

# GAN setting
ngpu = args.gpu
# beta1 = args.beta1_adam

# scene setting
# ContinuousFramesImageFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/train_640x360/images"     # png 640x360 images
# ContinuousFramesLabelFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/train_640x360/txt_annotations"    # labels. example: [480 340 541 339 542 360 481 361 car 0]
ContinuousFramesImageFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"     # png 640x360 images
ContinuousFramesLabelFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/txt_annotations_640x360"    # labels. example: [480 340 541 339 542 360 481 361 car 0]

### -----------------------------------------------------------    Initialize    ---------------------------------------------------------------------- ###
# set random seed 
Seed = args.seed  # 37564 7777
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

# # create working files
# subfolders = ['combined_images', 'visualize_results', 'patch_visualize', 'patch_pt']
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)
    os.chmod(args.work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
else:
    # 若有同名文件则删除
    shutil.rmtree(args.work_dir)
    print('--------------File exists! Existing file with the same name has been removed!--------------')
    # 删完再新建
    os.makedirs(args.work_dir)
    os.chmod(args.work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    # print('File Exists!')

# for subfolder in subfolders:
#     subfolder_path = os.path.join(args.work_dir, subfolder)
#     if not os.path.exists(subfolder_path):
#         os.makedirs(subfolder_path)
#         os.chmod(subfolder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

### -----------------------------------------------------------   DataLoader   ---------------------------------------------------------------------- ###
# DataLoader
# preparing dataset: continuous frames in UAV-ROD dataset
mean_RGB = [110.928 / 255.0, 107.967 / 255.0, 108.969 / 255.0]  # RGB color for scene dataset
std_RGB = [47.737 / 255.0, 48.588 / 255.0, 48.115 / 255.0]  # RGB color for scene dataset
mean_BGR = [108.969 / 255.0, 107.967 / 255.0, 110.928 / 255.0]  # BGR color for scene dataset
std_BGR = [48.115 / 255.0, 48.588 / 255.0, 47.737 / 255.0]  # BGR color for scene dataset
# ContinuousFramesNormalizer = transforms.Normalize(mean=mean, std=std)
mean_GeneratingPatch_BGR = [0.5, 0.5, 0.5]
std_GeneratingPatch_BGR = [0.5, 0.5, 0.5]

AdvImagesTransformer_BGR = transforms.Normalize(mean_BGR, std_BGR)

# preparing dataset: continuous frames in UAV-ROD dataset
zip_list = []
group_dict_list = []  # 用于存储group字典的
img_meta_keys = ['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'to_rgb']
bag_keys = ['Images_t', 'Annotations', 'imgs_metas']
SceneImage_files = os.listdir(ContinuousFramesImageFolder)
SceneAnnotation_files = os.listdir(ContinuousFramesLabelFolder)
common_files = list(set([os.path.splitext(f)[0] for f in SceneImage_files]) & set([os.path.splitext(f)[0] for f in SceneAnnotation_files]))
random.shuffle(common_files)
used_SceneImages_group = len(common_files) // image_BatchSize  # 对于全部的scene图片，一共used_SceneImages_group组，每组里面image_BatchSize张图片
for i in range(used_SceneImages_group):
    start_idx = i * image_BatchSize
    end_idx = start_idx + image_BatchSize
    zip_list.append(common_files[start_idx:end_idx])
# check
# print('zip_list:', zip_list)
name_bag_flag = 0
for name_bag in tqdm(zip_list):
    name_bag_flag += 1
    # name_bag: 包含图像（标签名字）的list，eg: ['DJI_0012_001080', 'DJI_0012_000540', 'DJI_0012_001050', 'DJI_0012_000300', 'DJI_0012_001140', 'DJI_0012_000000', 'DJI_0012_001110', 'DJI_0012_000900']
    coordinates_batch_list = []
    # labels_batch_list = []
    images_batch_list = []
    img_metas_batch = []
    group_dict = {key: None for key in bag_keys}
    for name in name_bag:
        # name表示一张图片的对应的信息
        coordinates_list_in1img = []
        labels_list_in1img = []
        img_path = os.path.join(ContinuousFramesImageFolder, name + '.png')
        label_path = os.path.join(ContinuousFramesLabelFolder, name + '.txt')
        img_meta = {key: None for key in img_meta_keys}
        # 产生img_metas字典
        img_meta['filename'] = img_path
        img_meta['ori_filename'] = name + '.png'
        img_meta['ori_shape'] = (360, 640, 3)  # before padding
        img_meta['img_shape'] = (360, 640, 3)
        img_meta['pad_shape'] = (384, 640, 3)
        img_meta['scale_factor'] = np.array([1., 1., 1., 1.], dtype=np.float32)
        img_meta['flip'] = False
        img_meta['flip_direction'] = None
        img_meta['img_norm_cfg'] = dict(mean=np.array([110.928, 107.967, 108.969], dtype=np.float32), std=np.array([47.737, 48.588, 48.115], dtype=np.float32))
        img_meta['to_rgb'] = True
        img_metas_batch.append(img_meta)
        # 读取图片文件
        img = cv2.imread(img_path) / 255.0
        # 图片padding
        # img = cv2.copyMakeBorder(img, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)  # (3,height,width)， BGR格式
        images_batch_list.append(img)
        # 读取label文件
        with open(label_path, 'r') as f:
            for line in f:
                elements = line.strip().split()
                coordinate = list(map(int, elements[:8]))
                # label = list(map(int, elements[9]))
                coordinates_list_in1img.append(coordinate)
                # labels_list_in1img.append(label)
        coordinates_batch_list.append(coordinates_list_in1img)
        # labels_batch_list.append(labels_list_in1img)
    images_batch_t = torch.stack(images_batch_list)  # (4,3,384,640)
    # 进行mask操作
    # image_mask_batch_t = zero_out_bounding_boxes_v2(images_batch_t, coordinates_batch_list)
    # 不进行mask
    image_mask_batch_t = images_batch_t
    group_dict['Images_t'] = image_mask_batch_t  # 存储batch_size中经过bounding box mask的场景图片, torch.Size([batch_size, 3, H, W])
    # group_dict['Images_t'] = images_batch_t
    group_dict['Annotations'] = coordinates_batch_list  # Annotations为原始图像上的bounding box坐标，为DOTA形式，顺时针四个点八个坐标
    # group_dict['Labels'] = labels_batch_list  # Annotations为原始图像上的labels
    # print('coordinates_batch_list:', coordinates_batch_list)
    # print('labels_batch_list:', labels_batch_list)
    # assert False
    group_dict['imgs_metas'] = img_metas_batch  # 存储batch_size张场景图片的相关信息
    # print(f'group_{name_bag_flag}:', group_dict)
    group_dict_list.append(group_dict)
print('Scene Bag Loading Complete!')


### ---------------------------------------------------------- Checkpoint & Init -------------------------------------------------------------------- ###
# Training preprocess
torch.cuda.empty_cache()
# generate random patch
# patch = torch.rand(1, 3, rowPatch_size, rowPatch_size).cuda()

# DSAP
netG = dcgan.DCGAN_G_Rect_2_gpu(patch_size, 100, 3, 64, ngpu, 0)
if netG_path != '':  # load checkpoint if needed
    netG.load_state_dict(torch.load(netG_path))
print(netG)
patch_seeds = torch.FloatTensor(1, 100, 1, 1).normal_(0, 1).cuda()  # 一次为一张图生成patch_per_image张补丁
generated_patch = netG(patch_seeds).data  # generated_patch: Tensor, torch.Size([patch_per_image, 3, patch_size, patch_size])
generated_patch.data = generated_patch.data.mul(0.5).add(0.5)

# advpatch
# patch_pt_file = "/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_advpatch_files/final_advpatch_alpha1.0_gamma0.8_VictimDetector_retinanet_o/patch_pt/patch_epoch100.pt"
# patch_origin_tensor = torch.load(patch_pt_file)  # 范围为[0, 1], 
# if patch_origin_tensor.ndim == 4:
#     patch_origin_tensor = patch_origin_tensor.squeeze(0)        # torch.Size([3. 64 ,64])


# init PatchTransformer and PatchApplier
if device == "cuda:0":
    patch_transformer = PatchTransformer(min_contrast=mini_contrast, 
                                         max_contrast=maxi_contrast, 
                                         min_brightness=mini_brightness, 
                                         max_brightness=maxi_brightness, 
                                         noise_factor=noise_fact).cuda()
    patch_applier = PatchApplier().cuda()
else:
    patch_transformer = PatchTransformer(min_contrast=mini_contrast, 
                                         max_contrast=maxi_contrast, 
                                         min_brightness=mini_brightness, 
                                         max_brightness=maxi_brightness, 
                                         noise_factor=noise_fact)
    patch_applier = PatchApplier()

### -----------------------------------------------------------    log file    ---------------------------------------------------------------------- ###
# experiment settings
experimental_settings_file_path = os.path.join(args.work_dir, 'ExperimentSettings.txt')
with open(experimental_settings_file_path, 'w') as f:
    for key, value in args_dict.items():
        f.write(f'{key}={value}\n')

### ---------------------------------------------------------- Training -------------------------------------------------------------------- ###
for scene_bag in tqdm(group_dict_list):     # group_dict_list为长度为1500的，包含
    # optimizer.zero_grad()

    # 一个scene_bag中包含image_batchsize张图片，为这些图片找到合适的位置
    image_mask_batch = scene_bag['Images_t']
    batch_images_annotations = scene_bag['Annotations']        # scene_bag_annotations表示batchsize张图片的坐标
    img_metas = scene_bag['imgs_metas']
    image_file_name = img_metas[0]['ori_filename']
    patch_positions_list = []                                   # patch_positions_list表示一个scene_bag（batch）中的fake positions
    for one_image_annotations in batch_images_annotations:      # one_image_annotations表示一个图里面的annotation
        one_image_annotations_le90 = []
        for annotation in one_image_annotations:            # annotation是指每个单独的bounding box坐标
            annotation_le90 = convert_to_le90(annotation)
            one_image_annotations_le90.append(annotation_le90)  # format: [[cx1, cy1, w1, h1, theta1], [cx2, cy2, w2, h2, theta2], ...]

        # find_patch_positions 函数可以生成patch_number个位置，是针对一张图的
        patch_position = find_patch_positions_v3(img_size=img_size,
                                                    bounding_box=one_image_annotations_le90,
                                                    patch_size=64,
                                                    mean_size=patch_ratio,
                                                    re_size=(-patch_ratio_range, patch_ratio_range),
                                                    patch_number=Patch_per_Image,
                                                    iteration_max=1000)
        patch_positions_list.append(patch_position)

    patch_positions_batch_t = torch.from_numpy(np.array(patch_positions_list)).cuda()  # (image_BatchSize,Patch_per_Image,5)
    fake_labels_batch_t = torch.zeros([image_BatchSize, Patch_per_Image, 1]).cuda()  # (image_BatchSize,Patch_per_Image,1)

    # 对images进行resize
    # fake_patch = patch.view(image_BatchSize, Patch_per_Image, 3, 64, 64)
    fake_patch = generated_patch.repeat(image_BatchSize, Patch_per_Image, 1, 1, 1)      # 对一个补丁进行重复，保证每一张图上都apply相同的补丁

    # 对fake_patch进行范围限制
    fake_patch = torch.clamp(fake_patch, 0, 1)

    # 进行apply
    adv_batch_masked, msk_batch = patch_transformer.forward4(adv_batch=fake_patch, 
                                                             ratio=patch_ratio, 
                                                             fake_bboxes_batch=patch_positions_batch_t, 
                                                             fake_labels_batch=fake_labels_batch_t, 
                                                             img_size=img_size, 
                                                             transform=False)
    p_img_batch = patch_applier(image_mask_batch.cuda(), adv_batch_masked.cuda())  # 这里p_img_batch还没做归一化

    # 对p_img_batch进行归一化
    p_img_batch_normalize = AdvImagesTransformer_BGR(p_img_batch)
    # 转换成RGB，这里转换成rgb其实没有什么作用，因为不需要经过检测器了。这里是为了和后面反归一化的时候对齐
    p_img_batch_normalize = p_img_batch_normalize[:, [2, 1, 0], :, :]  # RGB, torch.Size([batch_size,3, H, W]), 经过scene dataset对应mean, std的归一化


    # 进行p_img_batch_normalize可视化
    p_img_batch_normalize_copy = p_img_batch_normalize.clone()
    # 反归一化
    for i in range(p_img_batch_normalize_copy.size(0)):  # 遍历 batch 中的每个图像
        for c in range(p_img_batch_normalize_copy.size(1)):  # 遍历图像的每个通道
            p_img_batch_normalize_copy[i, c] = p_img_batch_normalize_copy[i, c].mul(std_RGB[c]).add(mean_RGB[c])  # Tensor, torch.Size([batch_size, 3, H, W]), RGB
    # RGB->BGR
    """
    # for pic_index in range(p_img_batch_normalize_copy.size(0)):       # 整个batch的batch_size张图片都进行存储
    #     adv_masked = p_img_batch_normalize_copy[pic_index, :, :, :]  # (3, 384, 640)
    #     adv_img_rgb = adv_masked.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
    #     adv_img_bgr = adv_img_rgb[:, :, [2, 1, 0]]
    #     adv_img_bgr = (np.ascontiguousarray(adv_img_bgr) * 255).astype(np.uint8)
    #     cv2.imwrite(f'vis_geni{gen_iterations}_picindex{pic_index}.jpg', adv_img_bgr)
    """
    adv_masked = p_img_batch_normalize_copy[0, :, :, :]  # (3, 384, 640)    # 选择一张进行存储
    adv_img_rgb = adv_masked.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
    adv_img_bgr = adv_img_rgb[:, :, [2, 1, 0]]
    adv_img_bgr = (np.ascontiguousarray(adv_img_bgr) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.work_dir, image_file_name), adv_img_bgr)
