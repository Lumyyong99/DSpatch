"""
adv.patch optimization
Optimize a patch in traditional way.
log 2025.1.8
生成矩形[32, 64]尺寸的advpatch，用于DSAP文章中的实验
log 2025.2.8
为室内实验做准备，针对uavrod数据集优化出能够在物理世界中使用的advpatch，具体方式为添加对比度、亮度、尺寸等的变换
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
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn

from ensemble_tools.detection_model import init_detector
from ensemble_tools.victim_model_inference import inference_det_loss_on_masked_images
from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier, TotalVariation  # 这里选择load_data_1中的函数
from ensemble_tools.utils0 import find_patch_positions_v4, zero_out_bounding_boxes_v2, convert_to_le90

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description='GAN-patch training settings')
parser.add_argument('--work_dir', default=None, help='folder to output files')
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument('--seed', default='2025', type=int, help='choose seed')
# config for adv patch optimization
parser.add_argument('--gpu', default=1, type=int, help='number of GPUs to use')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate of Generator')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--beta1_adam', default=0.5, type=float, help='beta1 parameter for adam')
# config for training process
parser.add_argument('--image_batchsize', default=4, type=int, help='load image_batchsize images in one iteration')
parser.add_argument('--patch_per_image', default=8, type=int, help='number of patch on each scene during training')
parser.add_argument('--epochs', default=100, type=int, help='total epoch number for training process')
parser.add_argument('--start_epoch', default=1, type=int, help='starting epoch number for training')
parser.add_argument('--save_interval', default=5, type=int, help='saving epoch after save_interval epochs')
parser.add_argument('--img_format', default='png', type=str, help='choose a image format, from [png, jpg]. Keep consistent with training dataset')
# victim detector
parser.add_argument('--detector', default='retinanet_o', type=str, help='victim detector. choose from [retinanet_o, faster_rcnn_o, gliding_vertex, oriented_rcnn, roi_trans, s2anet]')
# loss weight setting
parser.add_argument('--alpha', default=1.0, type=float, help='loss weight: detection loss loss_det')
parser.add_argument('--gamma', default=0.8, type=float, help='loss weight: tv_loss')
# transformation setting
parser.add_argument('--min_contrast', default=0.8, type=float, help='minimum contrast ratio boarder in transformation')
parser.add_argument('--max_contrast', default=1.2, type=float, help='maximum contrast ratio boarder in transformation')
parser.add_argument('--min_brightness', default=-0.1, type=float, help='minimum brightness boarder in transformation')
parser.add_argument('--max_brightness', default=0.1, type=float, help='maximum brightness boarder in transformation')
parser.add_argument('--noise_factor', default=0.1, type=float, help='noise factor')

args = parser.parse_args()

# check
args_dict = vars(args)
print('Experimental settings:\n', args_dict)

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
# victim detector setting
if args.detector == 'retinanet_o':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_uavrod_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/rotated_retinanet/latest.pth'
elif args.detector == 'faster_rcnn_o':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_uavrod_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/faster_rcnn_o/latest.pth'
elif args.detector == 'gliding_vertex':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/gliding_vertex/gliding_vertex_r50_fpn_1x_uavrod_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/gliding_vertex/latest.pth'
elif args.detector == 'oriented_rcnn':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_uavrod_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/oriented_rcnn/latest.pth'
elif args.detector == 'roi_trans':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/roi_trans/roi_trans_r50_fpn_1x_uavrod_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/roi_trans/latest.pth'
elif args.detector == 's2anet':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/s2anet/s2anet_r50_fpn_1x_uavrod_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/s2anet/latest.pth'
else:
    raise Exception('Wrong Detector!')

# patch setting
patch_size = [32, 64]
patch_h, patch_w = patch_size[0], patch_size[1]
image_BatchSize = args.image_batchsize  # 一次iteration中使用多少张scene图片，公式中的B
Patch_per_Image = args.patch_per_image  # N
GeneratePatch_batchsize = image_BatchSize * Patch_per_Image  # BN
patch_ratio = 1.0
patch_ratio_range = 0.1     # 物理域实验，这里patch_ratio_range改成0.1

# training setting
device = args.device
n_epochs = args.epochs
start_epoch = args.start_epoch
lr = args.lr
epoch_save = args.save_interval
img_format = '.' + args.img_format


# transformation setting
mini_contrast = args.min_contrast
maxi_contrast = args.max_contrast
mini_brightness = args.min_brightness
maxi_brightness = args.max_brightness
noise_fact = args.noise_factor

# GAN setting
ngpu = args.gpu
beta1 = args.beta1_adam

# scene setting
# ContinuousFramesImageFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
# ContinuousFramesLabelFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/txt_annotations_640x360"
ContinuousFramesImageFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/train_640x360/images"
ContinuousFramesLabelFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/train_640x360/txt_annotations"

### -----------------------------------------------------------    Detector    ---------------------------------------------------------------------- ###
start_time = time.time()
model = init_detector(config=DetectorCfgSource, checkpoint=DetectorCheckpoint, device=args.device)  # 这里从init_detector返回的model已经是.eval()模式的
finish_time = time.time()
print(f'Load detector in {finish_time - start_time} seconds.')

### -----------------------------------------------------------    Initialize    ---------------------------------------------------------------------- ###
# set random seed 
Seed = args.seed  # 37564 7777
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

# # create working files
subfolders = ['combined_images', 'visualize_results', 'patch_visualize', 'patch_pt']
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)
    os.chmod(args.work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
else:
    # 若有同名文件则删除
    shutil.rmtree(args.work_dir)
    print('--------------File exists! Existing file with the same name has been removed!--------------')

for subfolder in subfolders:
    subfolder_path = os.path.join(args.work_dir, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        os.chmod(subfolder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

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
        img_path = os.path.join(ContinuousFramesImageFolder, name + img_format)
        label_path = os.path.join(ContinuousFramesLabelFolder, name + '.txt')
        img_meta = {key: None for key in img_meta_keys}
        # 产生img_metas字典
        img_meta['filename'] = img_path
        img_meta['ori_filename'] = name + img_format
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
        # 判断图片是否需要进行padding，并执行padding
        scene_img_height = img.shape[0]
        if scene_img_height == 360:
            img = cv2.copyMakeBorder(img, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])    #将360扩成384
        elif scene_img_height == 384:
            img = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])    #不进行扩充
        else:
            raise Exception('Image Size Error! Image Size Should Be [360, 640] or [384, 640]')
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
    image_mask_batch_t = zero_out_bounding_boxes_v2(images_batch_t, coordinates_batch_list)
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
patch = torch.rand(1, 3, patch_h, patch_w).cuda()
patch.requires_grad_(True)

# create optimizer for GAN network
if args.adam:
    optimizer = optim.Adam([patch], lr=lr, betas=(beta1, 0.999))
else:
    optimizer = optim.RMSprop([patch], lr=lr)

# init PatchTransformer and PatchApplier
if device == "cuda:0":
    patch_transformer = PatchTransformer(min_contrast=mini_contrast, 
                                         max_contrast=maxi_contrast, 
                                         min_brightness=mini_brightness, 
                                         max_brightness=maxi_brightness, 
                                         noise_factor=noise_fact
                                        ).cuda()
    patch_applier = PatchApplier().cuda()
    total_variation = TotalVariation().cuda()
else:
    patch_transformer = PatchTransformer(min_contrast=mini_contrast, 
                                         max_contrast=maxi_contrast, 
                                         min_brightness=mini_brightness, 
                                         max_brightness=maxi_brightness, 
                                         noise_factor=noise_fact)
    patch_applier = PatchApplier()
    total_variation = TotalVariation()

### -----------------------------------------------------------    log file    ---------------------------------------------------------------------- ###
# experiment settings
experimental_settings_file_path = os.path.join(args.work_dir, 'ExperimentSettings.txt')
with open(experimental_settings_file_path, 'w') as f:
    for key, value in args_dict.items():
        f.write(f'{key}={value}\n')

# log settings
log_file_path = os.path.join(args.work_dir, 'logger.txt')
log_file = open(log_file_path, 'a')

### ---------------------------------------------------------- Training -------------------------------------------------------------------- ###
for epoch in range(n_epochs):
    epoch += 1
    ep_loss_det = 0
    ep_loss_tv = 0
    for scene_bag in tqdm(group_dict_list):
        optimizer.zero_grad()
        # 一个scene_bag中包含image_batchsize张图片，为这些图片找到合适的位置
        image_mask_batch = scene_bag['Images_t']
        batch_images_annotations = scene_bag['Annotations']        # scene_bag_annotations表示batchsize张图片的坐标
        img_metas = scene_bag['imgs_metas']
        patch_positions_list = []                                   # patch_positions_list表示一个scene_bag（batch）中的fake positions
        for one_image_annotations in batch_images_annotations:      # one_image_annotations表示一个图里面的annotation
            one_image_annotations_le90 = []
            for annotation in one_image_annotations:            # annotation是指每个单独的bounding box坐标
                annotation_le90 = convert_to_le90(annotation)
                one_image_annotations_le90.append(annotation_le90)  # format: [[cx1, cy1, w1, h1, theta1], [cx2, cy2, w2, h2, theta2], ...]

            # find_patch_positions 函数可以生成patch_number个位置，是针对一张图的
            patch_location_range = (360, 640)       # z这里意思是在范围(360, 640)内选择位置
            patch_position = find_patch_positions_v4(img_size=patch_location_range,
                                                     bounding_box=one_image_annotations_le90,
                                                     patch_size=[patch_w, patch_h],
                                                     mean_size=patch_ratio,
                                                     re_size=(-patch_ratio_range, patch_ratio_range),
                                                     patch_number=Patch_per_Image,
                                                     iteration_max=1000)
            patch_positions_list.append(patch_position)

        patch_positions_batch_t = torch.from_numpy(np.array(patch_positions_list)).cuda()  # (image_BatchSize,Patch_per_Image,5)
        fake_labels_batch_t = torch.zeros([image_BatchSize, Patch_per_Image, 1]).cuda()  # (image_BatchSize,Patch_per_Image,1)

        # 对images进行resize
        # fake_patch = patch.view(image_BatchSize, Patch_per_Image, 3, 64, 64)
        fake_patch = patch.repeat(image_BatchSize, Patch_per_Image, 1, 1, 1)

        # 对fake_patch进行范围限制
        fake_patch = torch.clamp(fake_patch, 0, 1)

        # 进行apply
        img_size = (384, 640)       # padding后的img_size
        adv_batch_masked, msk_batch = patch_transformer.forward4(adv_batch=fake_patch, 
                                                                 ratio=patch_ratio, 
                                                                 fake_bboxes_batch=patch_positions_batch_t, 
                                                                 fake_labels_batch=fake_labels_batch_t, 
                                                                 img_size=img_size, 
                                                                 transform=True)
        p_img_batch = patch_applier(image_mask_batch.cuda(), adv_batch_masked.cuda())  # 这里p_img_batch还没做归一化

        # 对p_img_batch进行归一化
        p_img_batch_normalize = AdvImagesTransformer_BGR(p_img_batch)
        # 转换成RGB
        p_img_batch_normalize = p_img_batch_normalize[:, [2, 1, 0], :, :]  # RGB, torch.Size([batch_size,3, H, W]), 经过scene dataset对应mean, std的归一化

        # 计算detection loss
        loss_det = inference_det_loss_on_masked_images(model=model, 
                                                       adv_images_batch_t=p_img_batch_normalize, 
                                                       img_metas=img_metas, 
                                                       patch_labels_batch_t=fake_labels_batch_t, 
                                                       patch_boxes_batch_t=patch_positions_batch_t, 
                                                       model_name=args.detector)
        ep_loss_det += loss_det.item()
        # 计算tv loss
        loss_tv = total_variation(fake_patch)
        ep_loss_tv += loss_tv.item()

        # update
        loss = args.alpha * loss_det + args.gamma * loss_tv

        loss.backward()
        optimizer.step()

        if epoch % epoch_save == 0:
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
            cv2.imwrite(os.path.join(args.work_dir, 'combined_images', f'vis_epoch{epoch}.jpg'), adv_img_bgr)

            # 进行patch进行存储
            patch_copy = patch.clone()  # patch_copy shape: torch.Size([1,3,patch_size, patch_size])
            # 存储成pt文件格式
            patch_pt_save_path = os.path.join(args.work_dir, 'patch_pt', f'patch_epoch{epoch}.pt')
            torch.save(patch_copy, patch_pt_save_path)
            # 进行可视化，存储成.jpg文件格式
            patch_copy_numpy = patch_copy.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
            patch_copy_numpy = (np.ascontiguousarray(patch_copy_numpy) * 255).astype(np.uint8)
            patch_vis_save_path = os.path.join(args.work_dir, 'patch_visualize', f'patch_epoch{epoch}.jpg')
            cv2.imwrite(patch_vis_save_path, patch_copy_numpy)

    ep_loss_det = ep_loss_det / len(group_dict_list)
    ep_loss_tv = ep_loss_tv / len(group_dict_list)
    epoch_logger = f'[{epoch}/{n_epochs}] loss_det: {ep_loss_det} loss_tv: {ep_loss_tv}'
    print(epoch_logger)
    log_file.write(epoch_logger + '\n')
    log_file.flush()
