"""
Test for GANpatch
使用训练好的GAN网络生成对抗性补丁，并且将补丁放入场景中。定量化测试我们所提方法的性能
"""
import argparse
from tqdm import tqdm
import os
import stat
import random
import numpy as np
import cv2
import torch

from ensemble_tools.utils0 import zero_out_bounding_boxes_v2

# scene setting
ContinuousFramesImageFolder = "/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/UAV-ROD-scene50-640x360/PNGImages_640x360"
ContinuousFramesLabelFolder = "/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/UAV-ROD-scene50-640x360/txt_annotations_640x360"
scene_image_batch_size = 1
work_dir = '/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/UAV-ROD-scene50-640x360/mask_images_640x360'


### ---------------------------------------------------------- Generating perturbed scenes -------------------------------------------------------------------- ###
# prepare folder
if not os.path.exists(work_dir):
    os.makedirs(work_dir)
    os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

# Reorganize scene dataset
# preparing dataset: continuous frames in UAV-ROD dataset, without mask
zip_list = []
group_dict_list = []  # 用于存储group字典的
img_meta_keys = ['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'to_rgb']
bag_keys = ['Images_t', 'Annotations', 'imgs_metas']
SceneImage_files = os.listdir(ContinuousFramesImageFolder)
SceneAnnotation_files = os.listdir(ContinuousFramesLabelFolder)
common_files = list(set([os.path.splitext(f)[0] for f in SceneImage_files]) & set([os.path.splitext(f)[0] for f in SceneAnnotation_files]))
random.shuffle(common_files)
used_SceneImages_group = len(common_files) // scene_image_batch_size  # 对于全部的scene图片，一共used_SceneImages_group组，每组里面image_BatchSize张图片
for i in range(used_SceneImages_group):
    start_idx = i * scene_image_batch_size
    end_idx = start_idx + scene_image_batch_size
    zip_list.append(common_files[start_idx:end_idx])
# check
# print('zip_list:', zip_list)
name_bag_flag = 0
for name_bag in tqdm(zip_list):
    name_bag_flag += 1
    # name_bag: 包含图像（标签名字）的list，eg: ['DJI_0012_001080', 'DJI_0012_000540', 'DJI_0012_001050', 'DJI_0012_000300', 'DJI_0012_001140', 'DJI_0012_000000', 'DJI_0012_001110', 'DJI_0012_000900']
    coordinates_batch_list = []
    images_batch_list = []
    img_metas_batch = []
    group_dict = {key: None for key in bag_keys}
    for name in name_bag:
        # name表示一张图片的对应的信息
        coordinates_list_in1img = []
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
        img = cv2.copyMakeBorder(img, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)  # (3,height,width)， BGR格式
        images_batch_list.append(img)
        # 读取label文件
        with open(label_path, 'r') as f:
            for line in f:
                elements = line.strip().split()
                coordinate = list(map(int, elements[:8]))
                coordinates_list_in1img.append(coordinate)
        coordinates_batch_list.append(coordinates_list_in1img)
    images_batch_t = torch.stack(images_batch_list)  # (4,3,384,640)
    # 进行mask操作
    image_mask_batch = zero_out_bounding_boxes_v2(images_batch_t, coordinates_batch_list)  # (4,3,384,640), 未经归一化
    # 放进字典里
    group_dict['Images_t'] = images_batch_t  # 存储batch_size中经过bounding box mask的场景图片, torch.Size([batch_size, 3, H, W])
    group_dict['Annotations'] = coordinates_batch_list  # Annotations为原始图像上的bounding box坐标，为DOTA形式，顺时针四个点八个坐标, len = 8
    group_dict['imgs_metas'] = img_metas_batch  # 存储batch_size张场景图片的相关信息，len = 8

    # 对mask图片进行保存
    adv_masked = images_batch_t[0, :, :, :]  # (3, 384, 640)
    adv_img_bgr = adv_masked.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
    adv_img_bgr = (np.ascontiguousarray(adv_img_bgr) * 255).astype(np.uint8)
    # 绘制patch_position边界框
    # patch_positions_batch = coordinates_batch_list[0]
    # for box in patch_positions_batch:
    #     pts = np.array(box, np.int0).reshape((-1, 1, 2))
    #     cv2.polylines(adv_img_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(os.path.join(work_dir, f'mask_image_{name_bag_flag}.jpg'), adv_img_bgr)
    # assert False

    # print(f'group_{name_bag_flag}:', group_dict)
    group_dict_list.append(group_dict)
print('Scene Bag Loading Complete!')

