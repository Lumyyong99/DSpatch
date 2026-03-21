"""
传入fake bounding boxes，image减去eps*data_grad，使得loss减小而达到攻击效果
I-FGSM算法针对单帧的脚本。在给定图片、给定位置的条件下做I-FGSM
生成相应的图片
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
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from shapely.geometry import Polygon

from ensemble_tools.detection_model import init_detector
from mmdet.apis import inference_detector
from ensemble_tools.victim_model_inference import inference_det_loss_on_masked_images
from ensemble_tools.utils0 import zero_out_bounding_boxes_v2, find_patch_positions_v2, find_patch_positions_v3, convert_to_le90
from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description='FGSM creation attack')
parser.add_argument('--work_dir', default=None, help='folder to output files')
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument('--seed', default='2024', type=int, help='choose seed')
# config for fake patch
# parser.add_argument('--custom_fake_object_size', action='store_true', default=False, help='whether to customize fake object size')
# parser.add_argument('--fake_width', default=65.0, type=float, help='fake object width')
# parser.add_argument('--fake_height', default=30.0, type=float, help='fake object height')
# parser.add_argument('--patch_ratio', default=0.8, type=float, help='patch zooming ratio. default=0.6')
# parser.add_argument('--patch_ratio_range', default=0.1, type=float, help='patch zooming ratio. default=0.1. meaning: [patch_ratio-patch_ratio_range, patch_ratio+patch_ratio_range')
# I-FGSM sestting
parser.add_argument('--perturbed_area_size', default=64, type=int, help='area size that adding perturbed')
parser.add_argument('--IFGSM_iteration', default=50, type=int, help='iteration number of I-FGSM.')
parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon of I-FGSM')
parser.add_argument('--alpha', default=0.2, type=float, help='alpha of I-FGSM')

# test setting
parser.add_argument('--test_epoch', default=1, type=int, help='test epoch.')    # 作图的时候test_epoch设成1
# parser.add_argument('--iou_threshold', default=0.7, type=float, help='iou threshold during I-FGSM test')
# parser.add_argument('--conf_threshold', default=0.1, type=float, help='conf threshold during I-FGSM test')



args = parser.parse_args()
# check
args_dict = vars(args)
print('Experimental setting: \n', args_dict)

# victim detector setting
victim_detector = 'RetinaNet-o'
DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_training/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_uavrod_le90.py'
DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_training/detection_model/detectors_640x360/rotated_retinanet/latest.pth'
# scene setting
ContinuousFramesImageFolder = "/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/UAV-ROD-scene50-640x360/PNGImages_640x360"
# ContinuousFramesImageFolder = "./drawing_test/Exp_p2/origin_scene_image"
# ContinuousFramesImageFolder = "./drawing_test/Exp_p2/origin_IFGSM/perturbed_images"

ContinuousFramesLabelFolder = "/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/UAV-ROD-scene50-640x360/txt_annotations_640x360"
# ContinuousFramesLabelFolder = "./drawing_test/Exp_p2/origin_scene_txt_annotations"
# FGSM setting
perturbed_area = args.perturbed_area_size
alpha = args.alpha
epsilon = args.epsilon

# patch setting
# patch_ratio = args.patch_ratio
# patch_ratio_range = args.patch_ratio_range

# image setting
image_file_name = 'DJI_0012_001260.png'
# image_file_name = 'DJI_0005_000540.png'
# image_file_name = 'DJI_0006_012030.png'
# image_file_name = 'DJI_0012_000720.png'
# image_file_name = 'DJI_0012_000960.png'
# image_file_name = 'DJI_0012_001470.png'


# test setting
use_cuda = True
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
image_BatchSize = 1  # 这里固定为1。FGSM每次单独对一张图片进行处理
Patch_per_Image = 4  # 这里固定为1，FGSM中一张图设定为一个fake bounding box
img_size = (384, 640)
patch_apply_range = (360, 640)
iteration_number = args.IFGSM_iteration
test_epoch = args.test_epoch
# test_iou_threshold = args.iou_threshold
# test_conf_threshold = args.conf_threshold

# transformer and applier setting
if torch.cuda.is_available():
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
else:
    patch_transformer = PatchTransformer()
    patch_applier = PatchApplier()

# scene mean and std settings
mean_BGR = [108.969 / 255.0, 107.967 / 255.0, 110.928 / 255.0]  # BGR color for scene dataset
std_BGR = [48.115 / 255.0, 48.588 / 255.0, 47.737 / 255.0]  # BGR color for scene dataset
mean_RGB = [110.928 / 255.0, 107.967 / 255.0, 108.969 / 255.0]  # RGB color for scene dataset
std_RGB = [47.737 / 255.0, 48.588 / 255.0, 48.115 / 255.0]  # RGB color for scene dataset
# normalizer
AdvImagesNormalizer_BGR = transforms.Normalize(mean_BGR, std_BGR)

### -----------------------------------------------------------    Detector    ---------------------------------------------------------------------- ###
start_time = time.time()
model = init_detector(config=DetectorCfgSource, checkpoint=DetectorCheckpoint, device=args.device)  # 这里从init_detector返回的model已经是.eval()模式的
finish_time = time.time()
print(f'Load detector in {finish_time - start_time} seconds.')

### -----------------------------------------------------------    Working folders    ---------------------------------------------------------------------- ###
# # create working files
subfolders = ['perturbed_images', 'visualize_results']
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)
    os.chmod(args.work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
else:
    # 若有同名文件则删除
    # shutil.rmtree(args.work_dir)
    # print('--------------File exists! Existing file with the same name has been removed!--------------')
    print('file exists!')

# for subfolder in subfolders:
#     subfolder_path = os.path.join(args.work_dir, subfolder)
#     if not os.path.exists(subfolder_path):
#         os.makedirs(subfolder_path)
#         os.chmod(subfolder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

# experiment settings log
experimental_settings_file_path = os.path.join(args.work_dir, 'ExperimentSettings.txt')
with open(experimental_settings_file_path, 'w') as f:
    for key, value in args_dict.items():
        f.write(f'{key}={value}\n')
    # victim detector
    f.write('Victim detector:' + victim_detector + '\n')


### -----------------------------------------------------------    Functions    ---------------------------------------------------------------------- ###
def FGSM_example(image, alpha, data_grad):
    """
    generate original FGSM example
    Parameters:
         image(Tensor): input clean image, torch.Size([1, 3, H, W])
         alpha(float): step in each iteration
         data_grad(Tensor): d(Loss)/d(input_image)
    Returns:
        perturbed_image(Tensor): perturbed image, torch.Size([1, 3, H, W])
    """
    sign_data_grad = data_grad.sign()
    grad_processed_image = image - alpha * sign_data_grad
    return grad_processed_image


def FGSM_example_clip(image, data_grad, source_image, alpha, epsilon):
    """
    generate original FGSM example, adding clip function to limit the range of image
    Parameters:
         image (Tensor): input clean image, torch.Size([1, 3, H, W])
         alpha (float): step in each iteration
         data_grad (Tensor): d(Loss)/d(input_image)
         epsilon (float): clip limit for the image
    Returns:
        perturbed_image(Tensor): perturbed image, torch.Size([1, 3, H, W])
    """
    sign_data_grad = data_grad.sign()
    grad_processed_image = image - alpha * sign_data_grad
    # return grad_processed_image
    # 进行clip：Clip(X,epsilon){X'}(x, y, z) = min{1, X(x, y, z}+epsilon, max{0, X(x, y, z)-epsilon, X'(x, y, z)}}
    # 这里clip的值要注意归一化
    mean_rgb_tensor = torch.tensor(mean_RGB).view(1, 3, 1, 1)
    std_rgb_tensor = torch.tensor(std_RGB).view(1, 3, 1, 1)
    zeros_normalize = (torch.zeros_like(image) - mean_rgb_tensor) / std_rgb_tensor
    ones_normalize = (torch.ones_like(image) - mean_rgb_tensor) / std_rgb_tensor
    bound1 = torch.max(zeros_normalize, torch.max(source_image - epsilon, grad_processed_image))
    grad_processed_image_clip = torch.min(ones_normalize, torch.min(image + epsilon, bound1))
    return grad_processed_image_clip


def denorm(normalized_image_batch, mean, std):
    """
    denorm a batch of normalized images. batch images, mean, std should be BGR or RGB at the same time.
    Parameters:
        normalized_image_batch(torch.Tensor): a batch of normalized images. torch.Size([batch_size, 3, img_height, img_width])
        mean(list): mean used for denormalization
        std(list): std used for denormalization
    Returns:
        denormalized_image_batch(torch.Tensor): a batch of denormalized images. torch.Size([batch_size, 3, img_height, img_width])
    """
    for i in range(normalized_image_batch.size(0)):  # 遍历 batch 中的每个图像
        for c in range(normalized_image_batch.size(1)):  # 遍历图像的每个通道
            normalized_image_batch[i, c] = normalized_image_batch[i, c].mul(std[c]).add(mean[c])  # Tensor, torch.Size([batch_size, 3, H, W])
    return normalized_image_batch


def I_FGSM_creation_attack(model, device, scene_bag, fake_positions, fake_labels, epsilon, alpha, num_iter):
    """
    I_FGSM算法测试。迭代多次的FGSM
    对于每张图片都要进行存储以及后续判定
    方法：在一个固定区域内添加扰动，并且基于fake_label和fake_box，计算loss并且生成对应的perturbed image。返回perturbed image
    函数运行一次处理一张图片的数据。scene_bag中仅包含一张图片
    Parameters:
        model (initial detector): victim detector
        device (torch.device): device used in experiments
        scene_bag (dict): 场景包。其中包含B*batch_number个图片的信息，包装成batch_number个dict，每个dict中B张图片
        epsilon (float): epsilon for I-FGSM, I-FGSM中epsilon表示clip的范围
        alpha (float): alpha in I-FGSM, alpha表示更新步长
    Returns:
        perturbed_images_list(list): perturbed images组成的list
    """
    # load image and img_meta from scene_bag
    image_scene = scene_bag['Images_t']  # (1, 3, 384, 640), BGR 格式, 范围(0,1)
    img_metas = scene_bag['imgs_metas']
    # 对img_mask_batch做归一化, BGR->RGB
    image_scene.to(device)
    image_normalize = AdvImagesNormalizer_BGR(image_scene)  # p_img_batch_normalize是BGR格式
    image_normalize_rgb = image_normalize[:, [2, 1, 0], :, :]  # BGR->RGB
    image_normalize_rgb.requires_grad = True

    # prepare source image
    source_image = image_normalize_rgb.clone()

    # 进行I-FGSM攻击
    pseudo_fake_patch = torch.ones((image_BatchSize, Patch_per_Image, 3, perturbed_area, perturbed_area)).to(device)
    for iter_index in range(num_iter):
        # 获取mask
        _, msk_batch = patch_transformer.forward2(adv_batch=pseudo_fake_patch,
                                                  fake_bboxes_batch=fake_positions,
                                                  fake_labels_batch=fake_labels,
                                                  img_size=img_size)  # msk_batch shape: torch.Size([1, patch_per_image, 3, img_height, img_width])

        # 对msk_batch进行融合，将msk融合在一张图上
        msk_batch_melt = torch.sum(msk_batch, dim=1)
        msk_batch_melt = torch.clamp(msk_batch_melt, 0, 1)

        # 计算loss
        # if args.custom_fake_object_size:
        #     fake_positions[:, :, 2] = args.fake_width
        #     fake_positions[:, :, 3] = args.fake_height
        loss = inference_det_loss_on_masked_images(model=model,
                                                   adv_images_batch_t=image_normalize_rgb,
                                                   img_metas=img_metas,
                                                   patch_labels_batch_t=fake_labels_batch_t,
                                                   patch_boxes_batch_t=patch_positions_batch_t)
        model.zero_grad()
        loss.backward()
        image_grad = image_normalize_rgb.grad.data

        # 得到带mask的data.grad
        msk_batch_melt = msk_batch_melt.cpu()
        # print(msk_batch_melt.device, image_grad.device)
        msk_perturbed_batch = torch.where((msk_batch_melt == 0), msk_batch_melt, image_grad)

        # 获取perturbed image
        image_normalize_rgb = FGSM_example_clip(image=image_normalize_rgb,
                                                data_grad=msk_perturbed_batch,
                                                source_image=source_image,
                                                alpha=alpha,
                                                epsilon=epsilon)  # perturbed_image是RGB格式，并且是归一化之后的

        image_normalize_rgb = image_normalize_rgb.detach()
        image_normalize_rgb.requires_grad = True
        # 对perturbed image进行存储
        image_normalize_rgb_copy = image_normalize_rgb.clone()
        image_rgb_copy = denorm(image_normalize_rgb_copy, mean_RGB, std_RGB)  # image_for_save: RGB
        image_rgb_copy_clamp = torch.clamp(image_rgb_copy, 0, 1)

    # 返回perturbed images(RGB，经过反归一化，取值范围为0~1), tensor形式，每一个tensor尺寸为[1, 3, H, W]
    return image_rgb_copy_clamp


### -----------------------------------------------------------    Dataloader    ---------------------------------------------------------------------- ###
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
    labels_batch_list = []
    images_batch_list = []
    img_metas_batch = []
    group_dict = {key: None for key in bag_keys}
    for name in name_bag:
        # name表示一张图片的对应的信息
        coordinates_list_in1img = []
        labels_list_in1img = []
        img_path = os.path.join(ContinuousFramesImageFolder, name + '.png')
        # img_path = 'xxx'
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
                label = list(map(int, elements[9]))
                coordinates_list_in1img.append(coordinate)
                labels_list_in1img.append(label)
        coordinates_batch_list.append(coordinates_list_in1img)
        labels_batch_list.append(labels_list_in1img)
    images_batch_t = torch.stack(images_batch_list)  # (4,3,384,640)
    # 进行mask操作
    # image_mask_batch_t = zero_out_bounding_boxes_v2(images_batch_t, coordinates_batch_list)
    # group_dict['Images_t'] = image_mask_batch_t  # 存储batch_size中经过bounding box mask的场景图片, torch.Size([batch_size, 3, H, W])
    group_dict['Images_t'] = images_batch_t
    group_dict['Annotations'] = coordinates_batch_list  # Annotations为原始图像上的bounding box坐标，为DOTA形式，顺时针四个点八个坐标
    group_dict['Labels'] = labels_batch_list  # Annotations为原始图像上的labels
    # print('coordinates_batch_list:', coordinates_batch_list)
    # print('labels_batch_list:', labels_batch_list)
    # assert False
    group_dict['imgs_metas'] = img_metas_batch  # 存储batch_size张场景图片的相关信息
    # print(f'group_{name_bag_flag}:', group_dict)
    group_dict_list.append(group_dict)
print('Scene Bag Loading Complete!')

# I_FGSM test
total_success_attacks = 0
total_patch_attack = 0
for epoch in tqdm(range(test_epoch)):
    epoch += 1
    img_index = 0
    for scene_bag in group_dict_list:
        scene_bag_image_metas = scene_bag['imgs_metas'][0]
        print(scene_bag_image_metas['ori_filename'], image_file_name)
        if scene_bag_image_metas['ori_filename'] != image_file_name:
            continue
        img_index += 1  # 更新图片编号
        # 为每张图找一个位置
        # 转换image_annotations至le90形式
        scene_bag_annotations = scene_bag['Annotations']
        for scene_bag_annotation in scene_bag_annotations:
            single_image_annotations_le90 = []
            for annotation in scene_bag_annotation:
                annotation_le90 = convert_to_le90(annotation)
                single_image_annotations_le90.append(annotation_le90)  # format: [[cx1, cy1, w1, h1, theta1], [cx2, cy2, w2, h2, theta2], ...]

        # 设置fake bounding box位置
        # patch_positions_list = []
        # for image_index in range(image_BatchSize):
            # patch_position = find_patch_positions_v3(img_size=patch_apply_range,
            #                                          bounding_box=single_image_annotations_le90,
            #                                          patch_size=64,
            #                                          mean_size=patch_ratio,
            #                                          re_size=(-patch_ratio_range, patch_ratio_range),
            #                                          patch_number=Patch_per_Image,
            #                                          iteration_max=1000)
            # patch_positions_list.append(patch_position)
        # patch_positions_batch_t = torch.from_numpy(np.array(patch_positions_list)).cuda()  # (image_BatchSize,Patch_per_Image,5)
        angle1 = 5
        angle2 = 5
        angle3 = 5
        angle4 = 5
        angle1_radius = angle1 * np.pi / 180.0
        angle2_radius = angle2 * np.pi / 180.0
        angle3_radius = angle3 * np.pi / 180.0
        angle4_radius = angle4 * np.pi / 180.0

        patch_ratio1 = 0.8
        patch_ratio2 = 0.8
        patch_ratio3 = 0.8
        patch_ratio4 = 0.8
        
        # patch_positions_batch_t = torch.Tensor([[[45, 239, perturbed_area * patch_ratio1, perturbed_area * patch_ratio1, angle1_radius],
        #                                          [109, 142, perturbed_area * patch_ratio2, perturbed_area * patch_ratio2, angle2_radius],
        #                                          [460, 149, perturbed_area * patch_ratio3, perturbed_area * patch_ratio3, angle3_radius],
        #                                          [470, 35, perturbed_area * patch_ratio4, perturbed_area * patch_ratio4, angle4_radius]]]).cuda()  # torch.Size([1, 1, 5])
        patch_positions_batch_t = torch.Tensor([[[56, 43, perturbed_area * patch_ratio1, perturbed_area * patch_ratio1, angle1_radius],
                                                 [243, 60, perturbed_area * patch_ratio2, perturbed_area * patch_ratio2, angle2_radius],
                                                 [326, 307, perturbed_area * patch_ratio3, perturbed_area * patch_ratio3, angle3_radius],
                                                 [528, 146, perturbed_area * patch_ratio4, perturbed_area * patch_ratio4, angle4_radius]]]).cuda()  # torch.Size([1, 1, 5])
        fake_labels_batch_t = torch.zeros([image_BatchSize, Patch_per_Image, 1]).cuda()  # (image_BatchSize,Patch_per_Image,1)

        perturbed_single_image_rgb = I_FGSM_creation_attack(model=model,
                                                            device=device,
                                                            scene_bag=scene_bag,
                                                            fake_positions=patch_positions_batch_t,
                                                            fake_labels=fake_labels_batch_t,
                                                            epsilon=epsilon,
                                                            alpha=alpha,
                                                            num_iter=iteration_number)

        perturbed_image_rgb = perturbed_single_image_rgb[0, :, :, :]  # (3, 384, 640)    # 选择一张进行存储
        per_img_rgb = perturbed_image_rgb.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
        per_img_bgr = per_img_rgb[:, :, [2, 1, 0]]  # RGB->BGR
        per_img_bgr = (np.ascontiguousarray(per_img_bgr) * 255).astype(np.uint8)

        perturbed_image_save_path = os.path.join(args.work_dir, 'image2.jpg')
        cv2.imwrite(perturbed_image_save_path, per_img_bgr)

        # # 进行定量检测
        # detection_results = inference_detector(model, perturbed_image_save_path)
        # # print('detection_results:', detection_results)
        # conf_thr = 0.01  # 确定一个阈值，
        # detection_results_np = detection_results[0]
        # detection_results_filtered = detection_results_np[detection_results_np[:, -1] > conf_thr]
        # # print('detection_results_filtered:', detection_results_filtered)
        # detection_results_annotations = detection_results_filtered[:, :-1]
        # # print('detection_results_annotations:', detection_results_annotations)
        # detection_results_annotations = [detection_results_annotations]
        # # print('detection_results_annotations_list:', detection_results_annotations)
        # patch_positions_list = patch_positions_batch_t.tolist()
        # # 进行classify
        # single_image_detection_result = detection_results_filtered
        # single_image_real_boxes_le90 = single_image_annotations_le90
        # single_image_fake_boxes_le90 = patch_positions_list[0]

        # detection_real_boxes, detection_fake_boxes, detection_unknown_boxes = classify_boxes(det_boxes=single_image_detection_result,
        #                                                                                      real_boxes=single_image_real_boxes_le90,
        #                                                                                      fake_boxes=single_image_fake_boxes_le90,
        #                                                                                      iou_threshold=0.1)

        # # 计算成功攻击的个数以及放入patch的个数
        # total_patch_attack += len(single_image_fake_boxes_le90)
        # single_image_success_attacks = count_successful_patch_attacks(det_fake_boxes=detection_fake_boxes,
        #                                                               gt_fake_boxes=single_image_fake_boxes_le90,
        #                                                               iou_threshold=test_iou_threshold,
        #                                                               conf_threshold=test_conf_threshold)
        # total_success_attacks += single_image_success_attacks

# # calculate total ASR
# print('total_success_attacks:', total_success_attacks)
# print('total_patch_attack:', total_patch_attack)
# ASR = total_success_attacks / total_patch_attack if total_patch_attack > 0 else 0
# print('ASR=', ASR)

# # save to file
# ASR_result_file_path = os.path.join(args.work_dir, 'ASR.txt')
# with open(ASR_result_file_path, 'w') as f:
#     f.write('ASR=' + str(ASR) + '\n')
