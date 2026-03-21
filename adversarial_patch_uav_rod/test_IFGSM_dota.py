"""
传入fake bounding boxes，image减去eps*data_grad，使得loss减小而达到攻击效果
对于同一张图，不同alpha下扰动的位置相同，而第一版是不同alpha, 相同图对应的扰动位置也不同
定量测试IFGSM方法。固定某一alpha,epsilon, iteration时候进行测试
log 2025.2.6
从test_IFGSM_uavrod.py修改而来，用于dota数据集的测试
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
from ensemble_tools.utils0 import zero_out_bounding_boxes_v2, find_patch_positions_v4, convert_to_le90
from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description='FGSM creation attack')
parser.add_argument('--work_dir', default=None, help='folder to output files')
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument('--seed', default='2025', type=int, help='choose seed')
# config for fake patch
parser.add_argument('--patch_ratio', default=1.0, type=float, help='default to be 1.0, when class=small-vehicle, patch_ratio=0.8')
parser.add_argument('--fake_class', default='plane', type=str, help='fake class')
# I-FGSM sestting
# parser.add_argument('--perturbed_area_size', default=64, type=int, help='area size that adding perturbed')
parser.add_argument('--ifgsm_iter', default=50, type=int, help='iteration number of I-FGSM.')
parser.add_argument('--epsilon', default=1.0, type=float, help='epsilon of I-FGSM')
parser.add_argument('--alpha', default=0.2, type=float, help='alpha of I-FGSM')
# test setting
parser.add_argument('--test_epoch', default=50, type=int, help='test epoch.')
parser.add_argument('--iou_threshold', default=0.5, type=float, help='iou threshold during I-FGSM test')
parser.add_argument('--conf_threshold', default=0.5, type=float, help='conf threshold during I-FGSM test')
# detector
parser.add_argument('--detector', default='retinanet_o', help='victim detector, select from []')


args = parser.parse_args()
# check
args_dict = vars(args)
print('Experimental setting: \n', args_dict)

# victim detector setting
victim_detector = args.detector
if args.detector == 'retinanet_o':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth'
elif args.detector == 'faster_rcnn_o':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth'
elif args.detector == 'gliding_vertex':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/gliding_vertex_r50_fpn_1x_dota_le90-12e7423c.pth'
elif args.detector == 'oriented_rcnn':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
elif args.detector == 'roi_trans':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth'
elif args.detector == 's2anet':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth'
else:
    raise Exception('Wrong Detector!')

# load patch
class_name = args.fake_class
num_classes = 15
if class_name == 'plane':
    perturbed_area = [48, 64]
elif class_name == 'large-vehicle':
    perturbed_area = [16, 48]
elif class_name == 'ship':
    perturbed_area = [16, 32]
elif class_name == 'helicopter':
    perturbed_area = [16, 64]
elif class_name == 'small-vehicle':
    perturbed_area = [16, 32]

if class_name == 'plane':
    class_i = 0
elif class_name == 'small-vehicle':
    class_i = 4
elif class_name == 'large-vehicle':
    class_i = 5
elif class_name == 'ship':  # 输入尺寸顺序为[h, w]
    class_i = 6
elif class_name == 'helicopter':
    class_i = 14

CLASSES2NUMBER = {'plane':0, 'baseball-diamond':1, 'bridge':2, 'ground-track-field':3,
                  'small-vehicle':4, 'large-vehicle':5, 'ship':6, 'tennis-court':7,
                  'basketball-court':8, 'storage-tank':9, 'soccer-ball-field':10,
                  'roundabout':11, 'harbor':12, 'swimming-pool':13, 'helicopter':14}


# scene setting
ContinuousFramesImageFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-{class_name}-propersize-scene50/images"
ContinuousFramesLabelFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-{class_name}-propersize-scene50/labelTxt"


# FGSM setting
perturbed_h, perturbed_w = perturbed_area[0], perturbed_area[1]
alpha = args.alpha
epsilon = args.epsilon

# patch setting
patch_ratio = args.patch_ratio
patch_ratio_range = 0.0
conf_thr=0.01

# test setting
use_cuda = True
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
scene_image_batch_size = 1  # 这里固定为1。FGSM每次单独对一张图片进行处理
Patch_per_Image = 1
img_size = (512, 512)
patch_apply_range = (512, 512)
iteration_number = args.ifgsm_iter
test_epoch = args.test_epoch
test_iou_threshold = args.iou_threshold
# test_conf_threshold = args.conf_threshold

# transformer and applier setting
if torch.cuda.is_available():
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
else:
    patch_transformer = PatchTransformer()
    patch_applier = PatchApplier()

# scene mean and std settings
mean_RGB = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]  # RGB color for scene dataset
std_RGB = [58.395 / 255.0, 57.12 / 255.0, 57.675 / 255.0]  # RGB color for scene dataset
mean_BGR = [103.53 / 255.0, 116.28 / 255.0, 123.675 / 255.0]  # BGR color for scene dataset
std_BGR = [57.675 / 255.0, 57.12 / 255.0, 58.395 / 255.0]  # BGR color for scene dataset
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
    shutil.rmtree(args.work_dir)
    print('--------------File exists! Existing file with the same name has been removed!--------------')

for subfolder in subfolders:
    subfolder_path = os.path.join(args.work_dir, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        os.chmod(subfolder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

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


def I_FGSM_creation_attack(model, device, scene_bag, fake_positions, position_labels, fake_labels, epsilon, alpha, num_iter):
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

        position_labels: 用于forward函数中的全0 position_labels, 与物体种类无关
        fake_labels: 用于inference过程中，与物体种类有关的labels
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
    pseudo_fake_patch = torch.ones((scene_image_batch_size, Patch_per_Image, 3, perturbed_h, perturbed_w)).to(device)
    for iter_index in range(num_iter):
        # 获取mask
        _, msk_batch = patch_transformer.forward4(adv_batch=pseudo_fake_patch, 
                                                  ratio=patch_ratio, 
                                                  fake_bboxes_batch=fake_positions,
                                                  fake_labels_batch=position_labels,
                                                  img_size=img_size,
                                                  transform=False)  # msk_batch shape: torch.Size([1, patch_per_image, 3, img_height, img_width])

        # 对msk_batch进行融合，将msk融合在一张图上
        msk_batch_melt = torch.sum(msk_batch, dim=1)
        msk_batch_melt = torch.clamp(msk_batch_melt, 0, 1)

        # 计算loss
        loss = inference_det_loss_on_masked_images(model=model,
                                                   adv_images_batch_t=image_normalize_rgb,
                                                   img_metas=img_metas,
                                                   patch_labels_batch_t=fake_labels,
                                                   patch_boxes_batch_t=patch_positions_batch_t,
                                                   model_name=victim_detector)
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


### -----------------------------------------------------------    Quantitative test    ---------------------------------------------------------------------- ###
def le90_to_polygon(box):
    """
    将box转换成Polygon形式
    Parameters:
        box(list): le90形式的box坐标, [cx, cy, w, h, theta], 其中theta为弧度制, 为负表示逆时针旋转,为正表示顺时针旋转
    Returns:
        rotated_box(Polygon): 形式为[x1, y1, x2, y2, x3, y3, x4, y4], 从左上角开始,按顺时针方向排列
    """
    cx, cy, w, h, theta = box
    # 转换为逆时针为正的形式,后面旋转矩阵需要逆时针旋转为正的形式, 但是输入是顺时针旋转为正
    theta = -theta
    # 矩形的四个顶点（相对中心点 (cx, cy) 的坐标）
    box = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    # 构建旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # 旋转并平移到中心点 (cx, cy)
    rotated_box = np.dot(box, rotation_matrix) + np.array([cx, cy])
    # print('rotated_box:', rotated_box)
    # 返回 shapely Polygon 对象
    return Polygon(rotated_box)


# 计算两个旋转矩形的交并比（IoU）
def calculate_rotated_iou(box1, box2):
    """
    calculate iou for single box1 and single box2
    Parameters:
        box1(list): le90 format box annotations
        box2(list): le90 format box annotations
    Returns:
        iou(float): iou of box1 and box2
    """
    polygon1 = le90_to_polygon(box1)
    polygon2 = le90_to_polygon(box2)
    # 计算交集和并集的面积
    if not polygon1.is_valid or not polygon2.is_valid:
        return 0.0
    inter_area = polygon1.intersection(polygon2).area
    union_area = polygon1.union(polygon2).area
    # 计算 IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def classify_boxes_dota_2(det_boxes, real_boxes, fake_boxes, iou_threshold=0.5, class_num=0):
    """
    对dota中某一个单独类别进行统计，输入中包含15个类别
    classify detection results of fake object and real object.
    Parameters:
        det_boxes(np.array): detection results, array of [[cx, cy, w, h, theta, conf]]. pay attention: here has parameter Confidence in bounding box
        real_boxes(list): real boxes, [[cx, cy, w, h, theta]]
        fake_boxes(torch.Tensor): fake object boxes
        iou_threshold(float): iou threshold for classify, 先判断物体是不是属于真实框，然后再判断是不是fake boundingbox
    Returns:
        real_targets(list): list of real targets get by detection results
        fake_targets(list): list of fake targets get by detection results
    """
    cls_det_boxes = det_boxes[class_num]
    cls_real_boxes = real_boxes[class_num]
    cls_fake_boxes = fake_boxes[class_num]


    real_targets = []
    fake_targets = []
    uncertain_targets = []
    for det_box in cls_det_boxes:
        iou_real_max = 0
        iou_fake_max = 0

        # seperate detection box coordinates and confidence
        det_box_coords = det_box[:5]
        conf = det_box[5]

        # 遍历真实目标框，计算与每个真实目标框的最大IoU
        if len(cls_real_boxes) == 0:
            raise Exception('cls_real_boxes = 0')
        for real_box in cls_real_boxes:
            iou_real = calculate_rotated_iou(det_box_coords, real_box)
            iou_real_max = max(iou_real_max, iou_real)

        # 遍历伪造目标框，计算与每个伪造目标框的最大IoU
        if len(cls_fake_boxes) == 0:
            raise Exception('cls_fake_boxes = 0')
        for fake_box in cls_fake_boxes:
            iou_fake = calculate_rotated_iou(det_box_coords, fake_box)
            iou_fake_max = max(iou_fake_max, iou_fake)

        # 分类检测结果框
        det_box_with_conf = det_box_coords.tolist() + [conf]  # det_box转换之前是np.array格式，转换后是list格式
        if iou_real_max >= iou_threshold and iou_real_max > iou_fake_max:
            real_targets.append(det_box_with_conf)  # 属于真实目标
        elif iou_fake_max >= iou_threshold and iou_fake_max > iou_real_max:
            fake_targets.append(det_box_with_conf)  # 属于伪造目标
        else:
            uncertain_targets.append(det_box_with_conf)  # 不确定或没有与任何目标重合的框

    return real_targets, fake_targets, uncertain_targets


def count_successful_patch_attacks_2(det_fake_boxes, gt_fake_boxes, conf_threshold=0.5, iou_threshold=0.5):
    """
    count success using detecting fake boxes and fake patch annotations. for 1 single image
    对一类物体进行统计
    Parameters:
        det_fake_boxes (list): detection results, array of [[cx, cy, w, h, theta, conf]]. pay attention: here has parameter Confidence in bounding box
        gt_fake_boxes (list): fake object boxes
        conf_threshold (float): confidence threshold
        iou_threshold (float): iou threshold
    Returns:
        real_targets(list): list of real targets get by detection results
        fake_targets(list): list of fake targets get by detection results
    """

    # print('\n')
    # print('det_fake_boxes:', det_fake_boxes)
    # print('gt_fake_boxes:', gt_fake_boxes)

    attack_success_number = 0
    for det_fake_box in det_fake_boxes:
        iou_fake_max = 0

        # sperate detection box coordinates and confidence
        det_fake_coords = det_fake_box[:5]
        det_fake_conf = det_fake_box[5]

        # 遍历伪造目标框，计算与每个伪造目标框的最大IoU
        for gt_fake_box in gt_fake_boxes:
            iou_fake = calculate_rotated_iou(det_fake_coords, gt_fake_box)
            iou_fake_max = max(iou_fake_max, iou_fake)

        # 计算attack_success_number
        if iou_fake_max >= iou_threshold and det_fake_conf >= conf_threshold:
            attack_success_number += 1

    # return attack success number in single image
    return attack_success_number


### -----------------------------------------------------------    Dataloader    ---------------------------------------------------------------------- ###
# preparing dataset: continuous frames in UAV-ROD dataset
zip_list = []
group_dict_list = []  # 用于存储group字典的
img_meta_keys = ['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'to_rgb']
bag_keys = ['Images_t', 'Annotations', 'imgs_metas', 'Annotations_classes']
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
    coordinates_batch_list_classes = []
    images_batch_list = []
    img_metas_batch = []
    group_dict = {key: None for key in bag_keys}
    for name in name_bag:
        # name表示一张图片的对应的信息
        coordinates_list_in1img = []
        coordinates_list_in1img_classes = [[] for _ in range(num_classes)]        # 这里需要接受15个类别物体的信息
        img_path = os.path.join(ContinuousFramesImageFolder, name + '.png')
        label_path = os.path.join(ContinuousFramesLabelFolder, name + '.txt')
        img_meta = {key: None for key in img_meta_keys}
        # 产生img_metas字典
        img_meta['filename'] = img_path
        img_meta['ori_filename'] = name + '.png'
        img_meta['ori_shape'] = (512, 512, 3)  # before padding
        img_meta['img_shape'] = (512, 512, 3)
        img_meta['pad_shape'] = (512, 512, 3)
        img_meta['scale_factor'] = np.array([1., 1., 1., 1.], dtype=np.float32)
        img_meta['flip'] = False
        img_meta['flip_direction'] = None
        img_meta['img_norm_cfg'] = dict(mean=np.array([123.675, 116.28, 103.53], dtype=np.float32), std=np.array([58.395, 57.12, 57.375], dtype=np.float32))
        img_meta['to_rgb'] = True
        img_metas_batch.append(img_meta)
        # 读取图片文件
        img = cv2.imread(img_path) / 255.0
        # # 图片padding
        # img = cv2.copyMakeBorder(img, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)  # (3,height,width)， BGR格式
        images_batch_list.append(img)
        # 读取label文件
        with open(label_path, 'r') as f:
            for line in f:
                elements = line.strip().split()
                coordinate = list(map(float, elements[:8]))
                gt_class = elements[8]
                gt_class_num = CLASSES2NUMBER[gt_class]
                coordinates_list_in1img_classes[gt_class_num].append(coordinate)
                coordinates_list_in1img.append(coordinate)
        coordinates_batch_list.append(coordinates_list_in1img)
        coordinates_batch_list_classes.append(coordinates_list_in1img_classes)
    images_batch_t = torch.stack(images_batch_list)  # (4,3,384,640)
    # 添加mask
    image_mask_batch = zero_out_bounding_boxes_v2(images_batch_t, coordinates_batch_list)  # (batchsize,3,512,512), 未经归一化
    # 放进字典里
    group_dict['Images_t'] = image_mask_batch  # 存储batch_size中经过bounding box mask的场景图片, torch.Size([batch_size, 3, H, W])
    group_dict['Annotations'] = coordinates_batch_list  # Annotations为原始图像上的bounding box坐标，为DOTA形式，顺时针四个点八个坐标, len = 8
    group_dict['Annotations_classes'] = coordinates_batch_list_classes  # Annotations为原始图像上的bounding box坐标，为DOTA形式，顺时针四个点八个坐标, len = 8
    group_dict['imgs_metas'] = img_metas_batch  # 存储batch_size张场景图片的相关信息，len = 8
    group_dict_list.append(group_dict)
print('Scene Bag Loading Complete!')

# I_FGSM test
# total_success_attacks_conf07 = 0
total_success_attacks_conf05 = 0
# total_success_attacks_conf03 = 0
# total_success_attacks_conf01 = 0
total_patch_attack = 0
for epoch in tqdm(range(test_epoch)):
    epoch += 1
    img_index = 0
    for scene_bag in group_dict_list:
        img_index += 1  # 更新图片编号

        scene_bag_image = scene_bag['Images_t']  # (1, 3, 384, 640), BGR 格式, 范围(0,1), 仅场景中的一张图
        scene_bag_annotations = scene_bag['Annotations']
        scene_bag_annotations_classes = scene_bag['Annotations_classes']
        scene_bag_img_metas = scene_bag['imgs_metas']

        # 为每张图找一个位置
        # 转换image_annotations至le90形式
        for scene_bag_annotation in scene_bag_annotations:
            single_image_annotations_le90 = []
            for annotation in scene_bag_annotation:
                annotation_le90 = convert_to_le90(annotation)
                single_image_annotations_le90.append(annotation_le90)  # format: [[cx1, cy1, w1, h1, theta1], [cx2, cy2, w2, h2, theta2], ...]

        # 设置fake bounding box位置
        patch_positions_list = []
        for image_index in range(scene_image_batch_size):
            patch_position = find_patch_positions_v4(img_size=patch_apply_range,
                                                     bounding_box=single_image_annotations_le90,
                                                     patch_size=[perturbed_w, perturbed_h],
                                                     mean_size=patch_ratio,
                                                     re_size=(-patch_ratio_range, patch_ratio_range),
                                                     patch_number=Patch_per_Image,
                                                     iteration_max=1000)
            patch_positions_list.append(patch_position)

        
        fake_labels_batch_t_0 = torch.ones([scene_image_batch_size, Patch_per_Image, 1]).cuda()  # (image_BatchSize,Patch_per_Image,1) 
        fake_labels_batch_t = fake_labels_batch_t_0 * class_i
        position_labels_batch_t = fake_labels_batch_t_0 * 0.0     # 训练过程只能针对一类物,该物体的position labels为全0

        patch_positions_batch_t = torch.from_numpy(np.array(patch_positions_list)).cuda()  # (image_BatchSize,Patch_per_Image,5)

        perturbed_single_image_rgb = I_FGSM_creation_attack(model=model,
                                                            device=device,
                                                            scene_bag=scene_bag,
                                                            fake_positions=patch_positions_batch_t,
                                                            position_labels=position_labels_batch_t,
                                                            fake_labels=fake_labels_batch_t,
                                                            epsilon=epsilon,
                                                            alpha=alpha,
                                                            num_iter=iteration_number)

        perturbed_image_rgb = perturbed_single_image_rgb[0, :, :, :]  # (3, 384, 640)    # 选择一张进行存储
        per_img_rgb = perturbed_image_rgb.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
        per_img_bgr = per_img_rgb[:, :, [2, 1, 0]]  # RGB->BGR
        per_img_bgr = (np.ascontiguousarray(per_img_bgr) * 255).astype(np.uint8)
        perturbed_image_save_path = os.path.join(args.work_dir, 'perturbed_images', f'IFGSM_epoch_{epoch}_img{img_index}.jpg')
        cv2.imwrite(perturbed_image_save_path, per_img_bgr)


        # 进行定量检测
        detection_results = inference_detector(model, perturbed_image_save_path)
        
        # 测试：只保留置信度大于0.3的检测框
        det_results = []
        for class_idx in range(num_classes):
            class_dets_np = detection_results[class_idx]        # class_detection_results_np
            if len(class_dets_np) != 0:
                dets_filtered = class_dets_np[class_dets_np[:, -1] > conf_thr]
                dets_annotations = dets_filtered
                det_results.append(dets_annotations)
            else:
                empty = np.empty((0,6),dtype=np.float32)
                det_results.append(empty)
        # check
        print('filtered detection results:', det_results)
        # assert False

        # real_boxes
        scene_bag_annotations_img = scene_bag_annotations_classes[0]  # TODO：注意：这里只适用于一张图片的情况！
        classes_annotations_le90 = []
        for class_annotations in scene_bag_annotations_img:
            single_image_annotations = []
            if len(class_annotations) != 0:
                for annotation in class_annotations:
                    annotation_le90 = convert_to_le90(annotation)
                    single_image_annotations.append(annotation_le90)
            classes_annotations_le90.append(single_image_annotations)
        # check
        print('real_class_annotations_batch_classes:', classes_annotations_le90)
        # assert False

        # fake_boxes
        fake_positions = [[] for _ in range(num_classes)]
        single_image_fake_boxes_le90 = patch_positions_list[0]      # 注意这里只能是一张图情况下使用
        fake_positions[class_i] = single_image_fake_boxes_le90
        # check
        print('fake_positions_classes:', fake_positions)
        
        assert False




        """
        # print('detection_results:', detection_results)
        conf_thr = 0.01  # 确定一个阈值，
        detection_results_np = detection_results[0]
        detection_results_filtered = detection_results_np[detection_results_np[:, -1] > conf_thr]
        # print('detection_results_filtered:', detection_results_filtered)
        detection_results_annotations = detection_results_filtered[:, :-1]
        # print('detection_results_annotations:', detection_results_annotations)
        detection_results_annotations = [detection_results_annotations]
        # print('detection_results_annotations_list:', detection_results_annotations)
        patch_positions_list = patch_positions_batch_t.tolist()
        # 进行classify
        single_image_detection_result = detection_results_filtered
        single_image_real_boxes_le90 = single_image_annotations_le90
        single_image_fake_boxes_le90 = patch_positions_list[0]

        detection_real_boxes, detection_fake_boxes, detection_unknown_boxes = classify_boxes(det_boxes=single_image_detection_result,
                                                                                             real_boxes=single_image_real_boxes_le90,
                                                                                             fake_boxes=single_image_fake_boxes_le90,
                                                                                             iou_threshold=0.1)

        # 计算成功攻击的个数以及放入patch的个数,ASRx一起进行计算
        total_patch_attack += len(single_image_fake_boxes_le90)
        # cal ASR0.7
        # single_image_success_attacks_conf07 = count_successful_patch_attacks(det_fake_boxes=detection_fake_boxes,
        #                                                                      gt_fake_boxes=single_image_fake_boxes_le90,
        #                                                                      iou_threshold=test_iou_threshold,
        #                                                                      conf_threshold=0.7)
        # total_success_attacks_conf07 += single_image_success_attacks_conf07

        # cal ASR0.5
        single_image_success_attacks_conf05 = count_successful_patch_attacks(det_fake_boxes=detection_fake_boxes,
                                                                             gt_fake_boxes=single_image_fake_boxes_le90,
                                                                             iou_threshold=test_iou_threshold,
                                                                             conf_threshold=0.5)
        total_success_attacks_conf05 += single_image_success_attacks_conf05

        # cal ASR0.3
        # single_image_success_attacks_conf03 = count_successful_patch_attacks(det_fake_boxes=detection_fake_boxes,
        #                                                                      gt_fake_boxes=single_image_fake_boxes_le90,
        #                                                                      iou_threshold=test_iou_threshold,
        #                                                                      conf_threshold=0.3)
        # total_success_attacks_conf03 += single_image_success_attacks_conf03

        # cal ASR0.1
        # single_image_success_attacks_conf01 = count_successful_patch_attacks(det_fake_boxes=detection_fake_boxes,
        #                                                                      gt_fake_boxes=single_image_fake_boxes_le90,
        #                                                                      iou_threshold=test_iou_threshold,
        #                                                                      conf_threshold=0.1)
        # total_success_attacks_conf01 += single_image_success_attacks_conf01

        """

# calculate total ASR
# print('conf0.7_total_success_attacks:', total_success_attacks_conf07)
print('conf0.5_total_success_attacks:', total_success_attacks_conf05)
# print('conf0.3_total_success_attacks:', total_success_attacks_conf03)
# print('conf0.1_total_success_attacks:', total_success_attacks_conf01)
print('total_patch_attack:', total_patch_attack)
# ASR07 = total_success_attacks_conf07 / total_patch_attack if total_patch_attack > 0 else 0
ASR05 = total_success_attacks_conf05 / total_patch_attack if total_patch_attack > 0 else 0
# ASR03 = total_success_attacks_conf03 / total_patch_attack if total_patch_attack > 0 else 0
# ASR01 = total_success_attacks_conf01 / total_patch_attack if total_patch_attack > 0 else 0
# print('ASR0.7=', ASR07)
print('ASR0.5=', ASR05)
# print('ASR0.3=', ASR03)
# print('ASR0.1=', ASR01)

# save to file
ASR_result_file_path = os.path.join(args.work_dir, 'ASR.txt')
with open(ASR_result_file_path, 'w') as f:
    # f.write('ASR0.7=' + str(ASR07) + '\n')
    f.write('ASR0.5=' + str(ASR05) + '\n')
    # f.write('ASR0.3=' + str(ASR03) + '\n')
    # f.write('ASR0.1=' + str(ASR01) + '\n')
