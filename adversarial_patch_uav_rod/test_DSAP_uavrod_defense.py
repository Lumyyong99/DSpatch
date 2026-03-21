"""
Test for GANpatch
使用训练好的GAN网络生成对抗性补丁，并且将补丁放入场景中。定量化测试我们所提方法的性能
log
2025.1.7
从test_VIDAP文件进行修改，主要是补丁改成矩形
"""
import argparse
from tqdm import tqdm
import os
import stat
import shutil
import random
import time
import numpy as np
import cv2
import torch
from shapely.geometry import Polygon

from ensemble_tools.detection_model import init_detector
from mmdet.apis import inference_detector
from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier
from ensemble_tools.utils0 import find_patch_positions_v4, weights_init, convert_to_le90
import ensemble_tools.GANmodels.dcgan as dcgan

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description='GAN-patch quantitative testing settings')
parser.add_argument('--work_dir', default=None, help='folder to output files')
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument('--seed', default='2025', type=int, help='choose seed')
# config for fake patch
parser.add_argument('--patch_per_image', default=2, type=int, help='number of patch on each scene during training')
# test setting
parser.add_argument('--test_epoch', default=50, type=int, help='test epoch')
parser.add_argument('--iou_threshold', default=0.5, type=float, help='iou threshold during I-FGSM test')
parser.add_argument('--conf_threshold', default=0.5, type=float, help='conf threshold during I-FGSM test')
# victim_model & target_model，victim_detector为受害者模型，source_detector指代针对该模型训练的netG
parser.add_argument('--detector', default='retinanet_o', help='detector')



args = parser.parse_args()

# check
args_dict = vars(args)
print('Experimental settings:\n', args_dict)


### -----------------------------------------------------------    Functions     ---------------------------------------------------------------------- ###
# calculate boxes iou
# 将 le90 形式的旋转矩形框转换为 shapely 的 Polygon
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


def classify_boxes(det_boxes, real_boxes, fake_boxes, iou_threshold=0.5):
    """
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
    real_targets = []
    fake_targets = []
    uncertain_targets = []
    for det_box in det_boxes:
        iou_real_max = 0
        iou_fake_max = 0

        # seperate detection box coordinates and confidence
        det_box_coords = det_box[:5]
        conf = det_box[5]

        # 遍历真实目标框，计算与每个真实目标框的最大IoU
        for real_box in real_boxes:
            iou_real = calculate_rotated_iou(det_box_coords, real_box)
            iou_real_max = max(iou_real_max, iou_real)

        # 遍历伪造目标框，计算与每个伪造目标框的最大IoU
        for fake_box in fake_boxes:
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


def count_successful_patch_attacks(det_fake_boxes, gt_fake_boxes, conf_threshold=0.5, iou_threshold=0.5):
    """
    count success using detecting fake boxes and fake patch annotations. for 1 single image
    Parameters:
        det_fake_boxes (list): detection results, array of [[cx, cy, w, h, theta, conf]]. pay attention: here has parameter Confidence in bounding box
        gt_fake_boxes (list): fake object boxes
        conf_threshold (float): confidence threshold
        iou_threshold (float): iou threshold
    Returns:
        real_targets(list): list of real targets get by detection results
        fake_targets(list): list of fake targets get by detection results
    """
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


### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
# victim detector setting
if args.detector == 'retinanet_o':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/rotated_retinanet_DSAP_defense/rotated_retinanet_obb_r50_fpn_1x_uavrod_le90_defense.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/rotated_retinanet-defense/latest.pth'
elif args.detector == 'faster_rcnn_o':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/rotated_faster_rcnn_DSAP_defense/rotated_faster_rcnn_r50_fpn_1x_uavrod_le90_defense.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/faster_rcnn_o-defense/latest.pth'
elif args.detector == 'gliding_vertex':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/gliding_vertex_DSAP_defense/gliding_vertex_r50_fpn_1x_uavrod_le90_defense.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/gliding_vertex-defense/latest.pth'
elif args.detector == 'oriented_rcnn':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/oriented_rcnn_DSAP_defense/oriented_rcnn_r50_fpn_1x_uavrod_le90_defense.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/oriented_rcnn-defense/latest.pth'
elif args.detector == 'roi_trans':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/roi_trans_DSAP_defense/roi_trans_r50_fpn_1x_uavrod_le90_defense.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/roi_trans-defense/latest.pth'
elif args.detector == 's2anet':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/s2anet_DSAP_defense/s2anet_r50_fpn_1x_uavrod_le90_defense.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/s2anet-defense/latest.pth'
else:
    raise Exception('Wrong Victim Detector!')
detector = args.detector


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
patch_h, patch_w = patch_size[0], patch_size[1]
test_images_total = 50
scene_image_batch_size = 1
patch_per_image = args.patch_per_image  # N
generating_patch_number = test_images_total * patch_per_image  # BN
patch_ratio = 1.0
patch_ratio_range = 0.01

# test setting
device = args.device
test_epoch = args.test_epoch
test_iou_threshold = args.iou_threshold
test_conf_threshold = args.conf_threshold

# GAN setting
ngpu = 1
nz = 100
ngf = 64
nc = 3
G_extra_layers = 0

# scene setting
ContinuousFramesImageFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
ContinuousFramesLabelFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/txt_annotations_640x360"

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

# create working files
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)
    os.chmod(args.work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
else:
    shutil.rmtree(args.work_dir)
    print('--------------File exists! Existing file with the same name has been removed!--------------')

subfolders = ['fake_patch', 'combined_images', 'visualize_results']  # fake_patch用于存储生成的50xPatch_per_Image个补丁
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
    f.write('total image number:' + str(test_images_total) + '\n')


### -----------------------------------------------------------   preparing   ---------------------------------------------------------------------- ###
# prepare generator
netG = dcgan.DCGAN_G_Rect_2_gpu(patch_size, nz, nc, ngf, ngpu, G_extra_layers)
netG.apply(weights_init)
if netG_path != '':  # load checkpoint if needed
    netG.load_state_dict(torch.load(netG_path))
print(netG)

# load to gpu
if device == "cuda:0":
    netG.cuda()
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
else:
    patch_transformer = PatchTransformer()
    patch_applier = PatchApplier()

### ---------------------------------------------------------- Generating perturbed scenes -------------------------------------------------------------------- ###
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
    # 放进字典里
    group_dict['Images_t'] = images_batch_t  # 存储batch_size中经过bounding box mask的场景图片, torch.Size([batch_size, 3, H, W])
    group_dict['Annotations'] = coordinates_batch_list  # Annotations为原始图像上的bounding box坐标，为DOTA形式，顺时针四个点八个坐标, len = 8
    group_dict['imgs_metas'] = img_metas_batch  # 存储batch_size张场景图片的相关信息，len = 8

    '''
    # 对图片和补丁进行可视化测试
    adv_masked = images_batch_t[0, :, :, :]  # (3, 384, 640)
    adv_img_bgr = adv_masked.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
    adv_img_bgr = (np.ascontiguousarray(adv_img_bgr) * 255).astype(np.uint8)
    # 绘制patch_position边界框
    patch_positions_batch = coordinates_batch_list[0]
    for box in patch_positions_batch:
        pts = np.array(box, np.int0).reshape((-1, 1, 2))
        cv2.polylines(adv_img_bgr, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(f'test_unzip_big.jpg', adv_img_bgr)
    assert False
    '''

    # print(f'group_{name_bag_flag}:', group_dict)
    group_dict_list.append(group_dict)
print('Scene Bag Loading Complete!')

### ---------------------------------------------------------- Generating perturbed scenes -------------------------------------------------------------------- ###
torch.cuda.empty_cache()

total_success_attacks = 0
total_patch_attack = 0
for epoch in tqdm(range(test_epoch)):
    epoch += 1
    scene_image_index = 0
    for scene_bag in tqdm(group_dict_list):
        patch_index = 0  # patch-index在每张图中重新开始计数
        scene_image_index += 1
        scene_bag_image = scene_bag['Images_t']  # (1, 3, 384, 640), BGR 格式, 范围(0,1), 仅场景中的一张图
        scene_bag_annotations = scene_bag['Annotations']
        scene_bag_img_metas = scene_bag['imgs_metas']
        # 转换annotations为le90, 转换结果为list，[[[cx1, cy1, w1, h1, theta1], [cx2, cy2, w2, h2, theta2], ...]]
        scene_bag_annotations_le90 = []
        for scene_bag_annotations_batch in scene_bag_annotations:
            single_image_annotations = []
            for annotation in scene_bag_annotations_batch:
                annotation_le90 = convert_to_le90(annotation)
                single_image_annotations.append(annotation_le90)
            scene_bag_annotations_le90.append(single_image_annotations)
        # print('scene_bag_annotations_le90:', scene_bag_annotations_le90)
        # 为每张图生成补丁
        # 每次新生成patch_seeds
        patch_seeds = torch.FloatTensor(patch_per_image, nz, 1, 1).normal_(0, 1).cuda()  # 一次为一张图生成patch_per_image张补丁
        generated_patch = netG(patch_seeds).data  # generated_patch: Tensor, torch.Size([patch_per_image, 3, patch_size, patch_size])
        # patch_position确定
        patch_apply_range = (360, 640)
        single_image_bbox = scene_bag_annotations_le90[0]  # single_image_bbox: 二维list TODO：注意：这里只适用于一张图片的情况！
        patch_positions_list = []
        for image_index in range(scene_image_batch_size):
            patch_position = find_patch_positions_v4(img_size=patch_apply_range, 
                                                     bounding_box=single_image_bbox, 
                                                     patch_size=[patch_w, patch_h],
                                                     mean_size=patch_ratio, 
                                                     re_size=(-patch_ratio_range, patch_ratio_range), 
                                                     patch_number=patch_per_image, iteration_max=1000)
            patch_positions_list.append(patch_position)
        patch_positions_batch_t = torch.from_numpy(np.array(patch_positions_list)).cuda()  # (scene_image_batch_size,patch_per_image,5)
        fake_labels_batch_t = torch.zeros([scene_image_batch_size, patch_per_image, 1]).cuda()  # (scene_image_batch_size,patch_per_image,1)

        """
        print('patch_positions_batch_t:', patch_positions_batch_t)
        # bounding box 与fake positions 检查
        scene_bag_annotations_le90 = torch.from_numpy(np.array(scene_bag_annotations_le90))
        background = torch.ones(1, 3, 384, 640)
        background = background[0, :, :, :]  # (3, 384, 640)
        background = background.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), BGR
        background = (np.ascontiguousarray(background) * 255).astype(np.uint8)
        # 绘制patch_position边界框
        scene_bag_bounding_box = scene_bag_annotations_le90[0, :, :].cpu().numpy().tolist()
        for box in scene_bag_bounding_box:
            rect = ((box[0], box[1]), (box[2], box[3]), 1 * box[4] * 180 / np.pi)  # cv2.boxPoints中顺时针旋转为正
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            cv2.drawContours(background, [rect], 0, (0, 0, 255), 2)
        # load fake bounding boxes
        fake_bounding_box =patch_positions_batch_t[0, :, :].cpu().numpy().tolist()
        for fake_box in fake_bounding_box:
            fake_rect = ((fake_box[0], fake_box[1]), (fake_box[2], fake_box[3]), 1 * fake_box[4] * 180 / np.pi)  # cv2.boxPoints中顺时针旋转为正
            fake_rect = cv2.boxPoints(fake_rect)
            fake_rect = np.int0(fake_rect)
            cv2.drawContours(background, [fake_rect], 0, (0, 255, 0), 2)
        cv2.imwrite(f'bounding_box_test_{scene_image_index}.jpg', background)
        """

        # 对生成的补丁进行apply
        fake_patch = generated_patch.view(scene_image_batch_size, patch_per_image, 3, patch_h, patch_w)
        # 调整fake_patch范围，从(-1,1)调整到(0,1)
        fake_patch.data = fake_patch.data.mul(0.5).add(0.5)
        # 对fake_patch进行存储
        fake_copy = fake_patch.clone()
        fake_copy = fake_copy.view(fake_patch.size(0) * fake_patch.size(1), fake_patch.size(2), fake_patch.size(3), fake_patch.size(4))
        for j in range(fake_copy.size(0)):
            patch_index += 1
            patch_tensor = fake_copy[j]  # torch.Size([3, patch_size, patch_size])
            patch_numpy = patch_tensor.permute(1, 2, 0).cpu().detach().numpy()
            patch_save = (np.ascontiguousarray(patch_numpy) * 255).astype(np.uint8)
            patch_save_path = os.path.join(args.work_dir, 'fake_patch', f'scene{scene_image_index}_patch{patch_index}.jpg')
            cv2.imwrite(patch_save_path, patch_save)

        # 进行apply
        img_size = (384, 640)
        adv_batch_masked, msk_batch = patch_transformer.forward4(adv_batch=fake_patch,
                                                                 ratio=patch_ratio, 
                                                                 fake_bboxes_batch=patch_positions_batch_t,
                                                                 fake_labels_batch=fake_labels_batch_t,
                                                                 img_size=img_size, 
                                                                 transform=False)
        patch_applied_image = patch_applier(scene_bag_image.cuda(), adv_batch_masked.cuda())  # 这里p_img_batch还没做归一化, BGR, torch.Size([scene_image_batch_size, 3, H, W])

        # # 存储patch_applied_image
        patch_applied_image_copy = patch_applied_image.clone()
        image_squeeze = patch_applied_image_copy[0, :, :, :]  # (3, 384, 640)
        image_squeeze_np = image_squeeze.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), BGR
        image_squeeze_np = (np.ascontiguousarray(image_squeeze_np) * 255).astype(np.uint8)
        perturbed_image_save_path = os.path.join(args.work_dir, 'combined_images', f'test_epoch_{epoch}_image_{scene_image_index}.jpg')
        cv2.imwrite(perturbed_image_save_path, image_squeeze_np)


        detection_results = inference_detector(model, perturbed_image_save_path)
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
        single_image_real_boxes_le90 = scene_bag_annotations_le90[0]
        single_image_fake_boxes_le90 = patch_positions_list[0]
        detection_real_boxes, detection_fake_boxes, detection_unknown_boxes = classify_boxes(det_boxes=single_image_detection_result,
                                                                                             real_boxes=single_image_real_boxes_le90,
                                                                                             fake_boxes=single_image_fake_boxes_le90,
                                                                                             iou_threshold=0.1)
        # 计算成功攻击的个数以及放入patch的个数
        total_patch_attack += len(single_image_fake_boxes_le90)
        single_image_success_attacks = count_successful_patch_attacks(det_fake_boxes=detection_fake_boxes,
                                                                      gt_fake_boxes=single_image_fake_boxes_le90,
                                                                      iou_threshold=test_iou_threshold,
                                                                      conf_threshold=test_conf_threshold)
        total_success_attacks += single_image_success_attacks

# calculate total ASR
print('total_success_attacks:', total_success_attacks)
print('total_patch_attack:', total_patch_attack)
ASR = total_success_attacks / total_patch_attack if total_patch_attack > 0 else 0
print('ASR=', ASR)

# save to file
ASR_result_file_path = os.path.join(args.work_dir, 'ASR.txt')
with open(ASR_result_file_path, 'w') as f:
    f.write('ASR=' + str(ASR) + '\n')
