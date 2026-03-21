# -*- coding: utf-8 -*-
"""
自动化放置最隐蔽的补丁，第二版本，给定图像合指定的位置，多次（>1000次）生成补丁并放在指定位置上，选择与原图LPIPS最小的一个
多进程处理，一次处理一张图片，四个固定位置
相当于就按照固定位置选择合适的补丁
log 2025.1.9
在AutomaticSelectPatch_2基础上进行修改,不使用多线程,减少io与度量器的新建,进行批处理。
log 2025.1.10
在AutomaticSelectPatch_3基础上进行修改,将3中原本处理一张图片的函数改成处理多张图片的函数。共处理50张图片，fake bounding box来源于uavrod_selected_patch_locations_scene50.json
AutomaticSelectPatch_4由AutomaticSelectPatch_3复制而来，添加筛选random, advpatch和IFGSM补丁的函数。还包括多检测器的

"""

import cv2
import os
import io
import stat
import random
from tqdm import tqdm
import time
import numpy as np
import torch
import shutil
import json
from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier
import ensemble_tools.GANmodels.dcgan as dcgan
import contextlib
from tqdm import tqdm
from ipdb import set_trace as st


# 测试IFGSM时将该部分注释掉
import pyiqa

# 测试IFGSM以外算法时将该部分注释掉
# from ensemble_tools.detection_model import init_detector
# from ensemble_tools.victim_model_inference import inference_det_loss_on_masked_images
# from torchvision import transforms

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

class MetricCalculator:
    def __init__(self, device):
        self.device = device
        with contextlib.redirect_stdout(io.StringIO()):
            self.lpips_metric = pyiqa.create_metric('lpips', device=device)
            self.psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, crop_border=4, color_space='ycbcr')
            self.ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, crop_border=4, color_space='ycbcr')
    
    def calculate_metrics(self, img_tensor1, img_tensor2):
        """直接计算张量之间的度量，避免文件IO"""
        lpips = self.lpips_metric(img_tensor1, img_tensor2)
        psnr = self.psnr_metric(img_tensor1, img_tensor2)
        ssim = self.ssim_metric(img_tensor1, img_tensor2)
        return {'lpips': lpips.item(), 'psnr': psnr.item(), 'ssim': ssim.item()}


def load_patches_from_json(file_path):
    with open(file_path, 'r') as f:
        patches_data = json.load(f)
    return patches_data


def find_patch_optimized(fake_position_t, patch_ratio, patch_h, patch_w,
                        total_generated_patches_num, iter_num,
                        device, patch_transformer, patch_applier,
                        generator, fake_labels_t, img_size, origin_image_tensor,
                        metric_calculator, init_lpips):
    """优化后的补丁查找函数"""
    # 批量生成补丁
    noise = torch.FloatTensor(total_generated_patches_num, 100, 1, 1).normal_(0, 1).to(device)

    with torch.no_grad():  # 添加no_grad上下文
        fake_patch_batch = generator(noise)
        fake_patch_batch = fake_patch_batch.mul(0.5).add(0.5)

    # 计算方差并选择补丁
    var_per_sample = torch.var(fake_patch_batch, dim=[1, 2, 3])
    _, indices = torch.topk(var_per_sample, iter_num, largest=False)
    selected_fake_patch_batch = fake_patch_batch[indices]

    # 初始化
    best_lpips = init_lpips
    best_mask = None
    
    # 批处理大小
    batch_size = 128  # 可以根据GPU内存调整
    num_batches = (iter_num + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, iter_num)
        current_batch_size = end_idx - start_idx        # current batch size: 32, 最后一个不一定是32

        # 处理当前批次的补丁
        current_patches = selected_fake_patch_batch[start_idx:end_idx].to(device)
        current_patches = current_patches.view(current_batch_size, 1, 3, patch_h, patch_w)

        # 批量转换补丁
        adv_batch_masked, _ = patch_transformer.forward4(
            adv_batch=current_patches,
            ratio=patch_ratio,
            fake_bboxes_batch=fake_position_t.repeat(current_batch_size, 1, 1),
            fake_labels_batch=fake_labels_t.repeat(current_batch_size, 1, 1),
            img_size=img_size,
            transform=False
        )

        # 批量应用补丁
        origin_batch = origin_image_tensor.repeat(current_batch_size, 1, 1, 1)
        patch_applied_images = patch_applier(origin_batch, adv_batch_masked)

        # 批量计算LPIPS
        for i in range(current_batch_size):
            metrics = metric_calculator.calculate_metrics(
                patch_applied_images[i:i+1],
                origin_image_tensor
            )
            
            if metrics['lpips'] < best_lpips:
                best_lpips = metrics['lpips']
                best_mask = adv_batch_masked[i:i+1].clone()
                best_patch = current_patches[i:i+1].clone()

        torch.cuda.empty_cache()

    # print(f"Select Complete! Min LPIPS = {best_lpips}")
    return best_mask, best_patch

def test_one_image():
    """
    该函数中坐标是框架图Fig2中的五个坐标
    """
    # 新建临时存储文件夹
    temp_work_dir = './vis_results_DSAP/select_diverse_stealthy_patches/fig1'
    if not os.path.exists(temp_work_dir):
        os.makedirs(temp_work_dir)
        os.chmod(temp_work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        shutil.rmtree(temp_work_dir)
        print('--------------File exists! Existing file with the same name has been removed!--------------')    # 先删除再新建
        os.makedirs(temp_work_dir)
        os.chmod(temp_work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # 初始化
    # original_image_path = '../../datasets/UAV-ROD/train_640x360/images/DJI_0006_006210.png'
    generator_checkpoint = './train_DSAP_uavrod/multidetectors/retinanet_r50/checkpoints/netG_epoch1200.pth'
    # response中
    original_image_path = '../../datasets/UAV-ROD/train_640x360/images/DJI_0012_001260.png'
    # generator_checkpoint = './train_DSAP_uavrod/train_NewSourceDomain_more_source_images_20250626/checkpoints/netG_epoch1200.pth'

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [32, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    # patch_ratio = 0.8
    patch_number = 2

    total_generated_patches_number = 12000        # 使用generator一次获得total_generated_patches_number张补丁，然后在这total_generated_patches_number张补丁中选择
    # selecting iteration number setting
    iter_num = 12000
    # other settings
    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    

    ### -----------------------------------------------------------    load images and generator    ---------------------------------------------------------------------- ###
    # prepare scene from file path
    origin_image = cv2.imread(original_image_path)  # BGR格式
    origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
    # 不做归一化，因为不需要过检测器
    origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
    origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])
    # load generator
    generator = dcgan.DCGAN_G_Rect_2_gpu(generate_patch_size, 100, 3, 64, 1, 0)
    # generator.apply(weights_init)
    generator.load_state_dict(torch.load(generator_checkpoint,map_location=generator.device))

    # TODO: 不要删除！这里的五个坐标是绘制Fig2图像adversarial scene里面的坐标，后续有可能还需要使用！
    # original_image_path = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360/DJI_0012_001260.png"
    # applying patch settings
    # 5 patches
    # angles = [-25, -25, -25, 95, 95]
    # angle1_radius, angle2_radius, angle3_radius, angle4_radius, angle5_radius = angles[0] * np.pi / 180.0, angles[1] * np.pi / 180.0, angles[2] * np.pi / 180.0, angles[3] * np.pi / 180.0, angles[4] * np.pi / 180.0
    # fake_positions_t = torch.Tensor([[[349, 47, generate_patch_w * ratios, generate_patch_h * ratios, angle1_radius],
    #                                   [335, 171, generate_patch_w * ratios, generate_patch_h * ratios, angle2_radius],
    #                                   [323, 310, generate_patch_w * ratios, generate_patch_h * ratios, angle3_radius],
    #                                   [434, 92, generate_patch_w * ratios, generate_patch_h * ratios, angle4_radius], 
    #                                   [414, 317, generate_patch_w * ratios, generate_patch_h * ratios, angle5_radius]]]).cuda()  # torch.Size([1, 1, 5]) 这个坐标不要轻易改！因为是新fig2中的五个位置的坐标
    
    # load position
    positions = [
        {'center': (326, 307), 'angle': -20},
        {'center': (352, 40), 'angle': -20}
        ]
    patch_number = len(positions)    # 一张图上放置多少张补丁
    ratio = 1.0                      # 补丁放缩尺寸，默认为1

    # 进行fake bounding box处理
    fake_locations_list = []
    for pos in positions:
        center = pos['center']
        angle_radius = pos['angle'] * np.pi / 180.0
        fake_position = [center[0], center[1], generate_patch_w * ratio, generate_patch_h * ratio, angle_radius]
        fake_locations_list.append(fake_position)
    fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

    
    # 初始化度量计算器（只初始化一次）
    metric_calculator = MetricCalculator(device)
    
    # LPIPS和最佳位置选取计算初始化
    best_lpips = float('inf')
    adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)
    best_patches = []

    for i in range(patch_number):
        fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)
        fake_label_t = torch.zeros([1, 1, 1]).to(device)
        
        # 使用优化后的函数
        adv_mask_batch, best_patch = find_patch_optimized(
            fake_position_t, ratio, generate_patch_h, generate_patch_w,
            total_generated_patches_number, iter_num,
            device, patch_transformer, patch_applier,
            generator, fake_label_t, img_size, origin_image_tensor,
            metric_calculator, best_lpips
        )
        
        if adv_mask_batch is not None:
            adv_mask_selected += adv_mask_batch
            best_patches.append(best_patch)


    # 最终处理（保持不变）
    scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
    scene_image = scene_with_patches[0].cpu()
    scene_image_np = scene_image.permute(1, 2, 0).numpy()
    scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(temp_work_dir, os.path.basename(original_image_path)), scene_image_np)

    # 对补丁进行存储
    for i in range(len(best_patches)):
        best_patch_tensor = best_patches[i]
        best_patch_tensor = best_patch_tensor.squeeze(0).squeeze(0).cpu()
        best_patch_tensor_cpu = best_patch_tensor.permute(1, 2, 0).numpy()
        best_patch_np = (np.ascontiguousarray(best_patch_tensor_cpu) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(temp_work_dir, f"patch_{i}.png"), best_patch_np)


def test_one_image_iharbour():
    """
    在创新港数据集上找最隐蔽的对抗性补丁
    """
    # 新建临时存储文件夹
    temp_work_dir = './vis_results_DSAP/DSAP_iharbour_adversarial_examples/fig2_road'
    if not os.path.exists(temp_work_dir):
        os.makedirs(temp_work_dir)
        os.chmod(temp_work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        shutil.rmtree(temp_work_dir)
        print('--------------File exists! Existing file with the same name has been removed!--------------')    # 先删除再新建
        os.makedirs(temp_work_dir)
        os.chmod(temp_work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # 初始化
    original_image_path = '../../datasets/iharbour_dataset_2/road_scene_1_640x320/images/frame_00126.jpg'
    # generator_checkpoint = './train_DSAP_uavrod_phys_iharbour_road/checkpoints/netG_epoch1200.pth'
    # generator_checkpoint = './train_DSAP_uavrod_phys_iharbour_road_alpha0.12/checkpoints/netG_epoch1200.pth'
    generator_checkpoint = './train_DSAP_uavrod_phys_iharbour_road/checkpoints/netG_epoch1200.pth'
    # generator_checkpoint = './train_DSAP_uavrod_phys_iharbour_grass_alpha0.15/checkpoints/netG_epoch1200.pth'

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [32, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]

    total_generated_patches_number = 6000        # 使用generator一次获得total_generated_patches_number张补丁，然后在这total_generated_patches_number张补丁中选择
    # selecting iteration number setting
    iter_num = 6000
    # other settings
    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    
    ### -----------------------------------------------------------    load images and generator    ---------------------------------------------------------------------- ###
    # prepare scene from file path
    origin_image = cv2.imread(original_image_path)  # BGR格式
    origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
    # 不做归一化，因为不需要过检测器
    origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
    origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])
    # load generator
    generator = dcgan.DCGAN_G_Rect_2_gpu(generate_patch_size, 100, 3, 64, 1, 0)
    # generator.apply(weights_init)
    generator.load_state_dict(torch.load(generator_checkpoint,map_location=generator.device))
    
    # load position
    positions = [
        {'center': (289, 173), 'angle': 88},
        {'center': (330, 111), 'angle': 88}
        ]
    patch_number = len(positions)    # 一张图上放置多少张补丁
    ratio = 0.9                    # 补丁放缩尺寸，默认为1

    # 进行fake bounding box处理
    fake_locations_list = []
    for pos in positions:
        center = pos['center']
        angle_radius = pos['angle'] * np.pi / 180.0
        fake_position = [center[0], center[1], generate_patch_w * ratio, generate_patch_h * ratio, angle_radius]
        fake_locations_list.append(fake_position)
    fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

    
    # 初始化度量计算器（只初始化一次）
    metric_calculator = MetricCalculator(device)
    
    # LPIPS和最佳位置选取计算初始化
    best_lpips = float('inf')
    adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)
    best_patches = []

    for i in range(patch_number):
        fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)
        fake_label_t = torch.zeros([1, 1, 1]).to(device)
        
        # 使用优化后的函数
        adv_mask_batch, best_patch = find_patch_optimized(
            fake_position_t, ratio, generate_patch_h, generate_patch_w,
            total_generated_patches_number, iter_num,
            device, patch_transformer, patch_applier,
            generator, fake_label_t, img_size, origin_image_tensor,
            metric_calculator, best_lpips
        )
        
        if adv_mask_batch is not None:
            adv_mask_selected += adv_mask_batch
            best_patches.append(best_patch)
    

    # 最终处理（保持不变）
    scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
    scene_image = scene_with_patches[0].cpu()
    scene_image_np = scene_image.permute(1, 2, 0).numpy()
    scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(temp_work_dir, os.path.basename(original_image_path)), scene_image_np)

    # 对补丁进行存储
    for i in range(len(best_patches)):
        best_patch_tensor = best_patches[0]
        best_patch_tensor = best_patch_tensor.squeeze(0).squeeze(0).cpu()
        best_patch_tensor_cpu = best_patch_tensor.permute(1, 2, 0).numpy()
        best_patch_np = (np.ascontiguousarray(best_patch_tensor_cpu) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(temp_work_dir, f"patch_{i}.png"), best_patch_np)

    
def test_one_image_DOTA():
    """
    根据test_one_image修改而来，适用于DOTA数据集
    """
    # 新建临时存储文件夹
    temp_work_dir = './vis_results_DSAP/DSAP_DOTA_adversarial_examples/fig4'
    if not os.path.exists(temp_work_dir):
        os.makedirs(temp_work_dir)
        os.chmod(temp_work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        shutil.rmtree(temp_work_dir)
        print('--------------File exists! Existing file with the same name has been removed!--------------')    # 先删除再新建
        os.makedirs(temp_work_dir)
        os.chmod(temp_work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # 初始化
    cls = 'ship'
    original_image_path = '../../datasets/DOTA-V1.0/DOTA-ship-propersize/images/P0707__1__412___412.png'

    if cls == 'plane':
        generator_checkpoint = './train_DSAP_dota/multidetectors/plane-newSourceImages/retinanet_o/checkpoints/netG_epoch1200.pth'
        generate_patch_size = [48, 64]
        ratio = 1.0
    elif cls == 'large-vehicle':
        generator_checkpoint = './train_DSAP_dota_LV_test15_newSourceImages_epochs1200/checkpoints/netG_epoch1200.pth'
        generate_patch_size = [16, 48]
        ratio = 1.0
    elif cls == 'small-vehicle':
        generator_checkpoint = './train_DSAP_dota/multidetectors/small-vehicle_newSourceImages_2/retinanet_o/checkpoints/netG_epoch1200.pth'
        generate_patch_size = [16, 32]
        ratio = 0.7
    elif cls == 'ship':
        generator_checkpoint = './train_DSAP_dota/multidetectors/ship_newSourceImages/retinanet_o/checkpoints/netG_epoch1200.pth'
        generate_patch_size = [16, 32]
        ratio = 1.2
    else:
        print('wrong fake object class!')
        assert False

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (512, 512)
    # patch setting
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]

    total_generated_patches_number = 12000        # 使用generator一次获得total_generated_patches_number张补丁，然后在这total_generated_patches_number张补丁中选择
    # selecting iteration number setting
    iter_num = 12000
    # other settings
    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    

    ### -----------------------------------------------------------    load images and generator    ---------------------------------------------------------------------- ###
    # prepare scene from file path
    origin_image = cv2.imread(original_image_path)  # BGR格式
    origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
    # 不做归一化，因为不需要过检测器
    origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
    origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])
    # load generator
    generator = dcgan.DCGAN_G_Rect_2_gpu(generate_patch_size, 100, 3, 64, 1, 0)
    # generator.apply(weights_init)
    generator.load_state_dict(torch.load(generator_checkpoint,map_location=generator.device))

    
    # load position
    positions = [
        {'center': (235, 163), 'angle': 30}
        # {'center': (369, 258), 'angle': 30}
        ]
    patch_number = len(positions)    # 一张图上放置多少张补丁
    # ratio = 1.0                      # 补丁放缩尺寸，默认为1

    # 进行fake bounding box处理
    fake_locations_list = []
    for pos in positions:
        center = pos['center']
        angle_radius = pos['angle'] * np.pi / 180.0
        fake_position = [center[0], center[1], generate_patch_w * ratio, generate_patch_h * ratio, angle_radius]
        fake_locations_list.append(fake_position)
    fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

    
    # 初始化度量计算器（只初始化一次）
    metric_calculator = MetricCalculator(device)
    
    # LPIPS和最佳位置选取计算初始化
    best_lpips = float('inf')
    adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)

    for i in range(patch_number):
        fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)
        fake_label_t = torch.zeros([1, 1, 1]).to(device)
        
        # 使用优化后的函数
        adv_mask_batch = find_patch_optimized(
            fake_position_t, ratio, generate_patch_h, generate_patch_w,
            total_generated_patches_number, iter_num,
            device, patch_transformer, patch_applier,
            generator, fake_label_t, img_size, origin_image_tensor,
            metric_calculator, best_lpips
        )
        
        if adv_mask_batch is not None:
            adv_mask_selected += adv_mask_batch

    # 最终处理（保持不变）
    scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
    scene_image = scene_with_patches[0].cpu()
    scene_image_np = scene_image.permute(1, 2, 0).numpy()
    scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(temp_work_dir, os.path.basename(original_image_path)), scene_image_np)


# TODO: 参考这里的补丁位置选择功能写法
def advpatch_apply_on_1_image():
    """
    寻找advpatch方法下,在一张图片的给定位置上放置
    """
    # 新建临时存储文件夹
    work_dir = './vis_results_DSAP/deploy_advpatch_on_1_image'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        print('--------------File exists! Existing file with the same name has been removed!--------------')
        # 调试过程使用
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 调试完成后
        # raise Exception('File exists! please name a new file name')

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [32, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    patch_ratio = 1.0
    

    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()


    ### -----------------------------------------------------------    load advpatch    ---------------------------------------------------------------------- ###
    pt_file = "./train_advpatch_uavrod/advpatch_retinanet_o_uavrod/patch_pt/patch_epoch300.pt"
    advpatch = torch.load(pt_file)
    # 对fake_patch进行范围限制
    advpatch = torch.clamp(advpatch, 0, 1)      # advpatch shape: [1, 3, 32, 64]

    # load position
    positions = [
        {'center': (322, 89), 'angle': -2},
        {'center': (524, 222), 'angle': -2}
        ]
    patch_number = len(positions)    # 一张图上放置多少张补丁

    ### -----------------------------------------------------------    load images and compute    ---------------------------------------------------------------------- ###
    original_image_path = '../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360/DJI_0012_000240.png'
    image_name = 'adding_advpatch.png'
    
    # prepare scene from file path
    origin_image = cv2.imread(original_image_path)  # BGR格式
    origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
    # 不做归一化，因为不需要过检测器
    origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
    origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])        # 一张图像

    # 进行fake bounding box处理
    fake_locations_list = []
    for pos in positions:
        center = pos['center']
        angle_radius = pos['angle'] * np.pi / 180.0
        fake_position = [center[0], center[1], generate_patch_w * patch_ratio, generate_patch_h * patch_ratio, angle_radius]
        fake_locations_list.append(fake_position)
    fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)
    adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)

    for i in range(patch_number):
        fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)    # torch.Size([1, 1, 5])
        fake_label_t = torch.zeros([1, 1, 1]).to(device)
        
        # 直接进行forward与
        adv_mask_batch, _ = patch_transformer.forward4(
            adv_batch=advpatch.unsqueeze(0),
            ratio=patch_ratio,
            fake_bboxes_batch=fake_position_t,
            fake_labels_batch=fake_label_t,
            img_size=img_size,
            transform=False
        )
        
        if adv_mask_batch is not None:
            adv_mask_selected += adv_mask_batch

    # 最终处理（保持不变）
    scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
    scene_image = scene_with_patches[0].cpu()
    scene_image_np = scene_image.permute(1, 2, 0).detach().numpy()
    scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(work_dir, image_name), scene_image_np)



def DSAP_find_50_patches(work_dir, netG, patch_ratio=1.0):
    """
    寻找DSAP方法下50张图的最匹配隐蔽性补丁
    """
    # 新建临时存储文件夹
    # work_dir = './vis_results_DSAP/deploy_patches_DSAP'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        print('--------------File exists! Existing file with the same name has been removed!--------------')
        # 调试过程使用
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 调试完成后
        # raise Exception('File exists! please name a new file name')

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [32, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    patch_number = 4    # 一张图上放置多少张补丁

    total_generated_patches_number = 2000        # 使用generator一次获得total_generated_patches_number张补丁，然后在这total_generated_patches_number张补丁中选择
    # selecting iteration number setting
    iter_num = 2000
    # other settings
    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    # load json file
    locations_json_file = './uavrod_selected_patch_locations_scene50.json'
    fake_locations = load_patches_from_json(locations_json_file)

    ### -----------------------------------------------------------    load generator    ---------------------------------------------------------------------- ###
    # generator_checkpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_DSAP_rect_patch/DSAP_retinanet_uavrod/checkpoints/netG_epoch1200.pth'
    generator_checkpoint = netG
    # load generator
    generator = dcgan.DCGAN_G_Rect_2_gpu(generate_patch_size, 100, 3, 64, 1, 0)
    # generator.apply(weights_init)
    generator.load_state_dict(torch.load(generator_checkpoint,map_location=generator.device))

    ### -----------------------------------------------------------    initialize lpips calculator    ---------------------------------------------------------------------- ###
    metric_calculator = MetricCalculator(device)
    best_lpips = float('inf')

    ### -----------------------------------------------------------    load images and compute    ---------------------------------------------------------------------- ###
    image_root = "/home/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
    flag = 0
    for image_name, data in fake_locations.items():
        original_image_path = os.path.join(image_root, image_name)
        patches = data.get('patches', [])
        
        # prepare scene from file path
        origin_image = cv2.imread(original_image_path)  # BGR格式
        origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
        # 不做归一化，因为不需要过检测器
        origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
        origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])        # 一张图像

        # 进行fake bounding box处理
        fake_locations_list = []
        for patch in patches:
            center = patch['center']
            angle_radius = patch['angle'] * np.pi / 180.0
            fake_position = [center[0], center[1], generate_patch_w * patch_ratio, generate_patch_h * patch_ratio, angle_radius]
            fake_locations_list.append(fake_position)
        fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

        adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)

        for i in range(patch_number):
            fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)
            fake_label_t = torch.zeros([1, 1, 1]).to(device)
            
            # 使用优化后的函数
            adv_mask_batch, _ = find_patch_optimized(
                fake_position_t, patch_ratio, generate_patch_h, generate_patch_w,
                total_generated_patches_number, iter_num,
                device, patch_transformer, patch_applier,
                generator, fake_label_t, img_size, origin_image_tensor,
                metric_calculator, best_lpips
            )
            # 返回值第一个位置是advpatchmask, 尺寸为torch.Size([1, 1, 3, 360, 640])，与原图保持一致；返回值第二个位置是选择出来该位置的最适合补丁，torch.Size([1, 1, 3, 32, 64])

            if adv_mask_batch is not None:
                adv_mask_selected += adv_mask_batch

        # 最终处理（保持不变）
        scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
        scene_image = scene_with_patches[0].cpu()
        scene_image_np = scene_image.permute(1, 2, 0).numpy()
        scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(work_dir, image_name), scene_image_np)

        # log
        flag += 1
        # print(f'{image_name} select completed! {flag} images have been processed!')

def advpatch_find_50_patches():
    """
    寻找advpatch方法下50张图的最匹配隐蔽性补丁
    """
    # 新建临时存储文件夹
    work_dir = './vis_results_DSAP/deploy_patches_advpatch/s2anet'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        print('--------------File exists! Existing file with the same name has been removed!--------------')
        # 调试过程使用
        # shutil.rmtree(work_dir)
        # os.makedirs(work_dir)
        # os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 调试完成后
        raise Exception('File exists! please name a new file name')

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [32, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    patch_ratio = 1.0
    patch_number = 4    # 一张图上放置多少张补丁

    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    # load json file
    locations_json_file = './uavrod_selected_patch_locations_scene50.json'
    fake_locations = load_patches_from_json(locations_json_file)

    ### -----------------------------------------------------------    load advpatch    ---------------------------------------------------------------------- ###
    pt_file = "/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/train_advpatch_uavrod/advpatch_s2anet_uavrod/patch_pt/patch_epoch300.pt"
    advpatch = torch.load(pt_file)
    # 对fake_patch进行范围限制
    advpatch = torch.clamp(advpatch, 0, 1)      # advpatch shape: [1, 3, 32, 64]


    ### -----------------------------------------------------------    load images and compute    ---------------------------------------------------------------------- ###
    image_root = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
    flag = 0
    for image_name, data in fake_locations.items():
        original_image_path = os.path.join(image_root, image_name)
        patches = data.get('patches', [])
        
        # prepare scene from file path
        origin_image = cv2.imread(original_image_path)  # BGR格式
        origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
        # 不做归一化，因为不需要过检测器
        origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
        origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])        # 一张图像

        # 进行fake bounding box处理
        fake_locations_list = []
        for patch in patches:
            center = patch['center']
            angle_radius = patch['angle'] * np.pi / 180.0
            fake_position = [center[0], center[1], generate_patch_w * patch_ratio, generate_patch_h * patch_ratio, angle_radius]
            fake_locations_list.append(fake_position)
        fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

        adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)

        for i in range(patch_number):
            fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)    # torch.Size([1, 1, 5])
            fake_label_t = torch.zeros([1, 1, 1]).to(device)
            
            # 直接进行forward与
            adv_mask_batch, _ = patch_transformer.forward4(
                adv_batch=advpatch.unsqueeze(0),
                ratio=patch_ratio,
                fake_bboxes_batch=fake_position_t,
                fake_labels_batch=fake_label_t,
                img_size=img_size,
                transform=False
            )
            
            if adv_mask_batch is not None:
                adv_mask_selected += adv_mask_batch

        # 最终处理（保持不变）
        scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
        scene_image = scene_with_patches[0].cpu()
        scene_image_np = scene_image.permute(1, 2, 0).detach().numpy()
        scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(work_dir, image_name), scene_image_np)

        # log
        flag += 1
        # print(f'{image_name} select completed! {flag} images have been processed!')



def NAP_find_50_patches():
    """
    寻找naturalistic adversarial patch方法下50张图的最匹配隐蔽性补丁
    """
    # 新建临时存储文件夹
    detector_name = 'retinane_o'
    # work_dir = f'../NAP-uav_rod/stealth_test/{detector_name}'
    work_dir = f'../NAP-uav_rod/stealth_test/retinanet_o_seed2025'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        print('--------------File exists! Existing file with the same name has been removed!--------------')
        # 调试过程使用
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 调试完成后
        # raise Exception('File exists! please name a new file name')

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [128, 128]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    patch_ratio = 0.5
    patch_number = 4    # 一张图上放置多少张补丁

    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    # load json file
    locations_json_file = './uavrod_selected_patch_locations_scene50.json'
    fake_locations = load_patches_from_json(locations_json_file)

    ### -----------------------------------------------------------    load advpatch    ---------------------------------------------------------------------- ###
    # NApatch_path = f'../NAP-uav_rod/exp/multidetectors2/{detector_name}/generated/generated-images-0500.png'
    NApatch_path = f'../NAP-uav_rod/exp/multiseeds/2025/generated/generated-images-0500.png'
    patch_image = cv2.imread(NApatch_path) / 255.0 # 先归一化到0~1之间
    advpatch = torch.from_numpy(patch_image.transpose(2, 0, 1)).unsqueeze(0)
    advpatch = advpatch.to(torch.float32).cuda()
    # st()

    # 对fake_patch进行范围限制
    advpatch = torch.clamp(advpatch, 0, 1)      # advpatch shape: [1, 3, 32, 64]


    ### -----------------------------------------------------------    load images and compute    ---------------------------------------------------------------------- ###
    image_root = "/home/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
    flag = 0
    for image_name, data in fake_locations.items():
        original_image_path = os.path.join(image_root, image_name)
        patches = data.get('patches', [])
        
        # prepare scene from file path
        origin_image = cv2.imread(original_image_path)  # BGR格式
        origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
        # 不做归一化，因为不需要过检测器
        origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
        origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])        # 一张图像

        # 进行fake bounding box处理
        fake_locations_list = []
        for patch in patches:
            center = patch['center']
            angle_radius = patch['angle'] * np.pi / 180.0
            fake_position = [center[0], center[1], generate_patch_w * patch_ratio, generate_patch_h * patch_ratio, angle_radius]
            fake_locations_list.append(fake_position)
        fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

        adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)

        for i in range(patch_number):
            fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)    # torch.Size([1, 1, 5])
            fake_label_t = torch.zeros([1, 1, 1]).to(device)
            
            # 直接进行forward与
            adv_mask_batch, _ = patch_transformer.forward4(
                adv_batch=advpatch.unsqueeze(0),
                ratio=patch_ratio,
                fake_bboxes_batch=fake_position_t,
                fake_labels_batch=fake_label_t,
                img_size=img_size,
                transform=False
            )
            
            if adv_mask_batch is not None:
                adv_mask_selected += adv_mask_batch

        # 最终处理（保持不变）
        scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
        scene_image = scene_with_patches[0].cpu()
        scene_image_np = scene_image.permute(1, 2, 0).detach().numpy()
        scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(work_dir, image_name), scene_image_np)

        # log
        flag += 1
        # print(f'{image_name} select completed! {flag} images have been processed!')


def AdvART_find_50_patches():
    """
    寻找AdvART方法下50张图的最匹配隐蔽性补丁
    """
    # 新建临时存储文件夹
    detector_name = 's2anet'
    work_dir = f'../AdvART/stealth_test/{detector_name}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        print('--------------File exists! Existing file with the same name has been removed!--------------')
        # 调试过程使用
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 调试完成后
        # raise Exception('File exists! please name a new file name')

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [64, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    patch_ratio = 1.0
    patch_number = 4    # 一张图上放置多少张补丁

    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    # load json file
    locations_json_file = './uavrod_selected_patch_locations_scene50.json'
    fake_locations = load_patches_from_json(locations_json_file)

    ### -----------------------------------------------------------    load advpatch    ---------------------------------------------------------------------- ###
    AdvARTpatch_path = f'../AdvART/train_AdvART_flowerdog_stealth/{detector_name}/patch_visualize/patch_epoch400.jpg'
    patch_image = cv2.imread(AdvARTpatch_path) / 255.0 # 先归一化到0~1之间
    advpatch = torch.from_numpy(patch_image.transpose(2, 0, 1)).unsqueeze(0)
    advpatch = advpatch.to(torch.float32).cuda()
    # st()

    # 对fake_patch进行范围限制
    advpatch = torch.clamp(advpatch, 0, 1)      # advpatch shape: [1, 3, 32, 64]


    ### -----------------------------------------------------------    load images and compute    ---------------------------------------------------------------------- ###
    image_root = "/home/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
    flag = 0
    for image_name, data in fake_locations.items():
        original_image_path = os.path.join(image_root, image_name)
        patches = data.get('patches', [])
        
        # prepare scene from file path
        origin_image = cv2.imread(original_image_path)  # BGR格式
        origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
        # 不做归一化，因为不需要过检测器
        origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
        origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])        # 一张图像

        # 进行fake bounding box处理
        fake_locations_list = []
        for patch in patches:
            center = patch['center']
            angle_radius = patch['angle'] * np.pi / 180.0
            fake_position = [center[0], center[1], generate_patch_w * patch_ratio, generate_patch_h * patch_ratio, angle_radius]
            fake_locations_list.append(fake_position)
        fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

        adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)

        for i in range(patch_number):
            fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)    # torch.Size([1, 1, 5])
            fake_label_t = torch.zeros([1, 1, 1]).to(device)
            
            # 直接进行forward与
            adv_mask_batch, _ = patch_transformer.forward4(
                adv_batch=advpatch.unsqueeze(0),
                ratio=patch_ratio,
                fake_bboxes_batch=fake_position_t,
                fake_labels_batch=fake_label_t,
                img_size=img_size,
                transform=False
            )
            
            if adv_mask_batch is not None:
                adv_mask_selected += adv_mask_batch

        # 最终处理（保持不变）
        scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
        scene_image = scene_with_patches[0].cpu()
        scene_image_np = scene_image.permute(1, 2, 0).detach().numpy()
        scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(work_dir, image_name), scene_image_np)

        # log
        flag += 1
        # print(f'{image_name} select completed! {flag} images have been processed!')






def SingleDSAP_find_50_patches():
    """
    一个DSAP补丁放置于50张图的情况
    """
    # 新建临时存储文件夹
    # retinanet
    # work_dir = './vis_results_DSAP/deploy_patches_SingleDSAP/retinanet'
    # faster-rcnn
    # work_dir = './vis_results_DSAP/deploy_patches_SingleDSAP/faster-rcnn'
    # gliding-vertex
    work_dir = './vis_results_DSAP/deploy_patches_SingleDSAP/gliding-vertex'

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        print('--------------Warnning: File exists!--------------')
        # 调试过程使用
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 调试完成后
        # raise Exception('File exists! please name a new file name')

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [32, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    patch_ratio = 1.0
    patch_number = 4    # 一张图上放置多少张补丁

    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    # load json file
    locations_json_file = './uavrod_selected_patch_locations_scene50.json'
    fake_locations = load_patches_from_json(locations_json_file)

    ### -----------------------------------------------------------    load SingleDSAP    ---------------------------------------------------------------------- ###
    # retinanet
    # generator_checkpoint = './train_DSAP_rect_patch/multidetectors/DSAP_retinanet_uavrod/checkpoints/netG_epoch1200.pth'
    # faster-rcnn
    # generator_checkpoint = './train_DSAP_rect_patch/multidetectors/DSAP_faster_rcnn_uavrod/checkpoints/netG_epoch1200.pth'
    # gliding-vertex
    generator_checkpoint = './train_DSAP_rect_patch/multidetectors/DSAP_gliding_vertex_uavrod/checkpoints/netG_epoch1200.pth'


    # load generator
    generator = dcgan.DCGAN_G_Rect_2_gpu(generate_patch_size, 100, 3, 64, 1, 0)
    # generator.apply(weights_init)
    generator.load_state_dict(torch.load(generator_checkpoint,map_location=generator.device))

    noise = torch.FloatTensor(1, 100, 1, 1).normal_(0, 1).to(device)        # 直接生成一个补丁
    with torch.no_grad():  # 添加no_grad上下文
        fake_patch = generator(noise)
        fake_patch = fake_patch.mul(0.5).add(0.5)



    ### -----------------------------------------------------------    load images and compute    ---------------------------------------------------------------------- ###
    image_root = "/home/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
    flag = 0
    for image_name, data in fake_locations.items():
        original_image_path = os.path.join(image_root, image_name)
        patches = data.get('patches', [])
        
        # prepare scene from file path
        origin_image = cv2.imread(original_image_path)  # BGR格式
        origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
        # 不做归一化，因为不需要过检测器
        origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
        origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])        # 一张图像

        # 进行fake bounding box处理
        fake_locations_list = []
        for patch in patches:
            center = patch['center']
            angle_radius = patch['angle'] * np.pi / 180.0
            fake_position = [center[0], center[1], generate_patch_w * patch_ratio, generate_patch_h * patch_ratio, angle_radius]
            fake_locations_list.append(fake_position)
        fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

        adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)

        for i in range(patch_number):
            fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)    # torch.Size([1, 1, 5])
            fake_label_t = torch.zeros([1, 1, 1]).to(device)
            
            # 直接进行forward与
            adv_mask_batch, _ = patch_transformer.forward4(
                adv_batch=fake_patch.unsqueeze(0),
                ratio=patch_ratio,
                fake_bboxes_batch=fake_position_t,
                fake_labels_batch=fake_label_t,
                img_size=img_size,
                transform=False
            )
            
            if adv_mask_batch is not None:
                adv_mask_selected += adv_mask_batch

        # 最终处理（保持不变）
        scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
        scene_image = scene_with_patches[0].cpu()
        scene_image_np = scene_image.permute(1, 2, 0).detach().numpy()
        scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(work_dir, image_name), scene_image_np)

        # log
        flag += 1
        # print(f'{image_name} select completed! {flag} images have been processed!')


def random_find_50_patches():
    """
    寻找random方法下50张图的最匹配隐蔽性补丁。尽管random补丁与检测器无关，还是分开都测试一下
    """
    # 新建临时存储文件夹
    work_dir = './vis_results_DSAP/deploy_patches_random/s2anet'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        print('--------------File exists! Existing file with the same name has been removed!--------------')
        # 调试过程使用
        shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 调试完成后
        # raise Exception('File exists! please name a new file name')

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    generate_patch_size = [32, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    patch_ratio = 1.0
    patch_number = 4    # 一张图上放置多少张补丁

    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
        patch_applier = PatchApplier().cuda()
    else:
        patch_transformer = PatchTransformer()
        patch_applier = PatchApplier()

    # load json file
    locations_json_file = './uavrod_selected_patch_locations_scene50.json'
    fake_locations = load_patches_from_json(locations_json_file)

    ### -----------------------------------------------------------    load random patch    ---------------------------------------------------------------------- ###
    random_patch = torch.rand([1, 3, generate_patch_h, generate_patch_w]).to(device)

    ### -----------------------------------------------------------    load images and compute    ---------------------------------------------------------------------- ###
    image_root = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
    flag = 0
    for image_name, data in fake_locations.items():
        original_image_path = os.path.join(image_root, image_name)
        patches = data.get('patches', [])
        
        # prepare scene from file path
        origin_image = cv2.imread(original_image_path)  # BGR格式
        origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
        # 不做归一化，因为不需要过检测器
        origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
        origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])        # 一张图像

        # 进行fake bounding box处理
        fake_locations_list = []
        for patch in patches:
            center = patch['center']
            angle_radius = patch['angle'] * np.pi / 180.0
            fake_position = [center[0], center[1], generate_patch_w * patch_ratio, generate_patch_h * patch_ratio, angle_radius]
            fake_locations_list.append(fake_position)
        fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)

        adv_mask_selected = torch.zeros(1, 1, 3, img_size[0], img_size[1]).to(device)

        for i in range(patch_number):
            fake_position_t = fake_positions_t[:, i, :].unsqueeze(1)    # torch.Size([1, 1, 5])
            fake_label_t = torch.zeros([1, 1, 1]).to(device)
            
            # 直接进行forward与
            adv_mask_batch, _ = patch_transformer.forward4(
                adv_batch=random_patch.unsqueeze(0),
                ratio=patch_ratio,
                fake_bboxes_batch=fake_position_t,
                fake_labels_batch=fake_label_t,
                img_size=img_size,
                transform=False
            )
            
            if adv_mask_batch is not None:
                adv_mask_selected += adv_mask_batch

        # 最终处理（保持不变）
        scene_with_patches = patch_applier(origin_image_tensor, adv_mask_selected)
        scene_image = scene_with_patches[0].cpu()
        scene_image_np = scene_image.permute(1, 2, 0).detach().numpy()
        scene_image_np = (np.ascontiguousarray(scene_image_np) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(work_dir, image_name), scene_image_np)

        # log
        flag += 1
        # print(f'{image_name} select completed! {flag} images have been processed!')


# IFGSM creation attack相关代码
# 该部分需要在yyx_ctr容器下运行，yyx_2容器缺少mmdetection相关的库
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
    mean_RGB = [110.928 / 255.0, 107.967 / 255.0, 108.969 / 255.0]  # RGB color for scene dataset
    std_RGB = [47.737 / 255.0, 48.588 / 255.0, 48.115 / 255.0]  # RGB color for scene dataset
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

def I_FGSM_creation_attack(model, device, 
                           scene_bag, img_size, 
                           normalizer, 
                           patch_number, patch_h, patch_w, patch_ratio,  
                           patch_transformer, 
                           fake_positions, fake_labels, 
                           epsilon, alpha, num_iter, 
                           detector_name, 
                           ):
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
    image_normalize = normalizer(image_scene)  # p_img_batch_normalize是BGR格式        normalizer: AdvImagesNormalizer_BGR
    image_normalize_rgb = image_normalize[:, [2, 1, 0], :, :]  # BGR->RGB
    image_normalize_rgb.requires_grad = True

    # prepare source image
    source_image = image_normalize_rgb.clone()

    # 进行I-FGSM攻击
    pseudo_fake_patch = torch.ones((1, patch_number, 3, patch_h, patch_w)).to(device)
    for _ in range(num_iter):
        # 获取mask
        _, msk_batch = patch_transformer.forward4(adv_batch=pseudo_fake_patch, 
                                                  ratio=patch_ratio, 
                                                  fake_bboxes_batch=fake_positions,
                                                  fake_labels_batch=fake_labels,
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
                                                   patch_boxes_batch_t=fake_positions,
                                                   model_name=detector_name)
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
        mean_RGB = [110.928 / 255.0, 107.967 / 255.0, 108.969 / 255.0]  # RGB color for scene dataset
        std_RGB = [47.737 / 255.0, 48.588 / 255.0, 48.115 / 255.0]  # RGB color for scene dataset
        image_normalize_rgb_copy = image_normalize_rgb.clone()
        image_rgb_copy = denorm(image_normalize_rgb_copy, mean_RGB, std_RGB)  # image_for_save: RGB
        image_rgb_copy_clamp = torch.clamp(image_rgb_copy, 0, 1)

    # 返回perturbed images(RGB，经过反归一化，取值范围为0~1), tensor形式，每一个tensor尺寸为[1, 3, H, W]
    return image_rgb_copy_clamp


def IFGSM_find_50_patches(model_name, file_name):
    """
    寻找IFGSM方法下50张图的最匹配隐蔽性补丁
    """
    # 新建存储文件夹     
    work_dir = os.path.join('./vis_results_DSAP/deploy_patches_IFGSM', file_name)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
        os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        print('--------------File exists! Existing file with the same name has been removed!--------------')
        # 调试过程使用
        # shutil.rmtree(work_dir)
        # os.makedirs(work_dir)
        # os.chmod(work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        # 调试完成后
        raise Exception('File exists! please name a new file name')

    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (384, 640)   # 注意这里是[384, 640]因为detector需要接收[384, 640]尺寸的图像
    mean_BGR = [108.969 / 255.0, 107.967 / 255.0, 110.928 / 255.0]  # BGR color for scene dataset
    std_BGR = [48.115 / 255.0, 48.588 / 255.0, 47.737 / 255.0]  # BGR color for scene dataset
    # mean_RGB = [110.928 / 255.0, 107.967 / 255.0, 108.969 / 255.0]  # RGB color for scene dataset
    # std_RGB = [47.737 / 255.0, 48.588 / 255.0, 48.115 / 255.0]  # RGB color for scene dataset
    AdvImagesNormalizer_BGR = transforms.Normalize(mean_BGR, std_BGR)
    # patch setting
    generate_patch_size = [32, 64]
    generate_patch_h, generate_patch_w = generate_patch_size[0], generate_patch_size[1]
    patch_ratio = 1.0
    patch_number = 4    # 一张图上放置多少张补丁

    device = 'cuda:0'
    Seed = 2025
    # random seed settings
    torch.manual_seed(Seed)
    torch.cuda.manual_seed(Seed)
    torch.cuda.manual_seed_all(Seed)
    np.random.seed(Seed)
    random.seed(Seed)
    # transformer and applier settings
    if device == "cuda:0":
        patch_transformer = PatchTransformer().cuda()
    else:
        patch_transformer = PatchTransformer()

    # load json file
    locations_json_file = './uavrod_selected_patch_locations_scene50.json'
    fake_locations = load_patches_from_json(locations_json_file)

    # IFGSM setting
    alpha = 0.15
    epsilon = 1.0
    iteration_number = 30

    # detector settings
    if model_name == 'retinanet_o':
        DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_uavrod_le90.py'
        DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/rotated_retinanet/latest.pth'
    elif model_name == 'faster_rcnn_o':
        DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_uavrod_le90.py'
        DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/faster_rcnn_o/latest.pth'
    elif model_name == 'gliding_vertex':
        DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/gliding_vertex/gliding_vertex_r50_fpn_1x_uavrod_le90.py'
        DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/gliding_vertex/latest.pth'
    elif model_name == 'oriented_rcnn':
        DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_uavrod_le90.py'
        DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/oriented_rcnn/latest.pth'
    elif model_name == 'roi_trans':
        DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/roi_trans/roi_trans_r50_fpn_1x_uavrod_le90.py'
        DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/roi_trans/latest.pth'
    elif model_name == 's2anet':
        DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/s2anet/s2anet_r50_fpn_1x_uavrod_le90.py'
        DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/s2anet/latest.pth'
    else:
        raise Exception('Wrong Victim Detector!')
    
    start_time = time.time()
    model = init_detector(config=DetectorCfgSource, checkpoint=DetectorCheckpoint, device="cuda:0")  # 这里从init_detector返回的model已经是.eval()模式的
    finish_time = time.time()
    # print(f'Load detector in {finish_time - start_time} seconds.')

    ### -----------------------------------------------------------    load images and compute    ---------------------------------------------------------------------- ###
    image_root = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
    # flag = 0
    # inti scenebag
    img_meta_keys = ['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'to_rgb']
    bag_keys = ['Images_t', 'imgs_metas']
    # flag = 0
    for image_name, data in tqdm(fake_locations.items()):
        img_metas_batch = []
        group_dict = {key: None for key in bag_keys}
        # flag += 1
        original_image_path = os.path.join(image_root, image_name)
        patches = data.get('patches', [])
        # 合成scenebag
        img_meta = {key: None for key in img_meta_keys}
        img_meta['filename'] = original_image_path
        img_meta['ori_filename'] = image_name
        img_meta['ori_shape'] = (360, 640, 3)  # before padding
        img_meta['img_shape'] = (360, 640, 3)
        img_meta['pad_shape'] = (384, 640, 3)
        img_meta['scale_factor'] = np.array([1., 1., 1., 1.], dtype=np.float32)
        img_meta['flip'] = False
        img_meta['flip_direction'] = None
        img_meta['img_norm_cfg'] = dict(mean=np.array([110.928, 107.967, 108.969], dtype=np.float32), std=np.array([47.737, 48.588, 48.115], dtype=np.float32))
        img_meta['to_rgb'] = True
        img_metas_batch.append(img_meta)        # batchsize = 1
        # 读取图片文件
        origin_image = cv2.imread(original_image_path)  # BGR格式
        origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
        # 图片padding
        origin_image = cv2.copyMakeBorder(origin_image, 0, 24, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        origin_image_tensor = torch.from_numpy(origin_image).permute(2, 0, 1).to(torch.float32)  # (3,height,width)， BGR格式
        origin_image_tensor = origin_image_tensor.unsqueeze(0).to('cpu')

        # bag中只需要image_t和images_mega
        group_dict['Images_t'] = origin_image_tensor        # torch.Size([1, 3, 384, 640])
        group_dict['imgs_metas'] = img_metas_batch

        # 进行fake bounding box处理
        fake_locations_list = []
        for patch in patches:
            center = patch['center']
            angle_radius = patch['angle'] * np.pi / 180.0
            fake_position = [center[0], center[1], generate_patch_w * patch_ratio, generate_patch_h * patch_ratio, angle_radius]
            fake_locations_list.append(fake_position)
        
        fake_positions_t = torch.Tensor(fake_locations_list).unsqueeze(0).to(device)        # torch.Size([1, 4, 5])
        fake_label_t = torch.zeros([1, patch_number, 1]).to(device)


        # 从这个地方开始，修改成生成IFGSM的代码
        # print('flag=', flag)
        perturbed_single_image_rgb = I_FGSM_creation_attack(model=model,device=device,
                                                            scene_bag=group_dict, img_size=img_size, 
                                                            normalizer=AdvImagesNormalizer_BGR, 
                                                            patch_transformer=patch_transformer, 
                                                            patch_number=patch_number, patch_h=generate_patch_h, patch_w=generate_patch_w, patch_ratio=patch_ratio, 
                                                            fake_positions=fake_positions_t,fake_labels=fake_label_t,
                                                            epsilon=epsilon,alpha=alpha,num_iter=iteration_number, 
                                                            detector_name=model_name)
        
        # 最终处理
        scene_image_rgb = perturbed_single_image_rgb[0, :, :, :]  # (3, 384, 640)    # 选择一张进行存储

        # 进行裁切，将尺寸为(3, 384, 640)的图像裁切成(3, 360, 640)的张量
        clipped_scene_image_rgb = scene_image_rgb[:, :360, :]

        sce_img_rgb = clipped_scene_image_rgb.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
        sce_img_bgr = sce_img_rgb[:, :, [2, 1, 0]]  # RGB->BGR
        sce_img_bgr = (np.ascontiguousarray(sce_img_bgr) * 255).astype(np.uint8)
        ifgsm_image_save_path = os.path.join(work_dir, image_name)
        cv2.imwrite(ifgsm_image_save_path, sce_img_bgr)


if __name__ == "__main__":
    # 获得6个检测器下IFGSM算法在给定位置（json文件）下的对抗样本。注意这里需要切换到yyx_ctr容器中进行
    # model_name = ['retinanet_o', 'faster_rcnn_o', 'gliding_vertex', 'oriented_rcnn', 'roi_trans', 's2anet']
    # file_name = ['retinanet', 'faster-rcnn', 'gliding_vertex', 'oriented-rcnn', 'roi-trans', 's2anet']
    # for i in range(len(model_name)):
    #     selected_model = model_name[i]
    #     save_file_name = file_name[i]
    #     IFGSM_find_50_patches(model_name=selected_model, file_name=save_file_name)

    # 对于DJI_0012_001260.png图片获得五个最佳补丁放置结果，用于绘制框架图Fig2
    # test_one_image()

    # 消融实验中不同weights下的对抗样本获取
    # alphas = [0.01, 0.05, 0.1, 0.5]
    # betas = [0.05, 0.1, 0.5, 1.0]
    # for alpha in alphas:
    #     for beta in betas:
    #         work_dir = f'./vis_results_DSAP/deploy_patches_DSAP_multiweights/alpha{alpha}_beta{beta}'
    #         netG = f'./train_DSAP_rect_patch/multiweights/alpha{alpha}_beta{beta}/checkpoints/netG_epoch1200.pth'
    #         DSAP_find_50_patches(work_dir=work_dir, netG=netG)

    
    # 不同patchnumber下隐蔽性补丁的获取
    # image_batchsizes = [1, 1, 2, 2]
    # patch_numbers_per_image = [1, 2, 2, 4]
    # len = len(image_batchsizes)
    # for i in range(len):
    #     image_batchsize = image_batchsizes[i]
    #     patch_per_image = patch_numbers_per_image[i]

    #     work_dir = f'./vis_results_DSAP/deploy_patches_DSAP_multipatchnumbers/imagebatchsize{image_batchsize}_patchperimage{patch_per_image}'
    #     netG = f'./train_DSAP_rect_patch/multipatchnumbers/ImageBatchSize{image_batchsize}_PatchPerImage{patch_per_image}/checkpoints/netG_epoch1200.pth'
    #     DSAP_find_50_patches(work_dir=work_dir, netG=netG)

    # # 不同patch size下隐蔽性补丁的获取
    # patch_ratio = 0.7
    # work_dir = f'./test_DSAP_multipatchsize_stealth/patchratio_{patch_ratio}'
    # netG = f'./train_DSAP_uavrod/multipatchsize/patchsize_{patch_ratio}/checkpoints/netG_epoch1200.pth'
    # DSAP_find_50_patches(work_dir=work_dir, netG=netG, patch_ratio=patch_ratio)


    # single DSAP补丁放置在50张图4个位置上，准备segmentor的训练数据
    # SingleDSAP_find_50_patches()

    # 用于绘制图1，在一个给定图像，2个给定位置上放置两个advpatch
    # advpatch_apply_on_1_image()

    # 用于绘制adversarial examples，在一个给定图像，多个给定位置上放置多个DSAP
    test_one_image()

    # 在dota数据集上挑选隐蔽性补丁，一张一张图进行挑选
    # test_one_image_DOTA()

    # 在创新港场景中寻找隐蔽性补丁
    # test_one_image_iharbour()

    # 在uav-rod-50数据集上放NAP
    # NAP_find_50_patches()

    # 在uav-rod-50数据集上放AdvART
    # AdvART_find_50_patches()
        



    

    

