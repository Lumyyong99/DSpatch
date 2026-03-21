"""
自动化挑选LPIPS最小的补丁放置位置，第一版本，给定补丁和图像，遍历找到最合适的位置，通过引入并行编程解决效率低的问题
多进程，一次处理一个补丁放在一张图片上, 16线程处理一张图片大概15-20min，cpu全占用
"""
import argparse
import cv2
import os
import stat
import random
import numpy as np
import torch
import glob
import pyiqa
import shutil
import math
import multiprocessing
from functools import partial
from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier
# cv2.setNumThreads(1)
# cv2.ocl.setUseOpenCL(False)
# torch.set_num_threads(6)

def run_task(task):
    return task()

def calculate_lpips(folder_gt, folder_restored):
    prefix=None
    lpips_score = []
    ssim_score = []
    psnr_score = []
    lpips_score = []
    img_list = sorted(glob.glob(os.path.join(folder_gt, '*')))
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, crop_border=4, color_space='ycbcr')
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, crop_border=4, color_space='ycbcr')

    fr_iqa_dict={}
    for i, img_path in enumerate(img_list):
        basename, ext = os.path.splitext(os.path.basename(img_path))
        if prefix is None:
            img_sr = os.path.join(folder_restored, basename + '.jpg')
            ssim_score.append(ssim_metric(img_sr,img_path))
            psnr_score.append(psnr_metric(img_sr,img_path))
            lpips_score.append(lpips_metric(img_sr,img_path))
    fr_iqa_dict['psnr']=sum(psnr_score).item() / len(psnr_score)
    fr_iqa_dict['ssim']=sum(ssim_score).item() / len(ssim_score)
    fr_iqa_dict['lpips']=sum(lpips_score).item() / len(lpips_score)
    print(fr_iqa_dict)

    return fr_iqa_dict['lpips']

def process_subregion(y_start, y_end, x_start, x_end, step,
                      rowPatch_size, ratio, angle_radius, 
                      device, 
                      patch_transformer, patch_applier, 
                      fake_patch, fake_labels_t, img_size, origin_image_tensor, 
                      temp_work_dir, process_folder, 
                      calculate_lpips, best_lpips, best_position):
    """
    计算[x_start, y_start, x_end, y_end]这个subregion区域内补丁最适合的位置与LPIPS
    """
    for y in range(y_start, y_end, step):
        for x in range(x_start, x_end, step):
            # 确定fake_bboxes_batch
            fake_positions_t = torch.Tensor([[[x, y, rowPatch_size * ratio, rowPatch_size * ratio, angle_radius]]]).to(device)  # torch.Size([1, 1, 5])

            # 循环中插入补丁并进行apply， fake_bboxes_batch滑动窗口有关
            adv_batch_masked, _ = patch_transformer.forward2(adv_batch=fake_patch, fake_bboxes_batch=fake_positions_t, fake_labels_batch=fake_labels_t, img_size=img_size)
            patch_applied_image = patch_applier(origin_image_tensor.to(device), adv_batch_masked.to(device))  # 这里p_img_batch还没做归一化, BGR, torch.Size([1, 3, H, W])

            # 存储图像
            patch_applied_image_copy = patch_applied_image.clone()
            image_squeeze = patch_applied_image_copy[0, :, :, :]  # (3, 384, 640)
            image_squeeze_np = image_squeeze.permute(1, 2, 0).cpu().detach().numpy()  # (360, 640, 3), BGR
            image_squeeze_np = (np.ascontiguousarray(image_squeeze_np) * 255).astype(np.uint8)
            perturbed_image_save_path = os.path.join(temp_work_dir, process_folder, 'folder_compare', f'pic1.jpg')
            cv2.imwrite(perturbed_image_save_path, image_squeeze_np)

            # 计算LPIPS值
            lpips_value = calculate_lpips(folder_gt=os.path.join(temp_work_dir,'folder_origin'), folder_restored=os.path.join(temp_work_dir, process_folder,'folder_compare'))

            # 如果当前LPIPS值更小，则更新最佳位置
            if lpips_value < best_lpips:
                best_lpips = lpips_value
                best_position = (x, y)
                # cv2.imwrite(os.path.join(temp_work_dir, 'best_match.jpg'), image_squeeze_np)
                
    return best_position, best_lpips

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')


    # 新建临时存储文件夹
    subfolders = ['folder_origin']        # folder_compare用于存储添加了补丁的jpg图像
    temp_work_dir = './temp_automatic_assign'
    if not os.path.exists(temp_work_dir):
        os.makedirs(temp_work_dir)
        os.chmod(temp_work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    # else:
    #     shutil.rmtree(temp_work_dir)
    #     print('--------------File exists! Existing file with the same name has been removed!--------------')    # 先删除再新建
    #     os.makedirs(temp_work_dir)
    #     os.chmod(temp_work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    for subfolder in subfolders:
        subfolder_path = os.path.join(temp_work_dir, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            os.chmod(subfolder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    original_image_path = "/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/drawing/Exp_p4/origin_scene_image/image4.jpg"
    target_folder = os.path.join(temp_work_dir,'folder_origin')
    new_jpg_name = 'pic1.jpg'
    # 将原图放到temp_work_dir中
    shutil.copy(original_image_path, target_folder)
    target_file = os.path.join(target_folder, new_jpg_name)
    origin_file = os.path.join(target_folder, os.path.basename(original_image_path))
    os.rename(origin_file, target_file)


    ### -----------------------------------------------------------    settings    ---------------------------------------------------------------------- ###
    # scene setting
    img_size = (360, 640)
    # patch setting
    rowPatch_size = 64
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
    # parallel setting
    num_processes = 4
    height_split = num_processes    # 按照进程数进行切分
    width_split = num_processes

    # 再次创建subfolder
    subregion_folders = []
    for i in range(height_split):
        for j in range(width_split):
            subregion_folders.append(f'subregion_{i}_{j}')
    for subregion_folder in subregion_folders:
        subregion_folder_path = os.path.join(temp_work_dir, subregion_folder)
        if not os.path.exists(subregion_folder_path):
            os.makedirs(subregion_folder_path)
            os.chmod(subregion_folder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    ### -----------------------------------------------------------    load images and patches    ---------------------------------------------------------------------- ###
    # prepare scene from file path
    origin_image = cv2.imread(original_image_path)  # BGR格式
    origin_image = origin_image.astype(np.float32) / 255.0  # [0, 255] to [0, 1]
    # 不做归一化，因为不需要过检测器
    origin_image = np.transpose(origin_image, (2, 0, 1))  # 将图片从(H ,W, 3)形式转换成(3, H, W)
    origin_image_tensor = torch.from_numpy(origin_image).unsqueeze(0).to(device)  # torch.Size([1, 3, H, W])
    # load patch
    patch1_pt_file = "/home/yyx/Adversarial/mmrotate-0.3.3/adversarial_patch_uav_rod/drawing/Exp_p4/weight_alpha0.1_beta1.0/patch_pt/patch_5.pt"
    patch1_origin_tensor = torch.load(patch1_pt_file)  # 范围为[0, 1], torch.Size([3, 64 ,64])
    patch_origin_tensor = patch1_origin_tensor.unsqueeze(0)     # torch.Size([1,3,64,64])
    # applying patch settings
    angle = 0
    angle_radius = angle * np.pi / 180.0
    ratio = 0.6
    fake_labels_t = torch.zeros([1, 1, 1]).to(device)  # (image_BatchSize,Patch_per_Image,1)
    fake_patch = patch_origin_tensor.view(1, 1, 3, rowPatch_size, rowPatch_size).to(device)

    # LPIPS和最佳位置选取计算初始化
    best_lpips = float('inf')
    best_position = (0, 0)

    # 滑动窗口遍历所有可能的位置
    h_start_coordinate = rowPatch_size // 2
    h_end_coordinate = img_size[0] - rowPatch_size // 2
    w_start_coordinate = rowPatch_size // 2
    w_end_coordinate = img_size[1] - rowPatch_size // 2
    step = 5
    h_split_range = (h_end_coordinate - h_start_coordinate) // height_split     # 每个subregion在h方向上的的尺寸
    w_split_range = (w_end_coordinate - w_start_coordinate) // width_split      # 每个subregion在w方向上的的尺寸

    tasks = []
    for i in range(height_split):
        for j in range(width_split):
            process_folder = f'subregion_{i}_{j}'
            subregion_work_folder = os.path.join(temp_work_dir, process_folder, 'folder_compare')
            if not os.path.exists(subregion_work_folder):
                os.makedirs(subregion_work_folder)
                os.chmod(subregion_work_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            
            y_start = h_start_coordinate + i * h_split_range
            y_end = h_start_coordinate + (i + 1) * h_split_range if i < height_split - 1 else h_end_coordinate
            x_start = w_start_coordinate + j * w_split_range
            x_end = w_start_coordinate + (j + 1) * w_split_range if j < width_split - 1 else w_end_coordinate
            task = partial(process_subregion, y_start, y_end, x_start, x_end, step, 
                           rowPatch_size, ratio, angle_radius, device, 
                           patch_transformer, patch_applier, 
                           fake_patch, fake_labels_t, img_size, origin_image_tensor, 
                           temp_work_dir, process_folder, 
                           calculate_lpips, best_lpips, best_position)
            tasks.append(task)

    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(run_task, tasks)

    # 获取最终的最佳位置和LPIPS值
    best_position = (0, 0)
    best_lpips = float('inf')
    for result in results:
        position, lpips_value = result
        if lpips_value < best_lpips:
            best_lpips = lpips_value
            best_position = position
            print('best_position:', best_position)


"""
# single image processing
# 统计循环总次数
y_loop_count = math.floor((h_end_coordinate - h_start_coordinate) / step)
x_loop_count = math.floor((w_end_coordinate - w_start_coordinate) / step)
total_iterations = y_loop_count * x_loop_count
iter_i = 0
for y in range(h_start_coordinate, h_end_coordinate, step):
    for x in range(w_start_coordinate, w_end_coordinate, step):
        iter_i += 1
        # 确定fake_bboxes_batch
        fake_positions_t = torch.Tensor([[[x, y, rowPatch_size * ratio, rowPatch_size * ratio, angle_radius]]]).cuda()  # torch.Size([1, 1, 5])
        
        # 循环中插入补丁并进行apply， fake_bboxes_batch滑动窗口有关
        adv_batch_masked, msk_batch = patch_transformer.forward2(adv_batch=fake_patch, fake_bboxes_batch=fake_positions_t, fake_labels_batch=fake_labels_t, img_size=img_size)
        patch_applied_image = patch_applier(origin_image_tensor.cuda(), adv_batch_masked.cuda())  # 这里p_img_batch还没做归一化, BGR, torch.Size([1, 3, H, W])

        # 存储图像
        patch_applied_image_copy = patch_applied_image.clone()
        image_squeeze = patch_applied_image_copy[0, :, :, :]  # (3, 384, 640)
        image_squeeze_np = image_squeeze.permute(1, 2, 0).cpu().detach().numpy()  # (360, 640, 3), BGR
        image_squeeze_np = (np.ascontiguousarray(image_squeeze_np) * 255).astype(np.uint8)
        perturbed_image_save_path = os.path.join(temp_work_dir, 'folder_compare', 'pic1.jpg')
        cv2.imwrite(perturbed_image_save_path, image_squeeze_np)

        # 计算LPIPS值
        lpips_value = calculate_lpips(folder_gt=os.path.join(temp_work_dir,'folder_origin'), folder_restored=os.path.join(temp_work_dir,'folder_compare'))

        # print循环次数
        print(f'iteration process: [{iter_i}/{total_iterations}]')


        # 如果当前LPIPS值更小，则更新最佳位置
        if lpips_value < best_lpips:
            best_lpips = lpips_value
            best_position = (y, x)
            cv2.imwrite(os.path.join(temp_work_dir, 'best_match.jpg'), image_squeeze_np)

print(f'best_location: ({x}, {y})')
print(f'best LPIPS: {best_lpips}')
"""
