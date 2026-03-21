import os
import cv2
import json
import numpy as np
import glob
import argparse
import pyiqa
import torch
from math import sqrt
from pathlib import Path

def calculate_patch_region(center, angle, patch_w=32, patch_h=64):
    """计算截取区域的正方形坐标"""
    diagonal = int(sqrt(patch_w**2 + patch_h**2))
    half_size = diagonal // 2
    
    x_min = max(0, center[0] - half_size)
    y_min = max(0, center[1] - half_size)
    x_max = center[0] + half_size
    y_max = center[1] + half_size
    
    return (x_min, y_min, x_max, y_max)

def crop_and_save_patches(args):
    """截取并保存图像块"""
    Path(args.output_orig).mkdir(parents=True, exist_ok=True)
    Path(args.output_adv).mkdir(parents=True, exist_ok=True)

    with open(args.json_path) as f:
        patch_info = json.load(f)

    patch_ratio = 1.5
    patch_w = 32 * patch_ratio  # 补丁原始宽度
    patch_h = 64 * patch_ratio  # 补丁原始高度


    patch_count = 0
    for img_name in patch_info:
        # 读取原始图像和对抗图像
        orig_img = cv2.imread(os.path.join(args.folder_orig, img_name))
        adv_img = cv2.imread(os.path.join(args.folder_adv, img_name))
        
        if orig_img is None or adv_img is None:
            print(f"Warning: Missing image {img_name}")
            continue

        # 处理每个补丁
        for i, patch in enumerate(patch_info[img_name]["patches"]):
            center = patch["center"]
            x_min, y_min, x_max, y_max = calculate_patch_region(center, patch["angle"], patch_w=patch_w, patch_h=patch_h)

            # 边界检查
            h, w = orig_img.shape[:2]


            # 1
            # if x_max > w or y_max > h:
            #     print(f"Warning: Patch {i} in {img_name} exceeds image boundary")
            #     continue
            # # 截取图像块
            # orig_patch = orig_img[y_min:y_max, x_min:x_max]
            # adv_patch = adv_img[y_min:y_max, x_min:x_max]



            # 2
            # # 计算实际截取坐标
            # x_start = max(0, x_min)
            # y_start = max(0, y_min)
            # x_end = min(w, x_max)
            # y_end = min(h, y_max)
            
            # # 创建空白画布
            # patch_size = x_max - x_min
            # orig_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            # adv_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            
            # # 计算偏移量
            # x_offset = x_start - x_min
            # y_offset = y_start - y_min
            
            # # 填充有效区域
            # valid_h = y_end - y_start
            # valid_w = x_end - x_start
            # if valid_h > 0 and valid_w > 0:
            #     orig_patch[y_offset:y_offset+valid_h, x_offset:x_offset+valid_w] = orig_img[y_start:y_end, x_start:x_end]
            #     adv_patch[y_offset:y_offset+valid_h, x_offset:x_offset+valid_w] = adv_img[y_start:y_end, x_start:x_end]
            # else:
            #     print(f"Warning: Patch {i} in {img_name} has no valid area")
            

            # 3
            # 计算截取区域的固定尺寸
            diagonal = int(np.sqrt(patch_w**2 + patch_h**2))  # 计算对角线长度

            # 计算截取区域坐标
            x_center, y_center = patch["center"]
            x_start = x_center - diagonal // 2
            x_end = x_start + diagonal
            y_start = y_center - diagonal // 2
            y_end = y_start + diagonal

            # 获取有效区域
            valid_x_start = max(x_start, 0)
            valid_y_start = max(y_start, 0)
            valid_x_end = min(x_end, w)
            valid_y_end = min(y_end, h)

            # 执行截取
            orig_patch = orig_img[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
            adv_patch = adv_img[valid_y_start:valid_y_end, valid_x_start:valid_x_end]

            # 计算需要填充的边界
            pad_left = abs(min(x_start, 0))
            pad_right = max(x_end - w, 0)
            pad_top = abs(min(y_start, 0))
            pad_bottom = max(y_end - h, 0)

            # 应用填充 (使用numpy.pad)
            orig_patch = np.pad(orig_patch, 
                            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                            mode='constant')
            adv_patch = np.pad(adv_patch,
                            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                            mode='constant')

            # 尺寸验证断言
            assert orig_patch.shape[:2] == (diagonal, diagonal), f"Invalid patch shape {orig_patch.shape}"


            # 保存图像块
            base_name = os.path.splitext(img_name)[0]
            cv2.imwrite(f"{args.output_orig}/{base_name}_patch{i}.png", orig_patch)
            cv2.imwrite(f"{args.output_adv}/{base_name}_patch{i}.png", adv_patch)
            patch_count += 1

    print(f"Total processed patches: {patch_count}")

def calculate_metrics(args):
    """计算图像质量指标"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化指标计算器
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    psnr_metric = pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr')
    ssim_metric = pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr')

    # 获取图像列表并排序
    orig_patches = sorted(glob.glob(os.path.join(args.output_orig, "*.png")))
    adv_patches = sorted(glob.glob(os.path.join(args.output_adv, "*.png")))

    # 确保配对正确
    assert len(orig_patches) == len(adv_patches), "Mismatched number of patches"

    metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": []
    }

    for orig_path, adv_path in zip(orig_patches, adv_patches):
        # 读取图像并转换为RGB
        orig = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
        adv = cv2.cvtColor(cv2.imread(adv_path), cv2.COLOR_BGR2RGB)

        # 转换为PyTorch张量
        orig_tensor = torch.from_numpy(orig).permute(2,0,1).float().unsqueeze(0)/255.0
        adv_tensor = torch.from_numpy(adv).permute(2,0,1).float().unsqueeze(0)/255.0

        # 计算指标
        metrics["psnr"].append(psnr_metric(adv_tensor, orig_tensor))
        metrics["ssim"].append(ssim_metric(adv_tensor, orig_tensor))
        metrics["lpips"].append(lpips_metric(adv_tensor, orig_tensor))

    # 计算平均指标
    final_metrics = {
        "PSNR": np.mean([x.item() for x in metrics["psnr"]]),
        "SSIM": np.mean([x.item() for x in metrics["ssim"]]),
        "LPIPS": np.mean([x.item() for x in metrics["lpips"]])
    }

    print("\nFinal Metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_orig", type=str, required=True, help="原始图像文件夹路径")
    parser.add_argument("--folder_adv", type=str, required=True, help="对抗图像文件夹路径") 
    parser.add_argument("--json_path", type=str, required=True, help="补丁信息JSON文件路径")
    parser.add_argument("--output_orig", type=str, default="original_patches", help="原始图像块输出路径")
    parser.add_argument("--output_adv", type=str, default="adversarial_patches", help="对抗图像块输出路径")
    
    args = parser.parse_args()

    # 第一步：截取并保存图像块
    print("Processing image patches...")
    crop_and_save_patches(args)
    
    # 第二步：计算图像质量指标
    print("\nCalculating metrics...")
    calculate_metrics(args)
