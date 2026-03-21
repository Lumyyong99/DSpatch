import os
from PIL import Image, ImageChops, ImageDraw
import cv2
import torch
import numpy as np
from PIL.ImageCms import profileToProfile


def crop_images_in_folder(folder_path, save_path):
    # 获取文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        # 检查文件是否为图片文件
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 打开图片
            image = Image.open(file_path)

            # 检查图片的尺寸是否为 (384, 640, 3)
            if image.size == (640, 384):
                # 裁剪图片：裁去底部的24像素
                cropped_image = image.crop((0, 0, 640, 360))

                # 保存裁剪后的图片，保存在原文件夹中
                cropped_image.save(os.path.join(save_path, f"{file_name}"))
                print(f"Processed {file_name}")
            else:
                print(f"Skipped {file_name}: size is not (384, 640, 3)")

def resize_images_in_folder(folder_path, save_path, target_size=(360, 640)):
    # 确保保存目录存在
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # 检查是否为图像文件（支持 jpg, png, jpeg 格式）
        if file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            # 读取图像
            image = cv2.imread(file_path)
            # 检查是否成功加载图像
            if image is None:
                print(f"无法加载图像: {file_name}")
                continue
            # 调整尺寸
            resized_image = cv2.resize(image, (target_size[1], target_size[0]))  # (width, height)
            # 生成保存路径
            save_image_path = os.path.join(save_path, file_name)
            # 保存图像
            cv2.imwrite(save_image_path, resized_image)
            print(f"process: {file_name} -> {save_image_path}")

def pt_to_image(pt_file, save_path, target_size=(64, 128)):
    # 读取 .pt 文件中的张量
    patch_t = torch.load(pt_file)       # torch.Size([1, 3, 32, 64])
    # 将张量从 [64, 32] 转换为 NumPy 数组
    patch_t_squeeze = patch_t[0]
    patch_np = patch_t_squeeze.permute(1, 2, 0).cpu().detach().numpy()
    # 调整尺寸，使用 OpenCV 进行 resize
    resized_patch = cv2.resize(patch_np, (target_size[1], target_size[0]))  # (width, height)
    # 将图像归一化到 [0, 255] 区间并转换为 uint8 类型
    resized_patch = np.ascontiguousarray(resized_patch * 255).astype(np.uint8)
    # 保存图片
    cv2.imwrite(save_path, resized_patch)
    print(f"patch saved: {save_path}")


def resize_patch_image(input_image_path, output_image_path, target_size=(64, 128)):
    # 读取图片
    image = cv2.imread(input_image_path)
    # 调整尺寸
    resized_image = cv2.resize(image, (target_size[1], target_size[0]))  # (width, height)
    # 保存为 PNG 格式
    cv2.imwrite(output_image_path, resized_image)
    print(f"patch image saved: {output_image_path}")


def rgb_to_cmyk_with_profile(rgb_image, cmyk_profile_path=None):
    """将RGB图像转换为CMYK模式，支持ICC配置文件"""
    if cmyk_profile_path and os.path.exists(cmyk_profile_path):
        # 使用ICC配置文件进行专业转换
        return profileToProfile(
            rgb_image,
            inputProfile='./system_color_settings/sRGB.icm',  # 内置sRGB配置文件
            outputProfile=cmyk_profile_path,
            renderingIntent=0,  # 感知渲染
            outputMode='CMYK'
        )
    else:
        # 回退到Pillow默认公式转换
        return rgb_image.convert('CMYK')

def create_comparison_image(rgb_path, output_path, cmyk_profile=None):
    # 加载原始RGB图像
    rgb_img = Image.open(rgb_path).convert('RGB')

    # 转换为CMYK
    cmyk_img = rgb_to_cmyk_with_profile(rgb_img, cmyk_profile)
    
    # 为展示将CMYK转换回RGB（模拟打印效果）
    preview_cmyk = cmyk_img.convert('RGB')
    
    # 创建对比图（左右拼接）
    width, height = rgb_img.size
    comparison = Image.new('RGB', (width*2, height))
    comparison.paste(rgb_img, (0, 0))
    comparison.paste(preview_cmyk, (width, 0))
    
    # 添加分割线
    draw = ImageDraw.Draw(comparison)
    draw.line([(width,0), (width,height)], fill='red', width=2)
    
    # 保存结果
    comparison.save(output_path, quality=95)
    print(f"对比图已保存至：{output_path}")


if __name__ == "__main__":
    # # crop
    # folder_path = '../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/mask_images_640x360'  # 替换为你的文件夹路径
    # save_path = '../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/cropped_masked_images_640x360'
    # crop_images_in_folder(folder_path, save_path)

    # # resize
    # folder_path = './vis_results_DSAP/uavrod_phys_exps/origin_images'
    # save_path = './vis_results_DSAP/uavrod_phys_exps/resized_uavrod_phys_images'
    # resize_images_in_folder(folder_path, save_path)

    # # load patch(tensor) to image
    # pt_file = "./train_advpatch_uavrod_phys/patch_pt/patch_epoch100.pt"  # 替换为实际的 .pt 文件路径
    # save_path = "./train_advpatch_uavrod_phys/print_phys_patch/uavrod_phys_patch_128x64.png"  # 替换为保存图片的路径
    # pt_to_image(pt_file, save_path)

    # resize patch image
    # input_image_path = './vis_results_DSAP/uavrod_phys_exps/imperceptible_patches_2/loc_3/imperceptive_patch/imp_patch.jpg'  # 替换为实际的图片路径
    # output_image_path = './vis_results_DSAP/uavrod_phys_exps/imperceptible_patches_2/print_patches/loc3_patch.png'  # 替换为保存图片的路径
    # resize_patch_image(input_image_path, output_image_path)

    # # resize uavrod phys scenes
    # input_image_path = './vis_results_DSAP/uavrod_phys_scenes/uavrod_phys_scene2.jpg'  # 替换为实际的图片路径
    # output_image_path = './vis_results_DSAP/uavrod_phys_scenes_resize/uavrod_phys_scene2.jpg'  # 替换为保存图片的路径
    # resize_patch_image(input_image_path, output_image_path,target_size=(360, 640))

    # resize uavrod adversarial phys scenes
    # input_image_path = './vis_results_DSAP/uavrod_phys_adv_scenes/Indoorphys1.jpg'  # 替换为实际的图片路径
    # output_image_path = './vis_results_DSAP/uavrod_phys_adv_scenes_resize/Indoorphys1.jpg'  # 替换为保存图片的路径
    # resize_patch_image(input_image_path, output_image_path,target_size=(360, 640))

    # rgb2cmyk
    # rgb_path = './test_DSAP_uavrod/multidetectors/retinanet_o-asr0.7/fake_patch/scene48_patch1.jpg'
    # output_path = './cmyk_rgb_compare.jpg'
    # cmyk_profile = './system_color_settings/CoatedFOGRA39.icc'
    # create_comparison_image(rgb_path, output_path, cmyk_profile)

    # 32x64 patches to 320x640
    folder_path = './20250321_print_patch'
    save_path = './20250321_print_patch_resize1080x2160'
    target_size = (1080, 2160)
    resize_images_in_folder(folder_path, save_path, target_size)