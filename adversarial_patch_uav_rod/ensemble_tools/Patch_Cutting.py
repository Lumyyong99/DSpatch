"""
从场景数据集中裁剪一定数量的patch作为 GAN网络的训练集
"""

import os
import stat
import cv2
import random


def extract_random_patches(image_folder, output_folder, patch_w, patch_h, number):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        os.chmod(output_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # 获取文件夹中所有图片的文件名
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # 计算每张图片需要截取的patch数量
    patches_per_image = number // len(image_files)
    patch_count = 0

    for image_file in image_files:
        # 读取图片
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape
        for _ in range(patches_per_image):
            # 随机生成patch的左上角坐标
            x = random.randint(0, img_w - patch_w)
            y = random.randint(0, img_h - patch_h)
            # 截取patch
            patch = image[y:y + patch_h, x:x + patch_w]
            # 生成patch的文件名并保存
            patch_filename = os.path.join(output_folder, f'patch_{patch_count}.png')
            cv2.imwrite(patch_filename, patch)
            patch_count += 1

    # 如果还有剩余的patch未生成，则从头开始补齐
    while patch_count < number:
        for image_file in image_files:
            if patch_count >= number:
                break
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            img_h, img_w, _ = image.shape
            x = random.randint(0, img_w - patch_w)
            y = random.randint(0, img_h - patch_h)
            patch = image[y:y + patch_h, x:x + patch_w]
            patch_filename = os.path.join(output_folder, f'patch_{patch_count}.png')
            cv2.imwrite(patch_filename, patch)
            patch_count += 1


if __name__ == '__main__':
    # 示例使用
    image_folder = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_PATCHES/DOTA_RESIZE/sub640/images'
    output_folder = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_PATCHES/patches_64x64'
    patch_w = 64
    patch_h = 64
    number = 10000

    extract_random_patches(image_folder, output_folder, patch_w, patch_h, number)
