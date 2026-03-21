import os
import random
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
import stat
"""
log 2025.2.20
重新选择源域图像，提出几个要求：
（1）不选择全黑或者全白的图像，避免污染源域图像
（2）每个类别分开选择，比如飞机就在DOTA-plane-propersize中进行选择
"""


def convert_to_le90(vertices):
    """
    convert DOTA annotations to le90 format annotations.
    Parameters:
        DOTA format annotations(list): [x1, y1, x2, y2, x3, y3, x4, y4]
    Returns:
        le90 annotations(list): [cx, cy, w, h, theta], theta belongs to [-pi/2, pi/2)
    """
    # Unpack vertices
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    # Calculate center
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4
    # Calculate edge vectors
    edges = [
        (x2 - x1, y2 - y1),
        (x3 - x2, y3 - y2),
        (x4 - x3, y4 - y3),
        (x1 - x4, y1 - y4)
    ]
    # Calculate lengths of edges
    lengths = [np.sqrt(dx ** 2 + dy ** 2) for dx, dy in edges]
    # Find width and height
    w, h = lengths[:2]
    # Calculate angle
    if w < h:
        w, h = h, w
    if lengths[0] > lengths[1]:
        theta = np.arctan2(edges[0][1], edges[0][0])
    else:
        theta = np.arctan2(edges[1][1], edges[1][0])
    # Normalize angle to [-pi/2, pi/2]
    if theta < -np.pi / 2:
        theta += np.pi
    elif theta > np.pi / 2:
        theta -= np.pi
    return [cx, cy, w, h, theta]

def filter_images_by_class(image_dir, label_dir, output_image_dir, output_label_dir, class_name, num_samples=200):
    # 确保输出文件夹存在
    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    Path(output_label_dir).mkdir(parents=True, exist_ok=True)

    # 获取所有图像和标签文件路径
    image_files = sorted(Path(image_dir).glob("*.png"))  # 假设图像文件是png格式
    label_files = sorted(Path(label_dir).glob("*.txt"))  # 假设标签文件是txt格式

    # 筛选包含指定类别的图像和标签
    valid_images = []
    valid_labels = []

    for image_file, label_file in zip(image_files, label_files):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if parts[8] == class_name:
                    valid_images.append(image_file)
                    valid_labels.append(label_file)
                    break  # 该图像包含目标物体类别，跳出检查

    # 随机选择一定数量的图像和标签
    if len(valid_images) < num_samples:
        print(f"找到的图像不足 {num_samples} 张，只选取了 {len(valid_images)} 张。")
        selected_images = valid_images
        selected_labels = valid_labels
    else:
        selected_images, selected_labels = zip(*random.sample(list(zip(valid_images, valid_labels)), num_samples))

    # 将选中的图像和标签复制到新的文件夹
    for image_file, label_file in zip(selected_images, selected_labels):
        # 复制图像
        shutil.copy(image_file, output_image_dir)
        # 复制标签
        shutil.copy(label_file, output_label_dir)

    print(f"成功从 {len(valid_images)} 张图像中随机选取了 {num_samples} 张包含 '{class_name}' 类别的图像。")

def total_median_calculate(class_name, label_folder):
    """
    计算给定class_name下物体尺寸的中位数，在全部标注中找
    """
    widths, heights = [], []
    for filename in tqdm(os.listdir(label_folder)):
        filepath = os.path.join(label_folder, filename)
        if not filename.endswith(".txt"):
            continue
        with open(filepath, 'r') as file:
            for line in file.readlines():
                parts = line.strip().split()
                bbox_vertices = list(map(float, parts[:8]))  # 角点坐标
                label = parts[8]  # 物体类别

                if label == class_name:
                    le90_list = convert_to_le90(bbox_vertices)
                    widths.append(le90_list[2])
                    heights.append(le90_list[3])
    if not widths or not heights:
        return None

    median_width = np.median(widths)
    median_height = np.median(heights)
    return [median_width, median_height]

def file_median_calculate(class_name, label_file):
    """
    计算给定class_name下物体尺寸的中位数，只在一个file中计算目标物体的尺寸中位数
    """
    widths, heights = [], []
    with open(label_file, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()
            bbox_vertices = list(map(float, parts[:8]))  # 角点坐标
            label = parts[8]  # 物体类别

            if label == class_name:
                le90_list = convert_to_le90(bbox_vertices)
                widths.append(le90_list[2])
                heights.append(le90_list[3])
    if not widths or not heights:
        return None

    median_width = np.median(widths)
    median_height = np.median(heights)
    return [median_width, median_height]

def select_proper_objects_scene(class_name, proper_size, input_folder, save_folder, box_rate=0.15,max_images=200):
    """
    选择包含合适尺寸物体的图像，共max_images张。
    合适尺寸指与proper_size差距不大。
    按照每张图片目标物体的中位数作为该图片中目标物体的平均尺寸进行比较
    proper_size输入格式为[w, h]
    save_folder文件夹下包括images和labelTxt
    """
    proper_w, proper_h = proper_size[0], proper_size[1]
    copied_images_flag = 0
    input_label_folder = os.path.join(input_folder, 'labelTxt')

    for filename in tqdm(os.listdir(input_label_folder)):
        if copied_images_flag >= max_images:
            break

        filepath = os.path.join(input_label_folder, filename)
        if not filename.endswith(".txt"):
            continue

        current_image_median_size = file_median_calculate(class_name=class_name, label_file=filepath)

        if current_image_median_size:
            current_image_median_width, current_image_median_height = current_image_median_size[0], current_image_median_size[1]
            width_diff = abs(current_image_median_width - proper_w)
            height_diff = abs(current_image_median_height - proper_h)

            if width_diff <= box_rate * proper_w and height_diff <= box_rate * proper_h:
                # 复制标签文件
                dest_label_path = os.path.join(save_folder, 'labelTxt', filename)
                shutil.copy(filepath, dest_label_path)
                # 复制图片文件
                image_filename = filename.replace('.txt','.png')
                image_filepath = os.path.join(input_folder, 'images', image_filename)
                dest_image_path = os.path.join(save_folder, 'images', image_filename)
                shutil.copy(image_filepath, dest_image_path)

                # 计数
                copied_images_flag += 1
    print(f'copied {copied_images_flag} images')



if __name__ == "__main__":
    # # 选择200张包含目标物体的场景图像
    # # 选择helicopter用这一部分
    # image_dir = "../../datasets/DOTA-V1.0/DOTA-sub/DOTA-sub512-gap100-rate1/images"  # 图像文件夹路径
    # label_dir = "../../datasets/DOTA-V1.0/DOTA-sub/DOTA-sub512-gap100-rate1/labelTxt"  # 标签文件夹路径
    # output_image_dir = "../../datasets/DOTA-V1.0/DOTA-helicopter/images"  # 输出图像文件夹路径
    # output_label_dir = "../../datasets/DOTA-V1.0/DOTA-helicopter/labelTxt"  # 输出标签文件夹路径
    # class_name = "helicopter"  # 目标物体类别名
    # num_samples = 200  # 需要选择的图像数量
    # filter_images_by_class(image_dir, label_dir, output_image_dir, output_label_dir, class_name, num_samples)

    # 计算数据集中plane物体的长宽中位数
    # class_name = 'plane'
    # label_folder = '/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-sub/DOTA-sub512-gap100-rate1/labelTxt'
    # median_size = total_median_calculate(class_name=class_name, label_folder=label_folder)
    # print(f'{class_name} object median size: [{median_size[0]}, {median_size[1]}]')

    """
    挑选200张目标物体与给定尺寸相差不大的图像
    plane: [75, 50]  ship: [40, 14]  large-vehicle: [55, 20]  helicoper: [66, 24]  small-vehicle: [35, 16], 注意这里不需要考虑补丁的尺寸，只与原本物体的中位数比较就行了
    生成补丁尺寸：plane:[64, 48], 
    注意，helicopter按这种方式选择不行。可能是由于helicopter场景本身偏少，包含helicopter物体类别的本身就只有184张
    """

    # 选择飞机场景
    class_name = 'plane'
    proper_size = [64, 48]
    input_folder = f'../../datasets/DOTA-V1.0/datasets/DOTA-V1.0/DOTA-plane-propersize'
    save_folder = f'../../datasets/DOTA-V1.0/DOTA-{class_name}-propersize-scene50'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.chmod(save_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    images_folder = os.path.join(save_folder, 'images')
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
        os.chmod(images_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    label_folder = os.path.join(save_folder, 'labelTxt')
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        os.chmod(label_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    select_proper_objects_scene(class_name, proper_size, input_folder, save_folder ,max_images=50)


