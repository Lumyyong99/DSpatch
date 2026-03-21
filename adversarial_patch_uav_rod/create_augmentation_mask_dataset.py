import os
from pathlib import Path
from PIL import Image
import glob
import shutil
from shutil import copyfile
import random
import cv2
import numpy as np
from tqdm import tqdm



def create_empty_txt_matching_images(src_img_dir, dst_txt_dir, image_extensions={'.jpg', '.png', '.jpeg', '.bmp', '.webp'}):
    """
    根据图片文件夹创建同名空txt文件
    
    参数：
    src_img_dir: 包含图片的源文件夹路径
    dst_txt_dir: 要创建txt文件的目标文件夹路径
    image_extensions: 要处理的图片扩展名集合
    """
    # 创建目标文件夹（如果不存在）
    Path(dst_txt_dir).mkdir(parents=True, exist_ok=True)
    
    # 遍历源文件夹中的所有文件
    for filename in os.listdir(src_img_dir):
        # 获取文件后缀并转换为小写
        file_ext = Path(filename).suffix.lower()
        
        # 检查是否为图片文件
        if file_ext in image_extensions:
            # 构建纯文件名（不带扩展名）
            stem = Path(filename).stem
            # 生成目标txt文件路径
            txt_path = Path(dst_txt_dir) / f"{stem}.txt"
            
            # 创建空文件（仅在文件不存在时创建）
            if not txt_path.exists():
                txt_path.touch()
                print(f"已创建：{txt_path}")
            else:
                print(f"已存在：{txt_path}")

def convert_jpg_to_png(source_folder, target_folder):
    """
    将源文件夹中的JPG图片转换为PNG格式到目标文件夹
    :param source_folder: 源图片文件夹路径
    :param target_folder: 目标存储文件夹路径
    """
    # 创建目标文件夹（如果不存在）
    os.makedirs(target_folder, exist_ok=True)
    
    # 获取所有JPG文件（包括子文件夹）
    jpg_files = glob.glob(os.path.join(source_folder, "**/*.jpg"), recursive=True)
    
    # 添加大写扩展名支持
    jpg_files += glob.glob(os.path.join(source_folder, "**/*.JPG"), recursive=True)
    jpg_files += glob.glob(os.path.join(source_folder, "**/*.jpeg"), recursive=True)
    jpg_files += glob.glob(os.path.join(source_folder, "**/*.JPEG"), recursive=True)

    for jpg_path in jpg_files:
        try:
            # 生成目标路径（保留原始目录结构）
            relative_path = os.path.relpath(jpg_path, source_folder)
            png_filename = os.path.splitext(relative_path)[0] + ".png"
            png_path = os.path.join(target_folder, png_filename)
            
            # 创建子目录结构
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            
            # 转换并保存图片
            with Image.open(jpg_path) as img:
                # 转换为RGB模式以处理带有alpha通道的JPEG
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(png_path, "PNG")
            
            print(f"转换成功: {os.path.basename(jpg_path)} => {os.path.basename(png_path)}")
            
        except Exception as e:
            print(f"转换失败 {jpg_path}: {str(e)}")


def merge_folders(src1, src2, dst, file_type, conflict_suffix="_dup"):
    """
    合并两个图片文件夹到新目录
    
    参数：
    src1: 第一个源文件夹路径
    src2: 第二个源文件夹路径
    dst: 目标文件夹路径
    file_type: str, select from 'png' or 'txt'
    conflict_suffix: 重名文件后缀模板
    """
    # 创建目标目录
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)
    
    # 文件计数器
    copied_count = 0
    renamed_count = 0

    # 遍历两个源目录
    for src in [Path(src1), Path(src2)]:
        for png_file in src.rglob(f"*.{file_type}"):
            # 保持相对路径
            relative_path = png_file.relative_to(src)
            target_file = dst_path / relative_path
            
            # 处理目录结构
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 智能处理文件冲突
            if target_file.exists():
                # 生成唯一文件名
                counter = 1
                while True:
                    new_name = f"{target_file.stem}{conflict_suffix}{counter}{target_file.suffix}"
                    candidate = target_file.with_name(new_name)
                    if not candidate.exists():
                        target_file = candidate
                        renamed_count += 1
                        break
                    counter += 1
            
            # 复制文件
            shutil.copy2(png_file, target_file)
            copied_count += 1
            
            print(f"已复制: {png_file} -> {target_file}")

    print(f"\n合并完成")
    print(f"总复制文件: {copied_count}")
    print(f"重命名文件: {renamed_count}")

def append_line_to_txt(folder_path, line_content, backup=False):
    """
    为指定文件夹内的所有txt文件追加特定行
    
    参数：
    folder_path: 目标文件夹路径
    line_content: 要追加的内容（自动添加换行符）
    backup: 是否创建备份文件（默认True）
    """
    target_folder = Path(folder_path)
    append_line = line_content + '\n'  # 确保换行
    
    processed = 0
    skipped = 0
    errors = []

    # 遍历所有txt文件（包括子目录）
    for txt_file in target_folder.rglob('*.txt'):
        try:
            # 创建备份（可选）
            if backup:
                backup_path = txt_file.with_suffix('.txt.bak')
                shutil.copy2(txt_file, backup_path)
            
            # 检查是否已包含该行
            with open(txt_file, 'r+', encoding='utf-8') as f:
                existing_content = f.read()
                if append_line in existing_content:
                    skipped += 1
                    continue
                
                # 追加新内容
                f.write(append_line)
                processed += 1
                
        except Exception as e:
            errors.append((str(txt_file), str(e)))
    
    # 输出统计信息
    print(f"处理完成！\n"
          f"成功处理文件数: {processed}\n"
          f"跳过重复文件数: {skipped}\n"
          f"错误文件数: {len(errors)}")
    
    # 显示错误详情
    if errors:
        print("\n错误详情：")
        for file, err in errors:
            print(f"- {file}: {err}")


def process_dataset(orig_img_dir, orig_label_dir, 
                    new_img_dir, new_label_dir,
                    mask_ratio=0.3, mask_color=(0, 0, 0)):
    """
    数据集增强处理函数
    :param orig_img_dir: 原始图像目录
    :param orig_label_dir: 原始标注目录
    :param new_img_dir: 新图像存储目录
    :param new_label_dir: 新标注存储目录
    :param mask_ratio: mask比例（默认30%）
    :param mask_color: mask填充颜色（默认黑色）
    """
    # 创建新目录
    os.makedirs(new_img_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)

    # 获取所有图像文件
    img_files = [f for f in os.listdir(orig_img_dir) if f.endswith('.png')]
    
    for img_file in tqdm(img_files, desc="Processing images"):
        # 路径设置
        img_path = os.path.join(orig_img_dir, img_file)
        label_path = os.path.join(orig_label_dir, img_file.replace('.png', '.txt'))
        new_img_path = os.path.join(new_img_dir, img_file)
        new_label_path = os.path.join(new_label_dir, img_file.replace('.png', '.txt'))

        # 读取图像和标注
        img = cv2.imread(img_path)
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 解析标注
        all_objects = []
        for line in lines:
            parts = line.strip().split()
            coords = list(map(int, parts[:8]))
            class_name = parts[8]
            difficulty = parts[9]
            all_objects.append((coords, class_name, difficulty))

        # 随机选择要mask的目标
        num_to_mask = max(1, int(len(all_objects) * mask_ratio))
        masked_indices = random.sample(range(len(all_objects)), num_to_mask)

        # 生成mask图像
        masked_img = img.copy()
        for idx in masked_indices:
            coords = np.array(all_objects[idx][0], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(masked_img, [coords], color=mask_color)

        # 保存新图像
        cv2.imwrite(new_img_path, masked_img)

        # 生成新标注（排除mask的目标）
        new_objects = [obj for i, obj in enumerate(all_objects) if i not in masked_indices]
        
        # 保存新标注
        with open(new_label_path, 'w') as f:
            for obj in new_objects:
                coords = ' '.join(map(str, obj[0]))
                line = f"{coords} {obj[1]} {obj[2]}\n"
                f.write(line)


# 使用示例
if __name__ == "__main__":
    # 为mask iamges创建空txt_annotations文件
    # source_dir = "../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/cropped_masked_images_640x360"  # 替换为你的图片文件夹路径
    # target_dir = "../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/cropped_masked_images_txt_annotations"  # 替换为目标txt文件夹路径
    # create_empty_txt_matching_images(source_dir, target_dir)

    # 替换文件夹下.jpg文件为.png文件
    # source_folder="../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/cropped_masked_images_640x360"
    # target_folder="../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/cropped_masked_images_640x360_png"
    # convert_jpg_to_png(source_folder, target_folder)

    # 合并图像文件夹，生成augmented dataset
    # src1='../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/cropped_masked_images_640x360_png'
    # src2='../../datasets/UAV-ROD/train_640x360/images'
    # dst='../../datasets/UAV-ROD/MaskAugmented_train_640x360/images'
    # file_type='png'
    # merge_folders(src1, src2, dst, file_type)

    # 合并图像文件夹，生成augmented dataset
    # src1='../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/cropped_masked_images_txt_annotations'
    # src2='../../datasets/UAV-ROD/train_640x360/txt_annotations'
    # dst='../../datasets/UAV-ROD/MaskAugmented_train_640x360/txt_annotations'
    # file_type='txt'
    # merge_folders(src1, src2, dst, file_type)

    # 为txt文件添加标签，为了避免训练过程中出现报错
    # target_folder = "../../datasets/UAV-ROD/UAV-ROD-scene50-640x360/cropped_masked_images_txt_annotations"  # 替换为实际路径
    # new_line = "0 0 0 1 1 1 1 0 car 0"
    # append_line_to_txt(target_folder, new_line)

    # 在原始数据集基础上添加mask进行增强，每张图像随机选取30%的物体进行添加
    orig_image_dir = "../../datasets/UAV-ROD/train_640x360/images"
    orig_label_dir = "../../datasets/UAV-ROD/train_640x360/txt_annotations"
    new_image_dir = "../../datasets/UAV-ROD/train_MaskAug_random0.3_640x360/images"
    new_label_dir = "../../datasets/UAV-ROD/train_MaskAug_random0.3_640x360/txt_annotations"
    process_dataset(
        orig_img_dir=orig_image_dir,
        orig_label_dir=orig_label_dir,
        new_img_dir=new_image_dir,
        new_label_dir=new_label_dir,
    )


