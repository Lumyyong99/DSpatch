import os
import shutil
import random

def copy_images(src_folders, dst_folder, total_images=2000):
    # 确保目标文件夹存在
    os.makedirs(dst_folder, exist_ok=True)
    
    # 计算每个文件夹需要复制的图片数量
    num_folders = len(src_folders)
    base_count = total_images // num_folders
    remainder = total_images % num_folders
    counts = [base_count + 1 if i < remainder else base_count for i in range(num_folders)]
    
    copied_count = 0
    
    for i, folder in enumerate(src_folders):
        # 获取文件夹中所有图片文件
        all_images = [f for f in os.listdir(folder) 
                     if os.path.isfile(os.path.join(folder, f)) and 
                     f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # 随机选择指定数量的图片
        selected_images = random.sample(all_images, min(counts[i], len(all_images)))
        
        # 复制选中的图片到目标文件夹
        for img in selected_images:
            src_path = os.path.join(folder, img)
            dst_path = os.path.join(dst_folder, img)
            
            # 处理文件名冲突
            counter = 1
            while os.path.exists(dst_path):
                name, ext = os.path.splitext(img)
                dst_path = os.path.join(dst_folder, f"{name}_{counter}{ext}")
                counter += 1
                
            shutil.copy2(src_path, dst_path)
            copied_count += 1
    
    print(f"成功复制 {copied_count} 张图片到 {dst_folder}")

# 使用示例
if __name__ == "__main__":
    # 设置源文件夹路径
    folder1 = "/home/Adversarial/datasets/iharbour_dataset_2/RoadSnip"  # 替换为实际路径
    folder2 = "/home/Adversarial/datasets/iharbour_dataset_2/grassnip"  # 替换为实际路径
    folder3 = "/home/Adversarial/datasets/UAV-ROD/source_domain_32x64"  # 替换为实际路径
    
    # 设置目标文件夹路径
    destination = "/home/Adversarial/datasets/UAV-ROD/source_domain_append_32x64"  # 替换为实际路径
    
    # 执行复制操作
    copy_images([folder1, folder2, folder3], destination)
