import os
import random
import cv2
import numpy as np
from tqdm import tqdm

class SimplePatchSampler:
    def __init__(self, src_dir, dst_dir, target_num=1000, patch_size=(48, 64)):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.target_num = target_num
        self.patch_size = patch_size
        self.counter = 0
        
        os.makedirs(dst_dir, exist_ok=True)
        self.file_list = self._pre_index_files()
    
    def _pre_index_files(self):
        """预筛选有效图像文件,选出比补丁尺寸大的图像"""
        valid_files = []
        for fname in os.listdir(self.src_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(self.src_dir, fname)
                img = cv2.imread(path)
                if img is not None and img.shape[0] >= self.patch_size[0] and img.shape[1] >= self.patch_size[1]:
                    valid_files.append(path)
        return valid_files
    
    def _has_solid_subregion(self, patch, sub_size=(16, 16), threshold=10):
        """
        检测是否存在纯色子区域
        参数：
        sub_size: 子区域大小
        threshold: 颜色方差阈值
        """
        h, w = patch.shape[:2]
        # 滑动窗口检测
        for y in range(0, h - sub_size[0] + 1, sub_size[0]//2):
            for x in range(0, w - sub_size[1] + 1, sub_size[1]//2):
                sub = patch[y:y+sub_size[0], x:x+sub_size[1]]
                if np.std(sub) < threshold:
                    return True
        return False
    
    def _is_valid_patch(self, patch):
        """自定义条件判断接口（示例：过滤低方差块）"""
        if np.std(patch) < 15:
            return False

        if self._has_solid_subregion(patch):
            return False
        
        return True

    def _sample_from_image(self, img_path, max_attempts=50):
        """从单张图像采样合格块"""
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        h, w = img.shape[:2]
        ph, pw = self.patch_size
        patches = []
        
        for _ in range(max_attempts):
            y = random.randint(0, h - ph)
            x = random.randint(0, w - pw)
            patch = img[y:y+ph, x:x+pw]
            
            if self._is_valid_patch(patch):
                patches.append(patch)
                if len(patches) >= 5:  # 单图最多取5个块防止偏斜
                    break
        
        return patches

    def run(self):
        """主运行逻辑"""
        with tqdm(total=self.target_num, desc='采样进度') as pbar:
            # 第一阶段：顺序处理预加载文件
            random.shuffle(self.file_list)  # 随机化处理顺序
            for path in self.file_list:
                if self.counter >= self.target_num:
                    break
                
                patches = self._sample_from_image(path)
                for patch in patches:
                    cv2.imwrite(os.path.join(self.dst_dir, f"patch_{self.counter:06d}.jpg"), patch)
                    self.counter += 1
                    pbar.update(1)
                    if self.counter >= self.target_num:
                        break

            # 第二阶段：如果未达目标，循环随机采样
            while self.counter < self.target_num:
                path = random.choice(self.file_list)
                patches = self._sample_from_image(path, max_attempts=10)
                
                for patch in patches:
                    if self.counter >= self.target_num:
                        break
                    cv2.imwrite(os.path.join(self.dst_dir, f"patch_{self.counter:06d}.jpg"), patch)
                    self.counter += 1
                    pbar.update(1)

if __name__ == "__main__":
    # select source domain images for dota dataset
    # sampler = SimplePatchSampler(
    #     src_dir="../../datasets/DOTA-V1.0/DOTA-ship-propersize/images",
    #     dst_dir="./DOTA_source_domain_images/SourceDomainImages_ship",
    #     target_num=2000,
    #     patch_size=(16, 32)
    # )
    # sampler.run()

    # select source domain images for uavrod datset
    sampler = SimplePatchSampler(
        src_dir="../../datasets/UAV-ROD/train_640x360/images",
        dst_dir="../../datasets/UAV-ROD/source_domain_32x64",
        target_num=2000,
        patch_size=(32, 64)
    )
    sampler.run()
