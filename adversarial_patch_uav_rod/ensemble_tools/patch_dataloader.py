import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset


class ForestPatch(Dataset):
    """
    load Forest style images for GAN training.
    Attention: load BGR format images
    Parameters:
        folder_path (str): the path to the folder containing the Forest style images.
    """

    def __init__(self, folder_path, patch_generating_transformer):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, forest_image_file) for forest_image_file in os.listdir(folder_path) if forest_image_file.endswith(('jpg', 'png'))]
        self.patch_generating_transformer = patch_generating_transformer

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        forest_img_path = self.image_files[index]
        forest_image = torch.from_numpy(cv2.imread(forest_img_path) / 255.0).permute(2, 0, 1).to(torch.float32)  # 读取的图片范围为[0,1]，才能用PatchGeneratingTransformer进行归一化。forest_image为BGR格式
        return self.patch_generating_transformer(forest_image)


class ScenePatch(Dataset):
    """
    load scene patch dataset for GAN training.
    Attention: load BGR format images
    Parameters:
        folder_path (str): the path to the folder containing the scene patch style images.
    """

    def __init__(self, folder_path, scene_patch_generating_transformer):
        self.folder_path = folder_path
        self.image_files = [os.path.join(folder_path, scene_patch_file) for scene_patch_file in os.listdir(folder_path) if scene_patch_file.endswith(('jpg', 'png'))]
        self.scene_patch_generating_transformer = scene_patch_generating_transformer

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        scene_patch_img_path = self.image_files[index]
        scene_patch_image = torch.from_numpy(cv2.imread(scene_patch_img_path) / 255.0).permute(2, 0, 1).to(torch.float32)  # 读取的图片范围为[0,1]，才能用PatchGeneratingTransformer进行归一化。forest_image为BGR格式
        return self.scene_patch_generating_transformer(scene_patch_image)
