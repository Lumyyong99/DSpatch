import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# 假设数据集的路径
data_dir = './forest2271'

# 定义数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


dataset = CustomImageDataset(root_dir=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# 初始化累加器
mean = torch.zeros(3)
std = torch.zeros(3)
total_pixels = 0


# 计算均值
for images in tqdm(dataloader):
    # 将图像展开为 [batch_size, channels, height * width]
    images = images.view(images.size(0), images.size(1), -1)
    # 累加每个通道的总和
    mean += images.mean(2).sum(0)
    total_pixels += images.size(0) * images.size(2)

mean /= total_pixels

# 计算方差
for images in tqdm(dataloader):
    images = images.view(images.size(0), images.size(1), -1)
    std += ((images - mean.view(1, 3, 1)) ** 2).sum([0, 2])

std = torch.sqrt(std / total_pixels)

# 输出结果
print(f"Mean: {mean}")
print(f"Std: {std}")
