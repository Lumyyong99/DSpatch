import torch
import torch.nn as nn
import json
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple
import os, stat

# --------------------- 网络结构定义 ---------------------
class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    """小型UNet变体，适用于像素级分类"""
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # 下采样路径
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = DoubleConv(128, 256)
        
        # 上采样路径
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        # 输出层
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.down1(x)    # [N, 64, H, W]
        x2 = self.pool1(x1)
        x2 = self.down2(x2)  # [N, 128, H/2, W/2]
        x3 = self.pool2(x2)
        
        # Bottleneck
        x3 = self.bottleneck(x3)  # [N, 256, H/4, W/4]
        
        # Decoder
        x = self.up3(x3)      # [N, 128, H/2, W/2]
        x = torch.cat([x, x2], dim=1)  # 跳跃连接
        x = self.conv3(x)     # [N, 128, H/2, W/2]
        
        x = self.up4(x)       # [N, 64, H, W]
        x = torch.cat([x, x1], dim=1)  # 跳跃连接
        x = self.conv4(x)     # [N, 64, H, W]
        
        # 输出
        x = self.out(x)       # [N, 1, H, W]
        return self.sigmoid(x)

# --------------------- 数据集定义 ---------------------
class PatchDataset(Dataset):
    def __init__(self, data_root: str, annotation_path: str, img_size=(640, 360)):
        """
        params:
            data_root: 包含所有PNG图像的文件夹路径
            annotation_path: JSON标注文件路径
        """
        self.data_root = Path(data_root)
        self.img_size = img_size
        
        # 加载标注数据
        with open(annotation_path, 'r') as f:
            self.annotations = json.load(f)
        
        # 获取有效图像路径列表
        self.img_paths = []
        for img_name in self.annotations.keys():
            img_path = self.data_root / img_name
            if img_path.exists() and img_path.suffix.lower() == '.png':
                self.img_paths.append(str(img_path))
            else:
                print(f"Warning: Missing image {img_name}")
        
        # 固定嵌入块尺寸 (w, h)
        self.patch_size = (64, 32)  # OpenCV使用(width, height)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[110.928 / 255.0, 107.967 / 255.0, 108.969 / 255.0], std=[47.737 / 255.0, 48.588 / 255.0, 48.115 / 255.0])     # transform应该是RGB的顺序
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = Path(img_path).name
        
        # 读取图像
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # 图像img是RGB格式
        h, w = img.shape[:2]
        
        # 初始化mask
        mask = np.zeros((h, w), dtype=np.float32)
        
        # 处理当前图像的标注
        for patch_info in self.annotations.get(img_name, {}).get('patches', []):
            # 解析旋转框参数
            center = patch_info['center']
            angle = patch_info['angle']
            
            # 构造OpenCV旋转矩形格式: ((center_x, center_y), (width, height), angle)
            rect = (
                (center[0], center[1]),   # 中心坐标
                self.patch_size,          # 固定尺寸 (w, h)
                angle                     # 旋转角度（度数，OpenCV兼容格式）
            )
            
            # 生成旋转矩形顶点坐标
            box = cv2.boxPoints(rect).astype(np.int32)
            
            # 在mask上绘制填充多边形
            cv2.fillPoly(mask, [box], color=1.0)
        
        # 图像预处理
        img_resized = cv2.resize(img, self.img_size)
        mask_resized = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        
        # 转换为Tensor
        img_tensor = self.transform(img_resized)
        mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0)  # [1, H, W]
        
        return img_tensor, mask_tensor

# --------------------- 训练代码 ---------------------
def train(model, dataloader, device, save_root, lr0, step, epochs=30):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr0)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.2)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            if i == 0:
                # test and visualize,只对第一次迭代进行可视化测试
                image1 = images[0]      # torch.Size([3, 360, 640])
                mask1 = masks[0]        # torch.Size([1, 360, 640])
                # 可视化image
                mean_RGB = [110.928 / 255.0, 107.967 / 255.0, 108.969 / 255.0]  # RGB color for uavrod datset
                std_RGB = [47.737 / 255.0, 48.588 / 255.0, 48.115 / 255.0]  # RGB color for uavrod datset
                for c in range(image1.size(0)):  # 遍历图像的每个通道
                    image1[c] = image1[c].mul(std_RGB[c]).add(mean_RGB[c])  # Tensor, torch.Size([3, H, W]), RGB
                adv_img_rgb = image1.permute(1, 2, 0).cpu().detach().numpy()  # (360, 640, 3), RGB
                adv_img_bgr = adv_img_rgb[:, :, [2, 1, 0]]
                adv_img_bgr = (np.ascontiguousarray(adv_img_bgr) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_root, 'GTimages', f'test_defence_segementor_GTimage_{epoch}.jpg'), adv_img_bgr)

                # 可视化mask
                adv_mask = mask1.squeeze(0).cpu().detach().numpy()  # [360, 640]
                adv_mask = (np.ascontiguousarray(adv_mask) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_root, 'GTmasks', f'./test_defence_segementor_GTmask_{epoch}.jpg'), adv_mask)
                # assert False

            # 前向传播
            outputs = model(images)     # torch.Size([4, 1, 360, 640])

            if i == 0:
                # outputs可视化,只对第一次迭代i=0进行可视化
                output = outputs[0]
                output_np = output.squeeze(0).cpu().detach().numpy()
                output_binary = (output_np > 0.5).astype(np.uint8) * 255
                output_vis = np.ascontiguousarray(output_binary)

                # 进行可视化
                red_overlay = np.zeros_like(adv_img_bgr)
                red_overlay[output_vis == 255] = (0, 0, 255)
                alpha = 0.5
                blended = cv2.addWeighted(adv_img_bgr, 1-alpha, red_overlay, alpha, 0)
                cv2.imwrite(os.path.join(save_root, 'seg_outputs', f'./test_defence_segementor_output_{epoch}.jpg'), blended)


            # 计算损失
            loss = criterion(outputs, masks)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # 调整学习率
        scheduler.step()

        # 保存模型
        model_save_path = os.path.join(save_root, 'segmentor_checkpoints', f'epoch_{epoch+1}_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_save_path)

        current_learning_rate = optimizer.param_groups[0]['lr']
        # 获取每个epoch的平均损失
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Lr: {current_learning_rate}')

# --------------------- 推理代码 ---------------------
"""
def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    '''应用mask到原图'''
    masked_img = image.copy()
    masked_img[mask > 0.5] = 0  # 将高置信度区域置黑
    return masked_img

def inference(model, img_path, device, img_size=(512, 512)):
    # 预处理
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]
    
    # 调整尺寸并归一化
    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img).unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        mask_pred = model(img_tensor).squeeze().cpu().numpy()
    
    # 后处理
    mask_pred = cv2.resize(mask_pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask = (mask_pred > 0.5).astype(np.uint8) * 255
    
    # 应用mask
    result = apply_mask(img, mask)
    return result, mask
"""

if __name__ == "__main__":
    # 硬件配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = UNet().to(device)
    
    # 数据路径配置
    # data_root = "./vis_results_DSAP/deploy_patches_DSAP/faster-rcnn"  # 包含50个PNG文件的文件夹

    # retinanet, advpatch
    # data_root = './vis_results_DSAP/deploy_patches_advpatch/retinanet'
    # faster-rcnn, advpatch
    # data_root = './vis_results_DSAP/deploy_patches_advpatch/faster-rcnn'
    # gliding-vertex, advpatch
    # data_root = './vis_results_DSAP/deploy_patches_advpatch/gliding-vertex'

    # retinanet, DSAP
    # data_root = ''
    # faster-rcnn, DSAP
    # data_root = './vis_results_DSAP/deploy_patches_SingleDSAP/faster-rcnn'
    # gliding-vertex, DSAP
    data_root = './vis_results_DSAP/deploy_patches_SingleDSAP/gliding-vertex'

    annotation_path = "./uavrod_selected_patch_locations_scene50.json"  # 标注文件路径
    
    # 初始化数据集
    dataset = PatchDataset(data_root, annotation_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 初始化训练可视化文件夹
    # save_root = './train_segmentor/segmentor_DSAP/segmentor_faster_rcnn'

    # retinanet, advpatch
    # save_root = './segmentor_training_advpatch_retinanet'
    # faster-rcnn, advpatch
    # save_root = './segmentor_training_advpatch_faster-rcnn'
    # gliding-vertex, advpatch
    # save_root = './segmentor_training_advpatch_gliding-vertex'

    # retinanet, DSAP
    # save_root = './segmentor_training_DSAP_retinanet'
    # faster-rcnn, DSAP
    # save_root = './segmentor_training_DSAP_faster-rcnn'
    # gliding-vertex, DSAP
    save_root = './segmentor_training_DSAP_gliding-vertex'

    subfolders = ['GTimages', 'GTmasks', 'seg_outputs', 'segmentor_checkpoints']
    if not os.path.exists(save_root):
        os.makedirs(save_root)
        os.chmod(save_root, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    else:
        # 若有同名文件则删除
        # shutil.rmtree(args.work_dir)
        print('--------------Warrning: File exists!--------------')
    for subfolder in subfolders:
        subfolder_path = os.path.join(save_root, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            os.chmod(subfolder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # 训练超参
    lr0 = 2e-4
    step = 8
    
    # 训练模型（其余代码保持不变）
    train(model, dataloader, device, save_root, lr0, step, epochs=10)
    
    # # 推理示例
    # test_img_path = "test_image.jpg"
    # result, mask = inference(model, test_img_path, device)
    
    # # 保存结果
    # cv2.imwrite("masked_image.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("predicted_mask.png", mask)
