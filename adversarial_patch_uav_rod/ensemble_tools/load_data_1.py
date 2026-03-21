import fnmatch

import math
import os
import sys
import time
from operator import itemgetter
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchgeometry as tgm

# from ensemble_tools.utils0 import *

import math


class TotalVariation(nn.Module):
    """
    TV loss calculation function
    TV loss: |(p(i,j)-p(i,j+1))|+|(p(i,j)-p(i+1,j))|  beta default to 1.0  后续修改？
    Parameters:
        beta (float): TV loss factor
    Returns:
        TV loss (Tensor): total variation.
    """

    def __init__(self, beta=1.0):
        super(TotalVariation, self).__init__()
        self.beta = beta

    def forward(self, adv_batch):
        """
        TotalVariation: calculates the total variation of a patch.
        Parameters:
            adv_batch (Tensor): adversarial patch that need to calculate tv loss. shape: torch.Size([batch_size, patch_per_image, 3, patch_w, patch_h])
        Returns:
            TV loss (Tensor): total variation.
        """
        adv_batch_copy = adv_batch.clone()
        adv_batch_flatten = adv_batch_copy.view(adv_batch_copy.size(0) * adv_batch.size(1), adv_batch_copy.size(2), adv_batch_copy.size(3), adv_batch_copy.size(4))
        TVLoss = 0
        for i in range(adv_batch_flatten.size(0)):
            adv_img = adv_batch_flatten[i]  # adv_img: torch.Size([3, patch_w, patch_h])
            channel, patch_w, patch_h = adv_img.size()
            # 计算图片宽度h方向，图片长度w方向的差值
            diff_h = adv_img[:, :, 1:] - adv_img[:, :, :-1] + 0.000001  # torch.Size([3, patch_w, patch_h - 1])
            diff_w = adv_img[:, 1:, :] - adv_img[:, :-1, :] + 0.000001  # torch.Size([3, patch_w - 1, patch_h])
            # 对diff_h, diff_w进行补全，方便后续计算
            diff_h_comp = torch.cat((diff_h, torch.zeros(channel, patch_w, 1).cuda()), dim=2)  # comp: dimension complete. torch.Size([3, patch_w, patch_h])
            diff_w_comp = torch.cat((diff_w, torch.zeros(channel, 1, patch_h).cuda()), dim=1)
            # calculate TV loss
            # tv = (diff_h_comp ** 2 + diff_w_comp ** 2) ** (self.beta / 2)
            tv = torch.abs(diff_h_comp) + torch.abs(diff_w_comp)
            TVLoss += torch.sum(tv)
        TVLoss = TVLoss / torch.numel(adv_batch_copy)
        return TVLoss


class PatchTransformer(nn.Module):
    """
    PatchTransformer: transforms batch of patches
    Module providing the functionality necessary to transform a batch of patches by: 
        - randomly adjusting brightness and contrast, 
        - adding random amount of noise,
        - rotating randomly, and 
        - resizing patches according to as size based on the batch of labels, 
          and pads them to the dimension of an image.
    """

    def __init__(self,
                 min_contrast=0.80,
                 max_contrast=1.20,
                 min_brightness=-0.10,
                 max_brightness=0.10,
                 noise_factor=0.10):
        super(PatchTransformer, self).__init__()
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.noise_factor = noise_factor

    def rect_occluding(self, num_rect=1, n_batch=8, n_feature=14, patch_size=300, with_cuda=True):
        """
        遮挡，在patch上设置num_rect块区域进行遮挡。n_batch表示batch中的图片数量，n_feature表示feature层的数量
        """
        if (with_cuda):
            device = 'cuda:0'
        else:
            device = 'cpu'
        tensor_img = torch.full((3, patch_size, patch_size), 0.0).to(device)
        for ttt in range(num_rect):
            # xs: x start   xe: x end   ys: y start     ye: y end
            xs = torch.randint(0, int(patch_size / 2), (1,))[0]
            xe = torch.randint(xs,
                               torch.min(torch.tensor(tensor_img.size()[-1]), xs + int(patch_size / 2)),
                               (1,))[0]
            ys = torch.randint(0, int(patch_size / 2), (1,))[0]
            ye = torch.randint(ys,
                               torch.min(torch.tensor(tensor_img.size()[-1]), ys + int(patch_size / 2)),
                               (1,))[0]
            tensor_img[:, xs:xe, ys:ye] = 0.5
        tensor_img_batch = tensor_img.unsqueeze(0)  ##  torch.Size([1, 3, 300, 300])
        tensor_img_batch = tensor_img_batch.expand(n_batch, n_feature, -1, -1, -1)  ##  torch.Size([8, 14, 3, 300, 300])
        return tensor_img_batch.to(device)

    def deg_to_rad(self, deg):
        return torch.tensor(deg * math.pi / 180.0).float().cuda()

    def rad_to_deg(self, rad):
        return torch.tensor(rad * 180.0 / math.pi).float().cuda()

    def resize_rotate(self, adv_patch, adv_batch, fake_bboxes_batch, fake_labels_batch, img_size):
        """
        patch resize and rotate according to fake_bboxes_batch
        Parameters:
            adv_patch (tensor): adv_patch after un-squeeze for original adv_patch input in 'forward' function on dim=0, torch.Size([1, 1, 3, patch_size_w,  patch_size_h])
            adv_batch (tensor): adv_batch after expand of adv_patch, torch.Size([1, patch_number, 3, patch_size, patch_size])
            fake_bboxes_batch (Tensor): fake bounding boxes. tensor.Size([1,patch_number, 5]), [cx,cy,w,h,theta], [w,h]分别沿图像横、纵方向，theta为正表示顺时针旋转
            fake_labels_batch (Tensor): fake labels. tensor.Size([1,patch_number, 1])
            img_size (tuple): original image size. (H, W)
        Returns:
            adv_batch_masked (tensor): patch on the image. 0 around patch location, and patch_value on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
            msk_batch (Tensor): mask of patch. 0 around patch location, and 1 on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
        """

        batch_size = torch.Size((fake_labels_batch.size(0), fake_labels_batch.size(1)))  # torch.Size([1,patch_number])

        cls_mask = fake_labels_batch.expand(-1, -1, 3)  # torch.Size([1, patch_number, 3]), 3应该是通道？
        cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([1, patch_number, 3, 1])
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([1, patch_number, 3, patch_size])
        cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([1, patch_number, 3, patch_size, 1])
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([1, patch_number, 3, patch_size, patch_size])
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # torch.Size([1, patch_number, 3, patch_size_w, patch_size_h]),但全是1

        # Pad patch and mask to image dimensions
        # Determine size of padding
        pad_w = (img_size[-1] - msk_batch.size(-2)) / 2  # img_size = (H, W) = (1536,2720), msk_batch: torch.Size([1, patch, 3, patch_size_w, patch_size_h])
        pad_h = (img_size[-2] - msk_batch.size(-1)) / 2
        mypad = nn.ConstantPad2d((int(pad_w + 0.5), int(pad_w), int(pad_h + 0.5), int(pad_h)), 0)
        adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([1, patch_number, 3, H, W])
        msk_batch = mypad(msk_batch)  # mks_batch size : torch.Size([1, patch_number, 3, H, W])

        # angle
        angle = fake_bboxes_batch[:, :, 4].view(np.prod(batch_size))  # angel: torch.Size([16])
        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)

        target_x = fake_bboxes_batch[:, :, 0].view(np.prod(batch_size))  # torch.Size([16])
        target_y = fake_bboxes_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([16])
        target_w = fake_bboxes_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([16])
        target_h = fake_bboxes_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([16])
        # 调整比例
        target_x = target_x / img_size[-1]
        target_y = target_y / img_size[-2]
        target_w = target_w / img_size[-1]
        target_h = target_h / img_size[-2]

        # 对target_y进行微调
        # target_y = target_y - 0.05

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([16, 3, 1536, 2720])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([16, 3, 1536, 2720])
        # print(f'adv_batch shape:{adv_batch.shape},msk_batch shape:{msk_batch.shape}')

        # 准备平移和旋转
        tx = (target_x - 0.5) * 2  # 乘以2是为了将[-0.5, 0.5]范围映射到[-1,1]范围
        ty = (target_y - 0.5) * 2
        sin = torch.sin(-angle)
        cos = torch.cos(-angle)
        aspect_ratio = img_size[-2] / img_size[-1]
        scale1 = (target_w / (current_patch_size / img_size[-1]))
        scale2 = (target_h / (current_patch_size / img_size[-2]))

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(np.prod(batch_size), 2, 3).fill_(0)  # torch.Size([112, 2, 3])
        theta[:, 0, 0] = cos / scale1
        theta[:, 0, 1] = sin / scale1 * aspect_ratio
        theta[:, 0, 2] = -cos * tx / scale1 - sin * ty / scale1 * aspect_ratio
        theta[:, 1, 0] = -sin / scale2 / aspect_ratio
        theta[:, 1, 1] = cos / scale2
        theta[:, 1, 2] = sin * tx / scale2 / aspect_ratio - cos * ty / scale2

        b_sh = adv_batch.shape  # b_sh = torch.Size([1*16, 3, 1536, 2720])
        grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([16, 1536, 2720, 2])
        adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([16, 3, 1536, 2720])
        msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([16, 3, 1536, 2720])
        # print(f'grid shape:{grid.shape}, adv_batch_t shape:{adv_batch_t.shape}, msk_batch_t shape:{msk_batch_t.shape}')

        # test_tensor_with_visualize(msk_batch_t)
        # assert False

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 2, 3, 1536, 2720])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 2, 3, 1536, 2720])

        return (adv_batch_t * msk_batch_t), msk_batch_t
    
    def resize_rotate_v2(self, adv_batch, fake_bboxes_batch, fake_labels_batch, img_size):
        """
        patch resize and rotate according to fake_bboxes_batch
        对于正方形的补丁能够映射到fake_bboxes_batch表示的区域内。
        只能映射正方形的补丁，不能映射非正方形的补丁
        Parameters:
            adv_batch (tensor): a batch of adv_patch, patch_batch_size=patch_number, torch.Size([batch_size, patch_number, 3, patch_h, patch_w]), 尺寸表示的是[行数，列数]
            fake_bboxes_batch (Tensor): fake bounding boxes. tensor.Size([1,patch_number, 5]), [cx,cy,w,h,theta], [w,h]分别沿图像横、纵方向，theta为正表示顺时针旋转
            fake_labels_batch (Tensor): fake labels. tensor.Size([1,patch_number, 1]), shoule be all zeros
            img_size (tuple): original image size. (H, W)
        Returns:
            adv_batch_masked (tensor): patch on the image. 0 around patch location, and patch_value on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
            msk_batch (Tensor): mask of patch. 0 around patch location, and 1 on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
        """

        batch_size = torch.Size((fake_labels_batch.size(0), fake_labels_batch.size(1)))  # torch.Size([batch_size,patch_number])
        current_patch_w, current_patch_h = adv_batch.size(-2), adv_batch.size(-1)     # adv_batch: torch.Size([batch_size, patch_number, 3, patch_h, patch_w])

        # 这一段主要生成msk_batch
        cls_mask = fake_labels_batch.expand(-1, -1, 3)  # cls_mask: torch.Size([batch_size, patch_number, 3])
        cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([batch_size, patch_number, 3, 1])
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([1, patch_number, 3, patch_h])
        cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([1, patch_number, 3, patch_h, 1])
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([1, patch_number, 3, patch_h, patch_w])
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # torch.Size([1, patch_number, 3, patch_h, patch_w]),但全是1

        # print('img_size:', img_size)
        # print('msk_batch shape:', msk_batch.shape)
        # print('adv_batch shape:', adv_batch.shape)

        # Pad patch and mask to image dimensions
        # Determine size of padding
        pad_w = (img_size[-1] - msk_batch.size(-1)) / 2  # img_size = (H, W) = (512, 512), msk_batch: torch.Size([1, patch, 3, patch_size_w, patch_size_h])
        pad_h = (img_size[-2] - msk_batch.size(-2)) / 2

        # print('pad_w:', pad_w, 'pad_h:', pad_h)
        mypad = nn.ConstantPad2d((int(pad_w + 0.5), int(pad_w), int(pad_h + 0.5), int(pad_h)), value=0)     # (w1, w2, h1, h2)表示左边、右边、上边、下边分别进行padding
        # mypad = nn.ConstantPad2d((0, 0, 0, 0), 0)
        adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([1, patch_number, 3, H, W])     # 0~1
        msk_batch = mypad(msk_batch)  # mks_batch size : torch.Size([1, patch_number, 3, H, W])

        # angle
        angle = fake_bboxes_batch[:, :, 4].view(np.prod(batch_size))  # angel: torch.Size([16])
        # Resizes and rotates
        target_x = fake_bboxes_batch[:, :, 0].view(np.prod(batch_size))  # torch.Size([16])
        target_y = fake_bboxes_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([16])
        target_w = fake_bboxes_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([16])
        target_h = fake_bboxes_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([16])

        # 输入img_size: [H, W]
        target_x = target_x / img_size[-1]
        target_y = target_y / img_size[-2]
        target_w = target_w / img_size[-1]
        target_h = target_h / img_size[-2]

        # 对target_y进行微调
        # target_y = target_y - 0.05

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([16, 3, 1536, 2720])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([16, 3, 1536, 2720])
        # print(f'adv_batch shape:{adv_batch.shape},msk_batch shape:{msk_batch.shape}')

        # 准备平移和旋转
        tx = (target_x - 0.5) * 2  # 乘以2是为了将[-0.5, 0.5]范围映射到[-1,1]范围
        ty = (target_y - 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)
       
        # 输入img_size: [W, H]
        aspect_ratio = img_size[-2] / img_size[-1]
        scale1 = (target_w / (current_patch_w / img_size[-1]))
        scale2 = (target_h / (current_patch_h / img_size[-2]))

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(np.prod(batch_size), 2, 3).fill_(0)  # torch.Size([batch_size*patch_per_image, 2, 3])

        theta[:, 0, 0] = cos / scale1
        theta[:, 0, 1] = sin / scale1 * aspect_ratio
        theta[:, 0, 2] = -cos * tx / scale1 - sin * ty / scale1 * aspect_ratio
        theta[:, 1, 0] = -sin / scale2 / aspect_ratio
        theta[:, 1, 1] = cos / scale2
        theta[:, 1, 2] = sin * tx / scale2 / aspect_ratio - cos * ty / scale2


        b_sh = adv_batch.shape  # b_sh = torch.Size([1*16, 3, 1536, 2720])
        grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([16, 1536, 2720, 2])
        adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([16, 3, 1536, 2720])
        msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([16, 3, 1536, 2720])
        # print(f'grid shape:{grid.shape}, adv_batch_t shape:{adv_batch_t.shape}, msk_batch_t shape:{msk_batch_t.shape}')

        # test_tensor_with_visualize(msk_batch_t)
        # assert False

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 2, 3, 1536, 2720])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 2, 3, 1536, 2720])

        return (adv_batch_t * msk_batch_t), msk_batch_t
   
    def resize_rotate_v3_1(self, adv_batch, ratio, fake_bboxes_batch, fake_labels_batch, img_size):
        """
        patch resize and rotate according to fake_bboxes_batch
        从resize_rotate_v2开始修改，使用该函数必须保证adv_batch的长宽比和fake_bboxes_batch中目标放置位置的长宽比一致
        Parameters:
            adv_batch (tensor): a batch of adv_patch, patch_batch_size=patch_number, torch.Size([batch_size, patch_number, 3, patch_h, patch_w]), 尺寸表示的是[行数，列数]
            ratio (float): 放缩过程中，如果adv_batch尺寸和fake_bboxes_batch规定位置尺寸一致，scale只能是常数。这里传入该常数
            fake_bboxes_batch (Tensor): fake bounding boxes. tensor.Size([1,patch_number, 5]), [cx,cy,w,h,theta], [w,h]分别沿图像横、纵方向，theta为正表示顺时针旋转
            fake_labels_batch (Tensor): fake labels. tensor.Size([1,patch_number, 1]), shoule be all zeros
            img_size (tuple): original image size. (H, W)
        Returns:
            adv_batch_masked (tensor): patch on the image. 0 around patch location, and patch_value on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
            msk_batch (Tensor): mask of patch. 0 around patch location, and 1 on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
        """

        batch_size = torch.Size((fake_labels_batch.size(0), fake_labels_batch.size(1)))  # torch.Size([batch_size,patch_number])
        current_patch_w, current_patch_h = adv_batch.size(-2), adv_batch.size(-1)     # adv_batch: torch.Size([batch_size, patch_number, 3, patch_h, patch_w])

        # 这一段主要生成msk_batch
        cls_mask = fake_labels_batch.expand(-1, -1, 3)  # cls_mask: torch.Size([batch_size, patch_number, 3])
        cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([batch_size, patch_number, 3, 1])
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))  # torch.Size([1, patch_number, 3, patch_h])
        cls_mask = cls_mask.unsqueeze(-1)  # torch.Size([1, patch_number, 3, patch_h, 1])
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))  # torch.Size([1, patch_number, 3, patch_h, patch_w])
        msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask  # torch.Size([1, patch_number, 3, patch_h, patch_w]),但全是1

        # print('img_size:', img_size)
        # print('msk_batch shape:', msk_batch.shape)
        # print('adv_batch shape:', adv_batch.shape)

        # Pad patch and mask to image dimensions
        # Determine size of padding
        pad_w = (img_size[-1] - msk_batch.size(-1)) / 2  # img_size = (H, W) = (512, 512), msk_batch: torch.Size([1, patch, 3, patch_size_w, patch_size_h])
        pad_h = (img_size[-2] - msk_batch.size(-2)) / 2

        # print('pad_w:', pad_w, 'pad_h:', pad_h)
        mypad = nn.ConstantPad2d((int(pad_w + 0.5), int(pad_w), int(pad_h + 0.5), int(pad_h)), value=0)     # (w1, w2, h1, h2)表示左边、右边、上边、下边分别进行padding
        # mypad = nn.ConstantPad2d((0, 0, 0, 0), 0)
        adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([1, patch_number, 3, H, W])     # 0~1
        msk_batch = mypad(msk_batch)  # mks_batch size : torch.Size([1, patch_number, 3, H, W])

        # angle
        angle = fake_bboxes_batch[:, :, 4].view(np.prod(batch_size))  # angel: torch.Size([16])
        # Resizes and rotates
        target_x = fake_bboxes_batch[:, :, 0].view(np.prod(batch_size))  # torch.Size([16])
        target_y = fake_bboxes_batch[:, :, 1].view(np.prod(batch_size))  # torch.Size([16])
        target_w = fake_bboxes_batch[:, :, 2].view(np.prod(batch_size))  # torch.Size([16])
        target_h = fake_bboxes_batch[:, :, 3].view(np.prod(batch_size))  # torch.Size([16])

        # 输入img_size: [H, W]
        target_x = target_x / img_size[-1]
        target_y = target_y / img_size[-2]
        target_w = target_w / img_size[-1]
        target_h = target_h / img_size[-2]

        # 对target_y进行微调
        # target_y = target_y - 0.05

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([16, 3, 1536, 2720])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])  # torch.Size([16, 3, 1536, 2720])
        # print(f'adv_batch shape:{adv_batch.shape},msk_batch shape:{msk_batch.shape}')

        # 准备平移和旋转
        tx = (target_x - 0.5) * 2  # 乘以2是为了将[-0.5, 0.5]范围映射到[-1,1]范围
        ty = (target_y - 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)
       
        # 输入img_size: [W, H]
        aspect_ratio = img_size[-2] / img_size[-1]
        # scale1 = (target_w / (current_patch_w / img_size[-1]))
        # scale2 = (target_h / (current_patch_h / img_size[-2]))
        # 输入补丁尺寸与规定位置长宽比一致
        scale1 = ratio
        scale2 = ratio

        # Theta = rotation,rescale matrix
        theta = torch.cuda.FloatTensor(np.prod(batch_size), 2, 3).fill_(0)  # torch.Size([batch_size*patch_per_image, 2, 3])

        theta[:, 0, 0] = cos / scale1
        theta[:, 0, 1] = sin / scale1 * aspect_ratio
        theta[:, 0, 2] = -cos * tx / scale1 - sin * ty / scale1 * aspect_ratio
        theta[:, 1, 0] = -sin / scale2 / aspect_ratio
        theta[:, 1, 1] = cos / scale2
        theta[:, 1, 2] = sin * tx / scale2 / aspect_ratio - cos * ty / scale2


        b_sh = adv_batch.shape  # b_sh = torch.Size([1*16, 3, 1536, 2720])
        grid = F.affine_grid(theta, adv_batch.shape)  # torch.Size([16, 1536, 2720, 2])
        adv_batch_t = F.grid_sample(adv_batch, grid)  # torch.Size([16, 3, 1536, 2720])
        msk_batch_t = F.grid_sample(msk_batch, grid)  # torch.Size([16, 3, 1536, 2720])
        # print(f'grid shape:{grid.shape}, adv_batch_t shape:{adv_batch_t.shape}, msk_batch_t shape:{msk_batch_t.shape}')

        # test_tensor_with_visualize(msk_batch_t)
        # assert False

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 2, 3, 1536, 2720])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])  # torch.Size([8, 2, 3, 1536, 2720])

        return (adv_batch_t * msk_batch_t), msk_batch_t


    def get_warpR(self, anglex, angley, anglez, fov, w, h):
        fov = torch.tensor(fov).float().cuda()
        w = torch.tensor(w).float().cuda()
        h = torch.tensor(h).float().cuda()
        z = torch.sqrt(w ** 2 + h ** 2) / 2 / torch.tan(self.deg_to_rad(fov / 2)).float().cuda()
        rx = torch.tensor([[1, 0, 0, 0],
                           [0, torch.cos(self.deg_to_rad(anglex)), -torch.sin(self.deg_to_rad(anglex)), 0],
                           [0, -torch.sin(self.deg_to_rad(anglex)), torch.cos(self.deg_to_rad(anglex)), 0, ],
                           [0, 0, 0, 1]]).float().cuda()
        ry = torch.tensor([[torch.cos(self.deg_to_rad(angley)), 0, torch.sin(self.deg_to_rad(angley)), 0],
                           [0, 1, 0, 0],
                           [-torch.sin(self.deg_to_rad(angley)), 0, torch.cos(self.deg_to_rad(angley)), 0, ],
                           [0, 0, 0, 1]]).float().cuda()
        rz = torch.tensor([[torch.cos(self.deg_to_rad(anglez)), torch.sin(self.deg_to_rad(anglez)), 0, 0],
                           [-torch.sin(self.deg_to_rad(anglez)), torch.cos(self.deg_to_rad(anglez)), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]]).float().cuda()
        r = torch.matmul(torch.matmul(rx, ry), rz)
        pcenter = torch.tensor([h / 2, w / 2, 0, 0]).float().cuda()
        p1 = torch.tensor([0, 0, 0, 0]).float().cuda() - pcenter
        p2 = torch.tensor([w, 0, 0, 0]).float().cuda() - pcenter
        p3 = torch.tensor([0, h, 0, 0]).float().cuda() - pcenter
        p4 = torch.tensor([w, h, 0, 0]).float().cuda() - pcenter
        dst1 = torch.matmul(r, p1)
        dst2 = torch.matmul(r, p2)
        dst3 = torch.matmul(r, p3)
        dst4 = torch.matmul(r, p4)
        list_dst = [dst1, dst2, dst3, dst4]
        org = torch.tensor([[0, 0],
                            [w, 0],
                            [0, h],
                            [w, h]]).float().cuda()
        dst = torch.zeros((4, 2)).float().cuda()
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]
        org = org.unsqueeze(0)
        dst = dst.unsqueeze(0)
        warpR = tgm.get_perspective_transform(org, dst).float().cuda()
        return warpR

    def forward(self, adv_patch, fake_bboxes_batch, fake_labels_batch, img_size):
        """
        Patch Transformer forward function
        Parameters:
            adv_patch (Tensor): The input adversarial patch. torch.Size([1, 3, patch_size_w, patch_size_h])
            fake_bboxes_batch (Tensor): fake bounding boxes. tensor.Size([1,patch_number, 5]), [cx,cy,w,h,theta], [w,h]分别沿图像横、纵方向，theta为正表示顺时针旋转
            fake_labels_batch (Tensor): fake labels. tensor.Size([1,patch_number, 1])
            img_size (tuple): origin image size, (H, W)
        Returns:
            adv_batch_masked (tensor): patch on the image. 0 around patch location, and patch_value on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
            msk_batch (Tensor): mask of patch. 0 around patch location, and 1 on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
        """
        adv_patch_size = adv_patch.size()[-1]  # patch_size

        if (adv_patch_size > min(img_size)):
            raise Exception('Patch size is bigger than image size!')

        adv_patch = adv_patch.unsqueeze(0)  # torch.Size([1,1,3,256,256])

        # print("adv_patch medianpooler size: "+str(adv_patch.size())) ## torch.Size([1, 3, 300, 300])
        # Make a batch of patches.
        # adv_patch = adv_patch.unsqueeze(0)#.unsqueeze(0)  ##  torch.Size([1, 1, 3, 300, 300])

        adv_batch = adv_patch.expand(fake_labels_batch.size(0), fake_labels_batch.size(1), -1, -1, -1)
        batch_size = torch.Size((fake_labels_batch.size(0), fake_labels_batch.size(1)))  # torch.Size([1,16])

        # Contrast, brightness and noise transforms
        # Create random contrast tensor
        # contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
        # contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        # contrast = contrast.cuda()
        # print("contrast size : "+str(contrast.size()))  ##  contrast size : torch.Size([8, 2, 3, 300, 300])

        # Create random brightness tensor
        # brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
        # brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        # brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) ##  brightness size : torch.Size([8, 2, 3, 300, 300])

        # Create random noise tensor
        # noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # print("noise size : "+str(noise.size()))  ##  noise size : torch.Size([8, 2, 3, 300, 300])

        ## Apply contrast/brightness/noise, clamp
        # if enable_no_random and not (enable_empty_patch):
        #     adv_batch = adv_batch
        # if not (enable_no_random) and not (enable_empty_patch):
        #     adv_batch = adv_batch * contrast + brightness + noise  # adv_batch.shape torch.Size([8,2,3,300,300])

        # resize and rotate
        adv_batch_masked, msk_batch = self.resize_rotate(adv_patch, adv_batch, fake_bboxes_batch, fake_labels_batch,
                                                         img_size)  # adv_batch torch.Size([1, 16, 3, 256, 256])   adv_batch_masked torch.Size([1, 16, 3, 1536, 2720])

        # print('fake_bboxes_batch:',fake_bboxes_batch)
        # print('msk_batch shape:',msk_batch.shape)
        # for i in range(msk_batch.size(0)):
        #     test_tensor_with_visualize(msk_batch[i],filename='msk_batch1')
        # assert False

        """
        if (with_projection):
            adv_batch = adv_batch_masked
            # # Rotating a Image
            b, f, c, h, w = adv_batch.size()
            adv_batch = adv_batch.view(b * f, c, h, w)
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            batch, channel, width, height = adv_batch.size()
            padding_borader = torch.nn.ZeroPad2d(50)
            input_ = padding_borader(adv_batch)
            # print("input_ "+str(input_.size())+"  "+str(input_.dtype))
            angle = np.random.randint(low=-50, high=51)
            mat = self.get_warpR(anglex=0, angley=angle, anglez=0, fov=42, w=width, h=height)
            mat = mat.expand(batch, -1, -1, -1)
            # print("image  "+str(self.image.dtype)+"  "+str(self.image.size()))
            # print("input_ "+str(input_.dtype)+"  "+str(input_.size()))
            # print("mat    "+str(mat.dtype)+"  "+str(mat.size()))
            adv_batch = tgm.warp_perspective(input_, mat, (input_.size()[-2], input_.size()[-1]))
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            adv_batch = adv_batch.view(b, f, c, input_.size()[-2], input_.size()[-1])
            # print("adv_batch "+str(adv_batch.size())+"  "+str(adv_batch.dtype))
            ##
            # Pad patch and mask to image dimensions
            # Determine size of padding
            pad = (img_size - adv_batch.size(-1)) / 2  # (416-300) / 2 = 58
            mypad = nn.ConstantPad2d((int(pad), int(pad), int(pad), int(pad)), 0)
            adv_batch = mypad(adv_batch)  # adv_batch size : torch.Size([8, 14, 3, 416, 416])
            adv_batch_masked = adv_batch
        """

        return adv_batch_masked, msk_batch

    def forward2(self, adv_batch, fake_bboxes_batch, fake_labels_batch, img_size):
        """
        Patch Transformer forward function. In forward2 function, adv_patches on one image are different. Pay attention: forward2 function is designed for single scene image.
        Parameters:
            adv_batch (Tensor): The input adversarial patch. torch.Size([batch_size, patch_number, 3, patch_size_w, patch_size_h])
            fake_bboxes_batch (Tensor): fake bounding boxes. tensor.Size([batch_size,patch_number, 5]), [cx,cy,w,h,theta], [w,h]分别沿图像横、纵方向，theta为正表示顺时针旋转
            fake_labels_batch (Tensor): fake labels. tensor.Size([batch_size,patch_number, 1])
            img_size (tuple): origin image size, (H, W)
        Returns:
            adv_batch_masked (tensor): patch on the image. 0 around patch location, and patch_value on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
            msk_batch (Tensor): mask of patch. 0 around patch location, and 1 on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
        """
        adv_patch_size = adv_batch.size()[-1]  # patch_size
        if adv_patch_size > min(img_size):
            raise Exception('Patch size is bigger than image size!')

        # resize and rotate
        adv_batch_masked, msk_batch = self.resize_rotate_v2(adv_batch, fake_bboxes_batch, fake_labels_batch,
                                                            img_size)  # adv_batch torch.Size([1, 16, 3, 256, 256])   adv_batch_masked torch.Size([1, 16, 3, 1536, 2720])

        return adv_batch_masked, msk_batch

    def forward3(self, adv_batch, fake_bboxes_batch, fake_labels_batch, img_size, transform=True):
        """
        Patch Transformer forward function. In forward3 function, adv_patches on one image are different. Pay attention: forward2 function is designed for single scene image.
        forward3 add transformation in forward2
        Parameters:
            adv_batch (Tensor): The input adversarial patch. torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])
            fake_bboxes_batch (Tensor): fake bounding boxes. tensor.Size([batch_size,patch_number, 5]), [cx,cy,w,h,theta], [w,h]分别沿图像横、纵方向，theta为正表示顺时针旋转
            fake_labels_batch (Tensor): fake labels. tensor.Size([batch_size,patch_number, 1])
            img_size (tuple): origin image size, (H, W)
            transform (bool): use brightness, noise, contrast transfomr or not
        Returns:
            adv_batch_masked (tensor): patch on the image. 0 around patch location, and patch_value on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
            msk_batch (Tensor): mask of patch. 0 around patch location, and 1 on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
        """
        adv_patch_size = (adv_batch.size(-2), adv_batch.size(-1))  # patch_size = (patch_h, patch_w)
        if adv_patch_size[0] > min(img_size) or adv_patch_size[1] > min(img_size):      # patch_w和patch_h都不能超过图片最小边长
            raise Exception('Patch size is bigger than image size!')

        batch_size = [adv_batch.size(0), adv_batch.size(1)]
        # Contrast, brightness and noise transforms
        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size[0], batch_size[1]).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size[0], batch_size[1]).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) ##  brightness size : torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # print("noise size : "+str(noise.size()))  ##  noise size : torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])

        ## Apply contrast/brightness/noise, clamp
        if transform:
            adv_batch = adv_batch * contrast + brightness + noise  # adv_batch.shape torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])
        else:
            adv_batch = adv_batch

        
        # 进行clamp
        adv_batch.data.copy_(torch.clamp(adv_batch.data, min=0., max=1. ))  #0~1

        # resize and rotate
        adv_batch_masked, msk_batch = self.resize_rotate_v2(adv_batch, 
                                                            fake_bboxes_batch, 
                                                            fake_labels_batch,
                                                            img_size)  # adv_batch torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])

        return adv_batch_masked, msk_batch
    
    def forward4(self, adv_batch, ratio, fake_bboxes_batch, fake_labels_batch, img_size, transform=True):
        """
        Patch Transformer forward function. In forward3 function, adv_patches on one image are different. Pay attention: forward2 function is designed for single scene image.
        forward4限制了补丁的尺寸，输入adv_batch的尺寸应该与fake_bboxes_batch所规定过的尺寸一致
        Parameters:
            adv_batch (Tensor): The input adversarial patch. torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])

            fake_bboxes_batch (Tensor): fake bounding boxes. tensor.Size([batch_size,patch_number, 5]), [cx,cy,w,h,theta], [w,h]分别沿图像横、纵方向，theta为正表示顺时针旋转
            fake_labels_batch (Tensor): fake labels. tensor.Size([batch_size,patch_number, 1])
            img_size (tuple): origin image size, (H, W)
            transform (bool): use brightness, noise, contrast transfomr or not
        Returns:
            adv_batch_masked (tensor): patch on the image. 0 around patch location, and patch_value on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
            msk_batch (Tensor): mask of patch. 0 around patch location, and 1 on patch location. torch.Size([1, patch_number, 3, img_size[0], img_size[1]])
        """
        adv_patch_size = (adv_batch.size(-2), adv_batch.size(-1))  # patch_size = (patch_h, patch_w)
        if adv_patch_size[0] > min(img_size) or adv_patch_size[1] > min(img_size):      # patch_w和patch_h都不能超过图片最小边长
            raise Exception('Patch size is bigger than image size!')

        batch_size = [adv_batch.size(0), adv_batch.size(1)]
        # Contrast, brightness and noise transforms
        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size[0], batch_size[1]).uniform_(self.min_contrast, self.max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        contrast = contrast.cuda()

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size[0], batch_size[1]).uniform_(self.min_brightness, self.max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
        brightness = brightness.cuda()
        # print("brightness size : "+str(brightness.size())) ##  brightness size : torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor
        # print("noise size : "+str(noise.size()))  ##  noise size : torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])

        ## Apply contrast/brightness/noise, clamp
        if transform:
            adv_batch = adv_batch * contrast + brightness + noise  # adv_batch.shape torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])
        else:
            adv_batch = adv_batch

        
        # 进行clamp
        adv_batch.data.copy_(torch.clamp(adv_batch.data, min=0., max=1. ))  #0~1

        # resize and rotate
        # adv_batch_masked, msk_batch = self.resize_rotate_v3_0(adv_batch, 
        #                                                     fake_bboxes_batch, 
        #                                                     fake_labels_batch,
        #                                                     img_size)  # adv_batch torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])
        
        adv_batch_masked, msk_batch = self.resize_rotate_v3_1(adv_batch, 
                                                              ratio, 
                                                              fake_bboxes_batch, 
                                                              fake_labels_batch,
                                                              img_size)  # adv_batch torch.Size([batch_size, patch_number, 3, patch_size_h, patch_size_w])

        return adv_batch_masked, msk_batch

        
class PatchApplier(nn.Module):
    """
    PatchApplier: applies adversarial patches to images.
    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):        # img_batch：torch.Size([batch_size, 3, H, W]), adv_batch: torch.Size([batch_size, patch_per_image, 3, rowPatch_size, rowPatch_size])
        for patch_index in range(adv_batch.size(1)):
            adv = adv_batch[:, patch_index, :, :, :]
            img_batch = torch.where((adv == 0), img_batch, adv)

        return img_batch


if __name__ == "__main__":
    import cv2
    import copy
    from utils0 import find_patch_positions_v2, convert_to_le90
    import torch

    """
    # load doggy patch
    img = cv2.imread('doggy.jpg') / 255.0
    img = cv2.resize(img, (64, 64))  # (256,360,3)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32).cuda()  # (1,3,height,width)
    # 1 patch on 1 image
    fake_bboxes_batch = torch.Tensor([1399, 722, 256, 100, -30 / 180 * math.pi]).unsqueeze(0).unsqueeze(0).cuda()  # (1,1,5), 角度正的是沿逆时针旋转
    fake_labels_batch = torch.zeros([1, 1, 1]).cuda()   # (1,1,1)
    PT = PatchTransformer()
    adv_batch_masked, msk_batch = PT(adv_patch=img,
                                     fake_bboxes_batch=fake_bboxes_batch,
                                     fake_labels_batch=fake_labels_batch,
                                     img_size=img_size)
    # adv_batch_masked: (1,10,3,1024,2000), msk_batch: (1,10,3,1024,1024)

    # 绘制adv_batch_masked与msk_batch
    test_batch = adv_batch_masked
    test_batch = test_batch.squeeze(0)
    combined_images = torch.zeros(3, 1024, 2000).cuda()
    for i in range(test_batch.size(0)):
        combined_images += test_batch[i]
    batch_img = combined_images.permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite('mask_test.jpg', (batch_img * 255).astype(np.uint8))

    # 绘制一个补丁时的图片
    # batch_img = adv_batch_masked.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # cv2.imwrite('out.jpg', (batch_img * 255).astype(np.uint8))
    """

    """
    img_size = (1530, 2720)
    patch_position = find_patch_positions_v2(img_size=img_size, patch_size=256, mean_size=0.5, re_size=(-0.1, 0.1), patch_number=10, iteration_max=1000)
    fake_bboxes_batch = torch.Tensor(patch_position).unsqueeze(0).cuda()  # (1,10,5)
    fake_labels_batch = torch.zeros([1, 10, 1]).cuda()
    # load doggy patch
    img = cv2.imread('doggy.jpg') / 255.0
    img = cv2.resize(img, (256, 256))  # (256,360,3)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32).cuda()  # (1,3,height,width)
    # 绘制多个补丁放置于一张图上的图片
    PT = PatchTransformer()
    adv_batch_masked, msk_batch = PT(adv_patch=img,
                                     fake_bboxes_batch=fake_bboxes_batch,
                                     fake_labels_batch=fake_labels_batch,
                                     img_size=img_size)

    print(adv_batch_masked.size())  # (1,10,3,1024,2000)

    batch_img = torch.max(adv_batch_masked.squeeze(0), dim=0)[0].permute(1, 2, 0).cpu().numpy()
    batch_img = (np.ascontiguousarray(batch_img) * 255).astype(np.uint8)

    for box in patch_position:
        rect = ((box[0], box[1]), (box[2], box[3]), -1 * box[4] * 180 / np.pi)  # cv2.boxPoints中顺时针旋转为正
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(batch_img, [rect], 0, (0, 0, 255), 2)

    cv2.imwrite('outpatch1.jpg', batch_img)
    """

    """
    # forward2函数测试
    img_size = (360, 640)
    patch_position = find_patch_positions_v2(img_size=img_size, patch_size=64, mean_size=0.5, re_size=(-0.1, 0.1), patch_number=10, iteration_max=1000)
    fake_bboxes_batch = torch.Tensor(patch_position).unsqueeze(0).cuda()  # (1,10,5)
    fake_labels_batch = torch.zeros([1, 10, 1]).cuda()
    # 读取多张补丁图片
    image_dir = './patch_batch_test'
    image_file = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    images_list = []
    for file in image_file:
        img = cv2.imread(os.path.join(image_dir, file)) / 255.0
        img = cv2.resize(img, (64, 64))
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32).cuda()  # (1,3,height,width)
        images_list.append(img)
    images = torch.stack(images_list).unsqueeze(0)  # (patch_number_on_single_image, 3, height, width), (1,10,3,64,64)
    # 进行transform以及apply
    PT = PatchTransformer()
    adv_batch_masked, msk_batch = PT.forward2(adv_batch=images,
                                              fake_bboxes_batch=fake_bboxes_batch,
                                              fake_labels_batch=fake_labels_batch,
                                              img_size=img_size)
    # print(adv_batch_masked.size())  # (1,10,3,360,640)
    batch_img = torch.max(adv_batch_masked.squeeze(0), dim=0)[0].permute(1, 2, 0).cpu().numpy()
    batch_img = (np.ascontiguousarray(batch_img) * 255).astype(np.uint8)
    # 绘制patch_position边界框
    for box in patch_position:
        rect = ((box[0], box[1]), (box[2], box[3]), -1 * box[4] * 180 / np.pi)  # cv2.boxPoints中顺时针旋转为正
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(batch_img, [rect], 0, (0, 0, 255), 2)
    cv2.imwrite('out_MultiImagePatch.jpg', batch_img)
    """

    """
    # 测试多补丁放多帧功能
    img_size = (1530, 2720)
    patch_position = find_patch_positions_v2(img_size=img_size, patch_size=256, mean_size=0.5, re_size=(-0.1, 0.1), patch_number=10, iteration_max=1000)
    print('patch position:', patch_position)
    fake_bboxes_batch = torch.tensor(patch_position).view(2, 5, 5)
    fake_labels_batch = torch.zeros([2, 5, 1]).cuda()
    # 读取多张补丁图片
    image_dir = './patch_batch_test'
    image_file = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    images_list = []
    for file in image_file:
        img = cv2.imread(os.path.join(image_dir, file)) / 255.0
        img = cv2.resize(img, (256, 256))
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32).cuda()  # (1,3,height,width)
        images_list.append(img)
    images = torch.stack(images_list).unsqueeze(0)  # (patch_number_on_single_image, 3, height, width) (1,10,3,64,64)
    # 对图片进行reshape
    images_batch = images.view(2, 5, 3, 256, 256)

    # 进行transform以及apply
    PT = PatchTransformer()
    adv_batch_masked, msk_batch = PT.forward2(adv_batch=images_batch,
                                              fake_bboxes_batch=fake_bboxes_batch,
                                              fake_labels_batch=fake_labels_batch,
                                              img_size=img_size)
    # print(adv_batch_masked.size())  # (2, 5, 3, 1530, 2720)
    # 进行可视乎
    for pic_index in range(adv_batch_masked.size(0)):
        adv_masked = adv_batch_masked[pic_index, :, :, :]  # (1, 5, 3, 1530, 2720)
        perturbed_img = torch.max(adv_masked.squeeze(0), dim=0)[0].permute(1, 2, 0).cpu().numpy()
        perturbed_img = (np.ascontiguousarray(perturbed_img) * 255).astype(np.uint8)
        # 绘制patch_position边界框
        perturbed_img_patch_position = fake_bboxes_batch[pic_index, :, :].numpy().tolist()
        for box in perturbed_img_patch_position:
            rect = ((box[0], box[1]), (box[2], box[3]), -1 * box[4] * 180 / np.pi)  # cv2.boxPoints中顺时针旋转为正
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            cv2.drawContours(perturbed_img, [rect], 0, (0, 0, 255), 2)
        cv2.imwrite(f'out_MultiImagePatch_MultiScene_{pic_index}.jpg', perturbed_img)
    """

    """
    # 测试total variation计算函数
    adv_batch = torch.rand(2, 2, 3, 8, 8)
    TV = TotalVariation(beta=0.5)
    tv_loss = TV(adv_batch)
    print('tv_loss:', tv_loss)
    """

    """
    # 测试forward4函数
    img_size = (360, 640)
    # 选择连续帧
    ContinuousFramesImageFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/PNGImages_640x360"
    ContinuousFramesLabelFolder = "/home/yyx/Adversarial/datasets/UAV-ROD/UAV-ROD-scene50-640x360/txt_annotations_640x360"
    image_BatchSize = 8
    image_files = os.listdir(ContinuousFramesImageFolder)
    annotation_files = os.listdir(ContinuousFramesLabelFolder)
    common_files = list(set([os.path.splitext(f)[0] for f in image_files]) & set([os.path.splitext(f)[0] for f in annotation_files]))
    selected_files = random.sample(common_files, image_BatchSize)
    # select corresponding image files and coordinate files, and complete pseudo img_metas dict
    coordinates_list = []
    images_list = []
    for file in selected_files:
        coordinates_list_in1img = []
        img_path = os.path.join(ContinuousFramesImageFolder, file + '.png')
        label_path = os.path.join(ContinuousFramesLabelFolder, file + '.txt')
        # 读取图片文件
        img = cv2.imread(img_path) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)  # (3,height,width)， BGR格式, 取值0到1
        images_list.append(img)
        # 读取label文件
        with open(label_path, 'r') as f:
            for line in f:
                elements = line.strip().split()
                coordinate = list(map(int, elements[:8]))
                coordinates_list_in1img.append(coordinate)
        coordinates_list.append(coordinates_list_in1img)        # dota格式，四个顶点坐标
    images_batch_t = torch.stack(images_list)  # (batch_size, 3, 640, 384)
    # print('images_batch_t shape:', images_batch_t.shape)

    
    # print('len coordinates_list:', len(coordinates_list))
    # print('coordinates:', coordinates_list)

    # 统计bounding box个数，需要获得bounding box数量的补丁，每一个物体上apply一个
    bounding_box_count = 0
    for bboxes_per_img in coordinates_list:
        for bbox in bboxes_per_img:
            bounding_box_count += 1
    

    # 转换image_annotations至le90形式
    batch_images_annotations_le90 = []      # len: batchsize
    for coordinates_list_in1image in coordinates_list:
        single_image_annotations_le90 = []
        for annotation in coordinates_list_in1image:
            annotation_le90 = convert_to_le90(annotation)
            single_image_annotations_le90.append(annotation_le90)
        batch_images_annotations_le90.append(single_image_annotations_le90)

    # print('len batch_images_annotations_le90:', len(batch_images_annotations_le90))
    # print('batch_images_annotations_le90:', batch_images_annotations_le90)

    
    # patch_position确定
    patch_ratio_range = (0.0, 0.0)
    patch_ratio = 1.0
    angle_perturb_range = (-5, 5)
    patch_positions_list = []
    for image_index in range(image_BatchSize):
        single_image_annotations_select_le90 = batch_images_annotations_le90[image_index]   # single_image_annotations_select_le90 二维
        bbox_positions = copy.deepcopy(single_image_annotations_select_le90)
        patch_positions_per_img = []
        for bbox_position in bbox_positions:
            resize_factor = np.random.uniform(*patch_ratio_range) + patch_ratio
            angle_perturbation = np.random.uniform(*angle_perturb_range) / 180 * math.pi      # 弧度制
            bbox_position[2], bbox_position[3] = bbox_position[3] * resize_factor, bbox_position[3] * resize_factor  # 生成一个方形的放置补丁区域，位置在bbox中间
            bbox_position[4] = bbox_position[4] + angle_perturbation
            patch_positions_per_img.append(bbox_position)
        patch_positions_list.append(patch_positions_per_img)        # 三维位置序列，list，len(patch_positions_list)=batch_size, 第二维度表示每张图里面的bbox
    adv_batch = torch.rand(bounding_box_count, 3, 64, 64)
    
    # 进行transform以及apply
    PT = PatchTransformer()
    adv_batch_masked, msk_batch = PT.forward4(adv_batch=adv_batch,bboxes_batch_list=patch_positions_list,img_size=img_size,transform=False)
    
    
    # print(adv_batch_masked.size())  # (1,10,3,360,640)
    batch_img = torch.max(adv_batch_masked.squeeze(0), dim=0)[0].permute(1, 2, 0).cpu().numpy()
    batch_img = (np.ascontiguousarray(batch_img) * 255).astype(np.uint8)
    # 绘制patch_position边界框
    for box in patch_position:
        rect = ((box[0], box[1]), (box[2], box[3]), -1 * box[4] * 180 / np.pi)  # cv2.boxPoints中顺时针旋转为正
        rect = cv2.boxPoints(rect)
        rect = np.int0(rect)
        cv2.drawContours(batch_img, [rect], 0, (0, 0, 255), 2)
    cv2.imwrite('out_MultiImagePatch.jpg', batch_img)
    """



    # 测试dota数据集下，forward3+resize_rotate函数
    # basic settings
    from utils0 import find_patch_positions_temp, zero_out_bounding_boxes_v2
    patch_size = [128, 64]  # 尺寸用（长，宽）表示
    patch_w, patch_h = patch_size[0], patch_size[1]

    # 正方形场景
    img_size = (512, 512)   # (H, W)
    # 选择连续帧
    ContinuousFramesImageFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-plane/DOTA-plane-200/images"
    ContinuousFramesLabelFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-plane/DOTA-plane-200/labelTxt"

    # 矩形场景
    # img_size = (360, 640)   # (H, W)
    # # 选择连续帧
    # ContinuousFramesImageFolder = f"/home/yyx/Adversarial/datasets/UAV-ROD/train_640x360/images"
    # ContinuousFramesLabelFolder = f"/home/yyx/Adversarial/datasets/UAV-ROD/train_640x360/txt_annotations"

    patch_ratio = 0.4
    patch_ratio_range = 0.01
    Patch_per_Image = 4
    image_BatchSize = 8
    img_format = '.png'
    fake_patch = torch.rand(image_BatchSize * Patch_per_Image, 3, patch_h, patch_w).cuda()  # 放在张量里面，倒数第二维度表示行数（宽），倒数第一列表示列数（长）
    patch_transformer = PatchTransformer().cuda()
    patch_applier = PatchApplier().cuda()
    mean_RGB = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]  # RGB color for scene dataset
    std_RGB = [58.395 / 255.0, 57.12 / 255.0, 57.675 / 255.0]  # RGB color for scene dataset
    mean_BGR = [103.53 / 255.0, 116.28 / 255.0, 123.675 / 255.0]  # BGR color for scene dataset
    std_BGR = [57.675 / 255.0, 57.12 / 255.0, 58.395 / 255.0]  # BGR color for scene dataset
    AdvImagesTransformer_BGR = transforms.Normalize(mean_BGR, std_BGR)
    
    image_files = os.listdir(ContinuousFramesImageFolder)
    annotation_files = os.listdir(ContinuousFramesLabelFolder)
    common_files = list(set([os.path.splitext(f)[0] for f in image_files]) & set([os.path.splitext(f)[0] for f in annotation_files]))
    coordinates_list = []
    images_list = []
    patch_positions_list = []
    
    file_log = []
    while len(images_list) < image_BatchSize:
        # 初始化设定
        file = random.choice(common_files)
        coordinates_list_in1img = []
        img_path = os.path.join(ContinuousFramesImageFolder, file + img_format)
        label_path = os.path.join(ContinuousFramesLabelFolder, file + '.txt')
        # 读取label文件
        with open(label_path, 'r') as f:
            for line in f:
                elements = line.strip().split()
                coordinate = list(map(float, elements[:8]))
                coordinates_list_in1img.append(coordinate)
        # 判断该图上是否能够正常放置补丁，并且按顺序放入list中
        single_image_annotations_le90 = []
        for annotation in coordinates_list_in1img:
            annotation_le90 = convert_to_le90(annotation)
            single_image_annotations_le90.append(annotation_le90)
        patch_position = find_patch_positions_temp(img_size=img_size,                             # patch_position里面是按
                                                   bounding_box=single_image_annotations_le90,
                                                   patch_size=(patch_w, patch_h),         # find positions这里的patch size是[长，宽]，使用过程中patch_size不可随意指定，长宽比只能与patch的形状一致！
                                                   mean_size=patch_ratio, 
                                                   re_size=(-patch_ratio_range, patch_ratio_range), 
                                                   patch_number=Patch_per_Image, 
                                                   iteration_max=1000)
        
        
        if patch_position == None:
            continue

        

        coordinates_list.append(coordinates_list_in1img)        # dota格式，四个顶点坐标

        patch_positions_list.append(patch_position)     # dota格式

        # 读取图片文件
        img = cv2.imread(img_path) / 255.0  # BGR格式
        img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)  # (3,height,width)， BGR格式
        resize = transforms.Resize(img_size)
        img = resize(transforms.ToPILImage()(img))
        img = transforms.ToTensor()(img)
        images_list.append(img)


    images_batch_t = torch.stack(images_list)  # (4,3,512,512)

    # 进行mask操作
    image_mask_batch = zero_out_bounding_boxes_v2(images_batch_t, coordinates_list)  # (batchsize,3,512,512), 未经归一化

    # 进行可视化检查
    # for pic_index in range(image_mask_batch.size(0)):
    #     image_mask = image_mask_batch[pic_index, :, :, :]  # (3, 384, 640)
    #     perturbed_img = image_mask.permute(1, 2, 0).cpu().numpy()   # (3, 384, 640), BGR
    #     perturbed_img = (np.ascontiguousarray(perturbed_img) * 255).astype(np.uint8)
    #     # 绘制patch_position边界框
    #     perturbed_img_patch_position = coordinates_list[pic_index]
    #     for box in perturbed_img_patch_position:
    #         pts = np.array([[box[i], box[i + 1]] for i in range(0, 7, 2)], np.int0)
    #         pts = pts.reshape((-1, 1, 2))
    #         cv2.polylines(perturbed_img, [pts], True, (0, 0, 255), 2)
    #     cv2.imwrite(f'out_MaskedImage_{pic_index}.jpg', perturbed_img)
    # assert False

    patch_positions_batch_t = torch.from_numpy(np.array(patch_positions_list)).cuda()  # (image_BatchSize,Patch_per_Image,5)

    fake_labels_batch_t = torch.ones([image_BatchSize, Patch_per_Image, 1]).cuda()  # (image_BatchSize,Patch_per_Image,1)   
    position_labels_batch_t = fake_labels_batch_t * 0.0     # 训练过程只能针对一类物体,该物体的position labels为全0
    
    fake_patch = fake_patch.view(image_BatchSize, Patch_per_Image, 3, patch_size[1], patch_size[0])

    # 调整fake_patch范围，从(-1,1)调整到(0,1)
    fake_patch.data = fake_patch.data.mul(0.5).add(0.5)     # fake_patch shape: torch.Size([image_BatchSize, patch_per_image, 3, row, col)

    # 进行apply
    # test forward 3
    adv_batch_masked, msk_batch = patch_transformer.forward4(adv_batch=fake_patch, 
                                                             ratio=patch_ratio, 
                                                             fake_bboxes_batch=patch_positions_batch_t, 
                                                             fake_labels_batch=position_labels_batch_t, 
                                                             img_size=img_size,
                                                             transform=False)


    # test forward 4
    # adv_batch_masked, msk_batch = patch_transformer.forward4(adv_batch=fake_patch, 
    #                                                          fake_bboxes_batch=patch_positions_batch_t, 
    #                                                          fake_labels_batch=position_labels_batch_t, 
    #                                                          img_size=img_size,
    #                                                          transform=False)
    
    # print('adv_batch_masked shape:', adv_batch_masked.shape)
    # print('image_Mask_batch shape:', image_mask_batch.shape)
    # assert False

    p_img_batch = patch_applier(image_mask_batch.cuda(), adv_batch_masked.cuda())  # 这里p_img_batch还没做归一化

    # 对p_img_batch进行归一化,
    p_img_batch_normalize = AdvImagesTransformer_BGR(p_img_batch)
    # 转换成RGB
    p_img_batch_normalize = p_img_batch_normalize[:, [2, 1, 0], :, :]  # RGB, torch.Size([batch_size,3, H, W]), 经过scene dataset对应mean, std的归一化


    # 进入loss_det计算环节之前的可视化检查
    p_img_batch_normalize_copy = p_img_batch_normalize.clone()
    # 反归一化
    for i in range(p_img_batch_normalize_copy.size(0)):  # 遍历 batch 中的每个图像
        for c in range(p_img_batch_normalize_copy.size(1)):  # 遍历图像的每个通道
            p_img_batch_normalize_copy[i, c] = p_img_batch_normalize_copy[i, c].mul(std_RGB[c]).add(mean_RGB[c])  # Tensor, torch.Size([batch_size, 3, H, W]), RGB
    # RGB->BGR
    for pic_index in range(p_img_batch_normalize_copy.size(0)):       # 整个batch的batch_size张图片都进行存储
        adv_masked = p_img_batch_normalize_copy[pic_index, :, :, :]  # (3, 384, 640)
        adv_img_rgb = adv_masked.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
        adv_img_bgr = adv_img_rgb[:, :, [2, 1, 0]]  # BGR
        adv_img_bgr = (np.ascontiguousarray(adv_img_bgr) * 255).astype(np.uint8)
        # 绘制patch_position边界框
        perturbed_img_patch_position = patch_positions_batch_t[pic_index, :, :].cpu().numpy().tolist()
        for box in perturbed_img_patch_position:
            rect = ((box[0], box[1]), (box[2], box[3]), 1 * box[4] * 180 / np.pi)  # cv2.boxPoints中顺时针旋转为正
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            cv2.drawContours(adv_img_bgr, [rect], 0, (0, 0, 255), 2)
    cv2.imwrite(f'fake_bboxes_test.jpg', adv_img_bgr)
    assert False
