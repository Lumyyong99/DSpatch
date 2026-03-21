import copy
import torch
import time
import os
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from torchvision.utils import make_grid
from PIL import Image
import random
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate
import math
import cv2


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def weights_init(m):
    """initialize GAN network weights"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def create_obb(cx, cy, width, height, angle):
    """
    通过cx, cy, width, height, angle生成obb。主要是针对ground truth bounding box的，用于后续判断是否会和patch发生碰撞
    返回polygon形式的obb
    """
    # Create a rectangle centered at (0, 0)
    rectangle = Polygon([(-width / 2, -height / 2),
                         (width / 2, -height / 2),
                         (width / 2, height / 2),
                         (-width / 2, height / 2)])

    # Rotate the rectangle
    rotated_rectangle = rotate(rectangle, angle, use_radians=True)
    # Translate the rectangle to the center (cx, cy)
    obb = translate(rotated_rectangle, cx, cy)
    return obb


def create_obb_tensor(bboxes_tensor):
    """
    输入形式变为bbox的张量形式
    返回list(<polygon>)形式的obb
    """
    bboxes_np = bboxes_tensor.numpy()
    obbs = []
    for bbox in bboxes_np:
        cx, cy, width, height, angle = bbox
        # Create a rectangle centered at (0, 0)
        rectangle = Polygon([(-width / 2, -height / 2),
                             (width / 2, -height / 2),
                             (width / 2, height / 2),
                             (-width / 2, height / 2)])
        # Rotate the rectangle
        rotated_rectangle = rotate(rectangle, angle, use_radians=True)
        # Translate the rectangle to the center (cx, cy)
        obb = translate(rotated_rectangle, cx, cy)
        obbs.append(obb)
    return obbs


def check_collision(patch, obbs):
    """
    判断patch和obb是否发生重叠
    """
    for obb in obbs:
        if patch.intersects(obb):
            return True
    return False


def find_patch_positions(img_size, obbs, patch_size, patch_number, iteration_max=100000):
    """
    找到符合要求的放置补丁的位置
    返回list((cx1,cy1),(cx2,cy2)...)
    """
    img_height, img_width = img_size
    patch_height, patch_width = patch_size
    patch_radius = np.sqrt((patch_width) ** 2 + (patch_height) ** 2) / 2

    valid_positions = []
    flag = 0
    while len(valid_positions) < patch_number:
        flag += 1
        # 判断迭代次数。如果次数过多则返回False
        if flag >= iteration_max:
            print('ERROR! Cannot find appropriate position in picture and do not apply fake patch!')
            return None

        # Randomly generate a center point for the patch within valid bounds
        cx = random.randint(int(patch_radius), img_width - int(patch_radius))
        cy = random.randint(int(patch_radius), img_height - int(patch_radius))
        # check distance
        too_close = False
        for (existing_cx, existing_cy) in valid_positions:
            if np.sqrt((cx - existing_cx) ** 2 + (cy - existing_cy) ** 2) < 2 * patch_radius:
                too_close = True
                break
        if too_close:
            continue

        # Create patch OBB at different angles and check for collisions
        collision_found = False
        for angle in np.linspace(0, 2 * np.pi, num=36, endpoint=False):  # Check at 10 degree intervals
            patch_obb = create_obb(cx, cy, patch_height, patch_width, angle)
            if check_collision(patch_obb, obbs):
                collision_found = True
                break
        if not collision_found:
            valid_positions.append((cx, cy))
    return valid_positions


def find_patch_positions_v2(img_size=(448, 798), patch_size=256, mean_size=0.5, re_size=(-0.25, 0.25), patch_number=10, iteration_max=1000):
    """
    寻找不相互重叠的patch_number个补丁位置\n
    针对一张图的补丁放置坐标
    Parameters:
        img_size (tuple): 原始图像的尺寸
        patch_size (int): 补丁尺寸
        mean_size (float): 对patch进行放缩的一个均值。0.5即是对补丁放缩0.5倍，原始256*256的补丁呗放缩成128*128
        re_size (tuple): 放缩范围，在mean_size上进行加减。re_size=(-0.25,0.25)意思是对补丁放缩尺寸为(0.5-0.25, 0.5+0.25)
        patch_number (int): 一个图上放置补丁的数量
        iteration_max (int): 寻找合适位置的最多迭代次数
    Returns:
        valid_positions (list): 合适的patch坐标，尺寸为（1, patch_number, 5），其中5表示[cx,cy,w,h,theta]，theta为弧度制角度，[w,h]顺序为沿x轴、沿y轴（长、高）
    """
    img_height, img_width = img_size

    valid_positions = []
    flag = 0
    cur_obb_list = []
    while len(valid_positions) < patch_number:
        flag += 1
        # 判断迭代次数。如果次数过多则返回False
        if flag >= iteration_max:
            print('ERROR! Cannot find appropriate position in picture and do not apply fake patch!')
            assert False

        cur_re_size_w = np.random.uniform(*re_size) + mean_size
        cur_re_size_h = np.random.uniform(*re_size) + mean_size
        angle = np.random.uniform(-45, 45) / 180 * math.pi

        patch_radius = np.sqrt((patch_size * cur_re_size_w) ** 2 + (patch_size * cur_re_size_h) ** 2) / 4

        cx = random.randint(int(patch_radius), img_width - int(patch_radius))
        cy = random.randint(int(patch_radius), img_height - int(patch_radius))

        patch_obb = create_obb(cx, cy, int(cur_re_size_w * patch_size), int(cur_re_size_h * patch_size), -1 * angle)  # polygon形式的框
        if check_collision(patch_obb, cur_obb_list):
            continue
        cur_obb_list.append(patch_obb)
        valid_positions.append([cx, cy, int(cur_re_size_w * patch_size), int(cur_re_size_h * patch_size), angle])
    return valid_positions


def find_patch_positions_v3(img_size=(448, 798), bounding_box=None, patch_size=256, mean_size=0.5, re_size=(-0.25, 0.25), patch_number=10, iteration_max=1000):
    """
    寻找不相互重叠的，并且与现有bounding box不重合的patch_number个补丁位置\n
    针对一张图的补丁放置坐标
    Parameters:
        img_size (tuple): 原始图像的尺寸
        bounding_box (list): 现有边界框，用于生成与现有边界框都不重合的补丁。一张图的边界框，形式为二元序列，[[cx1, cy1, w1, h1, theta1], [cx2, cy2, w2, h2, theta2], ...]
        patch_size (int): 补丁尺寸
        mean_size (float): 对patch进行放缩的一个均值。0.5即是对补丁放缩0.5倍，原始256*256的补丁呗放缩成128*128
        re_size (tuple): 放缩范围，在mean_size上进行加减。re_size=(-0.25,0.25)意思是对补丁放缩尺寸为(0.5-0.25, 0.5+0.25)
        patch_number (int): 一个图上放置补丁的数量
        iteration_max (int): 寻找合适位置的最多迭代次数
    Returns:
        valid_positions (list): 合适的patch坐标，尺寸为（1, patch_number, 5），其中5表示[cx,cy,w,h,theta]，theta为弧度制角度，[w,h]顺序为沿x轴、沿y轴（长、高）
    """
    img_height, img_width = img_size
    valid_positions = []
    flag = 0
    cur_obb_list = []
    # 将现有bbox放入cur_obb_list
    for existent_bbox in bounding_box:
        cx, cy, w, h, theta = existent_bbox
        patch_obb = create_obb(int(cx), int(cy), int(w), int(h), -1 * theta)
        cur_obb_list.append(patch_obb)
    while len(valid_positions) < patch_number:
        flag += 1
        # 判断迭代次数。如果次数过多则返回False
        if flag >= iteration_max:
            print('ERROR! Cannot find appropriate position in picture and do not apply fake patch!')
            return None
            # assert False

        cur_re_size_w = np.random.uniform(*re_size) + mean_size
        cur_re_size_h = np.random.uniform(*re_size) + mean_size
        angle = np.random.uniform(-45, 45) / 180 * math.pi      # 弧度制

        patch_radius = np.sqrt((patch_size * cur_re_size_w) ** 2 + (patch_size * cur_re_size_h) ** 2) / 2

        cx = random.randint(int(patch_radius), img_width - int(patch_radius))
        cy = random.randint(int(patch_radius), img_height - int(patch_radius))

        patch_obb = create_obb(cx, cy, int(cur_re_size_w * patch_size), int(cur_re_size_h * patch_size), -1 * angle)
        if check_collision(patch_obb, cur_obb_list):
            continue
        cur_obb_list.append(patch_obb)
        valid_positions.append([cx, cy, int(cur_re_size_w * patch_size), int(cur_re_size_h * patch_size), angle])
    return valid_positions


def find_patch_positions_v4(img_size=(448, 798), bounding_box=None, patch_size=[256, 128], mean_size=0.5, re_size=(-0.25, 0.25), patch_number=10, iteration_max=1000):
    """
    寻找不相互重叠的，并且与现有bounding box不重合的patch_number个补丁位置\n
    针对一张图的补丁放置坐标
    相比于find_patch_positions_v3, patch_size变成矩形，而非正方形。
    Parameters:
        img_size (tuple): 原始图像的尺寸
        bounding_box (list): 现有边界框，用于生成与现有边界框都不重合的补丁。一张图的边界框，形式为二元序列，[[cx1, cy1, w1, h1, theta1], [cx2, cy2, w2, h2, theta2], ...]
        patch_size (int): 补丁尺寸
        mean_size (float): 对patch进行放缩的一个均值。0.5即是对补丁放缩0.5倍，原始256*256的补丁呗放缩成128*128
        re_size (tuple): 放缩范围，在mean_size上进行加减。re_size=(-0.25,0.25)意思是对补丁放缩尺寸为(0.5-0.25, 0.5+0.25)
        patch_number (int): 一个图上放置补丁的数量
        iteration_max (int): 寻找合适位置的最多迭代次数
    Returns:
        valid_positions (list): 合适的patch坐标，尺寸为（1, patch_number, 5），其中5表示[cx,cy,w,h,theta]，theta为弧度制角度，[w,h]顺序为沿x轴、沿y轴（长、高）
    """
    img_height, img_width = img_size
    valid_positions = []
    flag = 0
    cur_obb_list = []
    # 将现有bbox放入cur_obb_list
    for existent_bbox in bounding_box:
        cx, cy, w, h, theta = existent_bbox
        patch_obb = create_obb(int(cx), int(cy), int(w), int(h), -1 * theta)
        cur_obb_list.append(patch_obb)
    while len(valid_positions) < patch_number:
        flag += 1
        # 判断迭代次数。如果次数过多则返回False
        if flag >= iteration_max:
            print('ERROR! Cannot find appropriate position in picture and do not apply fake patch!')
            return None

        cur_re_size_w = np.random.uniform(*re_size) + mean_size
        cur_re_size_h = np.random.uniform(*re_size) + mean_size
        angle = np.random.uniform(-45, 45) / 180 * math.pi      # 弧度制

        patch_radius = np.sqrt((patch_size[0] * cur_re_size_w) ** 2 + (patch_size[1] * cur_re_size_h) ** 2) / 2

        cx = random.randint(int(patch_radius), img_width - int(patch_radius))
        cy = random.randint(int(patch_radius), img_height - int(patch_radius))

        patch_obb = create_obb(cx, cy, int(cur_re_size_w * patch_size[0]), int(cur_re_size_h * patch_size[1]), -1 * angle)
        if check_collision(patch_obb, cur_obb_list):
            continue
        cur_obb_list.append(patch_obb)
        valid_positions.append([cx, cy, int(cur_re_size_w * patch_size[0]), int(cur_re_size_h * patch_size[1]), angle])
    return valid_positions

def find_patch_positions_temp(img_size=(448, 798), bounding_box=None, patch_size=[256, 128], mean_size=0.5, re_size=(-0.25, 0.25), patch_number=10, iteration_max=1000):
    """
    测试用，v4角度旋转控制在5°以内
    Parameters:
        img_size (tuple): 原始图像的尺寸
        bounding_box (list): 现有边界框，用于生成与现有边界框都不重合的补丁。一张图的边界框，形式为二元序列，[[cx1, cy1, w1, h1, theta1], [cx2, cy2, w2, h2, theta2], ...]
        patch_size (int): 补丁尺寸
        mean_size (float): 对patch进行放缩的一个均值。0.5即是对补丁放缩0.5倍，原始256*256的补丁呗放缩成128*128
        re_size (tuple): 放缩范围，在mean_size上进行加减。re_size=(-0.25,0.25)意思是对补丁放缩尺寸为(0.5-0.25, 0.5+0.25)
        patch_number (int): 一个图上放置补丁的数量
        iteration_max (int): 寻找合适位置的最多迭代次数
    Returns:
        valid_positions (list): 合适的patch坐标，尺寸为（1, patch_number, 5），其中5表示[cx,cy,w,h,theta]，theta为弧度制角度，[w,h]顺序为沿x轴、沿y轴（长、高）
    """
    img_height, img_width = img_size
    valid_positions = []
    flag = 0
    cur_obb_list = []
    # 将现有bbox放入cur_obb_list
    for existent_bbox in bounding_box:
        cx, cy, w, h, theta = existent_bbox
        patch_obb = create_obb(int(cx), int(cy), int(w), int(h), -1 * theta)
        cur_obb_list.append(patch_obb)
    while len(valid_positions) < patch_number:
        flag += 1
        # 判断迭代次数。如果次数过多则返回False
        if flag >= iteration_max:
            print('ERROR! Cannot find appropriate position in picture and do not apply fake patch!')
            return None

        cur_re_size_w = np.random.uniform(*re_size) + mean_size
        cur_re_size_h = np.random.uniform(*re_size) + mean_size
        angle = np.random.uniform(-5, 5) / 180 * math.pi      # 弧度制

        patch_radius = np.sqrt((patch_size[0] * cur_re_size_w) ** 2 + (patch_size[1] * cur_re_size_h) ** 2) / 2

        cx = random.randint(int(patch_radius), img_width - int(patch_radius))
        cy = random.randint(int(patch_radius), img_height - int(patch_radius))

        patch_obb = create_obb(cx, cy, int(cur_re_size_w * patch_size[0]), int(cur_re_size_h * patch_size[1]), -1 * angle)
        if check_collision(patch_obb, cur_obb_list):
            continue
        cur_obb_list.append(patch_obb)
        valid_positions.append([cx, cy, int(cur_re_size_w * patch_size[0]), int(cur_re_size_h * patch_size[1]), angle])
    return valid_positions



def zero_out_bounding_boxes(images_batch, boxes_batch):
    """
    Zero out the pixels inside the bounding boxes on the image.\n
    Parameters:
        images_batch (torch.Tensor): The input image tensor of shape (1, 3, 1536, 2720).
        boxes_batch (List): A list where each element is a bounding boxes tensor of shape (object_number, 5), for each image in the batch. Each row in the tensor represents (cx, cy, w, h, angle).
    Returns:
        images_batch_with_mask (torch.Tensor): The modified images tensor with bounding box areas set to zero.
    """
    batch_size, _, height, width = images_batch.shape
    images_batch = images_batch.cpu()

    for i in range(batch_size):
        boxes = boxes_batch[i]
        img = images_batch[i]
        for box in boxes:
            cx, cy, w, h, angle = box
            # Create a mask for the bounding box
            mask = np.zeros((height, width), dtype=np.uint8)
            # Calculate the coordinates of the rotated bounding box
            rect = ((cx.item(), cy.item()), (w.item(), h.item()), math.degrees(angle.item()))
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)
            # Draw the filled polygon on the mask
            cv2.fillPoly(mask, [box_points], 1)
            # Convert mask to torch tensor
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            mask_tensor = mask_tensor.expand_as(img)

            # Set the pixels inside the bounding box to zero
            img = img * (1 - mask_tensor)
        images_batch[i] = img
    # put image to cuda
    images_batch = images_batch.cuda()
    return images_batch


def zero_out_bounding_boxes_v2(images_batch, boxes_batch):
    """
    Zero out the pixels inside the bounding boxes on the image.\n
    Parameters:
        images_batch (torch.Tensor): The input image tensor of shape (batch_size, 3, height, width). BGR format images
        boxes_batch (List): A list where each element is a bounding boxes tensor of shape (object_number, 8), for each image. Each row in the tensor represents (x1, y1, x2, y2, x3, y3, x4, y4).
    Returns:
        images_batch_with_mask (torch.Tensor): The modified images tensor with bounding box areas set to zero. (batch_size, 3, height, width)
    """
    masked_images_list = []
    batch_size, _, height, width = images_batch.shape
    for i in range(batch_size):
        img = images_batch[i]  # (3, height, width), BGR
        boxes = boxes_batch[i]
        for box in boxes:
            mask = np.zeros((img.shape[-2], img.shape[-1]), dtype=np.uint8)
            pts = np.array(box, np.int0).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)
            mask_tensor = mask_tensor.expand_as(img)
            img = img * (1 - mask_tensor)
        images_batch[i] = img
    return images_batch


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
