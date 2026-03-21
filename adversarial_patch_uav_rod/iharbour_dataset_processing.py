import os
import cv2
from tqdm import tqdm
import numpy as np
import stat
from mmdet.apis import inference_detector
from ensemble_tools.detection_model import init_detector
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as T

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)


def resize_images(input_folder, output_folder, size):
    """
    resize image to size: (width, height)
    Parameters:
        input_folder (str): The path to the input folder containing images.
        output_folder (str): The path to the output folder where the resized images will be saved.
        size (tuple): The target width and height of the resized images.
    """
    os.makedirs(output_folder, exist_ok=True)
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_folder, filename)
            img = cv2.imread(input_path)
            resized_img = cv2.resize(img, size)
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)


# 亮度/对比度调整参数计算函数
def adjust_frames(img, b, c):
    """应用亮度/对比度调整"""
    # 矩阵乘法实现对比度调整
    img = img.astype(np.float32)
    img = c * img
    
    # 加法实现亮度调整
    img = img + b
    
    # 限制像素值在0-255范围
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def video_to_frames(video_path, output_dir, resize_dim=(640, 480), fps_extract=10, 
                    brightness=0, contrast=1.0, random_variation=False):
    """
    Split MP4 video to images, and then resize。
    Parameters:
        video_path (str): 输入视频文件的路径（例如，'input.mp4'）
        output_dir (str): 输出图像的保存目录
        resize_dim (tuple): 调整后的图像尺寸，默认为 (640, 480)
        fps_extract (int): 每秒钟帧数
        brightness (int): 亮度调整值 (-100 到 100)
        contrast (float): 对比度调整值 (0.0-3.0, 1.0为原始对比度)
        random_variation (bool): 是否为每帧生成随机亮度/对比度变化
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot Open {video_path} MP4 file")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print('Cannot get fps. default = 30')
        video_fps = 30

    frame_interval = int(round(video_fps / fps_extract))
    if frame_interval == 0:
        frame_interval = 1  # make sure frame_interval at least 1

    frame_count = 0
    extracted_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频读取完毕
        if frame_count % frame_interval == 0:
            # 调整对比度与亮度
            if random_variation:
                curr_brightness = brightness + np.random.randint(-20, 20)
                curr_contrast = contrast * np.random.uniform(0.8, 1.2)
            else:
                curr_brightness = brightness
                curr_contrast = contrast
            
            adjust_frame = adjust_frames(frame, curr_brightness, curr_contrast)

            # 调整图像尺寸
            resized_frame = cv2.resize(adjust_frame, resize_dim)
            # resized_frame = frame   # 不resize

            # 存储处理帧
            frame_filename = os.path.join(output_dir, f"frame_{extracted_count:05d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            print(f"save {frame_filename}",end='\r')
            extracted_count += 1
        frame_count += 1

    cap.release()
    print(f"Obtain {extracted_count} frames.")

def video_to_frames_given_total_images(video_path, output_dir, total_images, resize_dim=(640, 480), 
                                       brightness=0, contrast=1.0, random_variation=False):
    """
    将视频等时间间隔采样成指定数量的帧
    
    Parameters:
        video_path (str): 视频文件路径
        output_dir (str): 图像输出目录
        total_images (int): 需要采样的总帧数
        resize_dim (tuple): 输出图像尺寸 (宽, 高)
        brightness (int): 亮度调整值 (-100 到 100)
        contrast (float): 对比度调整值 (0.0-3.0, 1.0为原始对比度)
        random_variation (bool): 是否为每帧生成随机亮度/对比度变化
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 处理无效帧率
    if fps <= 0:
        fps = 30.0
        print(f"Warning: Invalid FPS. Using default FPS={fps}")
    
    # 计算视频总时长 (秒)
    duration_sec = total_frames / fps
    
    # 计算采样时间间隔
    if total_images <= 1:
        time_points = [0.0]
    else:
        time_points = np.linspace(0, duration_sec, total_images)
    
    # 存储采样的帧
    extracted_count = 0
    frame_index = 0

    # adjust_frames
    
    for t in time_points:
        # 计算对应帧位置
        target_frame = int(round(t * fps))
        target_frame = min(target_frame, total_frames - 1)
        
        # 定位到目标帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if ret:
            # 调整对比度与亮度
            if random_variation:
                curr_brightness = brightness + np.random.randint(-20, 20)
                curr_contrast = contrast * np.random.uniform(0.8, 1.2)
            else:
                curr_brightness = brightness
                curr_contrast = contrast
            
            adjust_frame = adjust_frames(frame, curr_brightness, curr_contrast)

            # 调整图像尺寸
            resized_frame = cv2.resize(adjust_frame, resize_dim)
            
            # 保存图像
            frame_filename = os.path.join(output_dir, f"frame_{frame_index:05d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            print(f"Saved frame at {t:.2f}s: {frame_filename}")
            
            extracted_count += 1
            frame_index += 1
        else:
            print(f"Warning: Failed to read frame at {t:.2f}s")
    
    cap.release()
    print(f"\nSuccessfully extracted {extracted_count}/{total_images} frames")
    print(f"Average sampling interval: {duration_sec/(total_images-1):.2f}s" 
          if total_images > 1 else "Single frame sampled")


def get_annotations(images_folder, annotations_folder):
    """
    Get annotations using pretrained rotated retinanet
    Parameters:
        images_folder (str): The folder path to the images folder
        annotations_folder (str): The folder path to the annotations file
    """
    config_path = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_uavrod_le90.py'
    checkpoint_path = '/home/yyx/Adversarial/mmrotate-0.3.3/UAV-ROD_models/detection_model/detectors_640x360/rotated_retinanet/latest.pth'
    device = 'cuda:0'
    # new an annotations_folder
    if not os.path.exists(annotations_folder):
        os.makedirs(annotations_folder)

    model = init_detector(config_path, checkpoint_path, device=device)
    for file in os.listdir(images_folder):
        print('file:', file)
        txt_file = file.replace('.jpg', '.txt')
        single_image_txt_annotations = os.path.join(annotations_folder, txt_file)
        file_path = os.path.join(images_folder, file)
        result = inference_detector(model, file_path)
        single_image_result = result[0]  # single_image_result: [[cx, cy, w, h, theta], [cx, cy, w, h, theta],...]
        with open(single_image_txt_annotations, 'w') as f:
            for bounding_box in single_image_result:
                vertex = le90_to_vertex(bounding_box)
                vertex_int = np.round(vertex).astype(int)  # float to int
                flattened_vertex_int = vertex_int.flatten()  # flatten
                f.write(' '.join(map(str, flattened_vertex_int)) + '\n')


def le90_to_vertex(box):
    """
    将box转换成八个顶点排列的形式
    Parameters:
        box (list): le90形式的box坐标, [cx, cy, w, h, theta, conf], 其中theta为弧度制, 为负表示逆时针旋转,为正表示顺时针旋转
    Returns:
        rotated_box (list): 形式为[x1, y1, x2, y2, x3, y3, x4, y4], 从左上角开始,按顺时针方向排列
    """
    cx, cy, w, h, theta, conf = box
    # 转换为逆时针为正的形式,后面旋转矩阵需要逆时针旋转为正的形式, 但是输入是顺时针旋转为正
    theta = -theta
    # 矩形的四个顶点（相对中心点 (cx, cy) 的坐标）
    box = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    # 构建旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    # 旋转并平移到中心点 (cx, cy)
    rotated_box = np.dot(box, rotation_matrix) + np.array([cx, cy])
    # print('rotated_box:', rotated_box)
    return rotated_box

# def generate_perturbed_origin_size_image():
def resize_pt_patch(pt_path, save_folder, resize_patch_size=(300,300)):
    """
    load pt file and resize to resize_patch_size. save at .png format
    :param pt_path: pt file path
    :param save_folder: save folder path
    :param resize_patch_size: target resize scale. use (300,300) or (360,360)
    :return: png image
    """
    # load pt tensor
    image_tensor = torch.load(pt_path)
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)  # 对于四维补丁，先将其转换为torch.Size([3, 64, 64])的尺寸
    # resize tensor to resize_patch_size
    image_tensor_resized = F.interpolate(image_tensor.unsqueeze(0), size=resize_patch_size, mode='bilinear', align_corners=False).squeeze(0)
    # convert numpy array [0-1] to png image [0-255]
    image = image_tensor_resized.permute(1, 2, 0).detach().cpu().numpy()
    image_255 = (image * 255).astype(np.uint8)  # range: (0,255)
    # save png
    save_path = os.path.join(save_folder,'patch_2.png')
    cv2.imwrite(save_path, image_255)

def resize_transform_pt_patch(pt_path, save_folder, png_file_name, resize_patch_size=(300,300), contrast_factor=1.0, brightness_factor=1.0):
    """
    load pt file and resize to resize_patch_size. adding contrast and brightness transform. save at .png format
    :param pt_path: pt file path
    :param save_folder: save folder path
    :param resize_patch_size: target resize scale. use (300,300) or (360,360)
    :return: png image
    """
    # load pt tensor
    image_tensor = torch.load(pt_path)
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)  # 对于四维补丁，先将其转换为torch.Size([3, 64, 64])的尺寸
    # apply contrast and brightness transformation
    # print(image_tensor)
    image_tensor = T.adjust_contrast(image_tensor, contrast_factor)
    # print(image_tensor)
    image_tensor = T.adjust_brightness(image_tensor, brightness_factor)
    # print(image_tensor)
    # assert False
    # resize tensor to resize_patch_size
    image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=resize_patch_size, mode='bilinear', align_corners=False).squeeze(0)
    # convert numpy array [0-1] to png image [0-255]
    image = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    image_255 = (image * 255).astype(np.uint8)  # range: (0,255)
    # save png
    save_path = os.path.join(save_folder,png_file_name)
    cv2.imwrite(save_path, image_255)


def resize_transforms_pt_patch(pt_path, save_folder, png_file_name, resize_patch_size=(300,300), contrast_factor=1.0, brightness_factor=1.0, noise_factor=1.0):
    """
    load pt file and resize to resize_patch_size. adding contrast and brightness transform. save at .png format
    :param pt_path: pt file path
    :param save_folder: save folder path
    :param resize_patch_size: target resize scale. use (300,300) or (360,360)
    :return: png image
    """
    # load pt tensor
    image_tensor = torch.load(pt_path)
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.squeeze(0)  # 对于四维补丁，先将其转换为torch.Size([3, 64, 64])的尺寸
    # apply contrast and brightness transformation
    # print(image_tensor)
    image_tensor = T.adjust_contrast(image_tensor, contrast_factor)
    # print(image_tensor)
    image_tensor = T.adjust_brightness(image_tensor, brightness_factor)
    # print(image_tensor)
    
    # add random Gaussian noise
    noise = torch.randn_like(image_tensor) * noise_factor
    image_tensor = image_tensor + noise

    # resize tensor to resize_patch_size
    image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=resize_patch_size, mode='bilinear', align_corners=False).squeeze(0)
    # convert numpy array [0-1] to png image [0-255]
    image = image_tensor.permute(1, 2, 0).detach().cpu().numpy()
    image_255 = (image * 255).astype(np.uint8)  # range: (0,255)
    # save png
    save_path = os.path.join(save_folder,png_file_name)
    cv2.imwrite(save_path, image_255)


def continuous_frames_to_video(frames_folder, output_file, fps):
    """
    melt continuous frames to an MP4 video, with fps frames per second
    :param frames_folder: continuous frames folder
    :param output_file: output mp4 file name
    :param fps: frames per second
    :return: None
    """
    images = sorted([img for img in os.listdir(frames_folder) if img.endswith(".jpg")], key=lambda x: int(x.split('_')[1].split('.')[0]))
    first_image_path = os.path.join(frames_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    height, width, layers = first_frame.shape

    # 定义视频编码和输出
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4编码
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # 循环读取图片并写入视频
    for image in tqdm(images):
        image_path = os.path.join(frames_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

def main_resize():
    # input_folder = './iharbour_dataset/images'
    # output_folder = './iharbour_dataset/images_384x640'
    input_folder = './iharbour_dataset/images/origin_images_11.11_2conceal_patches'
    output_folder = './iharbour_dataset/images_11.11_2conceal_960x720'
    # size = (640, 480)  # target width and height
    size = (960, 720)
    resize_images(input_folder, output_folder, size)


def main_video_to_images_given_fps():
    video_input_path = '../../datasets/indoor_attack_platform/origin/scene1_1920x1080.mp4'
    images_out_dir = '../../datasets/indoor_attack_platform/split_images/scene1_1920x1080'
    # size = (3840, 2160)       # 原图大小
    size = (1920, 1080)
    # size = (1280, 720)          
    # size = (960, 540)
    # size = (640, 360)           # 训练用

    # 等时间间隔切割
    frames_per_second = 10
    video_to_frames(video_path=video_input_path, output_dir=images_out_dir, resize_dim=size, fps_extract=frames_per_second)


def main_video_to_images_given_total_images():
    video_input_path = '../../datasets/iharbour_dataset_20250628/origins/30m_rotate_21s.mp4'
    images_out_dir = '../../datasets/iharbour_dataset_20250628/video_2_images_640x360/origin/30m_rotate_21s'
    # size = (3840, 2160)       # 原图大小
    # size = (1920, 1080)
    # size = (1280, 720)          
    # size = (960, 540)
    size = (640, 360)           # 训练用

    # 等时间间隔切割
    total_image_number = 150
    video_to_frames_given_total_images(video_path=video_input_path, output_dir=images_out_dir, total_images=total_image_number, resize_dim=size, 
                                       brightness=0, contrast=1.0, random_variation=False)



def main_get_annotations():
    images_folder = '../../datasets/iharbour_dataset_2/road_scene_1_640x320/images'
    annotations_folder = '../../datasets/iharbour_dataset_2/road_scene_1_640x320/txt_annotations'
    get_annotations(images_folder, annotations_folder)

def main_resize_pt_save_png_image():
    pt_path = './iharbour_dataset/generated_patches_64x64_EnhanceTransformation/patch_pt/patch_13.pt'
    # pt_path = './final_advpatch_alpha1.0_gamma0.8_contrast0.7-1.3_brightness-0.2-0.2_noise0.2_iharbour/patch_pt/patch_epoch100.pt'
    save_folder = './iharbour_dataset/print_patches/print_patch500x500'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.chmod(save_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    resize_pt_patch(pt_path, save_folder, resize_patch_size=(500, 500))

def main_saving_transform_patches():
    pt_path = './draw_UAVROD_patches/patch_pt/patch_75.pt'
    # pt_path = './final_advpatch_alpha1.0_gamma0.8_contrast0.7-1.3_brightness-0.2-0.2_noise0.2_iharbour/patch_pt/patch_epoch100.pt'
    save_folder = './drawing_test'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.chmod(save_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    save_file_name = 'patch2_noise.jpg'
    resize_transforms_pt_patch(pt_path, 
                               save_folder, 
                               png_file_name=save_file_name,
                               resize_patch_size=(64, 64), 
                               contrast_factor=1.0, 
                               brightness_factor=1.0,
                               noise_factor=0.01)


def main_resize_pt_save_png_image_transformation():
    """
    读取pt文件之后，引入contrast和brightness的transformation
    """
    pt_path = './iharbour_dataset/generated_patches/generated_patches_64x64_EnhanceTransformation/patch_pt/patch_45.pt'
    save_folder = './iharbour_dataset/print_patches/print_patch500x500_patch45'
    contrast_factors = [0.8, 1.0, 1.2]
    brightness_factors = [0.8, 1.1]
    index = 0
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        os.chmod(save_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    for contrast_factor  in contrast_factors:
        for brightness_factor in brightness_factors:
            index += 1
            save_file_name = f'patch_{index}_contrast_{contrast_factor}_brightness_{brightness_factor}.png'
            resize_transform_pt_patch(pt_path, 
                                      save_folder, 
                                      png_file_name=save_file_name,
                                      resize_patch_size=(500, 500), 
                                      contrast_factor=contrast_factor, 
                                      brightness_factor=brightness_factor)

def main_continuous_frames_to_video():
    frames_folder = './iharbour_dataset_DSAP/images_1280x720/road1_place2_height30m_angle90_cloud_visualize'
    output_mp4_file = './iharbour_dataset_DSAP/images_1280x720/road1_place2_height30m_angle90_cloud_visualize.mp4'
    fps = 30
    continuous_frames_to_video(frames_folder, output_mp4_file, fps)


if __name__ == '__main__':
    # main_resize()
    main_video_to_images_given_fps()     # given fps and obtain images from a video
    # main_video_to_images_given_total_images()  # given total image number and obtain images from a video.
    # main_get_annotations()
    # main_resize_pt_save_png_image()
    # main_continuous_frames_to_video()
    # main_resize_pt_save_png_image_transformation()
    # main_saving_transform_patches()
