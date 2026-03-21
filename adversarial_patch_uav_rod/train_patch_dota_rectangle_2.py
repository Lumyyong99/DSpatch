"""
GAN-Based Patch Optimization for DOTA
Optimize a naturalistic patch with GAN network. Based on DCGAN and WGAN.
相比于train_patch_dota_rectangle.py，不进行过分的resize，尽量使resize尺寸不大, 避免因为resize丢失信息
log 2025.1.16
进行dota数据集上的补丁训练?
"""
import argparse
from tqdm import tqdm
import os
import stat
import shutil
import random
import time
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils

from ensemble_tools.detection_model import init_detector
from ensemble_tools.victim_model_inference import inference_det_loss_on_masked_images
from ensemble_tools.load_data_1 import PatchTransformer, PatchApplier, TotalVariation
from ensemble_tools.utils0 import find_patch_positions_v4, zero_out_bounding_boxes_v2, weights_init, convert_to_le90
import ensemble_tools.GANmodels.dcgan as dcgan

from ensemble_tools.patch_dataloader import ForestPatch, ScenePatch

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(6)

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description='GAN-patch training settings')
parser.add_argument('--work_dir', default=None, help='folder to output files')
parser.add_argument('--patch_pattern', default='dota-scene', help="patch patten. Select from [dota-scene, forest].")
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
parser.add_argument('--seed', default='2025', type=int, help='choose seed')
# config for GAN network
parser.add_argument('--gpu', default=1, type=int, help='number of GPUs to use')
parser.add_argument('--latent_vector', default=100, type=int, help='latent z vector size')
parser.add_argument('--ndf', default=64, type=int, help='channel number of first layer in discriminator network')
parser.add_argument('--ngf', default=64, type=int, help='channel number of first layer in generator network')
parser.add_argument('--clamp_lower_d', default=-0.008, type=float, help='clamp lower for parameters in discriminator')
parser.add_argument('--clamp_upper_d', default=0.008, type=float, help='clamp upper for parameters in discriminator')
parser.add_argument('--Giters', default=1, type=int, help='number of updates of Generator per iteration')
parser.add_argument('--Diters', default=10, type=int, help='number of updates of Discriminator per iteration')
parser.add_argument('--lr_discriminator', default=0.0004, type=float, help='learning rate of Discriminator')
parser.add_argument('--lr_generator', default=0.0004, type=float, help='learning rate of Generator')
parser.add_argument('--G_extra_layers', default=0, type=int, help='Generator network extra layers')
parser.add_argument('--D_extra_layers', default=0, type=int, help='Discriminator network extra layers')
parser.add_argument('--resume_netD', default='', type=str, help='resume from this discriminator checkpoint')
parser.add_argument('--resume_netG', default='', type=str, help='resume from this generator checkpoint')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--beta1_adam', default=0.5, type=float, help='beta1 parameter for adam')
# training detector setting
parser.add_argument('--detector', default='retinanet_o', type=str, help='victim detector. choose from [retinanet_o, faster_rcnn_o, gliding_vertex, oriented_rcnn, roi_trans, s2anet]')
# config for fake patch
# parser.add_argument('--patch_size', default=64, type=int, help='size of patch in digital scene of 640*384')
parser.add_argument('--patch_ratio', default=1.0, type=float, help='patch zooming ratio. default=0.6')
parser.add_argument('--patch_ratio_range', default=0.0, type=float, help='patch zooming ratio. default=0.1. meaning: [patch_ratio-patch_ratio_range, patch_ratio+patch_ratio_range')
parser.add_argument('--fake_class', default='plane', type=str, help='fake class')
# config for training process
parser.add_argument('--image_batchsize', default=8, type=int, help='load image_batchsize images in one iteration')
parser.add_argument('--patch_per_image', default=10, type=int, help='number of patch on each scene during training')
parser.add_argument('--gan_batchsize', default=80, type=int, help='batchsize for training GAN of source domain. Default should equal to image_batchsize*patch_per_image')
parser.add_argument('--epochs', default=800, type=int, help='total epoch number for training process')
parser.add_argument('--start_epoch', default=1, type=int, help='starting epoch number for training')
parser.add_argument('--save_vis_interval', default=5, type=int, help='saving visualize image after save_interval epochs')
parser.add_argument('--save_ckpt_interval', default=100, type=int, help='saving chekpoints after save_interval epochs')
parser.add_argument('--img_format', default='png', type=str, help='choose a image format, from [png, jpg]. Keep consistent with training dataset')
# transformation setting
parser.add_argument('--min_contrast', default=0.8, type=float, help='minimum contrast ratio boarder in transformation')
parser.add_argument('--max_contrast', default=1.2, type=float, help='maximum contrast ratio boarder in transformation')
parser.add_argument('--min_brightness', default=-0.1, type=float, help='minimum brightness boarder in transformation')
parser.add_argument('--max_brightness', default=0.1, type=float, help='maximum brightness boarder in transformation')
parser.add_argument('--noise_factor', default=0.1, type=float, help='noise factor')
# loss weight setting
parser.add_argument('--alpha', default=0.15, type=float, help='loss weight: detection loss loss_det')
parser.add_argument('--beta', default=1.0, type=float, help='loss weight: errG loss')
parser.add_argument('--gamma', default=0.1, type=float, help='loss weight: tv_loss')

args = parser.parse_args()

# check
args_dict = vars(args)
print('Experimental settings:\n', args_dict)

### -----------------------------------------------------------    Functions     ---------------------------------------------------------------------- ###
# def get_width_height_mean(folder, class_name):
#     """
#     calculate mean (width, height) for object of a certain class
#     Parameters:
#         folder (str): folder of txt annotation files.
#         class_name (str): give a certain class
#     Returns:
#         avg_size (tuple): average size of the given object
#         avg_ratio (float): the ratio of width and height
#     """
#     widths, heights = [], []
#     # 遍历文件
#     for filename in os.listdir(folder):
#         label_path = os.path.join(folder, filename)
#         with open(label_path, 'r') as f:
#             for line in f:
#                 elements = line.strip().split()
#                 if elements[8] == class_name:
#                     vertices_annotation = list(map(float, elements[:8]))     # list[x1,y1,x2,y2,x3,y3,x4,y4]
#                     le90_annotation = convert_to_le90(vertices_annotation)
#                     widths.append(le90_annotation[2])
#                     heights.append(le90_annotation[3])
#     avg_width = sum(widths) / len(widths) if widths else 0
#     avg_height = sum(heights) / len(heights) if heights else 0

#     # 合成avg_size和avg_ratio
#     avg_size = (avg_width, avg_height)
#     avg_ratio = round(avg_width / avg_height)   # 四舍五入

#     return avg_size, avg_ratio


### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
# patch setting
# rowPatch_size = args.patch_size
image_BatchSize = args.image_batchsize  # 一次iteration中使用多少张scene图片，公式中的B
Patch_per_Image = args.patch_per_image  # N
GAN_training_batchsize = args.gan_batchsize
GeneratePatch_batchsize = image_BatchSize * Patch_per_Image  # BN

if args.patch_pattern == 'dota-scene':
    GAN_dataset = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_PATCHES/patches_64x64'
else:
    raise Exception("patch pattern currently not available.")
patch_ratio = args.patch_ratio
patch_ratio_range = args.patch_ratio_range

class_name = args.fake_class
if class_name == 'plane':
    patch_size = [48, 64]
elif class_name == 'large-vehicle':
    patch_size = [16, 48]
elif class_name == 'ship':
    patch_size = [16, 32]
elif class_name == 'helicopter':
    patch_size = [16, 64]
elif class_name == 'small-vehicle':
    patch_size = [16, 32]

if class_name == 'plane':
    class_idx = 0
elif class_name == 'small-vehicle':
    class_idx = 4
elif class_name == 'large-vehicle':
    class_idx = 5
elif class_name == 'ship':  # 输入尺寸顺序为[h, w]
    class_idx = 6
elif class_name == 'helicopter':
    class_idx = 14


# victim detector setting
if args.detector == 'retinanet_o':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/rotated_retinanet_obb_r50_fpn_1x_dota_le90-c0097bc4.pth'
elif args.detector == 'faster_rcnn_o':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth'
elif args.detector == 'gliding_vertex':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/gliding_vertex/gliding_vertex_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/gliding_vertex_r50_fpn_1x_dota_le90-12e7423c.pth'
elif args.detector == 'oriented_rcnn':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/checkpoints/oriented_rcnn_r50_fpn_1x_dota_le90-6d2b2ce0.pth'
elif args.detector == 'roi_trans':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/roi_trans/roi_trans_r50_fpn_1x_dota_le90.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/roi_trans_r50_fpn_1x_dota_le90-d1f0b77a.pth'
elif args.detector == 's2anet':
    DetectorCfgSource = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/configs/s2anet/s2anet_r50_fpn_1x_dota_le135.py'
    DetectorCheckpoint = '/home/yyx/Adversarial/mmrotate-0.3.3/DOTA_models/detection_models/pre-trained_models/s2anet_r50_fpn_1x_dota_le135-5dfcf396.pth'
else:
    raise Exception('Wrong Detector!')

# training setting
device = args.device
n_epochs = args.epochs
start_epoch = args.start_epoch
lrD = args.lr_discriminator
lrG = args.lr_generator
vis_save = args.save_vis_interval
ckpt_save = args.save_ckpt_interval
img_format = '.' + args.img_format

# GAN setting
ngpu = args.gpu
nz = args.latent_vector
ngf = args.ngf
ndf = args.ndf
nc = 3  # input patch channels
clamp_lower = args.clamp_lower_d
clamp_upper = args.clamp_upper_d
netG_path = args.resume_netG
netD_path = args.resume_netD
beta1 = args.beta1_adam

# transformation setting
mini_contrast = args.min_contrast
maxi_contrast = args.max_contrast
mini_brightness = args.min_brightness
maxi_brightness = args.max_brightness
noise_fact = args.noise_factor

# scene setting
# ContinuousFramesImageFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-{class_name}-propersize/images"
# ContinuousFramesLabelFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-{class_name}-propersize/labelTxt"
# if class_name == 'helicopter':
#     ContinuousFramesImageFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-{class_name}/images"
#     ContinuousFramesLabelFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-{class_name}/labelTxt"

# scene 50
ContinuousFramesImageFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-{class_name}-propersize-scene50/images"
ContinuousFramesLabelFolder = f"/home/yyx/Adversarial/datasets/DOTA-V1.0/DOTA-{class_name}-propersize-scene50/labelTxt"

# avg_size, patch_aspect_ratio = get_width_height_mean(folder=ContinuousFramesLabelFolder, class_name=class_name)
# print('patch_aspect_ratio:', patch_aspect_ratio)
# patch_size = [16, 16*patch_aspect_ratio]
# patch_ratio = avg_size[0] / patch_size[0]

patch_h, patch_w = patch_size[0], patch_size[1]



### -----------------------------------------------------------    Detector    ---------------------------------------------------------------------- ###
start_time = time.time()
model = init_detector(config=DetectorCfgSource, checkpoint=DetectorCheckpoint, device=args.device)  # 这里从init_detector返回的model已经�?eval()模式�?
finish_time = time.time()
print(f'Load detector in {finish_time - start_time} seconds.')

### -----------------------------------------------------------    Initialize    ---------------------------------------------------------------------- ###
# set random seed 
Seed = args.seed  # 37564 7777
torch.manual_seed(Seed)
torch.cuda.manual_seed(Seed)
torch.cuda.manual_seed_all(Seed)
np.random.seed(Seed)
random.seed(Seed)

# # create working files
subfolders = ['combined_images', 'NoRotatePatch', 'checkpoints']
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)
    os.chmod(args.work_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
else:
    # 若有同名文件则删除
    shutil.rmtree(args.work_dir)
    print('--------------File exists! Existing file with the same name has been removed!--------------')

for subfolder in subfolders:
    subfolder_path = os.path.join(args.work_dir, subfolder)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
        os.chmod(subfolder_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

### -----------------------------------------------------------   DataLoader   ---------------------------------------------------------------------- ###
# preparing dataset: continuous frames in dota dataset
mean_RGB = [123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0]  # RGB color for scene dataset
std_RGB = [58.395 / 255.0, 57.12 / 255.0, 57.675 / 255.0]  # RGB color for scene dataset
mean_BGR = [103.53 / 255.0, 116.28 / 255.0, 123.675 / 255.0]  # BGR color for scene dataset
std_BGR = [57.675 / 255.0, 57.12 / 255.0, 58.395 / 255.0]  # BGR color for scene dataset
# ContinuousFramesNormalizer = transforms.Normalize(mean=mean, std=std)
mean_GeneratingPatch_BGR = [0.5, 0.5, 0.5]
std_GeneratingPatch_BGR = [0.5, 0.5, 0.5]

# 设置归一化以及resize变换。针对torch.Size([3, width, height])的张量进行操作，使用BGR格式
PatchNormalizer = transforms.Compose(
    [
        transforms.Resize((patch_size[0], patch_size[1])),
        transforms.Normalize(mean_GeneratingPatch_BGR, std_GeneratingPatch_BGR)
    ]
)
AdvImagesTransformer_BGR = transforms.Normalize(mean_BGR, std_BGR)


# 实例化用于补丁图案训练的数据�?# 
if args.patch_pattern == 'dota-scene':
    PatchDataset = ScenePatch(GAN_dataset, PatchNormalizer)
else:
    raise Exception("patch pattern currently not available. Use forest or scene instead.")

PatchDataset = ScenePatch(GAN_dataset, PatchNormalizer)
PatchDataloader = DataLoader(PatchDataset, batch_size=GAN_training_batchsize, shuffle=False)

### -----------------------------------------------------------   GAN preparing   ---------------------------------------------------------------------- ###
# prepare GAN network
netG = dcgan.DCGAN_G_Rect_2(patch_size, nz, nc, ngf, ngpu, args.G_extra_layers)
netG.apply(weights_init)
netD = dcgan.DCGAN_D_Rect(patch_size, nz, nc, ndf, ngpu, args.D_extra_layers)
netD.apply(weights_init)
if netG_path != '':  # load checkpoint if needed
    netG.load_state_dict(torch.load(args.netG))
print(netG)
if netD_path != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)

# initialize GAN input
GAN_input = torch.FloatTensor(GAN_training_batchsize, 3, patch_size[0], patch_size[1])
noise = torch.FloatTensor(GAN_training_batchsize, nz, 1, 1)
fixed_noise = torch.FloatTensor(GAN_training_batchsize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1
if device == "cuda:0":
    netD.cuda()
    netG.cuda()
    GAN_input = GAN_input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

### ---------------------------------------------------------- Checkpoint & Init -------------------------------------------------------------------- ###
# Training preprocess
epoch_length = len(PatchDataloader)
torch.cuda.empty_cache()

# create optimizer for GAN network
if args.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=lrG)

# init PatchTransformer and PatchApplier
if device == "cuda:0":
    patch_transformer = PatchTransformer(
        min_contrast=mini_contrast,
        max_contrast=maxi_contrast,
        min_brightness=mini_brightness,
        max_brightness=maxi_brightness,
        noise_factor=noise_fact
    ).cuda()
    patch_applier = PatchApplier().cuda()
    total_variation = TotalVariation().cuda()
else:
    patch_transformer = PatchTransformer(
        min_contrast=mini_contrast,
        max_contrast=maxi_contrast,
        min_brightness=mini_brightness,
        max_brightness=maxi_brightness,
        noise_factor=noise_fact
    )
    patch_applier = PatchApplier()
    total_variation = TotalVariation()

### -----------------------------------------------------------    log file    ---------------------------------------------------------------------- ###
# experiment settings
experimental_settings_file_path = os.path.join(args.work_dir, 'ExperimentSettings.txt')
with open(experimental_settings_file_path, 'w') as f:
    for key, value in args_dict.items():
        f.write(f'{key}={value}\n')
    # network writing
    f.write('Generator:\n' + str(netG) + '\n')
    f.write('Discriminator:\n' + str(netD) + '\n')

# log settings
log_file_path = os.path.join(args.work_dir, 'logger.txt')
log_file = open(log_file_path, 'a')

### ---------------------------------------------------------- Training -------------------------------------------------------------------- ###
G_iters_num = 0
for epoch in range(start_epoch, n_epochs+1):
    ep_loss_det = 0
    ep_errD = 0
    ep_errD_real = 0
    ep_errD_fake = 0
    ep_errG = 0
    ep_loss_tv = 0
    i = 0
    data_iter = iter(PatchDataloader)
    while i < len(PatchDataloader):
        G_iters_num += 1
        # with torch.autograd.set_detect_anomaly(True):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        for dis_flag in range(args.Diters):
            # data iter
            try:
                data = data_iter.next()
            except StopIteration:
                data_iter = iter(PatchDataloader)
                data = data_iter.next()
            i += 1          # i表示discriminator迭代的次�?
            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)  # 限制discriminator参数的范围，参考WGAN
            netD.zero_grad()
            if device == "cuda:0":
                real_pics = data.cuda()
            GAN_input.resize_as_(real_pics).copy_(real_pics)  # GAN_input requires grad: False
            GAN_input_Variable = GAN_input.clone().requires_grad_(True)
            # train with real
            errD_real = netD(GAN_input_Variable)
            ep_errD_real += errD_real.data[0]
            errD_real.backward(one)

            # train with fake
            with torch.no_grad():
                noise.resize_(GAN_training_batchsize, nz, 1, 1).normal_(0, 1)
            fake = netG(noise).data.requires_grad_(True)  # torch.Size([GeneratePatch_batchsize, 3, rowPatch_size, rowPatch_size])
            # train with fake
            errD_fake = netD(fake)
            ep_errD_fake += errD_fake.data[0]
            errD_fake.backward(mone)
            errD = errD_real - errD_fake
            ep_errD += errD.data[0]
            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation

        # for gen_flag in tqdm(range(args.Giters)):
        for gen_flag in range(args.Giters):
            netG.zero_grad()
            noise.resize_(GAN_training_batchsize, nz, 1, 1).normal_(0, 1)
            GAN_input_noise_Variable = noise.clone().requires_grad_(True)
            fake_patch = netG(GAN_input_noise_Variable)  # fake_patch: generated patch. range (-1, 1), torch.Size([BN, 3, patch_width, patch_height]), BGR
            errG = netD(fake_patch)
            ep_errG += errG.data[0]

            
            # select corresponding image files and coordinate files, and complete pseudo img_metas dict
            image_files = os.listdir(ContinuousFramesImageFolder)
            annotation_files = os.listdir(ContinuousFramesLabelFolder)
            common_files = list(set([os.path.splitext(f)[0] for f in image_files]) & set([os.path.splitext(f)[0] for f in annotation_files]))
            coordinates_list = []
            images_list = []
            patch_positions_list = []
            img_metas = []
            img_meta_keys = ['filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg', 'to_rgb']
            
            file_log = []
            while len(images_list) < image_BatchSize:
                # 初始化            
                file = random.choice(common_files)
                coordinates_list_in1img = []
                img_path = os.path.join(ContinuousFramesImageFolder, file + img_format)
                label_path = os.path.join(ContinuousFramesLabelFolder, file + '.txt')
                img_meta = {key: None for key in img_meta_keys}
                # 读取label文件
                with open(label_path, 'r') as f:
                    for line in f:
                        elements = line.strip().split()
                        coordinate = list(map(float, elements[:8]))
                        coordinates_list_in1img.append(coordinate)
                # 判断该图上是否能够正常放置补丁，并且按顺序放入list�?                
                single_image_annotations_le90 = []
                for annotation in coordinates_list_in1img:
                    annotation_le90 = convert_to_le90(annotation)
                    single_image_annotations_le90.append(annotation_le90)
                patch_position = find_patch_positions_v4(img_size=(512, 512), 
                                                         bounding_box=single_image_annotations_le90,
                                                         patch_size=[patch_w, patch_h],         # find positions这里的patch size是[长，宽]
                                                         mean_size=patch_ratio, 
                                                         re_size=(-patch_ratio_range, patch_ratio_range), 
                                                         patch_number=Patch_per_Image, 
                                                         iteration_max=1000)
                

                if patch_position == None:
                    continue

                coordinates_list.append(coordinates_list_in1img)        # dota格式，四个顶点坐�?
                patch_positions_list.append(patch_position)     # dota格式

                # 产生img_metas字典
                img_meta['filename'] = img_path
                img_meta['ori_filename'] = file + img_format
                img_meta['ori_shape'] = (512, 512, 3)  # before padding
                img_meta['img_shape'] = (512, 512, 3)
                img_meta['pad_shape'] = (512, 512, 3)
                img_meta['scale_factor'] = np.array([1., 1., 1., 1.], dtype=np.float32)
                img_meta['flip'] = False
                img_meta['flip_direction'] = None
                img_meta['img_norm_cfg'] = dict(mean=np.array([123.675, 116.28, 103.53], dtype=np.float32), std=np.array([58.395, 57.12, 57.375], dtype=np.float32))
                img_meta['to_rgb'] = True
                img_metas.append(img_meta)


                # 读取图片文件
                img = cv2.imread(img_path) / 255.0  # BGR格式
                img = torch.from_numpy(img).permute(2, 0, 1).to(torch.float32)  # (3,height,width)�?BGR格式
                resize = transforms.Resize((512, 512))
                img = resize(transforms.ToPILImage()(img))
                img = transforms.ToTensor()(img)
                images_list.append(img)


            images_batch_t = torch.stack(images_list)  # (4,3,512,512)

            # 进行mask操作
            image_mask_batch = zero_out_bounding_boxes_v2(images_batch_t, coordinates_list)  # (batchsize,3,512,512), 未经归一化


            # # 进行可视化检查          
            # for pic_index in range(image_mask_batch.size(0)):
            #     image_mask = image_mask_batch[pic_index, :, :, :]  # (3, 384, 640)
            #     perturbed_img = image_mask.permute(1, 2, 0).cpu().numpy()   # (3, 384, 640), BGR
            #     perturbed_img = (np.ascontiguousarray(perturbed_img) * 255).astype(np.uint8)
            #     #     # 绘制patch_position边界            
            #     perturbed_img_patch_position = coordinates_list[pic_index]
            #     for box in perturbed_img_patch_position:
            #         pts = np.array([[box[i], box[i + 1]] for i in range(0, 7, 2)], np.int0)
            #         pts = pts.reshape((-1, 1, 2))
            #         cv2.polylines(perturbed_img, [pts], True, (0, 0, 255), 2)
            #     cv2.imwrite(f'out_MaskedImage_{pic_index}.jpg', perturbed_img)
            # assert False

            # 转换image_annotations至le90形式
            # batch_images_annotations_le90 = []      # len: batchsize
            # for coordinates_list_in1image in coordinates_list:
            #     single_image_annotations_le90 = []
            #     for annotation in coordinates_list_in1image:
            #         annotation_le90 = convert_to_le90(annotation)
            #         single_image_annotations_le90.append(annotation_le90)
            #     batch_images_annotations_le90.append(single_image_annotations_le90)

            # print(file_log)
            
            # patch_position确定
            # patch_location_range = (512, 512)
            # patch_positions_list = []
            # for image_index in range(image_BatchSize):
            #     single_image_annotations_select_le90 = batch_images_annotations_le90[image_index]
            #     patch_position = find_patch_positions_v3(img_size=patch_location_range, 
            #                                              bounding_box=single_image_annotations_select_le90,
            #                                              bounding_box=None,
            #                                              patch_size=64,
            #                                              mean_size=patch_ratio, 
            #                                              re_size=(-patch_ratio_range, patch_ratio_range), 
            #                                              patch_number=Patch_per_Image, 
            #                                              iteration_max=1000)
                
            #     patch_positions_list.append(patch_position)
            patch_positions_batch_t = torch.from_numpy(np.array(patch_positions_list)).cuda()  # (image_BatchSize,Patch_per_Image,5)

            # 对s2anet做处理，因为s2anet预训练的代码用的是le135格式，其与le90�?            
            if args.detector == 's2anet':
                patch_positions_batch_t_converted = patch_positions_batch_t.clone()

                # 提取宽高和角�?                # 提取宽高和角�?                
                w = patch_positions_batch_t[..., 2]  # shape=[image_BatchSize, Patch_per_Image]
                h = patch_positions_batch_t[..., 3]  # shape=[image_BatchSize, Patch_per_Image]
                theta = patch_positions_batch_t[..., 4]  # shape=[image_BatchSize, Patch_per_Image]
                
                # 找到需要转换的框（w < h的情况）
                mask = w < h
                
                # 对需要转换的�?
                # 1. 交换宽高
                patch_positions_batch_t_converted[mask, 2] = h[mask]  # 新的宽度为原来的高度
                patch_positions_batch_t_converted[mask, 3] = w[mask]  # 新的高度为原来的宽度
                # 2. 角度调整?                
                patch_positions_batch_t_converted[mask, 4] = theta[mask] + 90

                # 复制回去
                patch_positions_batch_t_0 = patch_positions_batch_t_converted
            else:
                patch_positions_batch_t_0 = patch_positions_batch_t

            fake_labels_batch_t_0 = torch.ones([image_BatchSize, Patch_per_Image, 1]).cuda()  # (image_BatchSize,Patch_per_Image,1)   
            fake_labels_batch_t = fake_labels_batch_t_0 * class_idx
            position_labels_batch_t = fake_labels_batch_t_0 * 0.0     # 训练过程只能针对一类物,该物体的position labels为全0
            
            # 对patch进行选择，从gen_batchsize中选择GeneratePatch_batchsize = image_BatchSize * Patch_per_Image个张�?            
            # 这里相当于要从data里面随机选image_batchsize * patch_per_image张补丁，需要考虑gan_batchsize与image_batchsize * patch_per_image的大小关�?            
            current_batchsize = fake_patch.size(0)
            if current_batchsize >= GeneratePatch_batchsize:
                indices = torch.randperm(current_batchsize)[:GeneratePatch_batchsize]
                fake_patch = fake_patch[indices]
            else:
                extra_needed = GeneratePatch_batchsize - current_batchsize
                extra_indices = torch.randint(0, current_batchsize, (extra_needed, ))
                fake_patch = torch.cat([fake_patch, fake_patch[extra_indices]], dim=0)
            
            # 对images进行resize
            fake_patch = fake_patch.view(image_BatchSize, Patch_per_Image, 3, patch_size[0], patch_size[1])


            # 调整fake_patch范围，从(-1,1)调整至(0,1)
            fake_patch.data = fake_patch.data.mul(0.5).add(0.5)

            # 进行apply
            img_size = (512, 512)
            # adv_batch_masked, msk_batch = patch_transformer.forward3(adv_batch=fake_patch, 
            #                                                          fake_bboxes_batch=patch_positions_batch_t, 
            #                                                          fake_labels_batch=position_labels_batch_t, 
            #                                                          img_size=img_size,
            #                                                          transform=True)    

            adv_batch_masked, msk_batch = patch_transformer.forward4(adv_batch=fake_patch, 
                                                                     ratio=patch_ratio,
                                                                     fake_bboxes_batch=patch_positions_batch_t, 
                                                                     fake_labels_batch=position_labels_batch_t, 
                                                                     img_size=img_size,
                                                                     transform=False)         

            p_img_batch = patch_applier(image_mask_batch.cuda(), adv_batch_masked.cuda())  # 这里p_img_batch还没做归一�?
            # 对p_img_batch进行归一化
            p_img_batch_normalize = AdvImagesTransformer_BGR(p_img_batch)
            # 转换成RGB
            p_img_batch_normalize = p_img_batch_normalize[:, [2, 1, 0], :, :]  # RGB, torch.Size([batch_size,3, H, W]), 经过scene dataset对应mean, std的归一�?
            # 计算detection loss
            

            # 进入loss_det计算环节之前的可视化检�?            
            # p_img_batch_normalize_copy = p_img_batch_normalize.clone()
            # # 反归一�?            
            # for i in range(p_img_batch_normalize_copy.size(0)):  # 遍历 batch 中的每个图像
            #     for c in range(p_img_batch_normalize_copy.size(1)):  # 遍历图像的每个通道
            #         p_img_batch_normalize_copy[i, c] = p_img_batch_normalize_copy[i, c].mul(std_RGB[c]).add(mean_RGB[c])  # Tensor, torch.Size([batch_size, 3, H, W]), RGB
            # # RGB->BGR
            # for pic_index in range(p_img_batch_normalize_copy.size(0)):       # 整个batch的batch_size张图片都进行存储
            #     adv_masked = p_img_batch_normalize_copy[pic_index, :, :, :]  # (3, 384, 640)
            #     adv_img_rgb = adv_masked.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
            #     adv_img_bgr = adv_img_rgb[:, :, [2, 1, 0]]  # BGR
            #     adv_img_bgr = (np.ascontiguousarray(adv_img_bgr) * 255).astype(np.uint8)
            #     # 绘制patch_position边界�?                
            #     perturbed_img_patch_position = patch_positions_batch_t[pic_index, :, :].cpu().numpy().tolist()
            #     for box in perturbed_img_patch_position:
            #         rect = ((box[0], box[1]), (box[2], box[3]), 1 * box[4] * 180 / np.pi)  # cv2.boxPoints中顺时针旋转为正
            #         rect = cv2.boxPoints(rect)
            #         rect = np.int0(rect)
            #         cv2.drawContours(adv_img_bgr, [rect], 0, (0, 0, 255), 2)
            # cv2.imwrite(f'fake_bboxes_test_{epoch}.jpg', adv_img_bgr)
            # assert False
            
            
            # 考虑s2anet的le135格式之前
            # loss_det = inference_det_loss_on_masked_images(model=model, 
            #                                                adv_images_batch_t=p_img_batch_normalize, 
            #                                                img_metas=img_metas, 
            #                                                patch_labels_batch_t=fake_labels_batch_t, 
            #                                                patch_boxes_batch_t=patch_positions_batch_t,
            #                                                model_name=args.detector)
            
            # 考虑s2anet le135
            loss_det = inference_det_loss_on_masked_images(model=model, 
                                                           adv_images_batch_t=p_img_batch_normalize, 
                                                           img_metas=img_metas, 
                                                           patch_labels_batch_t=fake_labels_batch_t, 
                                                           patch_boxes_batch_t=patch_positions_batch_t_0,
                                                           model_name=args.detector)

            ep_loss_det += loss_det.item()
            # 计算tv loss
            loss_tv = total_variation(fake_patch)
            ep_loss_tv += loss_tv.item()

            # update
            # optimizerG.zero_grad()
            loss = args.alpha * loss_det + args.beta * errG + args.gamma * loss_tv


            optimizerG.zero_grad()
            loss.backward()
            optimizerG.step()

        logger = '[%d/%d][%d/%d] loss_det: %f errD: %f errD_real: %f errD_fake: %f errG: %f loss_tv: %f' \
                  % (epoch, n_epochs, i, len(PatchDataloader), loss_det, errD, errD_real, errD_fake, errG, loss_tv)

        # logger = '[%d/%d][%d/%d] errD: %f errD_real: %f errD_fake: %f errG: %f' \
        #          % (epoch, n_epochs, i, len(PatchDataloader), errD, errD_real, errD_fake, errG)


        log_file.write(logger + '\n')
        log_file.flush()

    # log
    ep_loss_det = ep_loss_det / len(PatchDataloader) / args.Giters
    ep_errD = ep_errD / len(PatchDataloader) / args.Diters
    ep_errD_real = ep_errD_real / len(PatchDataloader) / args.Diters
    ep_errD_fake = ep_errD_fake / len(PatchDataloader) / args.Diters
    ep_errG = ep_errG / len(PatchDataloader) / args.Giters
    ep_loss_tv = ep_loss_tv / len(PatchDataloader) / args.Giters
    epoch_logger = '[%d/%d] loss_det: %f errD: %f errD_real: %f errD_fake: %f errG: %f loss_tv: %f' \
                   % (epoch, n_epochs, ep_loss_det, ep_errD, ep_errD_real, ep_errD_fake, ep_errG, ep_loss_tv)
    
    # epoch_logger = '[%d/%d] errD: %f errD_real: %f errD_fake: %f errG: %f' \
    #                % (epoch, n_epochs, ep_errD, ep_errD_real, ep_errD_fake, ep_errG)
    print(epoch_logger)

    if epoch % vis_save == 0:
        # 进行p_img_batch_normalize可视化        
        p_img_batch_normalize_copy = p_img_batch_normalize.clone()
        # 反归一化       
        for i in range(p_img_batch_normalize_copy.size(0)):  # 遍历 batch 中的每个图像
            for c in range(p_img_batch_normalize_copy.size(1)):  # 遍历图像的每个通道
                p_img_batch_normalize_copy[i, c] = p_img_batch_normalize_copy[i, c].mul(std_RGB[c]).add(mean_RGB[c])  # Tensor, torch.Size([batch_size, 3, H, W]), RGB

        adv_masked = p_img_batch_normalize_copy[0, :, :, :]  # (3, 384, 640)    # 选择一张进行存�?        
        adv_img_rgb = adv_masked.permute(1, 2, 0).cpu().detach().numpy()  # (384, 640, 3), RGB
        adv_img_bgr = adv_img_rgb[:, :, [2, 1, 0]]
        adv_img_bgr = (np.ascontiguousarray(adv_img_bgr) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.work_dir, 'combined_images', f'vis_gen_{epoch}.jpg'), adv_img_bgr)

        # 进行patch进行存储
        # 可视化real_samples
        real_pics_copy = data.clone()
        real_pics_copy.data = real_pics_copy.data.mul(0.5).add(0.5)  # torch.Size([batch_size,3,128,128])
        grid_real_pics = vutils.make_grid(real_pics_copy, nrow=8)  # torch.Size([3, patch_w + 2 x (num_column + 1), patch_h + 2 x (num_row + 1)])
        grid_real_pics_numpy = grid_real_pics.permute(1, 2, 0).cpu().detach().numpy()
        grid_real_pics_numpy = (np.ascontiguousarray(grid_real_pics_numpy) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.work_dir, 'NoRotatePatch', 'real_pic.jpg'), grid_real_pics_numpy)

        # 可视化fake_samples
        # fake_patch = fake_patch.view(image_BatchSize, Patch_per_Image, 3, patch_size[0], patch_size[1])
        # fake_patch.data = fake_patch.data.mul(0.5).add(0.5)


        fake = fake_patch.clone()
        fake = fake.view(fake_patch.size(0) * fake_patch.size(1), fake_patch.size(2), fake_patch.size(3), fake_patch.size(4))
        grid_fake_pics = vutils.make_grid(fake, nrow=8)  # torch.Size([3, patch_w + 2 x (num_column + 1), patch_h + 2 x (num_row + 1)])
        grid_fake_pics_numpy = grid_fake_pics.permute(1, 2, 0).cpu().detach().numpy()
        grid_fake_pics_numpy = (np.ascontiguousarray(grid_fake_pics_numpy) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.work_dir, 'NoRotatePatch', f'fake_samples_{epoch}.jpg'), grid_fake_pics_numpy)

    # do checkpointing
    if epoch % ckpt_save == 0:
        torch.save(netG.state_dict(), os.path.join(args.work_dir, 'checkpoints', f'netG_epoch{epoch}.pth'))
        torch.save(netD.state_dict(), os.path.join(args.work_dir, 'checkpoints', f'netD_epoch{epoch}.pth'))

