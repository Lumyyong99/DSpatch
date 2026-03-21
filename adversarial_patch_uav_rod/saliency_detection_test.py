import cv2
import os
import argparse
import stat

### -----------------------------------------------------------    Setting     ---------------------------------------------------------------------- ###
parser = argparse.ArgumentParser(description='saliency detection testing settings')
parser.add_argument('--input_folder', default=None, help='folder contains testing images')
parser.add_argument('--BW_output_folder', default=None, help='folder to BW output files')
parser.add_argument('--CL_output_folder', default=None, help='folder to CL output files')

parser.add_argument('--gt_folder', default=None, help='folder contains origin scene images')
parser.add_argument('--patch_folder', default=None, help='folder contains patch scene images')
args = parser.parse_args()

# 原始图片文件夹路径和存储saliency detection图片的文件夹路径
input_folder = args.input_folder
BW_output_folder = args.BW_output_folder  # 黑白显著性图保存路径
CL_output_folder = args.CL_output_folder  # 彩色显著性图保存路径

# 检查保存文件夹是否存在，不存在则创建
if not os.path.exists(BW_output_folder):
    os.makedirs(BW_output_folder)
    os.makedirs(CL_output_folder)
    os.chmod(args.BW_output_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(args.CL_output_folder, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

# 创建saliency检测器
saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()

# 遍历文件夹中的所有图片
for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):  # 仅处理图片文件
        # 构建图片的完整路径
        img_path = os.path.join(input_folder, filename)

        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            print(f"无法读取图片: {filename}")
            continue

        # 进行显著性检测
        (success, saliency_map) = saliency_detector.computeSaliency(image)

        if success:
            # 将显著性图像二值化
            saliency_map = (saliency_map * 255).astype("uint8")

            # 对显著性图进行伪彩色处理
            saliency_color_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

            # 保存显著性检测后的图片，文件名与原图相同
            BW_output_path = os.path.join(BW_output_folder, filename)
            cv2.imwrite(BW_output_path, saliency_map)
            CL_output_path = os.path.join(CL_output_folder, filename)
            cv2.imwrite(CL_output_path, saliency_color_map)


def compute_saliency_map(image):
    saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency_detector.computeSaliency(image)
    if success:
        return (saliency_map * 255).astype("uint8")
    return None

def calculate_l2_loss(folder1, folder2):
    l2_losses = []
    
    # 遍历文件夹中的所有图片
    for filename in os.listdir(folder1):
        if filename.endswith('.jpg'):  # 仅处理jpg文件
            img_path1 = os.path.join(folder1, filename)
            img_path2 = os.path.join(folder2, filename)
            
            # 读取原始图片
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)
            
            if img1 is None or img2 is None:
                print(f"无法读取图片: {filename}")
                continue
            
            # 计算显著性图
            saliency_map1 = compute_saliency_map(img1)
            saliency_map2 = compute_saliency_map(img2)
            
            if saliency_map1 is None or saliency_map2 is None:
                print(f"无法计算显著性图: {filename}")
                continue
            
            # 计算L2损失
            l2_loss = np.sqrt(np.mean((saliency_map1.astype(float) - saliency_map2.astype(float)) ** 2))
            l2_losses.append(l2_loss)
    
    # 计算均值
    if l2_losses:
        mean_l2_loss = np.mean(l2_losses)
        return mean_l2_loss
    else:
        return None

# 示例调用
path1 = args.gt_folder  # 原始图片文件夹1
path2 = args.patch_folder  # 原始图片文件夹2
mean_l2_loss = calculate_l2_loss(path1, path2)

if mean_l2_loss is not None:
    print(f"均值 L2损失: {mean_l2_loss:.2f}")
else:
    print("未计算到任何L2损失值。")