# 在pyiqa环境下运行，torchvision版本合格
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import json
import glob
import os
import torch.nn.functional as F
from ipdb import set_trace as st

def load_labels(json_path):
    with open(json_path, 'r') as f:
        labels = json.load(f)
    # 确保标签数量匹配
    assert len(labels) == 1000, f"标签数量应为1000，当前为{len(labels)}"
    return labels

# 替换为您的实际JSON文件路径
label_path = "./imagenet_classes.json"
labels = load_labels(label_path)

def get_topk_predictions(probabilities, k=5):
    """
    获取top-k预测结果
    
    参数:
    probabilities - 形状为[1000]的概率张量
    k - 要返回的顶部结果数量
    
    返回:
    topk_indices - 前k个索引
    topk_probs - 前k个概率值
    topk_labels - 前k个标签
    """
    topk_probs, topk_indices = torch.topk(probabilities, k)
    topk_labels = [labels[i.item()] for i in topk_indices]
    return topk_indices, topk_probs, topk_labels

def print_topk_predictions(topk_indices, topk_probs, topk_labels):
    print("\nTop 5 Predictions:")
    print("-" * 50)
    print(f"{'Rank':<5} | {'Class Index':<10} | {'Probability':<12} | Label")
    print("-" * 50)
    
    for rank, (idx, prob, label) in enumerate(zip(topk_indices, topk_probs, topk_labels)):
        idx_val = idx.item()
        prob_val = prob.item()
        
        # 截断过长的标签（保持输出整洁）
        display_label = label if len(label) <= 45 else label[:42] + "..."
        
        print(f"{rank+1:<5} | {idx_val:<10} | {prob_val * 100:>10.2f}%  | {display_label}")
    
    # 添加最高和最低置信度信息
    min_prob = probabilities.min().item() * 100
    mean_prob = probabilities.mean().item() * 100
    print("-" * 50)
    print(f"Highest confidence: {topk_probs[0].item() * 100:.2f}%")
    print(f"Lowest confidence: {min_prob:.2f}% (all < {mean_prob:.2f}%)")

def process_predictions(probabilities, label_path):
    # 加载标签
    labels = load_labels(label_path)
    
    # 获取top-5预测
    topk_indices, topk_probs, topk_labels = get_topk_predictions(probabilities)
    
    # 打印结果
    print_topk_predictions(topk_indices, topk_probs, topk_labels)
    
    # 可选：返回结果供进一步使用
    return {
        'indices': topk_indices,
        'probabilities': topk_probs,
        'labels': topk_labels
    }


# 1. 加载预训练模型
model = torchvision.models.vgg16(weights=torchvision.models.vgg.VGG16_Weights.IMAGENET1K_V1)
model.eval()  # 设置为评估模式

# 2. 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),                    # 调整尺寸
    transforms.CenterCrop(224),                # 中心裁剪
    transforms.ToTensor(),                     # 转为张量
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],           # ImageNet标准化参数
        std=[0.229, 0.224, 0.225]
    )
])

# 3. 加载并预处理图像
# image_folder = '../train_advpatch_uavrod/multiseeds/multiseedpatches'     # advpatch
# image_folder = '../train_NewSourceDomain_20250614/differentpatchesjpg'    # DSAP
image_folder = '../train_DSAP_uavrod/train_NewSourceDomain_more_source_images_20250626/different_patches_jpg_select'   # new DSAP
# image_folder = '../../AdvART/different_seeds_patches'    # AdvART
# image_folder ='../../NAP-uav_rod/different_seed_patch'     # NAP

image_extentions = ['jpg', 'jpeg', 'png']
image_paths = []
for ext in image_extentions:
    image_paths.extend(glob.glob(os.path.join(image_folder, f'*.{ext}')))
num_images = len(image_paths)
# 存储处理后的张量
processed_tensors = []
# 读取图片并进行保存
for i, image_path in enumerate(image_paths):
    try:
        image = Image.open(image_path)
        input_tensor = preprocess(image)
        processed_tensors.append(input_tensor)
        print(f'processed image: {i+1}/{num_images}: {os.path.basename(image_path)}')
    except Exception as e:
        print(f'error in processing {image_path}')
        continue


if processed_tensors:
    input_batch = torch.stack(processed_tensors)


# 4. 使用GPU加速（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_batch = input_batch.to(device)    # torch.Size([1, 3, 224, 224])
model = model.to(device)

# 5. 执行推理
with torch.no_grad():
    output = model(input_batch)


# 计算余弦相似度
normalized_tensor = F.normalize(output, p=2, dim=1)
similarity_matrix = torch.mm(normalized_tensor, normalized_tensor.t())

# 分析余弦相似度矩阵
def analyze_similarity(sim_matrix):
    """分析并打印相似度矩阵的统计信息"""
    batch_size = sim_matrix.size(0)
    
    print("\n相似度分析:")
    print("-" * 50)
    
    # 对角线是每个样本与自身的相似度（应为1.0）
    diagonal = sim_matrix.diag()
    print(f"自相似度: 均值={diagonal.mean().item():.4f} (范围: {diagonal.min().item():.4f}-{diagonal.max().item():.4f})")
    
    # 获取非对角线元素（不包括自身比较）
    mask = ~torch.eye(batch_size, dtype=torch.bool)
    off_diagonal = sim_matrix[mask]
    
    # 相似度统计
    print(f"样本间相似度: 均值={off_diagonal.mean().item():.4f}")
    print(f"最大相似度: {off_diagonal.max().item():.4f} (样本{torch.argmax(off_diagonal)//(batch_size-1)}和{torch.argmax(off_diagonal)%(batch_size-1)})")
    print(f"最小相似度: {off_diagonal.min().item():.4f}")
    
    # 打印每个样本与其他样本的平均相似度
    print("\n每个样本与其他样本的平均相似度:")
    for i in range(batch_size):
        # 排除与自身的比较
        others = torch.cat([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        print(f"样本 {i}: {others.mean().item():.4f} (范围: {others.min().item():.4f}-{others.max().item():.4f})")
    
    return {
        'matrix': sim_matrix,
        'stats': {
            'self_similarity_mean': diagonal.mean().item(),
            'inter_similarity_mean': off_diagonal.mean().item(),
            'max_similarity': off_diagonal.max().item(),
            'min_similarity': off_diagonal.min().item()
        }
    }

# 执行分析
analysis_result = analyze_similarity(similarity_matrix)

def visualize_similarity(sim_matrix, filename="cosine_similarity_matrix_DSAP_moresource.png", title="Cosine Similarity Matrix"):
    """可视化余弦相似度矩阵并保存为文件"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(10, 8))
    plt.imshow(sim_matrix.cpu().numpy(), cmap='viridis', vmin=-1, vmax=1)
    cbar = plt.colorbar(label='Cosine Similarity')
    cbar.set_label('Cosine Similarity', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    
    # 设置刻度标签
    plt.xticks(np.arange(sim_matrix.size(0)))
    plt.yticks(np.arange(sim_matrix.size(0)))
    
    # 添加数值标签（当矩阵较小时）
    if sim_matrix.size(0) <= 15:  # 样本数量不超过15时添加数值
        for i in range(sim_matrix.size(0)):
            for j in range(sim_matrix.size(1)):
                plt.text(j, i, f"{sim_matrix[i, j].item():.2f}", 
                         ha='center', va='center', color='w', fontsize=8)
    
    plt.tight_layout()
    
    # 保存图像（多种格式）
    ext = '.png'
    save_path = filename.replace('.png', ext) if ext != '.png' else filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"已保存相似度矩阵图像至: {save_path}")
    
    plt.close()  # 关闭图形，释放内存

    
visualize_similarity(similarity_matrix)





# # 6. 解析结果
# probabilities = torch.nn.functional.softmax(output[0], dim=0)

# # 7. 加载类别标签
# result = process_predictions(probabilities, label_path)





