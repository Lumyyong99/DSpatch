from PIL import Image
import os
import sys

def rotate_images(input_folder, output_folder, rotation=90):
    """
    旋转指定文件夹中的所有图片并保存到新文件夹
    
    参数:
    input_folder (str): 输入图片文件夹路径
    output_folder (str): 输出文件夹路径
    rotation (int): 旋转角度(90/180/270)
    """
    # 验证旋转角度
    valid_rotations = {90, 180, 270}
    if rotation not in valid_rotations:
        raise ValueError("旋转角度必须是90、180或270度")
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # 处理所有图片
    processed = 0
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(image_extensions):
            try:
                # 构建完整文件路径
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, filename)
                
                # 打开并旋转图片
                with Image.open(input_path) as img:
                    # 使用transpose确保无损旋转
                    if rotation == 90:
                        rotated = img.transpose(Image.ROTATE_90)
                    elif rotation == 180:
                        rotated = img.transpose(Image.ROTATE_180)
                    elif rotation == 270:
                        rotated = img.transpose(Image.ROTATE_270)
                    
                    # 保存图片（保留原始格式和元数据）
                    rotated.save(output_path, quality=95, subsampling=0)
                
                processed += 1
                print(f"已处理: {filename}")
                
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")
    
    print(f"\n完成! 共处理 {processed} 张图片")
    print(f"输出目录: {os.path.abspath(output_folder)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python rotate_images.py <输入文件夹> <输出文件夹> <旋转角度>")
        print("示例: python rotate_images.py input_images rotated_images 90")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    try:
        rotation_angle = int(sys.argv[3])
        rotate_images(input_dir, output_dir, rotation_angle)
    except ValueError as e:
        print(f"错误: {str(e)}")
        sys.exit(1)
