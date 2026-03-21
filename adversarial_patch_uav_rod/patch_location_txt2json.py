"""
log 2025.1.9
仅用于从uavrod_fake_patch_locations.txt生成对应的json文件。
"""
import json

def convert_txt_to_json(txt_path, json_path):
    """
    将原始txt格式转换为JSON格式
    
    参数:
    txt_path: 输入的txt文件路径
    json_path: 输出的JSON文件路径
    """
    # 初始化数据字典
    data_dict = {}
    current_image = None
    patches = []
    
    # 读取txt文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # 解析每一行
    for line in lines:
        line = line.strip()
        if not line:  # 跳过空行
            continue
            
        if line.endswith(('.jpg','.png')):  # 新的图片名称
            if current_image is not None:
                data_dict[current_image] = {"patches": patches}
            current_image = line
            patches = []
        else:  # 补丁信息
            # 解析坐标和角度
            parts = line.split(']')
            coords = eval(parts[0] + ']')  # 使用eval解析坐标列表
            angle = float(parts[1].strip())
            patches.append({
                "center": coords,
                "angle": angle
            })
    
    # 添加最后一个图片的补丁信息
    if current_image is not None:
        data_dict[current_image] = {"patches": patches}
    
    # 写入JSON文件
    with open(json_path, 'w') as f:
        json.dump(data_dict, f, indent=4)
    
    return data_dict

# 使用示例
txt_file = './vis_results_DSAP/uavrod_phys_exps/uavrod_phys_locations.txt'  # 输入的txt文件路径
json_file = './vis_results_DSAP/uavrod_phys_exps/uavrod_phys_locations.json'  # 输出的JSON文件路径

# 转换文件
data = convert_txt_to_json(txt_file, json_file)

# 打印转换后的数据示例
print("转换完成！JSON文件已保存到:", json_file)
print("\n数据示例:")
# 打印第一个图片的信息作为示例
first_image = list(data.keys())[0]
print(json.dumps({first_image: data[first_image]}, indent=4))