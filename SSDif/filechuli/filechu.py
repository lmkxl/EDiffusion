import os
from PIL import Image


def convert_images_to_png(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 检查文件是否为图像文件（支持常见格式）
        if filename.lower().endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.png')):
            # 打开图像文件
            with Image.open(file_path) as img:
                # 构建输出文件路径，确保保存为PNG格式
                output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

                # 将图像保存为PNG格式
                img.save(output_path, 'PNG')
                print(f"Converted and saved: {output_path}")


# 示例用法
input_folder = '/exp/home/mingkai.liu/data/ISIC/test/image1'  # 替换为输入文件夹路径
output_folder = '/exp/home/mingkai.liu/data/ISIC/test/images'  # 替换为输出文件夹路径
convert_images_to_png(input_folder, output_folder)