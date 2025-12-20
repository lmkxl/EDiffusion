import os
from PIL import Image


def resize_images(input_folder, output_folder, size=(1634, 1634)):
    # 检查输出文件夹是否存在，不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # 检查文件是否为图像文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            # 打开图像文件
            with Image.open(file_path) as img:
                # 调整图像大小
                resized_img = img.resize(size, Image.ANTIALIAS)

                # 构建输出文件路径，确保保存为PNG格式
                output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

                # 保存图像为PNG格式
                resized_img.save(output_path, 'PNG')
                print(f"Saved resized image to: {output_path}")


# 示例用法
input_folder = '/exp/home/mingkai.liu/data/Refuge/test/masks'  # 替换为输入文件夹路径
output_folder = '/exp/home/mingkai.liu/data/Refuge/test/masks1'  # 替换为输出文件夹路径
resize_images(input_folder, output_folder)