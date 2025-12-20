from PIL import Image
import os

def crop_and_convert_to_png(input_folder, output_folder=None):
    """
    将 input_folder 文件夹中的所有图片裁剪为 512x512 大小，并转换为 PNG 格式，保存到 output_folder。
    如果 output_folder 为 None，则覆盖原图。
    """
    target_size = (512, 512)

    # 如果指定了输出文件夹且文件夹不存在，则创建它
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 仅处理图片文件
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # 中心裁剪和调整大小
            width, height = img.size
            left = (width - target_size[0]) / 2
            top = (height - target_size[1]) / 2
            right = (width + target_size[0]) / 2
            bottom = (height + target_size[1]) / 2
            img_cropped = img.crop((left, top, right, bottom))

            # 转换为 PNG 格式并确定保存路径
            base_name = os.path.splitext(filename)[0]  # 获取文件名（不含扩展名）
            save_path = os.path.join(output_folder, f"{base_name}.png") if output_folder else os.path.join(input_folder, f"{base_name}.png")

            # 保存裁剪并转换后的图像为 PNG 格式
            img_cropped.save(save_path, "PNG")
            print(f"已裁剪并保存图片为 PNG 格式: {save_path}")

# 示例用法
input_folder = '/exp/home/mingkai.liu/data/Refuge/Test/mask1'  # 输入图片文件夹路径
output_folder = '/exp/home/mingkai.liu/data/Refuge/Test/masks'  # 输出文件夹路径（如果不指定则覆盖原图）

crop_and_convert_to_png(input_folder, output_folder)