import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm  # 进度条工具

def convert_masks_to_binary(input_folder, output_folder, threshold=128):
    """
    将文件夹中的mask图片转换为二值图并保存为PNG
    
    参数:
        input_folder: 包含原始mask图片的文件夹路径
        output_folder: 输出二值图的文件夹路径
        threshold: 二值化阈值(0-255)，默认128
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有图片文件
    valid_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    files = [f for f in os.listdir(input_folder) 
             if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"找到 {len(files)} 个mask文件，开始转换...")
    
    for filename in tqdm(files, desc="Processing Masks"):
        try:
            # 读取图片 (兼容不同格式)
            img_path = os.path.join(input_folder, filename)
            
            # 方法1: 使用OpenCV (自动处理透明度通道)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # 如果读取失败，尝试用Pillow
            if img is None:
                img = np.array(Image.open(img_path).convert('L'))
            
            # 二值化处理
            _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
            
            # 生成输出路径 (保持原名，强制.png后缀)
            output_path = os.path.join(output_folder, 
                                     os.path.splitext(filename)[0] + ".png")
            
            # 保存为PNG (无损压缩)
            cv2.imwrite(output_path, binary, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            
        except Exception as e:
            print(f"\n处理文件 {filename} 时出错: {str(e)}")
            continue

if __name__ == "__main__":
    # 使用示例
    input_dir = "/exp/home/mingkai.liu/data/FIVES/train/masks"  # 替换为你的mask文件夹路径
    output_dir = "/exp/home/mingkai.liu/data/FIVES/train/masks1"  # 替换为输出文件夹路径
    
    convert_masks_to_binary(input_dir, output_dir)
    
    print(f"\n转换完成！二值图已保存到: {output_dir}")