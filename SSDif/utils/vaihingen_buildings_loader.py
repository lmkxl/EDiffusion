import numpy as np
import os
import torch
#from torchvision.io import read_image
import torchvision.transforms.functional as TF

from PIL import Image
import torchvision.transforms as transforms

from torch.utils import data

#decode_segmap 将分割标签张量转换为彩色图像
def decode_segmap(seg, is_one_hot=False):
    colors = torch.tensor([
            [0, 0, 0],#黑色
            [255, 255, 255],#白色
        ], dtype=torch.uint8)#[0, 0, 0] 表示背景，[255, 255, 255] 表示前景。
    if is_one_hot:
        seg = torch.argmax(seg, dim=0)#如果 is_one_hot 为 True，则用 argmax 提取类别索引
    # convert classes to colors
    seg_img = torch.empty((seg.shape[0], seg.shape[1], 3), dtype=torch.uint8)
    for c in range(colors.shape[0]):
        seg_img[seg == c, :] = colors[c]
    return seg_img.permute(2, 0, 1) #将每个像素的类别标签映射为对应颜色，并返回重新排列的 RGB 图像张量

class VaihingenBuildingsLoader(data.Dataset):
    """Vaihingen Buildings dataloader"""

    def __init__(
            self,
            root,
            split="train", #数据分割类型（训练或测试）
            is_transform=True, #是否应用数据转换
            img_size=512, #图像尺寸，可以是整数或元组
            augmentations=None, #数据增强参数（默认无）
            img_norm=True, #是否归一化图像
    ):
        self.root = root
        if split == "val": # There is no separate validation data
            split = "test"
        self.split = split
        self.is_transform = is_transform
        self.n_classes = 2
        self.augmentations = augmentations
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.img_norm = img_norm
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.images = {}
        self.labels = {}

        self.setup() #调用 setup() 方法加载图像和标签路径

    #setup 方法根据图像数量加载并划分数据集
    def setup(self):
        n_train = 100 #定义训练集的样本数
        image_list = [] # 分别存储图像和标签的路径
        label_list = []
        for i in range(168):
            image_list.append(os.path.join(self.root, "building_{:03d}.png".format(i+1)))
            label_list.append(os.path.join(self.root, "building_mask_{:03d}.png".format(i+1)))
        self.images["train"] = image_list[:n_train]
        self.labels["train"] = label_list[:n_train]
        self.images["test"] = image_list[n_train:]
        self.labels["test"] = label_list[n_train:]
        #使用前 100 个样本作为训练集，剩下的作为测试集


    def __len__(self):
        return len(self.images[self.split])#返回数据集的样本数量，基于 split（train 或 test)

    #__getitem__ 方法获取指定 index 的图像和标签路径
    def __getitem__(self, index):
        img_path = self.images[self.split][index]
        lbl_path = self.labels[self.split][index]
        # Read image and label
        #img = read_image(img_path)
        #lbl = read_image(lbl_path).long()

        #使用 ToTensor() 将图像和标签转换为张量，并读取图像和标签
        totensor = transforms.ToTensor()
        img = totensor(Image.open(img_path))
        lbl = totensor(Image.open(lbl_path)).long()

        # Resize:将图像和标签调整为 self.img_size，使用双线性插值调整图像大小，标签则用最近邻插值
        img = TF.resize(img, self.img_size, antialias=True)
        lbl = TF.resize(lbl, self.img_size, interpolation=TF.InterpolationMode.NEAREST, antialias=True).squeeze(0).long()

        if self.split == "train":
            # 随机翻转
            if np.random.random() < 0.5:
                img = TF.vflip(img)
                lbl = TF.vflip(lbl)
            if np.random.random() < 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
            # 随机旋转
            if np.random.random() < 0.5:
                lbl = lbl.unsqueeze(0)
                angle = np.random.randint(-180, 180)
                img = TF.rotate(img, angle)
                lbl = TF.rotate(lbl, angle)
                lbl = lbl.squeeze(0)
            # 随机调整图像的对比度、饱和度和色调
            if np.random.random() < 0.5:
                img = TF.adjust_contrast(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_saturation(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_hue(img, np.random.random() * 0.05)

        # 图像归一化:如果 img_norm 为真，则按均值和标准差归一化图像。
        if self.img_norm:
            img = TF.normalize(img.float(), self.mean, self.std)

        return img, lbl
                
        
        
    
    