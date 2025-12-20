import numpy as np
import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as transforms
from torch.utils import data

def decode_segmap(seg, is_one_hot=False):
    colors = torch.tensor([
            [0, 0, 0],
            [255, 255, 255],
        ], dtype=torch.uint8)
    if is_one_hot:
        seg = torch.argmax(seg, dim=0)
    seg_img = torch.empty((seg.shape[0], seg.shape[1], 3), dtype=torch.uint8)
    for c in range(colors.shape[0]):
        seg_img[seg == c, :] = colors[c]
    return seg_img.permute(2, 0, 1)

class refugeDatasetLoader(data.Dataset):
    """refuge Dataset Loader"""

    def __init__(
            self,
            root,
            split="train",
            is_transform=True,
            img_size=(1634,1634),
            augmentations=None,
            img_norm=True,
    ):
        self.root = root
        self.split = split.lower()  # 'train' or 'test'
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.img_norm = img_norm
        self.n_classes = 2
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.augmentations = augmentations
        self.images = []
        self.labels = []

        self.setup()

    def setup(self):
        image_dir = os.path.join(self.root, self.split, "images")
        label_dir = os.path.join(self.root, self.split, "masks")

        # 获取 images 和 masks 文件列表
        image_files = sorted(os.listdir(image_dir))
        label_files = sorted(os.listdir(label_dir))

        for img_file, lbl_file in zip(image_files, label_files):
            self.images.append(os.path.join(image_dir, img_file))
            self.labels.append(os.path.join(label_dir, lbl_file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        lbl_path = self.labels[index]

        totensor = transforms.ToTensor()
        img = totensor(Image.open(img_path).convert("RGB"))
        lbl = totensor(Image.open(lbl_path)).long()

        # Resize 图像和标签
        img = TF.resize(img, self.img_size, antialias=True)
        lbl = TF.resize(lbl, self.img_size, interpolation=TF.InterpolationMode.NEAREST, antialias=True).squeeze(0).long()

        # 数据增强 (仅在训练模式下)
        if self.split == "train" and self.augmentations:
            if np.random.random() < 0.5:
                img = TF.vflip(img)
                lbl = TF.vflip(lbl)
            if np.random.random() < 0.5:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)
            if np.random.random() < 0.5:
                lbl = lbl.unsqueeze(0)
                angle = np.random.randint(-180, 180)
                img = TF.rotate(img, angle)
                lbl = TF.rotate(lbl, angle)
                lbl = lbl.squeeze(0)
            if np.random.random() < 0.5:
                img = TF.adjust_contrast(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_saturation(img, 0.75 + np.random.random() * 0.5)
                img = TF.adjust_hue(img, np.random.random() * 0.05)

        # 归一化
        if self.img_norm:
            img = TF.normalize(img.float(), self.mean, self.std)

        return img, lbl