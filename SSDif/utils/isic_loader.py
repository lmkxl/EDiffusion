"ISIC数据集导入"
import numpy as np
import os
import torch
#import torchvision.io.read_image
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

class ISICLoader(data.Dataset):
    """Vaihingen Buildings dataloader"""
    def __init__(
            self,
            root,
            split="train",
            is_transform=True,
            img_size=512,
            augmentations=None,
            img_norm=True,
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

        self.setup()
