#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import random
import shutil

# ===================== 可在这里改配置 =====================
CONFIG = {
    # 数据集根目录（包含 images/ 和 masks/ 子目录）
    "dataset_root": "/exp/home/mingkai.liu/data/Kvasir_SEG",

    # 输出目录（None 表示与 dataset_root 同级生成 "<root>_split"）
    "out_dir": '/exp/home/mingkai.liu/data/Kvasir_SEG_split',

    # 训练集比例（测试集 = 1 - ratio）
    "train_ratio": 0.8,

    # 随机种子（保证可复现）
    "seed": 42,
}
# ===================== 以上为可编辑区域 =====================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}  # 掩码常见格式

def collect_pairs(images_dir: Path, masks_dir: Path):
    pairs = []
    missing_masks = []
    missing_imgs = []

    # 遍历所有图片
    for img_path in images_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
            # 构造对应的 mask 文件名：stem + "_lesion" + .bmp
            mask_name = img_path.stem + ".jpg"
            mask_path = masks_dir / mask_name
            if mask_path.exists():
                pairs.append((img_path, mask_path))
            else:
                missing_masks.append(img_path.name)

    # 反查：有 mask 但没有对应的 image
    for mask_path in masks_dir.rglob("*"):
        if mask_path.is_file() and mask_path.suffix.lower() in MASK_EXTS:
            if mask_path.name.endswith(".jpg"):
                base_name = mask_path.stem.replace("", "")
                img_name = base_name + ".jpg"
                img_path = images_dir / img_name
                if not img_path.exists():
                    missing_imgs.append(mask_path.name)

    return pairs, missing_imgs, missing_masks

def split_pairs(pairs, train_ratio=0.8, seed=42):
    rnd = random.Random(seed)
    rnd.shuffle(pairs)
    n_total = len(pairs)
    n_train = int(round(n_total * train_ratio))
    train = pairs[:n_train]
    test  = pairs[n_train:]
    return train, test

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))  # 始终拷贝，不移动

def write_split(pairs, out_root: Path, subset_name: str):
    img_out = out_root / subset_name / "images"
    msk_out = out_root / subset_name / "masks"

    for img_path, msk_path in pairs:
        # 目标文件名沿用原文件名（含扩展名）
        safe_copy(img_path, img_out / img_path.name)
        safe_copy(msk_path, msk_out / msk_path.name)

def main():
    root = Path(CONFIG["dataset_root"]).resolve()
    images_dir = root / "images"
    masks_dir  = root / "masks"

    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise SystemExit(f"未找到 {images_dir} 或 {masks_dir} 目录。请确认数据结构为 root/images 与 root/masks。")

    out_root = (
        Path(CONFIG["out_dir"]).resolve()
        if CONFIG["out_dir"] else
        root.with_name(root.name + "_split")
    )

    pairs, missing_imgs, missing_masks = collect_pairs(images_dir, masks_dir)
    if len(pairs) == 0:
        raise SystemExit("没有找到可配对的样本，请检查命名规则是否正确。")

    train, test = split_pairs(
        pairs,
        train_ratio=CONFIG["train_ratio"],
        seed=CONFIG["seed"]
    )

    # 创建输出并写入
    write_split(train, out_root, "train")
    write_split(test,  out_root, "test")

    # 报告
    print("==== 划分完成 ====")
    print(f"总配对样本数: {len(pairs)}")
    print(f"训练集: {len(train)}  | 测试集: {len(test)}  | 训练占比: {len(train)/len(pairs):.3f}")
    if missing_imgs:
        print(f"有 {len(missing_imgs)} 个掩码没有找到对应图片。示例: {missing_imgs[:5]}")
    if missing_masks:
        print(f"有 {len(missing_masks)} 张图片没有找到对应掩码。示例: {missing_masks[:5]}")
    print(f"输出目录: {out_root}")

if __name__ == "__main__":
    main()