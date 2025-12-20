"""Evaluates the performance of a model"""
import logging
import math
import torch
import torch.nn.functional as F
import torchvision

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import JaccardIndex, F1Score ,Dice,Accuracy

from tqdm import tqdm
from skimage.filters import sobel
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import time

'''
from utils.cityscapes_loader import decode_segmap as decode_segmap_cityscapes
from utils.utils import diffuse, denoise_scale
from utils.uavid_loader import decode_segmap as decode_segmap_uavid
from utils.vaihingen_buildings_loader import decode_segmap as decode_segmap_vaihingen
'''
from utils.cityscapes_loader import decode_segmap as decode_segmap_cityscapes
from utils.utils import diffuse, denoise_scale
from utils.uavid_loader import decode_segmap as decode_segmap_uavid
from utils.vaihingen_buildings_loader import decode_segmap as decode_segmap_vaihingen

from utils.isicloarers import decode_segmap as decode_segmap_isic
from utils.refugeloader import decode_segmap as decode_segmap_refuge
from utils.monuseg_loader import decode_segmap as decode_segmap_monuseg
from utils.glas_loader import decode_segmap as decode_segmap_glas
from utils.hrf_loader import decode_segmap as decode_segmap_hrf
from utils.ph_loader import decode_segmap as decode_segmap_ph
from utils.five_loader import decode_segmap as decode_segmap_five
from utils.kvasir_loader import decode_segmap as decode_segmap_kvasir

from thop import profile, clever_format
from torchvision.transforms import ToPILImage

#计算加权交叉熵损失，用于处理类别不平衡问题
def segmentation_cross_entropy(predicted_segmentation, target_segmentation):
    """Returns Cross Entropy Loss"""
    #weights = torch.tensor([1.79, 1.0, 2.17, 1.17, 3.2, 27.14, 21.56, 190.25], dtype=torch.float32).to(target_segmentation.device)
    weights = torch.tensor([0.6,10.5], dtype=torch.float32).to(target_segmentation.device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights, reduction='sum')
    loss = criterion(predicted_segmentation, target_segmentation)
    return loss

#===========================Dice损失==================================
#对类别不平衡敏感，可以有效对齐小目标区域和背景
def dice_loss(logits, targets, smooth=1e-6):
    # Apply sigmoid to logits to get probabilities
    probs = torch.sigmoid(logits)

    # Flatten the tensors
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)

    # Compute intersection and union
    intersection = (probs_flat * targets_flat).sum()
    union = probs_flat.sum() + targets_flat.sum()

    # Compute Dice coefficient
    dice = (2.0 * intersection + smooth) / (union + smooth)

    # Return Dice loss
    return 1.0 - dice
#================================================


#================组合Dice损失和Bce损失===============================
def adaptive_combined_loss(logits, targets, smooth=1e-6):
    """
    Compute adaptive combined loss using Dice loss and Cross-Entropy loss.
    Args:
        logits (torch.Tensor): Raw model predictions (before sigmoid or softmax).
        targets (torch.Tensor): Ground truth binary masks.
        smooth (float): Smoothing factor for Dice loss.

    Returns:
        torch.Tensor: Combined loss.
    """
    # Compute Dice Loss
    probs = torch.sigmoid(logits)  # Apply sigmoid to get probabilities
    probs_flat = probs.view(-1)
    targets_flat = targets.view(-1)
    intersection = (probs_flat * targets_flat).sum()
    union = probs_flat.sum() + targets_flat.sum()
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

    # Compute Cross-Entropy Loss
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets)

    # Compute adaptive weights
    dice_weight = 1.0 / (dice_loss + smooth)
    ce_weight = 1.0 / (ce_loss + smooth)
    total_weight = dice_weight + ce_weight
    dice_weight /= total_weight
    ce_weight /= total_weight

    # Combined Loss
    total_loss = dice_weight * dice_loss + ce_weight * ce_loss
    return total_loss

#===========================================================


#计算均方误差 (MSE)，用于衡量预测和目标噪声之间的差异.
def noise_mse(noise_predicted, noise_target):
    """Returns MSE Loss"""
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(noise_predicted, noise_target)
    return loss

#计算总损失，这里只返回了分割交叉熵损失。
def compute_total_loss(segmentation_cross_entropy):
    """Returns total loss"""
    total_loss =  (1 * segmentation_cross_entropy)
    return total_loss

#将图像和分割结果写入 TensorBoard，以便可视化模型预测和真实标注的对比。
def write_images_to_tensorboard(writer, epoch, image=None, seg_diffused=None, seg_predicted=None, seg_gt=None, datasplit='validation', dataset_name='cityscapes'):
        """Writes images to TensorBoard"""
        # decode segmap based on dataset
        if dataset_name == 'cityscapes':
            decode_segmap = decode_segmap_cityscapes
        elif dataset_name == 'uavid':
            decode_segmap = decode_segmap_uavid
        elif dataset_name == 'vaihingen':
            decode_segmap = decode_segmap_vaihingen
        elif dataset_name == 'isic':
            decode_segmap = decode_segmap_isic
        elif dataset_name == 'refuge':
            decode_segmap = decode_segmap_refuge
        elif dataset_name == 'monuseg':
            decode_segmap = decode_segmap_monuseg
        elif dataset_name == 'glas':
            decode_segmap = decode_segmap_glas
        elif dataset_name == 'hrf':
            decode_segmap = decode_segmap_hrf
        elif dataset_name == 'ph':
            decode_segmap = decode_segmap_ph
        elif dataset_name == 'five':
            decode_segmap = decode_segmap_five
        elif dataset_name == 'kvasir':
            decode_segmap = decode_segmap_kvasir
            
        else:
            raise NotImplementedError('Dataset {} not implemented'.format(dataset_name))
        if image is not None:
            image = torchvision.utils.make_grid(image, normalize=True) # normalize to [0,1] and convert to uint8;make_grid:将多个图像拼接成网格形式便于展示
            writer.add_images('{}/image'.format(datasplit), image, epoch, dataformats='CHW')
        if seg_diffused is not None:
            seg_diffused = decode_segmap(seg_diffused, is_one_hot=True)
            writer.add_images('{}/seg_diffused'.format(datasplit), seg_diffused, epoch, dataformats='CHW')
        if seg_predicted is not None:
            seg_predicted = decode_segmap(seg_predicted, is_one_hot=True)
            writer.add_images('{}/seg_predicted'.format(datasplit), seg_predicted, epoch, dataformats='CHW')
        if seg_gt is not None:
            seg_gt = decode_segmap(seg_gt, is_one_hot=False)
            writer.add_images('{}/seg_gt'.format(datasplit), seg_gt, epoch, dataformats='CHW')

#在多个尺度上进行逐步去噪。
def denoise_loop_scales(model, device, network_config, images):
    """Denoises all scales for a single timestep"""
    #计算每个尺度对应的图像大小(smallest first)
    scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

    # 初始化预测图（随机噪声） (random noise)
    seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # 构建去噪结果的集合。
    seg_denoised_ensemble = torch.zeros(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Denoise whole segmentation map in steps
    for timestep in range(network_config.n_timesteps): # for each step
        
        for scale in range(network_config.n_scales): # for each scale
            # Resize to current scale
            images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
            seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False).softmax(dim=1)

            # Diffuse
            t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]) # time step
            seg_diffused = diffuse(seg_previous_scaled, t)  
            # Denoise
            seg_denoised = denoise_scale(model, device, seg_diffused, images_scaled, t, patch_size=network_config.max_patch_size)

            # Update the previous segmentation map
            seg_previous_scaled = seg_denoised
        
        # Add to ensemble
        if network_config.built_in_ensemble:
            if timestep == 0:
                seg_denoised_ensemble = seg_denoised
            else:
                seg_denoised_ensemble = seg_denoised_ensemble / 2 + seg_denoised / 2
            
            seg_previous_scaled = seg_denoised_ensemble
    
    return seg_denoised

#按线性时间步长在单一尺度上进行去噪。
def denoise_linear_scales(model, device, network_config, images):
    """Denoises one scale at a each timestep"""
    # 计算尺度大小(smallest first)
    scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

    # 每个尺度对应的时间步数。 (random noise)
    seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

    # Denoise whole segmentation map in steps
    for timestep in range(network_config.n_timesteps): # for each step
        # Get the current scale
        timesteps_per_scale = math.ceil(network_config.n_timesteps / network_config.n_scales)
        scale = timestep // timesteps_per_scale
        
        # Resize to current scale
        if timestep % timesteps_per_scale == 0:
            images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False) #在每个尺度上调整图像大小
            seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False)

        # Diffuse
        t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]) # time step

        seg_diffused = diffuse(seg_previous_scaled, t)
        # Denoise
        seg_denoised = denoise_scale(model, device, seg_diffused, images_scaled, t, patch_size=network_config.max_patch_size)

        # Update the previous segmentation map
        seg_previous_scaled = seg_denoised

    return seg_denoised

#根据配置选择去噪方法，支持循环和线性两种方式
def denoise(model, device, network_config, images):
        """Denoises the segmentation map"""
        if network_config.scale_procedure == 'loop':
            seg_denoised = denoise_loop_scales(model, device, network_config, images)
        elif network_config.scale_procedure == 'linear':
            seg_denoised = denoise_linear_scales(model, device, network_config, images)

        return seg_denoised


def save_images_to_folder(images, seg_predicted, seg_gt, it_total, output_dir='/exp/home/mingkai.liu/prodiffusmamba/remamba/output/preciates'):
    """
    保存图像、预测分割和地面真值分割到指定文件夹
    """
    # 创建保存目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        # 将预测结果、标签、原图保存为PNG图片
    # =========================================

    images = images.squeeze(0).cpu().detach().numpy()
    image_rgb = images.transpose(1, 2, 0)  # 将 [3, 256, 256] 转为 [256, 256, 3]
    image_rgb = (image_rgb * 255).astype(np.uint8)
    # 使用 PIL 保存为 PNG 图像
    #print(image_rgb.shape)
    img = Image.fromarray(image_rgb)

    # =======================================
    # 将图像转换为 NumPy 数组并归一化到 0-255 范围
    # label_gt = (label.numpy() * 255).astype(np.uint8)  # 归一化为 0-255 的整数
    #label_gt = (seg_gt * 255).astype(np.uint8)  # 直接使用 label 数组
    label_gt = ((seg_gt * 255).cpu().numpy()).astype(np.uint8)  # 直接使用 label 数组

    # 使用 PIL 保存为 PNG
    lab_gt = Image.fromarray(label_gt)
    # ========================================
    # 保存为 JPG 图片
    #print(seg_predicted.shape)
    #outprecs = torch.argmax(torch.softmax(seg_predicted, dim=1), dim=1, keepdim=True)

    seg_predicted_class = torch.argmax(seg_predicted, dim=0)  # 选择在类别维度（dim=0）上的最大值
    pred_image = seg_predicted_class.cpu().numpy().astype(np.uint8)
    #print(pred_image.shape)

    # 将结果转换为 numpy 数组
    pred_image = np.array(pred_image)

    # 处理颜色：0 为黑色，1 为白色
    pred_image[pred_image == 1] = 255  # 类别1（前景）变为白色
    pred_image[pred_image == 0] = 0  # 类别0（背景）保持为黑色
    to_pil = ToPILImage()
    pred_pil = to_pil(pred_image)
    # =========================================
    # 保存为PNG
    img.save(os.path.join(output_dir, f"{it_total}_img.png"))
    pred_pil.save(os.path.join(output_dir, f"{it_total}_pred.png"))
    lab_gt.save(os.path.join(output_dir, f"{it_total}_gt.png"))


class Evaluator:
    """模型评估"""
    def __init__(self, model, network_config, device, dataset_selection=None, test_data_loader=None, validation_data_loader=None, writer=None):
        self.model = model
        self.network_config = network_config
        self.device = device
        self.dataset_selection = dataset_selection
        self.test_data_loader = test_data_loader
        self.validation_data_loader = validation_data_loader
        self.writer = writer


    def evaluate(self, data_loader, epoch=1, is_test=True, ensemble=1): # epoch=None
        """Evaluates the model on the given dataset"""
        model = self.model
        network_config = self.network_config
        model.eval()

        # 创建保存图像的文件夹，如果没有的话
        #output_dir = '/exp/home/mingkai.liu/prosegdiffus/recursivediffusion/recursive_noise_diffusion/output/preciates'
        #os.makedirs(output_dir, exist_ok=True)

        if self.dataset_selection == 'cityscapes':
            ignore_index = 19
            n_ignore = 1
        else:
            ignore_index = None
            n_ignore = 0


        #=========================任务：task:"binary"：适用于二分类任务;"multiclass"：适用于多分类任务===============================
        jaccard_index = JaccardIndex(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index)
        jaccard_per_class = JaccardIndex(task="multiclass", num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index, average='none')
        #f1_score = F1Score(num_classes=data_loader.dataset.n_classes + n_ignore, mdmc_average='samplewise',task='multiclass')#================
        #accuracy = Accuracy(task='binary',num_classes=data_loader.dataset.n_classes + n_ignore, average='macro')  # 计算准确性
        accuracy = Accuracy(task='binary', num_classes=data_loader.dataset.n_classes + n_ignore,average='macro')  # 计算准确性
        #accuracy = Accuracy(task='multiclass', num_classes=data_loader.dataset.n_classes + n_ignore,average='macro')  # 计算准确性

        dice_score = Dice(task='binary', num_classes=data_loader.dataset.n_classes + n_ignore, ignore_index=ignore_index,average='macro') #=================

        j=1
        run_times=0
        with torch.no_grad():
            pbar_eval = tqdm(enumerate(data_loader), total=len(data_loader), desc='{}'.format('Test' if is_test else 'Validation'), leave=is_test, bar_format='{l_bar}{bar:50}{r_bar}')
            for it, samples in pbar_eval:
                # Unpack the samples
                images, seg_gt = samples

                #start_time = time.time()
                seg_denoised = denoise(model, self.device, network_config, images)
                #end_time = time.time()
                # 计算并输出运行时间
                #execution_time = end_time - start_time
                #run_times=run_times+execution_time
                #print(f"模型运行时间: {execution_time:.4f}秒")
                #print(f"模型总推理时间: {run_times:.4f}秒")

                #save_images_to_folder(images[0], seg_denoised[0], seg_gt[0], j)
                #j=j+1

                # Ensamble
                for i in range(ensemble-1):
                    seg_denoised += denoise(model, self.device, network_config, images)
                seg_denoised /= ensemble

                # Compute loss
                seg_predicted = seg_denoised.view(seg_denoised.shape[0], seg_denoised.shape[1], -1).argmax(dim=1)
                seg_target = seg_gt.view(seg_gt.shape[0], -1)
                jaccard_index.update(seg_predicted, seg_target)
                jaccard_per_class.update(seg_predicted, seg_target)
                accuracy.update(seg_predicted, seg_target)  # 计算准确性
                #f1_score.update(seg_predicted, seg_target)

                dice_score.update(seg_predicted, seg_target)#===========

                # 将图像和指标写入日志
                if self.writer is not None:
                    if it < 8: 
                        write_images_to_tensorboard(self.writer, epoch, image=images[0], seg_predicted=seg_denoised[0], seg_gt=seg_gt[0], datasplit='validation/{}'.format(it))
                
            
        # 总体指标
        jaccard_index_total = jaccard_index.compute()
        jaccard_per_class_total = jaccard_per_class.compute()
        #f1_score_total = f1_score.compute()
        accuracy_total = accuracy.compute()  # 计算准确性

        dice_score_total = dice_score.compute() #===========

        # 文本报告
        #report = 'Jaccard index: {:.4f} | F1 score: {:.4f} | Dice score: {:.4f}'.format(jaccard_index_total, f1_score_total , dice_score_total)#=============
        report = 'Jaccard index: {:.4f} | Acc score: {:.4f} | Dice score: {:.4f}'.format(jaccard_index_total,accuracy_total,dice_score_total)  # =============
        #report = 'Jaccard index: {:.4f} | F1 score: {:.4f}'.format(jaccard_index_total,f1_score_total)
        report_per_class = 'Jaccard index per class: {}'.format(jaccard_per_class_total)
        if self.writer is None:
            logging.log(logging.WARNING, report)
            logging.log(logging.WARNING, report_per_class)
        else:
            logging.info('{} | {} | {}'.format("Test" if is_test else "Validation | Epoch: {}".format(epoch), report, report_per_class))

        # 写入 tensorboard
        if self.writer is not None:
            self.writer.add_scalar('{}/JaccardIndex'.format('test' if is_test else 'validation'), jaccard_index_total, epoch)
            #self.writer.add_scalar('{}/F1Score'.format('test' if is_test else 'validation'), f1_score_total, epoch)
            self.writer.add_scalar('{}/Accuracy'.format('test' if is_test else 'validation'), accuracy_total, epoch)
            self.writer.add_scalar('{}/DiceScore'.format('test' if is_test else 'validation'), dice_score_total, epoch)#===============

        
    def validate(self, epoch):
        """Evaluates the model on the validation dataset"""
        self.evaluate(self.validation_data_loader, epoch, is_test=False)


    def test(self, ensemble=1):
        """Evaluates the model on the test dataset"""
        self.evaluate(self.test_data_loader, is_test=True, ensemble=ensemble)


