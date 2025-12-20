import logging
import math
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
#from tensorboard import SummaryWriter #新加的

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

#from utils.evaluation import Evaluator, segmentation_cross_entropy, noise_mse, write_images_to_tensorboard
#from utils.utils import diffuse, get_patch_indices, dynamic_range
from utils.evaluation import Evaluator, segmentation_cross_entropy, noise_mse, write_images_to_tensorboard,adaptive_combined_loss
from utils.utils import diffuse, get_patch_indices, dynamic_range

class TrainerConfig:
    """
    Config settings (hyperparameters) for training.
    """
    # optimization parameters
    max_epochs = 100
    #batch_size = 2 #用于vilgen\hrf
    batch_size = 4 #用于ISIC\refuge
    learning_rate = 1e-5
    momentum = None #设置动量
    weight_decay = 0.001 
    grad_norm_clip = 0.95

    # learning rate decay params
    lr_decay = True #表示是否启用学习率衰减
    lr_decay_gamma = 0.98 #学习率衰减率

    # network
    network = 'unet'

    # diffusion other settings
    train_on_n_scales = None #设置模型训练的尺度数量
    #not_recursive = False #是否使用递归去噪；若为true,则不使用递归去噪
    not_recursive = True

    # checkpoint settings
    #checkpoint_dir = 'output/checkpoints/'
    checkpoint_dir = 'output/checkpoints/'
    log_dir = 'output/logs/'
    load_checkpoint =None       #测试需要设置none 和检查点文件
    #load_checkpoint = '/exp/home/mingkai.liu/prodiffusmamba/remamba/output/checkpoints/20251021-1212_ph_d0.29_t35/20251021-1212_unet_e46.pt'
    #load_checkpoint = '/exp/home/mingkai.liu/prodiffusmamba/recursivemamba/output/checkpoints/20250614-2137_unet/20250614-2137_unet_e66.pt'

    checkpoint = None
    weights_only = False

    #==========================数据集=====================
    dataset_selection = 'isic'


    # other
    eval_every = 2 #表示评估频率
    save_every = 2 #保存检查点的频率
    seed = 0
    n_workers = 8 #数据加载的并行线程数量

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

    def save_config_file(self, filename):
        Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
        logging.info("Saving TrainerConfig file: {}".format(filename))
        with open(filename, 'w') as f:
            for k,v in vars(self).items():
                f.write("{}={}\n".format(k,v))

#负责整个训练过程的管理
class Trainer:

    def __init__(self, model, network_config, config, train_data_loader, validation_data_loader=None):
        self.model = model #训练的模型
        self.network_config = network_config #网络配置
        self.config = config #训练的配置
        self.train_data_loader = train_data_loader #训练数据加载器
        self.validation_data_loader = validation_data_loader
        self.device = config.device #训练设备

    #创建运行名称：使用当前时间和网络名称生成运行名称，方便识别不同的训练任务
    def create_run_name(self):
        """Creates a unique run name based on current time and network"""
        self.run_name = '{}_{}'.format(time.strftime("%Y%m%d-%H%M"), self.config.network)

    #保存检查点:将模型、优化器和调度器的状态保存到文件中，以便训练恢复
    def save_checkpoint(self, model, optimizer, scheduler, epoch, id=None):
        """Saves a model checkpoint"""
        if id is None:
            id = "e{}".format(epoch)
        path = os.path.normpath(self.config.checkpoint_dir + "{}/{}_{}.pt".format(self.run_name, self.run_name, id)) # path/time_network/time_network_epoch.pt
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        logging.info("Saving checkpoint: {}".format(path))
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}, path)

    #获取优化器:定义优化器为 AdamW。如果存在检查点并且 weights_only 为 False，则加载优化器状态
    def get_optimizer(self):
        """Defines the optimizer"""
        # optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, betas=(0.9, 0.999), weight_decay=self.config.weight_decay)
        if (self.config.checkpoint is not None) and (self.config.weights_only is False):
            optimizer.load_state_dict(self.config.checkpoint['optimizer_state_dict'])
        return optimizer

    #获取学习率调度器:定义指数衰减学习率调度器，同样检查是否从保存状态中恢复
    def get_scheduler(self, optimizer):
        """Defines the learning rate scheduler"""
        scheduler = ExponentialLR(optimizer, gamma=self.config.lr_decay_gamma) #动态调整优化器的学习率；ExponentialLR 是 PyTorch 提供的一种学习率调度器，它通过对初始学习率乘以一个固定的因子（gamma）来实现学习率的指数衰减。
        if (self.config.checkpoint is not None) and (self.config.weights_only is False):
            scheduler.load_state_dict(self.config.checkpoint['scheduler_state_dict'])
        return scheduler

    #多尺度去噪循环:逐尺度和时间步处理图像，通过迭代地调整尺度来去噪
    def denoise_loop_scales(self, model, network_config, config, images, seg_gt_one_hot, optimizer, scaler):
        """Denoises all scales for a single timestep：表示单个时间步的所有尺度去噪"""
        # Calculate scale sizes (smallest first)：计算各尺度的尺寸；images.shape[2]：图像的高；images.shape[3]：图像的宽
        scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

        # 随机初始化一个与 images 尺寸匹配的预测图 seg_previous_scaled，作为初始去噪结果
        seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

        # 逐步对整个分割图进行去噪
        for timestep in range(network_config.n_timesteps): # for each step
            loss_per_scale = torch.zeros(network_config.n_scales) #用于存储每个尺度上的损失

            for scale in range(network_config.n_scales): # 遍历每个尺度 scale，只在 train_on_n_scales 指定的范围内进行
                # break if we don't want to train on all scales:如果我们不想进行各种规模的训练，就休息一下
                if scale > config.train_on_n_scales - 1:
                    break
                # Resize to current scale:调整图像和标签到当前尺度:使用双线性插值将 images、seg_gt_one_hot 和 seg_previous_scaled 调整到当前尺度大小
                images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False) #训练图像
                seg_gt_scaled = F.interpolate(seg_gt_one_hot, size=scale_sizes[scale], mode='bilinear', align_corners=False) #真实标签
                seg_previous_scaled = F.interpolate(seg_previous_scaled, size=scale_sizes[scale], mode='bilinear', align_corners=False) #初始化的预测分割图

                #获取图像块的索引:patch_indices 包含图像的块坐标和块大小，用于在去噪过程中逐块处理图像；图像切块，分块可以降低显存需求，提高计算效率
                #scale_sizes[scale]:之前计算的各尺度图像大小；network_config.max_patch_size：分块的最大尺寸；overlap=False：指定是否允许分块之间有重叠
                patch_indices = get_patch_indices(scale_sizes[scale], network_config.max_patch_size, overlap=False)

                # Create a new tensor to store the denoised segmentation map:创建一个新的张量来存储去噪的分割图
                seg_denoised = torch.zeros(seg_previous_scaled.shape)#初始化一个全零张量 seg_denoised，为后续的分块去噪提供存储容器，最终累积每个图像块的去噪结果以生成完整的去噪分割图像。
                # Create a tensor to store the number of times a pixel has been denoised:创建一个张量来存储像素去噪的次数
                n_denoised = torch.zeros(seg_previous_scaled.shape)#记录每个像素被处理的次数

                for x, y, patch_size in patch_indices: # 遍历每个图像块
                    # Get the patch:将图像、标签和先前的预测图的当前块提取出来，并移至 CUDA 设备
                    '''
                    img_patch = images_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().cuda(non_blocking=True)
                    seg_gt_patch = seg_gt_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().cuda(non_blocking=True)
                    seg_patch_previous = seg_previous_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().cuda(non_blocking=True).softmax(dim=1)
                    '''
                    img_patch = images_scaled[:, :, x:x + patch_size, y:y + patch_size].detach().to(self.device, non_blocking=True)
                    seg_gt_patch = seg_gt_scaled[:, :, x:x + patch_size, y:y + patch_size].detach().to(self.device, non_blocking=True)
                    seg_patch_previous = seg_previous_scaled[:, :, x:x + patch_size, y:y + patch_size].detach().to(self.device, non_blocking=True).softmax(dim=1)

                    #若 not_recursive 为真，则更新 seg_patch_previous 为 seg_gt_patch
                    if config.not_recursive:
                        if timestep + scale > 0:
                            seg_patch_previous = seg_gt_patch

                    # Diffuse
                    #t = torch.tensor([(network_config.n_timesteps - (timestep + scale/network_config.n_scales)) / network_config.n_timesteps]).cuda(non_blocking=True) # time step
                    t = torch.tensor([(network_config.n_timesteps - (timestep + scale / network_config.n_scales)) / network_config.n_timesteps]).to(self.device,non_blocking=True)
                    seg_patch_diffused = diffuse(seg_patch_previous, t).detach() # diffuse segmentation map：对先前的分割预测图 (seg_patch_previous) 添加时间步 t 对应的噪声，生成扩散后的分割图；并将其从计算图中分离（detach），以避免梯度传播。
                    #目标噪声在扩散模型中用于指导模型学习噪声的去除；扩散后的分割图 seg_patch_diffused 与真实分割标签 seg_gt_patch 之间的差值
                    noise_gt = seg_patch_diffused - seg_gt_patch # The noise added in the diffusion process + the error from the previous step:计算 noise_gt 作为预测的目标噪声

                    # 清零梯度
                    optimizer.zero_grad()

                    # 使用 autocast 进行前向传播，预测噪声 noise_predicted
                    #autocast()是pytorch方法（amp自动混合精度）：自动将模型中的部分计算切换到半精度（float16），以减少内存占用并加快运算。同时保留某些计算（如损失计算或归一化层）在全精度（float32）下进行，以保持训练的稳定性
                    with torch.cuda.amp.autocast():
                        # Forward pass:使用模型对扩散后的图像进行前向传播，预测噪声
                        noise_predicted = model(seg_patch_diffused, img_patch, t) # predict the noise
                        seg_patch_denoised = seg_patch_diffused - noise_predicted # denoise the patch(去噪)

                        # 计算损失
                        losses = {}
                        noise_mse_loss = noise_mse(noise_predicted, noise_gt)
                        losses['noise_mse'] = noise_mse_loss #用于将噪声预测的均方误差损失保存到 losses 字典

                        # noise_combine_loss = adaptive_combined_loss(noise_predicted, noise_gt)
                        #losses['noise_combine'] = noise_combine_loss

                        # seg_cross_entropy_loss = segmentation_cross_entropy(seg_patch_denoised, seg_gt_patch.argmax(dim=1))
                        # losses['seg_cross_entropy'] = seg_cross_entropy_loss

                        total_loss = noise_mse_loss
                        #total_loss = noise_combine_loss

                    # Backward pass
                    # total_loss.backward():利用自动混合精度（AMP）技术对总损失 (total_loss) 进行反向传播，并进行缩放操作，以稳定梯度计算和优化过程
                    scaler.scale(total_loss).backward()
                    
                    # Clip the gradients
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

                    # Update the parameters
                    # optimizer.step() 
                    scaler.step(optimizer)#执行优化器更新，并自动反缩放梯度

                    # Update the scale for the next iteration.
                    scaler.update()#动态调整放大因子

                    # 将去噪块添加到分割图中
                    #更新去噪图像块:将去噪后的图像块添加到 seg_denoised，并记录在 n_denoised 中
                    seg_patch_denoised = seg_patch_denoised.detach().cpu() # detach from the graph
                    seg_denoised[:, :, x:x+patch_size, y:y+patch_size] += seg_patch_denoised
                    n_denoised[:, :, x:x+patch_size, y:y+patch_size] += 1 

                
                # 平均去噪块
                seg_denoised = seg_denoised / n_denoised

                # # Adjust range
                # seg_denoised = dynamic_range(seg_denoised)

                # 更新上一尺度结果
                seg_previous_scaled = seg_denoised
        
        return seg_denoised, losses #返回最后一个时间步的分割去噪图（即分割mask）和损失


    #线性去噪:类似于 denoise_loop_scales，但它在每个时间步只处理一个尺度，并按时间步顺序调整尺度。
    def denoise_linear_scales(self, model, network_config, config, images, seg_gt_one_hot, optimizer, scaler):
        """Denoises one scale at a each timestep"""
        # 根据图像原始大小 images.shape[2:] 和总尺度数 network_config.n_scales，计算每个尺度的分辨率。
        scale_sizes = [(images.shape[2] // (2**(network_config.n_scales - i -1)), images.shape[3] // (2**(network_config.n_scales - i -1))) for i in range(network_config.n_scales)]

        # Initialize first prediction (random noise)
        seg_previous_scaled = torch.rand(images.shape[0], network_config.n_classes, images.shape[2], images.shape[3])

        # 逐步对整个分割图进行去噪
        for timestep in range(network_config.n_timesteps): # for each step
            # 时间步被均匀分配到每个尺度。如果总时间步数为 100，尺度数为 4，则每个尺度分配 25 个时间步。
            #math.ceil():向上取整函数
            timesteps_per_scale = math.ceil(network_config.n_timesteps / network_config.n_scales)
            scale = timestep // timesteps_per_scale
            
            # Resize to current scale：在时间步切换到新尺度时，对输入图像、分割标签和先前预测图进行缩放。
            if timestep % timesteps_per_scale == 0:
                images_scaled = F.interpolate(images, size=scale_sizes[scale], mode='bilinear', align_corners=False)
                seg_gt_scaled = F.interpolate(seg_gt_one_hot, size=scale_sizes[scale], mode='nearest')
                seg_previous_scaled = F.interpolate(seg_previous_scaled.float(), size=scale_sizes[scale], mode='bilinear', align_corners=False)

                patch_indices = get_patch_indices(scale_sizes[scale], network_config.max_patch_size, overlap=False)#划分当前尺度的图像块（无重叠）

            # Create a new tensor to store the denoised segmentation map
            seg_denoised = torch.zeros(seg_previous_scaled.shape)
            # Create a tensor to store the number of times a pixel has been denoised
            n_denoised = torch.zeros(seg_previous_scaled.shape)

            for x, y, patch_size in patch_indices: # for each patch
                # Get the patch

                img_patch = images_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(self.device).contiguous()
                seg_gt_patch = seg_gt_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(self.device).contiguous()
                seg_patch_previous = seg_previous_scaled[:, :, x:x+patch_size, y:y+patch_size].detach().to(self.device).contiguous()
                '''
                img_patch = images_scaled[:, :, x:x + patch_size, y:y + patch_size].detach().cuda(non_blocking=True).contiguous()
                seg_gt_patch = seg_gt_scaled[:, :, x:x + patch_size, y:y + patch_size].detach().cuda(non_blocking=True).contiguous()
                seg_patch_previous = seg_previous_scaled[:, :, x:x + patch_size, y:y + patch_size].detach().cuda(non_blocking=True).contiguous()
                '''
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Diffused
                t = torch.tensor([(network_config.n_timesteps - timestep) / network_config.n_timesteps]).to(self.device) # time step
                #t = torch.tensor([(network_config.n_timesteps - timestep) / network_config.n_timesteps]).cuda(non_blocking=True)  # time step
                seg_patch_diffused = diffuse(seg_patch_previous, t).detach() # diffuse segmentation map
                # ================mask+noise;====================================
                seg_gt_diffused = diffuse(seg_gt_patch, t).detach()
                # ===============================================================
                noise_gt = seg_patch_diffused - seg_gt_patch # The noise added in the diffusion process + the error from the previous step

                #==================================
                with torch.cuda.amp.autocast():
                    # Forward pass:使用模型对扩散后的图像进行前向传播，预测噪声
                    noise_predicted = model(seg_patch_diffused, img_patch, t)  # predict the noise#===================
                    seg_patch_denoised = seg_patch_diffused - noise_predicted  # denoise the patch(去噪)

                    # 计算损失
                    losses = {}
                    noise_mse_loss = noise_mse(noise_predicted, noise_gt)
                    losses['noise_mse'] = noise_mse_loss  # 用于将噪声预测的均方误差损失保存到 losses 字典
                    #seg_cross_entropy_loss = segmentation_cross_entropy(seg_patch_denoised, seg_gt_patch.argmax(dim=1))#交叉熵损失
                    #losses['seg_cross_entropy'] = seg_cross_entropy_loss
                    #total_loss =seg_cross_entropy_loss

                    total_loss = noise_mse_loss

                # Backward pass
                # total_loss.backward():利用自动混合精度（AMP）技术对总损失 (total_loss) 进行反向传播，并进行缩放操作，以稳定梯度计算和优化过程
                scaler.scale(total_loss).backward()
                # Update the parameters
                # optimizer.step()
                scaler.step(optimizer)  # 执行优化器更新，并自动反缩放梯度

                # Update the scale for the next iteration.
                scaler.update()  # 动态调整放大因子
                #====================================
                '''
                # Forward pass
                noise_predicted = model(seg_patch_diffused, img_patch, t) # predict the noise
                seg_patch_denoised = seg_patch_diffused - noise_predicted # denoise the patch

                # Compute loss
                losses = {}
                noise_mse_loss = noise_mse(noise_predicted, noise_gt)
                losses['noise_mse'] = noise_mse_loss
                #seg_cross_entropy_loss = segmentation_cross_entropy(seg_patch_denoised, seg_gt_patch.argmax(dim=1))#交叉熵损失
                #losses['seg_cross_entropy'] = seg_cross_entropy_loss
                #total_loss = noise_mse_loss + seg_cross_entropy_loss
                total_loss = noise_mse_loss
                

                # Backward pass
                total_loss.backward()
                
                # Clip the gradients：对模型参数的梯度进行裁剪（梯度归一化），以防止梯度爆炸。
                # model.parameters()：需要裁剪梯度的模型参数，一般通过 model.parameters() ；config.grad_norm_clip：获取最大梯度范数，即梯度裁剪的阈值
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)

                # Update the parameters
                optimizer.step() 
                '''

                # Add the denoised patch to the segmentation map
                seg_patch_denoised = seg_patch_denoised.detach().cpu() # detach from the graph
                seg_denoised[:, :, x:x+patch_size, y:y+patch_size] += seg_patch_denoised
                n_denoised[:, :, x:x+patch_size, y:y+patch_size] += 1 

            # Average the denoised patches
            seg_denoised = seg_denoised / n_denoised

            # Update the previous segmentation map
            seg_previous_scaled = seg_denoised

        return seg_denoised, losses ##返回最后一个时间步的分割去噪图（即分割mask）和损失

    #根据配置选择使用哪种去噪方法，并返回去噪结果和损失。
    def denoise_and_backprop(self, model, network_config, config, images, seg_gt_one_hot, optimizer, scaler):
        """Denoises and backpropagates the error"""
        if network_config.scale_procedure == 'loop':
            seg_denoised, losses = self.denoise_loop_scales(model, network_config, config, images, seg_gt_one_hot, optimizer, scaler)
        elif network_config.scale_procedure == 'linear':
            seg_denoised, losses = self.denoise_linear_scales(model, network_config, config, images, seg_gt_one_hot, optimizer, scaler)

        return seg_denoised, losses

    #主训练方法，执行整个训练流程。
    def train(self):
        """Trains the model"""
        self.create_run_name()
        model = self.model
        network_config = self.network_config
        config = self.config
        optimizer = self.get_optimizer() #初始化 optimizer（优化器)
        scaler = GradScaler() # GradScaler 实例，用于自动混合精度训练（自动调整浮点精度）
        scheduler = self.get_scheduler(optimizer) #初始化 scheduler（学习率调度器）以控制学习率变化
        writer = SummaryWriter(log_dir=(config.log_dir + self.run_name)) #SummaryWriter 实例，用于 TensorBoard 日志记录
        #评估器:用于在训练过程中的验证，帮助评估模型性能
        evaluator = Evaluator(model, network_config, self.device, dataset_selection=config.dataset_selection, validation_data_loader=self.validation_data_loader, writer=writer)

        #保存配置文件:将当前训练配置保存到文件中，路径基于检查点目录和运行名称。
        config.save_config_file(os.path.normpath(config.checkpoint_dir + "{}/{}_config.txt".format(self.run_name, self.run_name)))

        #run_epoch 是内部函数，用于定义每个训练周期的具体操作。
        def run_epoch():
            model.train()#设置模型为训练模式

            #初始化进度条:pbar_epoch 是用于可视化批次进度的 tqdm 进度条。显示当前周期数、总周期数、以及训练进度。
            pbar_epoch = tqdm(enumerate(self.train_data_loader), total=len(self.train_data_loader), desc='Epoch {}/{}'.format(epoch+1, config.max_epochs), leave=False, bar_format='{l_bar}{bar:50}{r_bar}')

            for it, samples in pbar_epoch:#遍历数据加载器的批次
                # Unpack the samples
                # 遍历 train_data_loader 的每个批次，将批次索引和数据样本分别赋值给 it 和 samples。
                images, seg_gt = samples ## images 和 seg_gt 分别为输入图像和对应的分割标签。
                #seg_gt_one_hot将分割标签转换为 one-hot 编码,并调整维度排列，以符合模型输入格式。
                seg_gt_one_hot = F.one_hot(seg_gt, num_classes=network_config.n_classes+1).permute(0,3,1,2)[:,:-1,:,:].float() # make one hot (if remove void class [:,:-1,:,:])移除 "void" 类，并将张量类型转换为浮点数。

                # 在多尺度上去噪并反向传播误差，得到去噪后的分割图 seg_denoised 和损失 losses。denoise_and_backprop：集成去噪和反向传播的核心方法，负责模型的主要训练逻辑。
                seg_denoised, losses = self.denoise_and_backprop(model, network_config, config, images, seg_gt_one_hot, optimizer, scaler)

                # 记录损失到 TensorBoard,每 20 次批次记录一次损失，调用 writer.add_scalar 将各损失名称和值添加到 TensorBoard，以便可视化训练损失。
                it_total = it + epoch*len(self.train_data_loader)
                if it_total % 20 == 0 and it_total > 0:
                    for loss_name, loss in losses.items():
                        writer.add_scalar('train/{}'.format(loss_name), loss, it_total)
                
                #定期写入图像到 TensorBoard,每 20个批次，将当前批次的图像、预测分割结果和标签写入 TensorBoard，以便在训练过程中检查图像。
                if it % 20 == 0:
                    write_images_to_tensorboard(writer, it_total, image=images[0], seg_predicted=seg_denoised[0], seg_gt=seg_gt[0], datasplit='train', dataset_name=config.dataset_selection)
            
            scheduler.step() #更新学习率，通过调度器 scheduler 调整当前学习率。
            

        #总训练循环:使用 tqdm 显示总训练进度，通过 config.max_epochs 设置最大训练周期数
        with logging_redirect_tqdm():
            pbar_total = tqdm(range(config.max_epochs), desc='Total', bar_format='{l_bar}{bar:50}{r_bar}')
            for epoch in pbar_total:
                # 调用 run_epoch() 执行每个周期的训练
                run_epoch()

                # 每隔 save_every 个周期调用 save_checkpoint 保存模型、优化器和调度器状态。
                if (epoch+1) % config.save_every == 0:
                    self.save_checkpoint(model, optimizer, scheduler, epoch+1)
                
                # 定期验证模型：如果有验证数据加载器，每隔 eval_every 个周期调用 evaluator.validate 进行模型验证，评估模型性能。
                if self.validation_data_loader is not None:
                    if (epoch+1) % config.eval_every == 0:
                        evaluator.validate(epoch+1)

            #结束训练和清理：将所有待写入的日志刷新到磁盘，并关闭 SummaryWriter 以结束训练的日志记录
            writer.flush()
            writer.close()