import torch
import torch.nn as nn
import einops
from .main_blocks import *
from einops import rearrange, reduce
#from timm.layers.activations import *
#from timm.layers import DropPath, trunc_normal_
import math
from torch.utils.cpp_extension import load
import torch
import torch.nn as nn
from torch.nn import functional as F
#from timm.layers import DropPath, create_act_layer,  LayerType
import numpy as np
import torchvision
from typing import Callable, Dict, Optional, Type
import pickle
import os




def q_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
    #====================
    if patch_resolution is None:
        side = int(math.sqrt(N))
        assert side * side == N, "Cannot infer patch resolution: N is not a square number"
        patch_resolution = (side, side)
    #===================
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)

def telu(input):
    return input * torch.tanh(torch.exp(input))

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1/4, shift_pixel=1, hidden_rate=2, 
                 key_norm=True):
        super().__init__()
        self.n_embd = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self):
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            xx = q_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        #k = torch.square(telu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x
    

if __name__ == "__main__":
    # 参数设置
    '''
    B = 2  # batch size
    H, W = 8, 8  # patch grid size
    N = H * W
    C = 64  # embedding dimension

    # 模拟输入
    x = torch.randn(B, N, C)
    '''
    x = torch.randn(1, 256, 64, 64)
    print("x:", x.shape)  # 输出张量的形状
    # Prepare for MHSA: (B, C, H, W) -> (B, N, C) where N=H*W
    B, C, H, W = x.shape
    x = x.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
    print("Input shape:", x.shape)

    # 初始化模型
    model = VRWKV_ChannelMix(
        n_embd=C,
        channel_gamma=1/4,
        shift_pixel=1,
        hidden_rate=2,
        key_norm=True
    )

    # 前向传播
    out = model(x, patch_resolution=(H, W))
    print("Output shape:", out.shape)  # 输出张量的形状
    
    # Step 3: Reshape back to (B, C, H, W)
    B, n_patch, hidden = out.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidde
    h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    attn_output = out.permute(0, 2, 1)
    attn_output = out.contiguous().view(B, hidden, h, w)
    
    print("Reshaped output shape:", attn_output.shape)  # 输出张量的形状
    print("======================")

    # 输出结果
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Sample output (first batch, first patch):\n{out[0, 0]}")