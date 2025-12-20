import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6', if_devide_out=True, use_norm=True)

    def forward(self, input, h, w):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        # 动态设置 h 和 w
        self.block.input_h = h
        self.block.input_w = w
        output = self.block(input)
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v7', if_devide_out=True, use_norm=True)

    def forward(self, input0, input1, h, w):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        # 动态设置 h 和 w
        self.block.input_h = h
        self.block.input_w = w
        output = self.block(input0, extra_emb=input1)
        return output + skip


class FusionMamba(nn.Module):
    def __init__(self, dim, depth=1, final=False):
        super().__init__()
        self.final = final  # 是否输出最终融合结果
        self.spa_mamba_layers = nn.ModuleList([])  # 空间特征处理层
        self.spe_mamba_layers = nn.ModuleList([])  # 光谱特征处理层

        # 创建多个单层 MambaBlock
        for _ in range(depth):
            self.spa_mamba_layers.append(SingleMambaBlock(dim))
            self.spe_mamba_layers.append(SingleMambaBlock(dim))

        self.spa_cross_mamba = CrossMambaBlock(dim)
        self.spe_cross_mamba = CrossMambaBlock(dim)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.shape

        feat1 = rearrange(feat1, 'b c h w -> b (h w) c')
        feat2 = rearrange(feat2, 'b c h w -> b (h w) c')

        for spa_layer, spe_layer in zip(self.spa_mamba_layers, self.spe_mamba_layers):
            feat1 = spa_layer(feat1, h, w)  # 空间特征处理
            feat2 = spe_layer(feat2, h, w)  # 光谱特征处理

        spa_fusion = self.spa_cross_mamba(feat1, feat2, h, w)  # 融合空间特征
        spe_fusion = self.spe_cross_mamba(feat2, feat1, h, w)  # 融合光谱特征

        fusion = self.out_proj((spa_fusion + spe_fusion) / 2)

        feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=h, w=w)
        feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=h, w=w)
        output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)

        if self.final:
            return output  # 返回最终融合结果
        else:
            return (feat1 + output) / 2, (feat2 + output) / 2  # 返回中间特征与融合结果


