import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm.modules.mamba_simple import Mamba


class SingleMambaBlock(nn.Module):
    def __init__(self, dim, dropout_prob=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6', if_devide_out=True, use_norm=True)
        self.dropout = nn.Dropout(p=dropout_prob)  # 添加 Dropout

    def forward(self, input, h, w):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        # 动态设置 h 和 w
        self.block.input_h = h
        self.block.input_w = w
        output = self.block(input)
        output = self.dropout(output)  # 应用 Dropout
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim, dropout_prob=0.5):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v7', if_devide_out=True, use_norm=True)
        self.dropout = nn.Dropout(p=dropout_prob)  # 添加 Dropout

    def forward(self, input0, input1, h, w):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        # 动态设置 h 和 w
        self.block.input_h = h
        self.block.input_w = w
        output = self.block(input0, extra_emb=input1)
        output = self.dropout(output)  # 应用 Dropout
        return output + skip


class FusionMamba(nn.Module):
    def __init__(self, dim, depth=1, final=False, dropout_prob=0.5):
        super().__init__()
        self.final = final  # 是否输出最终融合结果
        self.spa_mamba_layers = nn.ModuleList([])  # 空间特征处理层
        self.spe_mamba_layers = nn.ModuleList([])  # 光谱特征处理层

        # 创建多个单层 MambaBlock
        for _ in range(depth):
            self.spa_mamba_layers.append(SingleMambaBlock(dim, dropout_prob=dropout_prob))
            self.spe_mamba_layers.append(SingleMambaBlock(dim, dropout_prob=dropout_prob))

        # 创建交叉融合 MambaBlock
        self.spa_cross_mamba = CrossMambaBlock(dim, dropout_prob=dropout_prob)
        self.spe_cross_mamba = CrossMambaBlock(dim, dropout_prob=dropout_prob)

        # 输出投影层
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(p=dropout_prob)  # 在输出投影层后添加 Dropout

    def forward(self, feat1, feat2):
        # 动态从输入中获取 H 和 W
        b, c, h, w = feat1.shape

        # 将输入重排为 (B, N, C)，其中 N=H*W
        feat1 = rearrange(feat1, 'b c h w -> b (h w) c', h=h, w=w)
        feat2 = rearrange(feat2, 'b c h w -> b (h w) c', h=h, w=w)

        # 单层空间和光谱特征处理
        for spa_layer, spe_layer in zip(self.spa_mamba_layers, self.spe_mamba_layers):
            feat1 = spa_layer(feat1, h, w)  # 空间特征处理
            feat2 = spe_layer(feat2, h, w)  # 光谱特征处理

        # 空间和光谱特征交叉融合
        spa_fusion = self.spa_cross_mamba(feat1, feat2, h, w)  # 融合空间特征
        spe_fusion = self.spe_cross_mamba(feat2, feat1, h, w)  # 融合光谱特征

        # 融合结果投影，取平均
        fusion = self.out_proj((spa_fusion + spe_fusion) / 2)
        fusion = self.dropout(fusion)  # 在投影后应用 Dropout

        # 将特征重排回原始形状 (B, C, H, W)
        feat1 = rearrange(feat1, 'b (h w) c -> b c h w', h=h, w=w)
        feat2 = rearrange(feat2, 'b (h w) c -> b c h w', h=h, w=w)
        output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)

        # 最终输出
        if self.final:
            return output  # 返回最终融合结果
        else:
            return (feat1 + output) / 2, (feat2 + output) / 2  # 返回中间特征与融合结果


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat1 = torch.randn(4, 256, 32, 32)
    feat2 = torch.randn(4, 256, 32, 32)
    feat1 = feat1.to(device)  # 将输入1移动到 GPU
    feat2 = feat2.to(device)  # 将输入2移动到 GPU
    model = FusionMamba(dim=256, depth=2, final=True, dropout_prob=0.5)  # 设置 Dropout 概率为 0.5
    model = model.to(device)  # 将模型移动到 GPU
    output = model(feat1, feat2)
    print(output.shape)  # 输出形状: [4, 256, 32, 32]