from recursivediffusion.recursive_noise_diffusion.networks.manyscale import MultiScaleEdgeFusionModule
import torch
from torch import nn, einsum
def renorm(dim_in, dim_out):
    return nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=1, stride=1, padding=0)

def exists(x):
    return x is not None

x=torch.randn(8, 32, 64, 64)
#cov=nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0)
#fusion=MultiScaleEdgeFusionModule(in_channels=64, out_channels=128)
#x=fusion(y)
fusion=MultiScaleEdgeFusionModule(in_channels=32, out_channels=64)
re=renorm(64 ,32)
print("=========================")
print(x.shape)
x=fusion(x)
print(0000)
print(x.shape)
print(1111)
x=re(x)
print(x.shape)
print("结束")
