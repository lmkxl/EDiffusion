n_scales=3
train_on_n_scales=4
'''
n_timesteps=25
for timestep in range(n_timesteps):
    print(timestep)
'''
import torch
images = torch.randn(1, 128, 512, 512)
scale_sizes = [(images.shape[2] // (2**(n_scales - i -1)), images.shape[3] // (2**(n_scales - i -1))) for i in range(n_scales)]
print(scale_sizes)

for scale in range(n_scales):  # 遍历每个尺度 scale，只在 train_on_n_scales 指定的范围内进行
    # break if we don't want to train on all scales:如果我们不想进行各种规模的训练，就休息一下
    print(scale)
    if scale > train_on_n_scales - 1:
        break