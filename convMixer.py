import torch
import torch.nn as nn
import numpy as np

class Residual(nn.Module)
    def __init__(self, fn)
        super().__init__()
        self.fn = fn

    def forward(self, x)
        return self.fn(x) + x
class Squeeze(nn.Module)
    def __init__(self)
        super().__init__()

    def forward(self, x)
        return torch.squeeze(x)
    

def ConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1)
    return nn.Sequential(
        nn.Conv3d(1, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm3d(dim),
        [nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv3d(dim, dim, kernel_size, groups=dim, padding=4),
                    nn.GELU(),
                    nn.BatchNorm3d(dim)
                )),
                nn.Conv3d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm3d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool3d((1,1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes),
        Squeeze(),
    )