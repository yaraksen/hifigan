from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple

class ResBlock(nn.Module):
    def __init__(self,
                residual_kernel: int,
                residual_dilations: List[List[int]], # D_r[n]
                conv_channels: int,
                **kwargs):
        super().__init__()
        self.norm = nn.utils.weight_norm
        self.m_max = len(residual_dilations)
        self.l_max = len(residual_dilations[0])
        self.res_blocks = self.get_res_blocks(residual_kernel, residual_dilations, conv_channels)
    
    def get_res_blocks(self, residual_kernel, residual_dilations, conv_channels):
        blocks = [nn.ModuleList([None for l in range(self.l_max)]) for m in range(self.m_max)]
        for m in range(self.m_max):
            for l in range(self.l_max):
                blocks[m][l] = nn.Sequential(
                    nn.LeakyReLU(0.1),
                    self.norm(nn.Conv1d(
                        in_channels=conv_channels,
                        out_channels=conv_channels,
                        padding="same",
                        kernel_size=residual_kernel,
                        dilation=residual_dilations[m][l]
                    )))
        return nn.ModuleList(blocks)

    def forward(self, x, **batch):
        for m in range(self.m_max):
            res = x
            for l in range(self.l_max):
                x = self.res_blocks[m][l](x)
            x = res + x
        return x


class MRF(nn.Module):
    def __init__(self,
                residual_kernels: List[int],
                residual_dilations: List[List[int]],
                in_channels: int,
                **kwargs):
        super().__init__()
        self.max_n = len(residual_kernels)
        self.res_blocks = self.get_res_blocks(residual_kernels, residual_dilations, in_channels)
        
    def get_res_blocks(self, residual_kernels, residual_dilations, in_channels):
        blocks = []
        for n in range(self.max_n):
            blocks.append(ResBlock(
                residual_kernels[n],
                residual_dilations[n],
                conv_channels=in_channels
            ))
        return nn.ModuleList(blocks)

    def forward(self, x, **batch):
        out = self.res_blocks[0](x)
        for n in range(1, self.max_n):
            out = out + self.res_blocks[n](x)
        return out


class Generator(nn.Module):
    def __init__(self,
                 upscale_kernels: List[int],
                 upscale_init_height: int,
                 **kwargs):
        super().__init__()

        self.norm = nn.utils.weight_norm
        self.conv_in = self.norm(nn.Conv1d(
            in_channels=80,
            out_channels=upscale_init_height,
            kernel_size=7,
            padding="same"
        ))

        self.transpose_blocks = self.get_transpose_blocks(upscale_kernels, upscale_init_height, **kwargs)
        
        self.relu = nn.LeakyReLU(0.1)
        self.conv_out = self.norm(nn.Conv1d(
            in_channels=upscale_init_height // (2 ** len(upscale_kernels)),
            out_channels=1,
            kernel_size=7,
            padding="same"
        ))
        self.tanh = nn.Tanh()

    def get_transpose_blocks(self, upscale_kernels, upscale_init_height, **kwargs):
        blocks = []
        for l in range(len(upscale_kernels)):
            blocks.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                self.norm(nn.ConvTranspose1d(
                    in_channels=(upscale_init_height // (2 ** l)),
                    out_channels=(upscale_init_height // (2 ** (l + 1))),
                    padding=(upscale_kernels[l] - upscale_kernels[l] // 2) // 2,
                    kernel_size=upscale_kernels[l],
                    stride=(upscale_kernels[l] // 2),
                )),
                MRF(in_channels=(upscale_init_height // (2 ** (l + 1))), **kwargs)
            ))
        return nn.ModuleList(blocks)
    
    def forward(self, mel, **batch):
        x = self.conv_in(mel)
        for i in range(len(self.transpose_blocks)):
            x = self.transpose_blocks[i](x)
        x = self.tanh(self.conv_out(self.relu(x)))
        return x
