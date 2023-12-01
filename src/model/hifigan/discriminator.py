from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple
# from src.model.fastspeech2.encoder import Encoder

class SubMPD(nn.Module):
    def __init__(self,
                 period: int,
                 **kwargs):
        super().__init__()
        self.period = period
        self.norm = nn.utils.weight_norm
        self.l_max = 4

        self.conv_blocks = self.get_conv_blocks()
        self.conv1 = self.norm(nn.Conv2d(
            in_channels=2**(5 + self.l_max - 1),
            out_channels=1024,
            padding="same",
            kernel_size=(5, 1)
        ))
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = self.norm(nn.Conv2d(
            in_channels=1024,
            out_channels=1,
            padding="same",
            kernel_size=(3, 1)
        ))

    def get_conv_blocks(self):
        blocks = []
        in_channels = 1
        for l in range(self.l_max):
            blocks += [
                self.norm(nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=2**(5 + l),
                    padding=(2, 0),
                    kernel_size=(5, 1),
                    stride=(3, 1)
                )),
                nn.LeakyReLU(0.1)
            ]
            in_channels = 2**(5 + l)
        return nn.ModuleList(blocks)

    def pad_to_multiple_and_reshape(self, x):
        pad_value = self.period - (x.shape[1] % self.period)
        return F.pad(x, (0, pad_value), mode="reflect").reshape(-1, self.period)
    
    def forward(self, wav, **batch):
        x = self.pad_to_multiple_and_reshape(wav)
        layer_activations = []
        # B x T//p x p
        for l in range(self.l_max):
            x = self.conv_blocks[l](x)
            layer_activations.append(x)
        x = self.relu(self.conv1(x))
        layer_activations.append(x)
        x = self.conv2(x)
        layer_activations.append(x)
        return x, layer_activations


class MPD(nn.Module):
    def __init__(self,
                 periods: List[int],
                 **kwargs):
        super().__init__()

        self.sub_blocks = self.get_sub_blocks(periods)
    
    def get_sub_blocks(self, periods):
        sub_blocks = [SubMPD(
            period=periods[i]
        ) for i in range(len(periods))]
        return nn.ModuleList(sub_blocks)

    def forward(self, x, tag: str, **batch) -> dict:
        logits, activations = [], []
        for sub_mpd in self.sub_blocks:
            l, a = sub_mpd(x)
            logits.append(l)
            activations.append(a)

        return {
                f"{tag}_mpd_activations": activations,
                f"{tag}_mpd_logits": logits
            }


class SubMSD(nn.Module):
    def __init__(self,
                 spectral_norm: bool,
                 **kwargs):
        super().__init__()
        
        self.norm = nn.utils.spectral_norm if spectral_norm else nn.utils.weight_norm

        blocks_channels = [128, 128, 256, 512, 1024, 1024]
        blocks_groups = [4, 16, 16, 16, 16]
        self.relu = nn.LeakyReLU(0.1)

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                self.norm(nn.Conv1d(1, 128, kernel_size=15, stride=1, padding=7, padding_mode="reflect")),
                self.relu,
            ),
            *[
                nn.Sequential(
                    self.norm(nn.Conv1d(blocks_channels[i], blocks_channels[i + 1], kernel_size=41, stride=4, padding=20, groups=blocks_groups[i])),
                    self.relu
                )
                for i in range(len(blocks_channels) - 1)
            ],
            nn.Sequential(
                self.norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                self.relu,
            ),
            self.norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)),
        ])
    
    def forward(self, x, **batch):
        # x is avg pooled
        layer_activations = []
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x)
            layer_activations.append(x)
        return torch.flatten(x, 1, -1), layer_activations


class MSD(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 **kwargs):
        super().__init__()

        self.sub_blocks = self.get_sub_blocks(num_blocks)
    
    def get_sub_blocks(self, num_blocks):
        sub_blocks = [SubMSD(
            spectral_norm=(i == 0)
        ) for i in range(num_blocks)]
        return nn.ModuleList(sub_blocks)

    def forward(self, x, tag: str, **batch) -> dict:
        logits, activations = [], []
        for i, sub_msd in enumerate(self.sub_blocks):
            if i != 0:
                x = F.avg_pool1d(x,
                                 kernel_size=4,
                                 stride=2,
                                 padding=2)
            l, a = sub_msd(x)
            logits.append(l)
            activations.append(a)

        return {
                f"{tag}_msd_activations": activations,
                f"{tag}_msd_logits": logits
            }

        
            




