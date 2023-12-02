from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple
from src.model.hifigan.generator import Generator
from src.model.hifigan.discriminator import MPD, MSD
from src.model.hifigan.utils import MelSpectrogramConfig, MelSpectrogram


def get_conv_shape(I, K, P, S, D=1):
    # from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    numer = torch.tensor(I + 2 * P - D * (K - 1) - 1, dtype=torch.float64)
    return torch.floor(numer / S + 1)


class HiFiGAN(nn.Module):
    def __init__(self,
                 mpd_periods: List[int],
                 msd_num_scales: int,
                 **kwargs):
        super().__init__()
        self.gen = Generator(**kwargs)
        self.mel_creator = MelSpectrogram(MelSpectrogramConfig())

        self.mpd = MPD(mpd_periods)
        self.msd = MSD(msd_num_scales)
    
    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def generator(self, real_mels, real_wavs, **batch):
        # real_mels = self.mel_creator(real_wavs)
        fake_wavs = self.gen(real_mels)
        fake_mels = self.mel_creator(fake_wavs)
        return {
            "fake_wavs": fake_wavs,
            "fake_mels": fake_mels
            }

    def discriminator(self, real_wavs, fake_wavs, **batch) -> dict:
        out = dict()
        out.update(self.mpd(real_wavs, tag="real"))
        out.update(self.mpd(fake_wavs, tag="fake"))
        out.update(self.msd(real_wavs, tag="real"))
        out.update(self.msd(fake_wavs, tag="fake"))
        return out