import json
import logging
import os
import shutil
from pathlib import Path
import time
import numpy as np

from tqdm import tqdm
import torchaudio

from torch import randint
from torch.utils.data import Dataset
from torch import from_numpy
from src.utils import ROOT_PATH
from glob import glob
from src.model.hifigan.utils import MelSpectrogramConfig, MelSpectrogram
# from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)

class HiFiLJSpeech(Dataset):
    def __init__(self, data_path: str):
        data_path = Path(data_path)
        wav_path = data_path / "wavs"
        self.wav_files = list(wav_path.glob("**/*.wav"))
        self.mel_creator = MelSpectrogram(MelSpectrogramConfig())

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav, _ = torchaudio.load(self.wav_files[idx])
        print(wav.shape)
        wav_len = 22272
        audio_start = randint(low=0, high=wav.shape[1] - wav_len, size=(1,))
        wav = wav[:, audio_start: audio_start + wav_len]
        return {
            "real_wavs": wav,
            "real_mels": self.mel_creator(wav.detach())
        }
