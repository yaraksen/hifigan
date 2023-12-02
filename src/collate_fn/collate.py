import logging
from typing import List
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch
from torch import tensor
import numpy as np
# import tqdm
# import os
# import time
# from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}
    result_batch["real_wavs"] = torch.cat([rec["real_wavs"] for rec in dataset_items], axis=0).unsqueeze(1)
    result_batch["real_mels"] = torch.cat([rec["real_mels"] for rec in dataset_items], axis=0)
    # print(result_batch["real_wavs"].shape, result_batch["real_mels"].shape)
    return result_batch