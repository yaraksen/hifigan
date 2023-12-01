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
    print(dataset_items[0]["real_wavs"].shape, dataset_items[0]["real_mels"].shape)
    result_batch["real_wavs"] = torch.as_tensor([rec["real_wavs"].squeeze(0) for rec in dataset_items])
    result_batch["real_mels"] = torch.as_tensor([rec["real_mels"] for rec in dataset_items])
    return result_batch