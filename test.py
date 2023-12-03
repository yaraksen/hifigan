import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH
from src.utils.object_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
import src.metric as module_metric
from src.utils import MetricTracker
from glob import glob
import torchaudio
from src.model.hifigan.utils import MelSpectrogramConfig, MelSpectrogram
import torch.nn.functional as F 

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"
        

def main(config, wavs_path: str, out_dir: str):
    logger = config.get_logger("test")
    wavs_path = Path(wavs_path)
    out_dir = Path(out_dir)

    # define cpu or gpu if possible
    device_id = 0
    device = torch.device('cpu') # torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(device)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    mel_creator = MelSpectrogram(MelSpectrogramConfig())
    
    with torch.no_grad():
        for file in tqdm(wavs_path.glob("*.wav"), desc=f"Processing..."):
            audio, sr = torchaudio.load(file)
            audio = F.pad(audio, (0, 256 - audio.shape[1] % 256), value=0)
            mel = mel_creator(audio)
            wav = model.gen(mel).squeeze(0)
            assert wav.shape == audio.shape
            torchaudio.save(out_dir / file.name, wav, sample_rate=sr)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=1,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )
    args.add_argument(
        "-wp",
        "--wavs_path",
        default="wavs",
        type=str,
        help="File with checkpoint of vocoder model",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    # model_config = Path(args.resume).parent / "config.json"

    with open(args.config) as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # # update with addition configs from `args.config` if provided
    # if args.config is not None:
    #     with Path(args.config).open() as f:
    #         config.config.update(json.load(f))
    
    # # if `--test-data-folder` was provided, set it as a default test set
    # if args.test_data_folder is not None:
    #     test_data_folder = Path(args.test_data_folder).absolute().resolve()
    #     assert test_data_folder.exists()
    #     config.config["data"] = {
    #         "test": {
    #             "batch_size": args.batch_size,
    #             "num_workers": args.jobs,
    #             "datasets": [
    #                 {
    #                     "type": "SpeechSeparationDataset",
    #                     "args": {
    #                         "part": "",
    #                         "data_dir": test_data_folder
    #                     },
    #                 }
    #             ],
    #         }
    #     }

    # if config.config.get("data", {}).get("test", None) is None:
    #     assert config.config.get("data", {}).get("test-clean", None) is not None
    #     assert config.config.get("data", {}).get("test-other", None) is not None

    main(config, args.wavs_path, args.output)
