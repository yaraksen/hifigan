import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.base.base_text_encoder import BaseTextEncoder
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker



class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            G_optimizer,
            D_optimizer,
            G_scheduler,
            D_scheduler,
            config,
            device,
            dataloaders,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model,
                        criterion,
                        G_optimizer,
                        D_optimizer,
                        G_scheduler,
                        D_scheduler,
                        config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}

        if self.len_epoch == 1:
            self.log_step = 1
        else:
            self.log_step = 100

        print('self.log_step:', self.log_step)
        print('self.len_epoch:', self.len_epoch)

        metric_keys = ["G_loss", "G_mel_loss", "G_fm_loss", "G_adv_loss", "D_loss"]
        self.train_metrics = MetricTracker(
            "G grad norm", "D grad norm", *[m for m in metric_keys], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        keys_to_gpu = ["real_wavs", "real_mels"]
        for tensor_for_gpu in keys_to_gpu:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)

        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch - 1)
        ):  
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            
            global_step = self.len_epoch * epoch + batch_idx
            if global_step % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate G", self.G_optimizer.param_groups[0]['lr']
                )
                self.writer.add_scalar(
                    "learning rate D", self.D_optimizer.param_groups[0]['lr']
                )
                rand_idx = torch.randint(low=0, high=batch["fake_wavs"].shape[0], size=(1,))
                self._log_audio(batch["fake_wavs"][rand_idx], "fake_wavs")
                self._log_audio(batch["real_wavs"][rand_idx], "real_wavs")
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()

            if batch_idx + 1 >= self.len_epoch:
                break
        log = last_train_metrics

        if self.G_scheduler is not None:
            self.G_scheduler.step()
        if self.D_scheduler is not None:
            self.D_scheduler.step()

        return log

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        
        # D optimizing
        self.D_optimizer.zero_grad()

        G_outputs = self.model.generator(**batch)
        batch.update(G_outputs)
        print(batch["real_wavs"].shape, batch["fake_wavs"].shape)

        D_outputs = self.model.discriminator(real_wavs=batch["real_wavs"], fake_wavs=batch["fake_wavs"].detach())
        batch.update(D_outputs)

        D_loss = self.criterion.D_loss(**batch)
        batch.update(D_loss)
        D_loss["D_loss"].backward()
        self.D_optimizer.step()

        # G optimizing
        self.G_optimizer.zero_grad()

        D_outputs = self.model.discriminator(**batch)
        batch.update(D_outputs)

        G_loss = self.criterion.G_loss(**batch)
        batch.update(G_loss)
        G_loss["G_loss"].backward()
        self.G_optimizer.step()

        for k, v in batch.items():
            if "loss" in k:
                metrics.update(k, v.item())
        
        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))
    
    def _log_audio(self, audio: torch.Tensor, tag: str):
        # audio = random.choice(audio_batch.cpu())
        self.writer.add_audio(tag, audio.cpu(), sample_rate=16000)

    # @torch.no_grad()
    # def get_grad_norm(self, norm_type=2):
    #     parameters = self.model.parameters()
    #     if isinstance(parameters, torch.Tensor):
    #         parameters = [parameters]
    #     parameters = [p for p in parameters if p.grad is not None]
    #     total_norm = torch.norm(
    #         torch.stack(
    #             [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
    #         ),
    #         norm_type,
    #     )
    #     return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
