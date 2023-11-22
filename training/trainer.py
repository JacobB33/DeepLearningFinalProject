"""
Taken from the pytorch DDP series
"""
from configs.configs import *
from dataclasses import asdict
from collections import OrderedDict
from typing import Optional, Any, Dict
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

import fsspec


class Trainer:
    """This class is a training class for a distributed training run. It is paramatarized by the train config. 
    Includes evaluation runs.
    """

    def __init__(self, trainer_config: TrainerConfig, model, optimizer, train_dataset, test_dataset=None):
        self.config = trainer_config
        # set torchrun variables
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        # data stuff
        # self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.test_loader = self._prepare_dataloader(test_dataset) if test_dataset else None
        # initialize train states
        self.epochs_run = 0
        self.model = model.to(self.local_rank)
        self.optimizer = optimizer
        self.save_every = self.config.save_every

        # initialize amp. Should speed up by 2x or 3x
        if self.config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        # load snapshot if available. only necessary on the first node.
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        self._load_snapshot()
        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank])

        if trainer_config.use_wandb:
            import wandb
            if self.global_rank == 0:
                wandb.init(project="ddp-series")
                wandb.log({"config": asdict(self.config)})


    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=self.config.shuffle,
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset)
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return

        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, train: bool = True) -> float:
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                                               enabled=(self.config.use_amp)):
            _, loss = self.model(source, targets)

        if train:
            # Set to none is faster as it doesn't overwrite the memory
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                self.optimizer.step()

        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        dataloader.sampler.set_epoch(epoch)
        losses = []
        for iter, (source, targets) in enumerate(dataloader):
            step_type = "Train" if train else "Eval"
            # import IPython; IPython.embed()
            source = source.to(self.local_rank)
            targets = targets.to(self.local_rank)
            batch_loss = self._run_batch(source, targets, train)
            losses.append(batch_loss)
            if iter % 100 == 0:
                print(f"[GPU{self.global_rank}] Epoch {epoch} | Iter {iter} | {step_type} Loss {batch_loss:.5f}")

        return sum(losses) / len(losses)

    def _save_snapshot(self, epoch):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch
        )
        # save snapshot
        snapshot = asdict(snapshot)
        torch.save(snapshot, self.config.snapshot_path)

        print(f"Snapshot saved at epoch {epoch}")

    def train(self):
        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1
            batch_avg_loss = self._run_epoch(epoch, self.train_loader, train=True)

            # eval run
            if self.test_loader:
                test_avg_loss = self._run_epoch(epoch, self.test_loader, train=False)
                test_loss = torch.tensor([test_avg_loss])
                dist.reduce(test_loss, 0, dist.ReduceOp.SUM)
            if self.global_rank == 0:
                if self.test_loader:
                    test_loss = test_loss / self.world_size
                    wandb.log({"loss": batch_avg_loss, "test_loss": test_loss.item()})
                else:
                    wandb.log({"loss": batch_avg_loss})
                if epoch % self.save_every == 0:
                    self._save_snapshot(epoch)
