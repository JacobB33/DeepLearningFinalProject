from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Any, Dict, List
import torch


@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    shuffle: bool = False
    
    # wandb stuff
    use_wandb: bool = False
    run_name: str= None
    # Used to have async data loading. if 0, Synchronus
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None
    use_lr_scheduler: bool = None


@dataclass
class ModelConfig:
    upscale_schedule: List[int]
    num_transformer_layers: int
    nhead: int



@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int


@dataclass
class OptimizerConfig:
    optimizer: str = None
    learning_rate: float = None
    weight_decay: float = None


@dataclass
class DataConfig:
    train_percentage: float

@dataclass 
class LRSchedulerConfig:
    type: str = None
    step_size: int = None
    gamma: float = None
    
    mode: str = None
    patience: int = None
    threshold: float = None
    min_lr: float = None
    cooldown: int = None
    
    
@dataclass
class WarmUpConfig:
    warmup_steps: int = None
    warmup_lr: float = None


def create_lr_scheduler(optimizer, config: LRSchedulerConfig, warmup_config: WarmUpConfig = None):
    if warmup_config:
        warmup_lr = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=warmup_config.warmup_lr, total_iters=warmup_config.warmup_steps)
    
    if config.type == "step":
        lr_scheduler =  torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.type == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config.gamma, patience=config.patience, mode=config.mode, threshold=config.threshold, min_lr=config.min_lr,
                                                                  cooldown=config.cooldown)
    else:
        raise ValueError(f"Scheduler {config.scheduler} not supported")
    
    if warmup_config is None:
        return lr_scheduler
    else:
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_lr, lr_scheduler], milestones=[warmup_config.warmup_steps])
    
def create_optimizer(model, config: OptimizerConfig):
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Optimizer {config.optimizer} not supported")
