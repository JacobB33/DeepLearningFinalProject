from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Any, Dict

@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    shuffle: bool = False
    use_wandb: bool = False
    # Used to have async data loading. if 0, Synchronus
    data_loader_workers: int = None
    grad_norm_clip: float = None
    snapshot_path: Optional[str] = None
    save_every: int = None
    use_amp: bool = None

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

def create_optimizer(model, config: OptimizerConfig):
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Optimizer {config.optimizer} not supported")

