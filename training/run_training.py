import yaml

from networks.encoder import BrainScanEmbedder
from trainer import Trainer
from training.configs import *

import os
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group
from training.data import get_train_dataset


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_train_objs(model_config: ModelConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig, compile: bool):
    dataset = get_train_dataset()
    train_len = int(len(dataset) * data_cfg.train_percent)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    model = BrainScanEmbedder(model_config)
    optimizer = create_optimizer(model, opt_cfg)
    
    return model, optimizer, train_set, test_set

def main(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'), yaml.FullLoader)
    ddp_setup()
    print(cfg)

    model_config = ModelConfig(**cfg['model_config'])
    opt_config = OptimizerConfig(**cfg['optimizer_config'])
    data_config = DataConfig(**cfg['data_config'])
    trainer_config = TrainerConfig(**cfg['trainer_config'])

    model, optimizer, train_data, test_data = get_train_objs(model_config, opt_config, data_config, cfg['compile'])
    trainer = Trainer(trainer_config, model, optimizer, train_data, test_data)
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    main('/home/jacob/projects/DeepLearningFinalProject/training/configs/encoder_config.yaml')