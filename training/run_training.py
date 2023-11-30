# command: OMP_NUM_THREADS=32 torchrun --nnodes 1 --nproc_per_node 8 ./training/run_training.py

import yaml

from networks.encoder import BrainScanEmbedder, FancyBrainScanEmbedder, plzWork
from trainer import Trainer
from training.configs import *
import torch.nn as nn

import os
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group
from training.data import get_train_dataset
import random
# import pickle
# torchrun --nnodes 1 --nproc_per_node 1 ./training/run_training.py 
def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_train_test(train_len):
    dataset = get_train_dataset()
    
    indicies = list(range(len(dataset)))
    random.shuffle(indicies)
    train_idx, test_idx = indicies[:train_len], indicies[train_len:]
    train_set, test_set = torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)

    
    return train_set, test_set, test_idx

        
def get_train_objs(cfg):
    model_config = ModelConfig(**cfg['model_config'])
    opt_cfg = OptimizerConfig(**cfg['optimizer_config'])
    data_cfg = DataConfig(**cfg['data_config'])
        
    train_set, test_set, test_idx = get_train_test(int(len(get_train_dataset())*data_cfg.train_percentage))    
    cfg['test_idx'] = test_idx
    if cfg['model_type'] == 'normal':
        model = BrainScanEmbedder(model_config)
    elif cfg['model_type'] == 'fancy':
        model = FancyBrainScanEmbedder(model_config)
    elif cfg['model_type'] == 'plzwork':
        model = plzWork(model_config)
    else:
        raise ValueError(f"Model type {cfg['model_type']} not supported")

    if cfg['compile']:
        model = torch.compile(model)
    
    optimizer = create_optimizer(model, opt_cfg)
    if cfg['compile'] == True:
        compile()
    lr_scheduler = None
    if 'lr_scheduler_config' in cfg:
        lr_scheduler_config = LRSchedulerConfig(**cfg['lr_scheduler_config'])
        warmup_config = None
        if 'warmup_config' in cfg:
            warmup_config = WarmUpConfig(**cfg['warmup_config'])
        
        lr_scheduler = create_lr_scheduler(optimizer, lr_scheduler_config, warmup_config)


    return model, optimizer, train_set, test_set, lr_scheduler

def main(cfg_path):
    cfg = yaml.load(open(cfg_path, 'r'), yaml.FullLoader)
    ddp_setup()
    print(cfg)
    
    trainer_config = TrainerConfig(**cfg['trainer_config'])
  
    
    model, optimizer, train_data, test_data, lr_scheduler = get_train_objs(cfg)
    
    
    trainer = Trainer(trainer_config, model, optimizer, train_data, test_data, lr_scheduler, cfg)
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    main('./training/configs/encoder_config.yaml')
    # main('./training/configs/fancy_encoder_config.yaml')
    # main('./training/configs/plzwork_config.yaml')