from networks.encoder import Encoder
from trainer import Trainer
from trainer import TrainerConfig
    

import os
import torch
from torch.utils.data import random_split
from torch.distributed import init_process_group, destroy_process_group

from omegaconf import DictConfig
import hydra


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def get_train_objs(encoder_config, opt_cfg: OptimizerConfig, data_cfg: DataConfig, compile: bool):
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)
    
    return model, optimizer, train_set, test_set

@hydra.main(version_base=None, config_path=".", config_name="encoder_config")
def main(cfg: DictConfig):
    ddp_setup()

    encoder_config = cfg['encoder_config']
    opt_config = cfg['optimizer_config']
    data_config = cfg['data_config']
    
    model, optimizer, train_data, test_data = get_train_objs(encoder_config, opt_cfg, data_cfg, cfg['compile'])
    trainer = Trainer(trainer_cfg, model, optimizer, train_data, test_data)
    trainer.train()

    destroy_process_group()


if __name__ == "__main__":
    main()