import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from training.configs import ModelConfig
from training.networks.building_blocks import ResBlock, DoubleConv, EncodeBS

class BrainScanEmbedder(nn.Module):
    """This needs to be batched input of channel, brainscan or (c, 7604). Outputs (c, 77, 1024)"""
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        upscale_schedule = model_config.upscale_schedule

        self.encoder = EncodeBS(num_output_channels=upscale_schedule[0])
        module = []
        
        for i in range(1, len(upscale_schedule)):
            module.append(DoubleConv(upscale_schedule[i-1], upscale_schedule[i]))
            module.append(nn.TransformerEncoderLayer(d_model=1024, nhead=model_config.nhead))
            module.append(ResBlock(upscale_schedule[i]))
        module.append(DoubleConv(upscale_schedule[-1], upscale_schedule[-1], activation=False))
        self.backbone = nn.Sequential(*module)
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=model_config.nhead)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_config.num_transformer_layers)

    def forward(self, source, targets=None):
        output = self.encoder(source)
        # output = self.transformer_encoder(output)
        output = self.backbone(output)
        if targets is None:
            return output
        loss = F.mse_loss(output, targets)
        return output, loss

