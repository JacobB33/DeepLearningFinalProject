import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from training.configs import ModelConfig
from training.networks.building_blocks import ResBlock, DoubleConv, EncodeBS
from training.networks.fancy_building_blocks import AttnBlock, ResnetBlock


class BrainScanEmbedder(nn.Module):
    """This needs to be batched input of channel, brainscan or (c, 7604). Outputs (c, 77, 1024)"""
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        upscale_schedule = model_config.upscale_schedule

        self.encoder = EncodeBS(num_output_channels=upscale_schedule[0])
        module = []
        
        for i in range(1, len(upscale_schedule)):
            module.append(DoubleConv(upscale_schedule[i-1], upscale_schedule[i]))
            module.append(nn.TransformerEncoderLayer(d_model=1024, nhead=model_config.nhead, dropout=0))
            # module.append(nn.Tanh())
            module.append(ResBlock(upscale_schedule[i]))
        module.append(nn.TransformerEncoderLayer(d_model=1024, nhead=model_config.nhead, dropout=0))
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


class FancyBrainScanEmbedder(nn.Module):
    """This needs to be batched input of channel, brainscan or (c, 7604). Outputs (c, 77, 1024)"""
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        upscale_schedule = model_config.upscale_schedule

        self.encoder = EncodeBS(num_output_channels=upscale_schedule[0])
        
        module = []
        for i in range(1, len(upscale_schedule)):
            module.append(ResnetBlock(in_channels=upscale_schedule[i-1], dropout=0))
            module.append(AttnBlock(in_channels=upscale_schedule[i-1]))
            module.append(ResnetBlock(in_channels=upscale_schedule[i-1], out_channels=upscale_schedule[i], dropout=0))
        self.backbone = nn.Sequential(*module)
        
    def forward(self, source, targets=None):
        output = self.encoder(source)
        # output = self.transformer_encoder(output)
        output = self.backbone(output)
        if targets is None:
            return output
        loss = F.mse_loss(output, targets)
        return output, loss





if __name__ == '__main__':
    test_tensor = torch.randn(32, 7604).to('cuda')  
    model = FancyBrainScanEmbedder(ModelConfig(upscale_schedule=[8, 16, 32, 64, 77], num_transformer_layers=1, nhead=8)).to('cuda')
    output = model(test_tensor)
    print(output.shape)
    