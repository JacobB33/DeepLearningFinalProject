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
            module.append(ResBlock(upscale_schedule[i]))
        module.append(DoubleConv(upscale_schedule[-1], upscale_schedule[-1], activation=False))
        self.backbone = nn.Sequential(*module)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=model_config.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_config.num_transformer_layers)

    def forward(self, source, targets=None):
        output = self.encoder(source)
        output = self.transformer_encoder(output)
        output = self.backbone(output)
        if targets is None:
            return output
        loss = F.mse_loss(output, targets)
        return output, loss

class NewBrainScanEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_dim = 7604
        self.fc1 = nn.Linear(input_dim, input_dim*8, bias=False)
        self.fc2 = nn.Linear(input_dim*8, input_dim*8, bias=False)
        self.fc3 = nn.Linear(input_dim*8, input_dim*16, bias=False)
        self.fc4 = nn.Linear(input_dim*16, input_dim*16, bias=False)
        self.fc5 = nn.Linear(input_dim*16, input_dim*77)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn1 = nn.BatchNorm1d(input_dim*8)
        self.bn2 = nn.BatchNorm1d(input_dim*8)
        self.bn3 = nn.BatchNorm1d(input_dim*16)
        self.bn4 = nn.BatchNorm1d(input_dim*16)
        
    
    def forward(self, source, targets=None):
        # 1 to 8
        output = self.fc1(source)
        output = self.bn1(output)
        output = self.relu(output)
        
        # 8
        skip = output
        # 8 to 8
        output = self.fc2(output)
        output = self.bn2(output)
        output = output + skip
        output = self.relu(output)
        
        # 8 to 16
        output = self.fc3(output)
        output = self.bn3(output)
        output = self.relu(output)
        
        skip = output
        # 16 to 16
        output = self.fc4(output)
        output = self.bn4(output)
        output = output + skip
        output = self.relu(output)
        
        # 16 to 77
        output = self.fc5(output)
       
        
        output = output.reshape(output.shape[0], 77, -1)
        
        if targets is None:
            return output
        else:
            loss = F.mse_loss(output, targets)
            return output, loss
        
        
        
        
        
        
        
        
        
    
    