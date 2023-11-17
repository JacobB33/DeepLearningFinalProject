import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from building_blocks import ResBlock, DoubleConv, EncodeBS

class BrainScanEmbedder(nn.Module):
    """This needs to be batched input of channel, brainscan or (c, 7604). Outputs (c, 77, 1024)"""
    def __init__(self, upscale_schedule):
        super().__init__()
        self.encoder = EncodeBS(upscale_schedule[0])
        module = []
        
        for i in range(1, len(upscale_schedule)):
            module.append(DoubleConv(upscale_schedule[i-1], upscale_schedule[i]))
            module.append(ResBlock(upscale_schedule[i]))
        module.append(DoubleConv(upscale_schedule[-1], upscale_schedule[-1], activation=False))
        self.backbone = nn.Sequential(*module)
        