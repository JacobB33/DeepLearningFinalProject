import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os



class ResBlock(nn.Module):
    """This is a residual block. It is 2 1d convolutions with a residual connection."""
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv_1 = nn.Conv1d(channels, channels, kernel_size, stride, padding, bias=False)
        torch.nn.init.kaiming_normal_(self.conv_1.weight, nonlinearity="relu")
        self.conv_2 = nn.Conv1d(channels, channels, kernel_size, stride, padding, bias=False)
        torch.nn.init.kaiming_normal_(self.conv_2.weight, nonlinearity="relu")
        self.batch_norm_1 = nn.BatchNorm1d(channels)
        self.batch_norm_2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        # self.silu = nn.SiLU(inplace=True)
        
    def forward(self, x):
        residual = x
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu(x)
        # x = self.silu(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x += residual
        x = self.relu(x)
        # x = self.silu(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, activation=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        layers = [
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ]
        if activation:
            # layers.append(nn.ReLU(inplace=True))
            layers.append(nn.SiLU(inplace=True))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class EncodeBS(nn.Module):
    def __init__(self, input_size=7604, num_output_channels=8):
        super().__init__()
        # 1024 is the embedding size of the clip model
        self.encoder1 = nn.Linear(input_size, 1024*num_output_channels)
        # self.encoder2 = nn.Linear(1024*num_output_channels, 1024*num_output_channels)
        # self.encoder3 = nn.Linear(1024*num_output_channels, 1024*num_output_channels)

        self.relu = nn.ReLU(inplace=True)
        # self.silu = nn.SiLU(inplace=True)
        self.num_output_channels = num_output_channels
    
    def forward(self, x):
        x = self.encoder1(x)
        # using swoosh activations
        # x = self.silu(x)
        x = self.relu(x)
        # skip = x
        # x = self.encoder2(x)
        # x = self.relu(x)
        # x = self.encoder3(x)
        # x = x + skip
        # self.relu(x)
        x = x.reshape(x.shape[0], self.num_output_channels, -1)
        return x

class AttentionModule(nn.Module):   
    def __init__(self, embedding_dim, nhead):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=nhead)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self.attention(q, k, v)[0]

if __name__ == "__main__":  
    test_tensor = torch.randn(32, 7604)
    encoder = EncodeBS()
    output = encoder(test_tensor)
    print(output.shape)
    block = ResBlock(8)
    output = block(output)
    
    print(output.shape) 
    
    upscale = DoubleConv(8, 16)
    print(upscale(output).shape)
    