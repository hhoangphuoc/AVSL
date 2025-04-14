# NOTE: DEPRECATED: This file is no longer used in the current implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

#----------------------------------------------------------------------------------------------------------
#                                       ConvFeatureExtractionModel
#----------------------------------------------------------------------------------------------------------
class ConvFeatureExtractionModel(nn.Module):
    """
    Convolutional feature extraction model for audio signals.
    Based on the AV-HuBERT implementation but using PyTorch instead.
    """
    def __init__(self, conv_layers: List[Tuple[int, int, int]], dropout: float = 0.0):
        super().__init__()
        
        def block(n_in, n_out, k, stride):
            return nn.Sequential(
                nn.Conv1d(n_in, n_out, k, stride=stride, bias=False),
                nn.GroupNorm(1, n_out),
                nn.GELU(),
            )
        
        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "conv_layers should be a list of tuples (dim, kernel_size, stride)"
            (dim, k, stride) = cl
            self.conv_layers.append(block(in_d, dim, k, stride))
            in_d = dim

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T)
        x = x.unsqueeze(1)  # (B, 1, T)
        
        for conv in self.conv_layers:
            x = conv(x)
        
        return self.dropout(x)
#----------------------------------------------------------------------------------------------------------
