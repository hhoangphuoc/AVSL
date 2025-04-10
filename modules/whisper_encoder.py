import torch
import torch.nn as nn
from transformers.models.whisper.modeling_whisper import WhisperEncoder as HuggingFaceWhisperEncoder
from transformers.models.whisper.configuration_whisper import WhisperConfig

class WhisperEncoder(nn.Module):
    """
    Whisper Encoder
    """
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.encoder = HuggingFaceWhisperEncoder(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)