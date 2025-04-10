import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from argparse import Namespace
from typing import Optional
import os

# LayerNorm definition (can be shared or imported)
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

class AVHuBERTVisualEncoder(nn.Module):
    """
    Visual Encoder from AV-HuBERT model
    """
    pass
    
class AVHuBERTAudioEncoder(nn.Module):
    """
    Audio Encoder from AV-HuBERT model
    """
    pass

class AVHuBERTEncoderWrapper(nn.Module):
    """
    A wrapper for loading and using a pre-trained AV-HuBERT model 
    (from fairseq) as the visual encoder.
    Includes a projection layer to match the target embedding dimension.
    """
    def __init__(self,
                 av_hubert_code_path: str, 
                 av_hubert_ckpt_path: str,
                 output_embed_dim: int,
                 av_hubert_output_dim: int = 1024, # Default for AV-HuBERT Large
                 freeze_encoder: bool = True):
        super().__init__()

        if not os.path.exists(av_hubert_ckpt_path):
             raise FileNotFoundError(f"AV-HuBERT checkpoint not found at {av_hubert_ckpt_path}")
        if not os.path.isdir(av_hubert_code_path):
             raise NotADirectoryError(f"AV-HuBERT code directory not found at {av_hubert_code_path}")

        self.av_hubert_code_path = av_hubert_code_path
        self.av_hubert_ckpt_path = av_hubert_ckpt_path
        self.output_embed_dim = output_embed_dim
        self.av_hubert_output_dim = av_hubert_output_dim
        self.freeze_encoder = freeze_encoder

        # Load the AV-HuBERT model using fairseq utilities
        # Ensure the user_dir points to the directory containing the model definition (e.g., avhubert)
        user_dir_ns = Namespace(user_dir=self.av_hubert_code_path)
        utils.import_user_module(user_dir_ns)
        
        print(f"Loading AV-HuBERT model from: {self.av_hubert_ckpt_path}")
        # Determine if loading weights is intended (might depend on ckpt naming)
        # load_weights = False if "no_weights" in self.av_hubert_ckpt_path else True 
        # For simplicity, assume weights are always loaded if ckpt exists
        # The fairseq loading function might handle this implicitly or require arguments.
        # Adjust `arg_overrides` if necessary based on AV-HuBERT loading requirements.
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [self.av_hubert_ckpt_path],
            # arg_overrides={'load_weights': load_weights} # Example override
        )
        
        # Extract the encoder part. The exact attribute might vary ('encoder' or the model itself).
        # Check the loaded `models[0]` structure if unsure.
        # Assuming a structure similar to the reference code:
        # If the checkpoint is from fine-tuning (often contains 'ft'), it might be models[0].encoder
        # If it's a base pre-trained model, it might be models[0] itself.
        if hasattr(models[0], 'encoder') and 'ft' in self.av_hubert_ckpt_path:
             self.encoder = models[0].encoder
             print("Loaded fine-tuned AV-HuBERT encoder.")
        else:
             self.encoder = models[0]
             print("Loaded base AV-HuBERT model as encoder.")

        # Freeze the encoder parameters if requested
        if self.freeze_encoder:
            print("Freezing AV-HuBERT encoder parameters.")
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval() # Set to eval mode if frozen

        # Projection layer
        self.projection = nn.Linear(self.av_hubert_output_dim, self.output_embed_dim)
        # Optional: LayerNorm after projection? Diagram doesn't explicitly show it here.
        # self.projection_ln = LayerNorm(self.output_embed_dim) 

    def forward(self, 
                video_input: torch.Tensor, 
                video_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through AV-HuBERT encoder and projection.

        Args:
            video_input: Tensor of video features/frames (B, T, C or B, C, T, H, W depending on model input).
                         Check AV-HuBERT documentation for expected input format.
                         Assuming (B, T, H, W, C) or similar for ResNet frontend based models.
                         The reference implementation seems to pass (B, T, F) pre-extracted features.
                         Let's assume input is (B, T, FeatureDim) for now, matching ref.
            video_padding_mask: Optional padding mask for the video sequence (B, T).

        Returns:
            projected_features: Tensor of shape (B, T_out, output_embed_dim).
        """
        
        # Pass video through the AV-HuBERT model.
        # The exact call depends on the loaded model type (base vs fine-tuned).
        # Using the structure from whisper-flamingo ref:
        if hasattr(self.encoder, 'extract_features'): # Common in fairseq base models
            # Base model: often requires a dictionary input and specific flags
            encoder_out_dict = self.encoder.extract_features( 
                source={'video': video_input, 'audio': None}, 
                padding_mask=video_padding_mask, 
                mask=False, # Usually False during inference/feature extraction
                # features_only=True # May or may not be needed depending on version
            )
            # Output structure can vary, common keys are 'x' or 'encoder_out'
            if 'x' in encoder_out_dict:
                visual_features = encoder_out_dict['x'] # Shape (B, T, D)
            elif 'encoder_out' in encoder_out_dict: # Often list or tensor
                 visual_features = encoder_out_dict['encoder_out'] 
                 if isinstance(visual_features, list): visual_features = visual_features[0]
                 # Check if permutation is needed (e.g., T, B, D -> B, T, D)
                 if visual_features.shape[0] != video_input.shape[0] and visual_features.shape[1] == video_input.shape[0]:
                     visual_features = visual_features.permute(1, 0, 2)
            else:
                 raise KeyError(f"Could not find expected output key ('x' or 'encoder_out') in AV-HuBERT output: {encoder_out_dict.keys()}")
        
        elif 'ft' in self.av_hubert_ckpt_path and hasattr(self.encoder, 'forward'): # Fine-tuned model
            # Fine-tuned model might have a simpler forward or expect dict
            try:
                # Try dictionary input first (matches ref)
                encoder_out_dict = self.encoder(source={'video': video_input, 'audio': None}, padding_mask=video_padding_mask)
                if 'encoder_out' not in encoder_out_dict:
                    raise KeyError("Expected 'encoder_out' key in fine-tuned model output dict.")
                visual_features = encoder_out_dict['encoder_out'] # Typically (T, B, D)
                # Permute T, B, D -> B, T, D
                visual_features = visual_features.permute(1, 0, 2)
            except TypeError: 
                # Fallback if it doesn't accept dict input
                # This path is less likely based on reference code
                print("AV-HuBERT encoder forward failed with dict input, trying direct tensor input.")
                visual_features = self.encoder(video_input, padding_mask=video_padding_mask)
                # Check output format, might be tuple or tensor, adjust accordingly
                if isinstance(visual_features, tuple): visual_features = visual_features[0]
                # Ensure B, T, D format
                if visual_features.shape[0] != video_input.shape[0] and visual_features.shape[1] == video_input.shape[0]:
                     visual_features = visual_features.permute(1, 0, 2)
        else:
             raise NotImplementedError("Unsupported AV-HuBERT model structure or forward method.")

        # Apply projection
        projected_features = self.projection(visual_features)

        # Apply optional LayerNorm
        # if hasattr(self, 'projection_ln'):
        #     projected_features = self.projection_ln(projected_features)

        # Ensure encoder is kept in eval mode if frozen
        if self.freeze_encoder and self.encoder.training:
            self.encoder.eval()
            
        return projected_features 