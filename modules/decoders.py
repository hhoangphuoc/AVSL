import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import (
    WhisperDecoderLayer, 
    WhisperAttention
)
from transformers.utils import logging
logger = logging.get_logger(__name__)

# LayerNorm implementation consistent with Hugging Face Whisper
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)

class GatedCrossAttentionLayer(nn.Module):
    """
    Implements the Gated Cross Attention mechanism shown in the diagram.
    Query: Decoder hidden state
    Key/Value: Visual features (e.g., from AV-HuBERT)
    Includes parallel gated paths for cross-attention and feed-forward network.
    """
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.dropout = dropout

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Cross Attention components
        self.cross_attn = WhisperAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            is_decoder=True, # Using is_decoder=True for consistency, though no causal mask needed here
            bias=True
        )
        # Use a distinct LayerNorm instance if needed, or reuse existing ones carefully
        # self.cross_attn_layer_norm = LayerNorm(self.embed_dim) # Input LayerNorm handled outside
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        # Feed Forward (FFW) components (MLP)
        # Using Whisper's standard MLP structure for consistency
        # Input LayerNorm handled outside
        self.ff_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim * 4)
        self.activation_fn = nn.GELU()
        self.fc2 = nn.Linear(self.embed_dim * 4, self.embed_dim)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))


    def forward(
        self,
        hidden_states: torch.Tensor, # Query (from decoder self-attention output)
        key_value_states: torch.Tensor, # Key/Value (visual features)
        attention_mask: torch.Tensor | None = None, # Optional mask for visual features
    ) -> torch.Tensor:

        # --- Gated Cross-Attention Path ---
        # NOTE: LayerNorm before this module is expected to be applied outside
        residual = hidden_states # Store residual for the gated addition

        # Cross Attention
        # Input LayerNorm is applied before calling this module
        hidden_states_norm = self.cross_attn_layer_norm(hidden_states) # NOTE: Apply LayerNorm if neededS
        attn_output, _ = self.cross_attn(
            hidden_states=hidden_states, # Query (Q)
            key_value_states=key_value_states, # Visual Features (K, V)
            attention_mask=attention_mask, # Mask for visual features if any padding
        )
        # Dropout is handled within WhisperAttention

        # ------------------- First tanh gate -------------------
        gated_attn_output = attn_output * self.attn_gate.tanh() 


        # --- Gated Feed-Forward Path ---
        # Note: LayerNorm before this module is expected to be applied outside
        # Apply LayerNorm if needed here, or ensure it's done before this module
        hidden_states_norm = self.ff_layer_norm(hidden_states)
        ff_output = self.fc1(hidden_states_norm)
        ff_output = self.activation_fn(ff_output)
        # Dropout can be added here if needed, WhisperDecoderLayer adds it after MLP
        ff_output = self.fc2(ff_output)

        # ------------------- Second tanh gate -------------------
        gated_ff_output = ff_output * self.ff_gate.tanh() 

        # --- Combine Paths ---
        # Add the outputs of the two gated paths to the original residual
        # This differs slightly from adding them sequentially as in the ref code's ResidualAttentionBlock
        # but matches the parallel structure suggested by the diagram's (+) symbols.
        output = gated_attn_output + gated_ff_output

        return output


class GatedWhisperDecoderLayer(WhisperDecoderLayer):
    """
    This class is the GatedWhisperDecoderLayer that adapted for Audio-Visual task to the decoder layer of WhisperDecoder,
    This layer use Gated Cross Attention methods (through `GatedCrossAttentionLayer`) to integrate visual features with audio features.

    Other Whisper Layers including: 
    - Self-Attention Layer
    - Cross-Attention Layer
    - MLP Layer
    are all inherited from the original WhisperDecoderLayer.
    """
    def __init__(self, config):
        super().__init__(config)
        
        # --------------- NOTE: Added Gated Cross Attention Layer to WhisperDecoderLayer ---------------
        self.gated_cross_attn = GatedCrossAttentionLayer(
            embed_dim=self.embed_dim,
            n_heads=config.decoder_attention_heads,
            dropout=config.dropout # Use the main dropout rate
        )
        # Add a LayerNorm specifically for the input to the gated cross-attention
        self.gated_cross_attn_layer_norm = LayerNorm(self.embed_dim)
        # --------------------------------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None, # Audio features
        encoder_attention_mask: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None, # NOTE: Added visual features (new input)
        visual_attention_mask: torch.Tensor | None = None, # NOTE: Added optional mask for visual features
        layer_head_mask: torch.Tensor | None = None,
        cross_attn_layer_head_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = True,
    ) -> torch.Tensor:

        if visual_features is None:
             raise ValueError("`visual_features` must be provided to AudioVisualWhisperDecoder")

        residual = hidden_states

        # ------------------------------- Self-Attention Layer -------------------------------
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # --------------------------------------------------------------------------------------------

        # -------------------------------------------- Gated Cross-Attention Layer --------------------------------------------
        residual = hidden_states
        hidden_states_norm = self.gated_cross_attn_layer_norm(hidden_states)
        gated_output = self.gated_cross_attn(
            hidden_states=hidden_states_norm,
            key_value_states=visual_features,
            attention_mask=visual_attention_mask, # Use specific mask if provided
        )
        # Apply dropout AFTER the gated addition? Or assume dropout is handled inside?
        # Let's add dropout after the addition for consistency with other blocks.
        gated_output = nn.functional.dropout(gated_output, p=self.dropout, training=self.training)
        hidden_states = residual + gated_output

        # -------------------------------------------- Cross-Attention Layer --------------------------------------------
        # Same as original WhisperDecoderLayer
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

            # add cross-attn to positions 1..N
            present_key_value = present_key_value + cross_attn_present_key_value

        # -------------------------------------------- MLP Layer --------------------------------------------   
        # Same as original WhisperDecoderLayer
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # --------------------------------------------------------------------------------------------
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights) # Maybe add gated_attn_weights?

        if use_cache:
            outputs += (present_key_value,)

        return outputs 

from transformers.models.whisper.modeling_whisper import WhisperDecoder
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, Tuple, Union
import torch.utils.checkpoint

class AudioVisualWhisperDecoder(WhisperDecoder):
    """
    Whisper Decoder modified to use Gated Cross Attention layers for visual features.
    Inherits from transformers.WhisperDecoder and replaces the standard layers
    with AudioVisualWhisperDecoderLayer.
    """
    def __init__(self, config):
        # Initialize the parent WhisperDecoder but skip its layer creation momentarily
        # We achieve this by temporarily setting config.decoder_layers to 0
        # then calling the super().__init__ and finally creating our custom layers.
        original_num_layers = config.decoder_layers
        config.decoder_layers = 0
        super().__init__(config)
        config.decoder_layers = original_num_layers # Restore config value

        # Re-create the layers using our modified layer class
        self.layers = nn.ModuleList([GatedWhisperDecoderLayer(config) for _ in range(config.decoder_layers)])

        # Ensure other parts like embed_tokens, embed_positions, layernorm_embedding are initialized correctly by super()
        self.gradient_checkpointing = False # Keep consistent with parent

    def forward(
        self,
        input_ids=None,
        attention_mask=None,  # This is the decoder attention mask (for self-attention)
        encoder_hidden_states=None, # Audio features
        encoder_attention_mask=None, # Mask for audio features
        visual_features=None, # Visual features (new input)
        visual_attention_mask=None, # Optional mask for visual features
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if attention_mask is None and input_shape[-1] > 1:
             attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, src_seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = self._expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # expand visual attention mask
        if visual_features is not None and visual_attention_mask is not None:
             # [bsz, src_seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
             # Assuming visual_attention_mask has shape [bsz, visual_seq_len]
             visual_attention_mask = self._expand_mask(visual_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        all_cross_attentions = ()
        next_decoder_cache = ()

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                assert attn_mask.size()[0] == (
                    len(self.layers)
                ), f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = torch.rand(1)

            # Note: LayerDrop logic might need adjustment if different dropout is desired for gated attention path
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask, # Causal mask
                    encoder_hidden_states, # Audio features
                    encoder_attention_mask,
                    visual_features, # Visual features passed here
                    visual_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None, # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask, # Causal mask
                    encoder_hidden_states=encoder_hidden_states, # Audio features
                    encoder_attention_mask=encoder_attention_mask,
                    visual_features=visual_features, # Pass visual features
                    visual_attention_mask=visual_attention_mask, # Pass visual mask
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)
                    # Note: Gated cross-attention weights are not currently captured here

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        ) 