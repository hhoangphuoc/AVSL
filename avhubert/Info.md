Model Architecture described as follows:
```bash
Model: AV2TextForConditionalGeneration(
  (model): AV2TextModel(
    (encoder): AVHubertModel(
      (feature_extractor_audio): SubModel(
        (proj): Linear(in_features=104, out_features=1024, bias=True)
      )
      (feature_extractor_video): SubModel(
        (resnet): ResEncoder(
          (frontend3D): Sequential(
            (0): Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
            (1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): PReLU(num_parameters=64)
            (3): MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=1, ceil_mode=False)
          )
          (trunk): ResNet(
            (layer1): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu1): PReLU(num_parameters=64)
                (relu2): PReLU(num_parameters=64)
                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (1): BasicBlock(
                (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu1): PReLU(num_parameters=64)
                (relu2): PReLU(num_parameters=64)
                (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (layer2): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu1): PReLU(num_parameters=128)
                (relu2): PReLU(num_parameters=128)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (downsample): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu1): PReLU(num_parameters=128)
                (relu2): PReLU(num_parameters=128)
                (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (layer3): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu1): PReLU(num_parameters=256)
                (relu2): PReLU(num_parameters=256)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (downsample): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu1): PReLU(num_parameters=256)
                (relu2): PReLU(num_parameters=256)
                (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (layer4): Sequential(
              (0): BasicBlock(
                (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu1): PReLU(num_parameters=512)
                (relu2): PReLU(num_parameters=512)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (downsample): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
              )
              (1): BasicBlock(
                (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                (relu1): PReLU(num_parameters=512)
                (relu2): PReLU(num_parameters=512)
                (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (avgpool): AdaptiveAvgPool2d(output_size=1)
          )
        )
        (proj): Linear(in_features=512, out_features=1024, bias=True)
      )
      (post_extract_proj): Linear(in_features=2048, out_features=1024, bias=True)
      (dropout_input): Dropout(p=0.1, inplace=False)
      (dropout_features): Dropout(p=0.1, inplace=False)
      (encoder): AVHubertEncoder(
        (pos_conv_embed): AVWav2Vec2PositionalConvEmbedding(
          (conv): ParametrizedConv1d(
            1024, 1024, kernel_size=(128,), stride=(1,), padding=(64,), groups=16
            (parametrizations): ModuleDict(
              (weight): ParametrizationList(
                (0): _WeightNorm()
              )
            )
          )
          (padding): Wav2Vec2SamePadLayer()
          (activation): GELUActivation()
        )
        (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (layers): ModuleList(
          (0-23): 24 x AVHubertEncoderLayer(
            (attention): Wav2Vec2Attention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (dropout): Dropout(p=0.1, inplace=False)
            (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (feed_forward): AVWav2Vec2FeedForward(
              (intermediate_dropout): Dropout(p=0.0, inplace=False)
              (intermediate_dense): Linear(in_features=1024, out_features=4096, bias=True)
              (intermediate_act_fn): GELUActivation()
              (output_dense): Linear(in_features=4096, out_features=1024, bias=True)
              (output_dropout): Dropout(p=0.1, inplace=False)
            )
            (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
    )
    (decoder): AVTransformerDecoder(
      (embed_tokens): Embedding(1000, 1024, padding_idx=1)
      (embed_positions): Speech2TextSinusoidalPositionalEmbedding()
      (layers): ModuleList(
        (0-8): 9 x AVTransformerDecoderLayer(
          (self_attn): AVTransformerAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (activation_fn): ReLU()
          (self_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder_attn): AVTransformerAttention(
            (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
            (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (encoder_attn_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (fc1): Linear(in_features=1024, out_features=4096, bias=True)
          (fc2): Linear(in_features=4096, out_features=1024, bias=True)
          (final_layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
      (layer_norm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
    (lm_head): Linear(in_features=1024, out_features=1000, bias=False)
  )
  (lm_head): Linear(in_features=1024, out_features=1000, bias=False)
)
Tokenizer: Speech2TextTokenizer(name_or_path='nguyenvulebinh/AV-HuBERT-MuAViC-en', vocab_size=1000, model_max_length=1000000000000000019884624838656, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=True, added_tokens_decoder={
        0: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        1: AddedToken("<pad>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        3: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}
)
Config: AV2TextConfig {
  "_name_or_path": "/export/data1/data/binhnguyen/workspaces/AV-HuBERT-S2S/model-bin/muavic/en/hf",
  "activation_dropout": 0.0,
  "activation_function": "relu",
  "adapter_attn_dim": null,
  "adapter_kernel_size": 3,
  "adapter_stride": 2,
  "add_adapter": false,
  "apply_spec_augment": true,
  "architectures": [
    "AV2TextForConditionalGeneration"
  ],
  "attention_dropout": 0.1,
  "attn_implementation": null,
  "audio_dropout": 0.5,
  "audio_feat_dim": 104,
  "bos_token_id": 1,
  "classifier_proj_size": 256,
  "codevector_dim": 256,
  "contrastive_logits_temperature": 0.1,
  "conv_bias": false,
  "conv_channels": 1024,
  "conv_dim": [
    512,
    512,
    512,
    512,
    512,
    512,
    512
  ],
  "conv_kernel": [
    10,
    3,
    3,
    3,
    3,
    2,
    2
  ],
  "conv_kernel_sizes": [
    5,
    5
  ],
  "conv_stride": [
    5,
    2,
    2,
    2,
    2,
    2,
    2
  ],
  "ctc_loss_reduction": "sum",
  "ctc_zero_infinity": false,
  "d_model": 1024,
  "decoder_attention_heads": 8,
  "decoder_attn_implementation": "eager",
  "decoder_ffn_dim": 4096,
  "decoder_hidden_size": 1024,
  "decoder_layerdrop": 0.0,
  "decoder_layers": 9,
  "decoder_start_token_id": 2,
  "diversity_loss_weight": 0.1,
  "do_stable_layer_norm": false,
  "dropout": 0.1,
  "dropout_features": 0.1,
  "dropout_input": 0.1,
  "encoder_attention_heads": 16,
  "encoder_embed_dim": 1024,
  "encoder_ffn_dim": 2048,
  "encoder_hidden_size": 1024,
  "encoder_layerdrop": 0.0,
  "encoder_layers": 12,
  "eos_token_id": 2,
  "feat_extract_activation": "gelu",
  "feat_extract_norm": "group",
  "feat_proj_dropout": 0.1,
  "feat_quantizer_dropout": 0.0,
  "feature_grad_mult": 0.1,
  "final_dim": 256,
  "final_dropout": 0.0,
  "freeze_feat_extract_train": true,
  "hidden_act": "gelu",
  "hidden_dropout": 0.1,
  "init_std": 0.02,
  "initializer_range": 0.02,
  "input_channels": 1,
  "input_feat_per_channel": 80,
  "intermediate_size": 4096,
  "is_encoder_decoder": true,
  "label_rate": 25,
  "layer_norm_eps": 1e-05,
  "layerdrop": 0.0,
  "logit_temp": 0.1,
  "mask_channel_length": 10,
  "mask_channel_min_space": 1,
  "mask_channel_other": 0.0,
  "mask_channel_prob": 0.0,
  "mask_channel_selection": "static",
  "mask_feature_length": 10,
  "mask_feature_min_masks": 0,
  "mask_feature_prob": 0.0,
  "mask_length_audio": 10,
  "mask_length_image": 5,
  "mask_min_space": 1,
  "mask_other": 0.0,
  "mask_prob_audio": 0.8,
  "mask_prob_image": 0.3,
  "mask_selection": "static",
  "mask_time_length": 10,
  "mask_time_min_masks": 2,
  "mask_time_min_space": 1,
  "mask_time_other": 0.0,
  "mask_time_prob": 0.0,
  "mask_time_selection": "static",
  "masking_type": "input",
  "max_source_positions": 6000,
  "max_target_positions": 2048,
  "modality_dropout": 0.5,
  "modality_fuse": "concat",
  "model_type": "speech_to_text",
  "no_mask_channel_overlap": false,
  "no_mask_overlap": false,
  "no_mask_time_overlap": false,
  "num_adapter_layers": 3,
  "num_classes": 2004,
  "num_codevector_groups": 2,
  "num_codevectors_per_group": 320,
  "num_conv_layers": 2,
  "num_conv_pos_embedding_groups": 16,
  "num_conv_pos_embeddings": 128,
  "num_dictionaries": 1,
  "num_feat_extract_layers": 7,
  "num_hidden_layers": 24,
  "num_negatives": 100,
  "output_hidden_size": 1024,
  "pad_token_id": 1,
  "proj_codevector_dim": 256,
  "resnet_relu_type": "prelu",
  "resnet_weights": null,
  "sample_rate": 25,
  "scale_embedding": null,
  "selection_type": "same_seq",
  "sim_type": "cosine",
  "skip_masked": false,
  "skip_nomask": false,
  "sub_encoder_layers": 0,
  "target_glu": false,
  "tdnn_dilation": [
    1,
    2,
    3,
    1,
    1
  ],
  "tdnn_dim": [
    512,
    512,
    512,
    512,
    1500
  ],
  "tdnn_kernel": [
    5,
    3,
    3,
    1,
    1
  ],
  "torch_dtype": "float32",
  "transformers_version": "4.48.3",
  "untie_final_proj": true,
  "use_cache": true,
  "use_weighted_layer_sum": false,
  "vocab_size": 1000,
  "xvector_output_dim": 512
}
```