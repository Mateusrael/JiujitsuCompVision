"""DINOv3 model identity and offline configuration; no ML imports required."""

DINOV3_WEIGHTS_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3_TRANSFORMERS_VERSION = "4.57.1"

# The released ViT-B/16 configuration in the Transformers implementation.
# Keep every architecture/preprocessing-relevant value explicit so checkpoint
# reconstruction does not request the model's configuration from the Hub.
# https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/dinov3
DINOV3_BACKBONE_CONFIG = {
    "patch_size": 16,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "hidden_act": "gelu",
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-5,
    "rope_theta": 100.0,
    "image_size": 224,
    "num_channels": 3,
    "query_bias": True,
    "key_bias": False,
    "value_bias": True,
    "proj_bias": True,
    "mlp_bias": True,
    "layerscale_value": 1.0,
    "drop_path_rate": 0.0,
    "use_gated_mlp": False,
    "num_register_tokens": 4,
    "pos_embed_shift": None,
    "pos_embed_jitter": None,
    "pos_embed_rescale": 2.0,
}
