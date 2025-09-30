#!/usr/bin/env python3
"""
Script to convert Mirage checkpoint from original codebase to diffusers format.
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file
from dataclasses import dataclass, asdict
from typing import Tuple, Dict


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusers.models.transformers.transformer_mirage import MirageTransformer2DModel
from diffusers.pipelines.mirage import MiragePipeline

@dataclass(frozen=True)
class MirageBase:
    context_in_dim: int = 2304
    hidden_size: int = 1792
    mlp_ratio: float = 3.5
    num_heads: int = 28
    depth: int = 16
    axes_dim: Tuple[int, int] = (32, 32)
    theta: int = 10_000
    time_factor: float = 1000.0
    time_max_period: int = 10_000


@dataclass(frozen=True)
class MirageFlux(MirageBase):
    in_channels: int = 16
    patch_size: int = 2


@dataclass(frozen=True)
class MirageDCAE(MirageBase):
    in_channels: int = 32
    patch_size: int = 1


def build_config(vae_type: str) -> dict:
    if vae_type == "flux":
        cfg = MirageFlux()
    elif vae_type == "dc-ae":
        cfg = MirageDCAE()
    else:
        raise ValueError(f"Unsupported VAE type: {vae_type}. Use 'flux' or 'dc-ae'")

    config_dict = asdict(cfg)
    config_dict["axes_dim"] = list(config_dict["axes_dim"])  # type: ignore[index]
    return config_dict



def create_parameter_mapping(depth: int) -> dict:
    """Create mapping from old parameter names to new diffusers names."""

    # Key mappings for structural changes
    mapping = {}

    # RMSNorm: scale -> weight
    for i in range(depth):
        mapping[f"blocks.{i}.qk_norm.query_norm.scale"] = f"blocks.{i}.qk_norm.query_norm.weight"
        mapping[f"blocks.{i}.qk_norm.key_norm.scale"] = f"blocks.{i}.qk_norm.key_norm.weight"
        mapping[f"blocks.{i}.k_norm.scale"] = f"blocks.{i}.k_norm.weight"

        # Attention: attn_out -> attention.to_out.0
        mapping[f"blocks.{i}.attn_out.weight"] = f"blocks.{i}.attention.to_out.0.weight"

    return mapping


def convert_checkpoint_parameters(old_state_dict: Dict[str, torch.Tensor], depth: int) -> Dict[str, torch.Tensor]:
    """Convert old checkpoint parameters to new diffusers format."""

    print("Converting checkpoint parameters...")

    mapping = create_parameter_mapping(depth)
    converted_state_dict = {}

    # First, print available keys to understand structure
    print("Available keys in checkpoint:")
    for key in sorted(old_state_dict.keys())[:10]:  # Show first 10 keys
        print(f"  {key}")
    if len(old_state_dict) > 10:
        print(f"  ... and {len(old_state_dict) - 10} more")

    for key, value in old_state_dict.items():
        new_key = key

        # Apply specific mappings if needed
        if key in mapping:
            new_key = mapping[key]
            print(f"  Mapped: {key} -> {new_key}")

        # Handle img_qkv_proj -> split to to_q, to_k, to_v
        if "img_qkv_proj.weight" in key:
            print(f"  Found QKV projection: {key}")
            # Split QKV weight into separate Q, K, V projections
            qkv_weight = value
            q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)

            # Extract layer number from key (e.g., blocks.0.img_qkv_proj.weight -> 0)
            parts = key.split(".")
            layer_idx = None
            for i, part in enumerate(parts):
                if part == "blocks" and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = parts[i + 1]
                    break

            if layer_idx is not None:
                converted_state_dict[f"blocks.{layer_idx}.attention.to_q.weight"] = q_weight
                converted_state_dict[f"blocks.{layer_idx}.attention.to_k.weight"] = k_weight
                converted_state_dict[f"blocks.{layer_idx}.attention.to_v.weight"] = v_weight
                print(f"  Split QKV for layer {layer_idx}")

                # Also keep the original img_qkv_proj for backward compatibility
                converted_state_dict[new_key] = value
        else:
            converted_state_dict[new_key] = value

    print(f"✓ Converted {len(converted_state_dict)} parameters")
    return converted_state_dict


def create_transformer_from_checkpoint(checkpoint_path: str, config: dict) -> MirageTransformer2DModel:
    """Create and load MirageTransformer2DModel from old checkpoint."""

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load old checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    old_checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if isinstance(old_checkpoint, dict):
        if "model" in old_checkpoint:
            state_dict = old_checkpoint["model"]
        elif "state_dict" in old_checkpoint:
            state_dict = old_checkpoint["state_dict"]
        else:
            state_dict = old_checkpoint
    else:
        state_dict = old_checkpoint

    print(f"✓ Loaded checkpoint with {len(state_dict)} parameters")

    # Convert parameter names if needed
    model_depth = int(config.get("depth", 16))
    converted_state_dict = convert_checkpoint_parameters(state_dict, depth=model_depth)

    # Create transformer with config
    print("Creating MirageTransformer2DModel...")
    transformer = MirageTransformer2DModel(**config)

    # Load state dict
    print("Loading converted parameters...")
    missing_keys, unexpected_keys = transformer.load_state_dict(converted_state_dict, strict=False)

    if missing_keys:
        print(f"⚠ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {unexpected_keys}")

    if not missing_keys and not unexpected_keys:
        print("✓ All parameters loaded successfully!")

    return transformer




def create_scheduler_config(output_path: str):
    """Create FlowMatchEulerDiscreteScheduler config."""

    scheduler_config = {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": 1000,
        "shift": 1.0
    }

    scheduler_path = os.path.join(output_path, "scheduler")
    os.makedirs(scheduler_path, exist_ok=True)

    with open(os.path.join(scheduler_path, "scheduler_config.json"), "w") as f:
        json.dump(scheduler_config, f, indent=2)

    print("✓ Created scheduler config")


def create_vae_config(vae_type: str, output_path: str):
    """Create VAE config based on type."""

    if vae_type == "flux":
        vae_config = {
            "_class_name": "AutoencoderKL",
            "latent_channels": 16,
            "block_out_channels": [128, 256, 512, 512],
            "down_block_types": [
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D",
                "DownEncoderBlock2D"
            ],
            "up_block_types": [
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D",
                "UpDecoderBlock2D"
            ],
            "scaling_factor": 0.3611,
            "shift_factor": 0.1159,
            "use_post_quant_conv": False,
            "use_quant_conv": False
        }
    else:  # dc-ae
        vae_config = {
            "_class_name": "AutoencoderDC",
            "latent_channels": 32,
            "encoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "decoder_block_out_channels": [128, 256, 512, 512, 1024, 1024],
            "encoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock"
            ],
            "decoder_block_types": [
                "ResBlock",
                "ResBlock",
                "ResBlock",
                "EfficientViTBlock",
                "EfficientViTBlock",
                "EfficientViTBlock"
            ],
            "encoder_layers_per_block": [2, 2, 2, 3, 3, 3],
            "decoder_layers_per_block": [3, 3, 3, 3, 3, 3],
            "encoder_qkv_multiscales": [[], [], [], [5], [5], [5]],
            "decoder_qkv_multiscales": [[], [], [], [5], [5], [5]],
            "scaling_factor": 0.41407,
            "upsample_block_type": "interpolate"
        }

    vae_path = os.path.join(output_path, "vae")
    os.makedirs(vae_path, exist_ok=True)

    with open(os.path.join(vae_path, "config.json"), "w") as f:
        json.dump(vae_config, f, indent=2)

    print("✓ Created VAE config")


def create_text_encoder_config(output_path: str):
    """Create T5GemmaEncoder config."""

    text_encoder_config = {
        "model_name": "google/t5gemma-2b-2b-ul2",
        "model_max_length": 256,
        "use_attn_mask": True,
        "use_last_hidden_state": True
    }

    text_encoder_path = os.path.join(output_path, "text_encoder")
    os.makedirs(text_encoder_path, exist_ok=True)

    with open(os.path.join(text_encoder_path, "config.json"), "w") as f:
        json.dump(text_encoder_config, f, indent=2)

    print("✓ Created text encoder config")


def create_tokenizer_config(output_path: str):
    """Create GemmaTokenizerFast config and files."""

    tokenizer_config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "added_tokens_decoder": {
            "0": {"content": "<pad>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "1": {"content": "<eos>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "2": {"content": "<bos>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "3": {"content": "<unk>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "106": {"content": "<start_of_turn>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True},
            "107": {"content": "<end_of_turn>", "lstrip": False, "normalized": False, "rstrip": False, "single_word": False, "special": True}
        },
        "additional_special_tokens": ["<start_of_turn>", "<end_of_turn>"],
        "bos_token": "<bos>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<eos>",
        "extra_special_tokens": {},
        "model_max_length": 256,
        "pad_token": "<pad>",
        "padding_side": "right",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "GemmaTokenizer",
        "unk_token": "<unk>",
        "use_default_system_prompt": False
    }

    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>"
    }

    tokenizer_path = os.path.join(output_path, "tokenizer")
    os.makedirs(tokenizer_path, exist_ok=True)

    with open(os.path.join(tokenizer_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    with open(os.path.join(tokenizer_path, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2)

    print("✓ Created tokenizer config (Note: tokenizer.json and tokenizer.model files need to be provided separately)")


def create_pipeline_components(vae_type: str, output_path: str):
    """Create all pipeline components with proper configs."""

    create_scheduler_config(output_path)
    create_vae_config(vae_type, output_path)
    create_text_encoder_config(output_path)
    create_tokenizer_config(output_path)


def create_model_index(vae_type: str, output_path: str):
    """Create model_index.json for the pipeline."""

    if vae_type == "flux":
        vae_class = "AutoencoderKL"
    else:  # dc-ae
        vae_class = "AutoencoderDC"

    model_index = {
        "_class_name": "MiragePipeline",
        "_diffusers_version": "0.31.0.dev0",
        "_name_or_path": os.path.basename(output_path),
        "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        "text_encoder": ["transformers", "T5GemmaEncoder"],
        "tokenizer": ["transformers", "GemmaTokenizerFast"],
        "transformer": ["diffusers", "MirageTransformer2DModel"],
        "vae": ["diffusers", vae_class],
    }

    model_index_path = os.path.join(output_path, "model_index.json")
    with open(model_index_path, "w") as f:
        json.dump(model_index, f, indent=2)

    print("✓ Created model_index.json")


def main(args):
    # Validate inputs
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    config = build_config(args.vae_type)

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    print(f"✓ Output directory: {args.output_path}")

    # Create transformer from checkpoint
    transformer = create_transformer_from_checkpoint(args.checkpoint_path, config)

    # Save transformer
    transformer_path = os.path.join(args.output_path, "transformer")
    os.makedirs(transformer_path, exist_ok=True)

    # Save config
    with open(os.path.join(transformer_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save model weights as safetensors
    state_dict = transformer.state_dict()
    save_file(state_dict, os.path.join(transformer_path, "diffusion_pytorch_model.safetensors"))
    print(f"✓ Saved transformer to {transformer_path}")

    # Create other pipeline components
    create_pipeline_components(args.vae_type, args.output_path)

    # Create model index
    create_model_index(args.vae_type, args.output_path)

    # Verify the pipeline can be loaded
    try:
        pipeline = MiragePipeline.from_pretrained(args.output_path)
        print("Pipeline loaded successfully!")
        print(f"Transformer: {type(pipeline.transformer).__name__}")
        print(f"VAE: {type(pipeline.vae).__name__}")
        print(f"Text Encoder: {type(pipeline.text_encoder).__name__}")
        print(f"Scheduler: {type(pipeline.scheduler).__name__}")

        # Display model info
        num_params = sum(p.numel() for p in pipeline.transformer.parameters())
        print(f"✓ Transformer parameters: {num_params:,}")

    except Exception as e:
        print(f"Pipeline verification failed: {e}")
        return False

    print("Conversion completed successfully!")
    print(f"Converted pipeline saved to: {args.output_path}")
    print(f"VAE type: {args.vae_type}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mirage checkpoint to diffusers format")

    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the original Mirage checkpoint (.pth file)"
    )

    parser.add_argument(
        "--output_path", type=str, required=True, help="Output directory for the converted diffusers pipeline"
    )

    parser.add_argument(
        "--vae_type",
        type=str,
        choices=["flux", "dc-ae"],
        required=True,
        help="VAE type to use: 'flux' for AutoencoderKL (16 channels) or 'dc-ae' for AutoencoderDC (32 channels)",
    )

    args = parser.parse_args()

    try:
        success = main(args)
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
