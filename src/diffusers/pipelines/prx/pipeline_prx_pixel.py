# Copyright 2025 The Photoroom and The HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from diffusers.models import AutoencoderDC, AutoencoderKL
from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.pipelines.prx.pipeline_prx import PRXPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# PRXPixel is a 1024px model.
PRX_PIXEL_DEFAULT_RESOLUTION = 1024
# Number of text tokens used at training time (the Qwen tokenizer's own ``model_max_length`` is far larger).
PRX_PIXEL_DEFAULT_MAX_TOKENS = 256


class PRXPixelPipeline(PRXPipeline):
    r"""
    Pipeline for text-to-image generation with the PRXPixel model.

    PRXPixel is a pixel-space variant of [`PRXPipeline`]: it denoises raw RGB directly (the VAE is an identity /
    absent component), conditions on a Qwen3-VL text encoder rather than T5Gemma, and feeds the latent resolution
    into the timestep modulation (`resolution_embeds=True` on the [`PRXTransformer2DModel`]). The denoising loop,
    prompt encoding, latent preparation and CFG handling are all inherited from [`PRXPipeline`]; only the component
    types, the text-token budget, the (lighter) prompt cleaning, and the default resolution differ.

    This pipeline inherits from [`PRXPipeline`]. Check the superclass documentation for the generic methods (text
    encoding, latent preparation, the `__call__` signature, ...).

    Args:
        transformer ([`PRXTransformer2DModel`]):
            The PRX denoiser. For PRXPixel this is built with `in_channels=3`, a bottleneck `img_in`, and
            `resolution_embeds=True`.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler used to denoise the (pixel-space) latents.
        text_encoder ([`PreTrainedModel`]):
            The Qwen3-VL text backbone used to encode prompts (the vision tower is discarded). Must return a
            `last_hidden_state`.
        tokenizer ([`PreTrainedTokenizerBase`]):
            Tokenizer for `text_encoder` (typically loaded via `AutoTokenizer`).
        vae ([`AutoencoderKL`] or [`AutoencoderDC`], *optional*):
            Optional VAE. PRXPixel operates in pixel space, so this is usually `None` (an identity VAE).
        default_sample_size (`int`, *optional*, defaults to 1024):
            Default height/width used when none is provided to `__call__`.
        prompt_max_tokens (`int`, *optional*, defaults to 256):
            Number of text tokens the prompt is padded/truncated to before encoding.
    """

    def __init__(
        self,
        transformer: PRXTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer | PreTrainedTokenizerBase,
        vae: AutoencoderKL | AutoencoderDC | None = None,
        default_sample_size: int | None = PRX_PIXEL_DEFAULT_RESOLUTION,
        prompt_max_tokens: int = PRX_PIXEL_DEFAULT_MAX_TOKENS,
        noise_scale: float = 2.0,
    ):
        super().__init__(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            default_sample_size=default_sample_size,
        )
        # Pin the text-token budget; the Qwen tokenizer's model_max_length is otherwise far too large.
        self.tokenizer_max_length = prompt_max_tokens
        # The Qwen3-VL embedding tower was trained without the DeepFloyd cleaning; use light cleaning only.
        self.skip_text_cleaning = True
        # PRXPixel predicts the clean sample x0 (converted to velocity each step), not the velocity directly.
        self.prediction_type = "x_prediction_flow_matching"
        # PRXPixel trains with a non-unit initial-noise scale; sampling must start from randn * noise_scale.
        self.noise_scale = noise_scale

    @property
    def vae_scale_factor(self):
        # PRXPixel operates directly in RGB pixel space (identity / no VAE): no spatial compression.
        if self.vae is None:
            return 1
        return super().vae_scale_factor
