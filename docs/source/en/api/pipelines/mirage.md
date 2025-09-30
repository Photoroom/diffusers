<!-- Copyright 2025 The HuggingFace Team. All rights reserved.
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
# limitations under the License. -->

# MiragePipeline

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
</div>

Mirage is a text-to-image diffusion model using a transformer-based architecture with flow matching for efficient high-quality image generation. The model uses T5Gemma as the text encoder and supports both Flux VAE (AutoencoderKL) and DC-AE (AutoencoderDC) for latent compression.

Key features:

- **Transformer Architecture**: Uses a modern transformer-based denoising model with attention mechanisms optimized for image generation
- **Flow Matching**: Employs flow matching with Euler discrete scheduling for efficient sampling
- **Flexible VAE Support**: Compatible with both Flux VAE (8x compression, 16 latent channels) and DC-AE (32x compression, 32 latent channels)
- **T5Gemma Text Encoder**: Uses Google's T5Gemma-2B-2B-UL2 model for text encoding with strong text-image alignment
- **Efficient Architecture**: ~1.3B parameters in the transformer, enabling fast inference while maintaining quality
- **Modular Design**: Text encoder and VAE weights are loaded from HuggingFace, keeping checkpoint sizes small

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

## Loading the Pipeline

Mirage checkpoints only store the transformer and scheduler weights locally. The VAE and text encoder are automatically loaded from HuggingFace during pipeline initialization:

```py
from diffusers import MiragePipeline

# Load pipeline - VAE and text encoder will be loaded from HuggingFace
pipe = MiragePipeline.from_pretrained("path/to/mirage_checkpoint")
pipe.to("cuda")

prompt = "A digital painting of a rusty, vintage tram on a sandy beach"
image = pipe(prompt, num_inference_steps=28, guidance_scale=4.0).images[0]
image.save("mirage_output.png")
```

### Manual Component Loading

You can also load components individually:

```py
import torch
from diffusers import MiragePipeline
from diffusers.models import AutoencoderKL, AutoencoderDC
from diffusers.models.transformers.transformer_mirage import MirageTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import T5GemmaModel, GemmaTokenizerFast

# Load transformer
transformer = MirageTransformer2DModel.from_pretrained(
    "path/to/checkpoint", subfolder="transformer"
)

# Load scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "path/to/checkpoint", subfolder="scheduler"
)

# Load T5Gemma text encoder
t5gemma_model = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2")
text_encoder = t5gemma_model.encoder
tokenizer = GemmaTokenizerFast.from_pretrained("google/t5gemma-2b-2b-ul2")

# Load VAE - choose either Flux VAE or DC-AE
# Flux VAE (16 latent channels):
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")
# Or DC-AE (32 latent channels):
# vae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers")

pipe = MiragePipeline(
    transformer=transformer,
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    vae=vae
)
pipe.to("cuda")
```

## VAE Variants

Mirage supports two VAE configurations:

### Flux VAE (AutoencoderKL)
- **Compression**: 8x spatial compression
- **Latent channels**: 16
- **Model**: `black-forest-labs/FLUX.1-dev` (subfolder: "vae")
- **Use case**: Balanced quality and speed

### DC-AE (AutoencoderDC)
- **Compression**: 32x spatial compression
- **Latent channels**: 32
- **Model**: `mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers`
- **Use case**: Higher compression for faster processing

The VAE type is automatically determined from the checkpoint's `model_index.json` configuration.

## Generation Parameters

Key parameters for image generation:

- **num_inference_steps**: Number of denoising steps (default: 28). More steps generally improve quality at the cost of speed.
- **guidance_scale**: Classifier-free guidance strength (default: 4.0). Higher values produce images more closely aligned with the prompt.
- **height/width**: Output image dimensions (default: 512x512). Can be customized in the checkpoint configuration.

```py
# Example with custom parameters
image = pipe(
    prompt="A serene mountain landscape at sunset",
    num_inference_steps=28,
    guidance_scale=4.0,
    height=1024,
    width=1024,
    generator=torch.Generator("cuda").manual_seed(42)
).images[0]
```

## Memory Optimization

For memory-constrained environments:

```py
import torch
from diffusers import MiragePipeline

pipe = MiragePipeline.from_pretrained("path/to/checkpoint", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()  # Offload components to CPU when not in use

# Or use sequential CPU offload for even lower memory
pipe.enable_sequential_cpu_offload()
```

## MiragePipeline

[[autodoc]] MiragePipeline
  - all
  - __call__

## MiragePipelineOutput

[[autodoc]] pipelines.mirage.pipeline_output.MiragePipelineOutput
