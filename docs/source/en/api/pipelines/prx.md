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

# PRX


PRX is a family of efficient text-to-image diffusion models by Photoroom. The flagship model, **PRXPixel** ([`PRXPixelPipeline`]), generates 1024px images directly in pixel space: a ~7B transformer denoises raw RGB without any VAE, conditioned on a Qwen3-VL text encoder, and feeds the generation resolution into the timestep modulation. It uses flow matching and predicts the clean image at each step (x-prediction).

Earlier PRX versions ([`PRXPipeline`]) operate in a VAE latent space (Flux VAE with 8x compression, or DC-AE with 32x compression) with a ~1.3B simplified MMDIT transformer where text tokens don't update through the blocks, and Google's T5Gemma-2B-2B-UL2 model for text encoding.

## Available models

**PRXPixel is the flagship model.** The other checkpoints are earlier latent-space versions of PRX at 256/512px with different VAE configurations; the distilled variants generate in 8 steps.

| Model | Resolution | Fine-tuned | Distilled | Description | Suggested prompts | Suggested parameters | Recommended dtype |
|:-----:|:-----------------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| [`Photoroom/prxpixel-t2i`](https://huggingface.co/Photoroom/prxpixel-t2i)| 1024 | No | No | Flagship pixel-space model (~7B transformer, no VAE) with a Qwen3-VL text encoder, loaded with [`PRXPixelPipeline`] | Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-256-t2i`](https://huggingface.co/Photoroom/prx-256-t2i)| 256 | No | No | Base model pre-trained at 256 with Flux VAE|Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-256-t2i-sft`](https://huggingface.co/Photoroom/prx-256-t2i-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with Flux VAE | Can handle less detailed prompts|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i`](https://huggingface.co/Photoroom/prx-512-t2i)| 512 | No | No | Base model pre-trained at 512 with Flux VAE |Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-sft`](https://huggingface.co/Photoroom/prx-512-t2i-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with Flux VAE | Can handle less detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-sft-distilled)| 512 | Yes | Yes | 8-step distilled model from [`Photoroom/prx-512-t2i-sft`](https://huggingface.co/Photoroom/prx-512-t2i-sft) | Can handle less detailed prompts in natural language|8 steps, cfg=1.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae)| 512 | No | No | Base model pre-trained at 512 with [Deep Compression Autoencoder (DC-AE)](https://hanlab.mit.edu/projects/dc-ae)|Works best with detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae-sft`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft)| 512 | Yes | No | Fine-tuned on the [Alchemist dataset](https://huggingface.co/datasets/yandex/alchemist) dataset with [Deep Compression Autoencoder (DC-AE)](https://hanlab.mit.edu/projects/dc-ae) | Can handle less detailed prompts in natural language|28 steps, cfg=5.0| `torch.bfloat16` |
| [`Photoroom/prx-512-t2i-dc-ae-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft-distilled)| 512 | Yes | Yes | 8-step distilled model from [`Photoroom/prx-512-t2i-dc-ae-sft-distilled`](https://huggingface.co/Photoroom/prx-512-t2i-dc-ae-sft-distilled) | Can handle less detailed prompts in natural language|8 steps, cfg=1.0| `torch.bfloat16` |

Refer to [this](https://huggingface.co/collections/Photoroom/prx-models-68e66254c202ebfab99ad38e) collection for more information.

## Loading the pipeline

Load the pipeline with [`~DiffusionPipeline.from_pretrained`]. [`PRXPixelPipeline`] denoises raw RGB directly, so no VAE is loaded or needed. It requires `transformers >= 4.57` (the version that introduced `Qwen3VLTextModel`).

```py
import torch
from diffusers import PRXPixelPipeline

pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A front-facing portrait of a lion in the golden savanna at sunset."
image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
image.save("prxpixel_output.png")
```

### Latent-space generation (earlier PRX versions)

Use [`PRXPipeline`] for the earlier latent-space checkpoints; the VAE and text encoder are loaded as part of the pipeline.

```py
import torch
from diffusers import PRXPipeline

pipe = PRXPipeline.from_pretrained("Photoroom/prx-512-t2i-sft", torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A front-facing portrait of a lion in the golden savanna at sunset."
image = pipe(prompt, num_inference_steps=28, guidance_scale=5.0).images[0]
image.save("prx_output.png")
```

### Manual Component Loading

Load components individually to customize the pipeline, for instance to use quantized models (shown here for the latent-space [`PRXPipeline`]).

```py
import torch
from diffusers.pipelines.prx import PRXPipeline
from diffusers.models import AutoencoderKL, AutoencoderDC
from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import T5GemmaModel, GemmaTokenizerFast
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as BitsAndBytesConfig

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
# Load transformer
transformer = PRXTransformer2DModel.from_pretrained(
    "checkpoints/prx-512-t2i-sft",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

# Load scheduler
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    "checkpoints/prx-512-t2i-sft", subfolder="scheduler"
)

# Load T5Gemma text encoder
t5gemma_model = T5GemmaModel.from_pretrained("google/t5gemma-2b-2b-ul2",
                                            quantization_config=quant_config,
                                            torch_dtype=torch.bfloat16)
text_encoder = t5gemma_model.encoder.to(dtype=torch.bfloat16)
tokenizer = GemmaTokenizerFast.from_pretrained("google/t5gemma-2b-2b-ul2")
tokenizer.model_max_length = 256

# Load VAE - choose either Flux VAE or DC-AE
# Flux VAE
vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev",
                                    subfolder="vae",
                                    quantization_config=quant_config,
                                    torch_dtype=torch.bfloat16)

pipe = PRXPipeline(
    transformer=transformer,
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    vae=vae
)
pipe.to("cuda")
```


## Memory Optimization

For memory-constrained environments:

```py
import torch
from diffusers import PRXPixelPipeline

pipe = PRXPixelPipeline.from_pretrained("Photoroom/prxpixel-t2i", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()  # Offload components to CPU when not in use

# Or use sequential CPU offload for even lower memory
pipe.enable_sequential_cpu_offload()
```

## PRXPixelPipeline

[[autodoc]] PRXPixelPipeline
  - all
  - __call__

## PRXPipeline

[[autodoc]] PRXPipeline
  - all
  - __call__

## PRXPipelineOutput

[[autodoc]] pipelines.prx.pipeline_output.PRXPipelineOutput
