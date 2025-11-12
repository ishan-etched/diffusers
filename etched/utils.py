from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

MODEL_ID = "genmo/mochi-1-preview"
PIPELINE_VARIANT = "bf16"
PIPELINE_DTYPE = torch.bfloat16
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 848
VAE_SPATIAL_SCALE_FACTOR = 8
VAE_TEMPORAL_SCALE_FACTOR = 6
TOKENIZER_MAX_LENGTH = 256
NUM_VIDEOS_PER_PROMPT = 1


@dataclass(frozen=True)
class GenerationSettings:
    prompt: str
    negative_prompt: Optional[str]
    num_frames: int
    num_inference_steps: int
    height: int
    width: int
    guidance_scale: float
    fps: int
    seed: int
    output_path: str


@dataclass(frozen=True)
class Part1Artifacts:
    settings: GenerationSettings
    prompt_embeds: torch.Tensor
    prompt_attention_mask: Optional[torch.Tensor]
    negative_prompt_embeds: Optional[torch.Tensor]
    negative_prompt_attention_mask: Optional[torch.Tensor]
    latents: torch.Tensor
    num_videos_per_prompt: int
    attention_kwargs: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class Part2Artifacts:
    denoised_latents: torch.Tensor


def pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
