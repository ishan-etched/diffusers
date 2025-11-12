from dataclasses import dataclass
from pathlib import Path
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
SNAPSHOT_DIR = Path("artifacts")
DEFAULT_ENCODE_SNAPSHOT = SNAPSHOT_DIR / "encode_artifacts.pt"
DEFAULT_DENOISE_SNAPSHOT = SNAPSHOT_DIR / "denoise_artifacts.pt"


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


@dataclass(frozen=True)
class EncodeArtifacts:
    settings: GenerationSettings
    prompt_embeds: torch.Tensor
    prompt_attention_mask: Optional[torch.Tensor]
    negative_prompt_embeds: Optional[torch.Tensor]
    negative_prompt_attention_mask: Optional[torch.Tensor]
    latents: torch.Tensor
    num_videos_per_prompt: int
    attention_kwargs: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class DenoiseArtifacts:
    settings: GenerationSettings
    denoised_latents: torch.Tensor


def pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _save_snapshot(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def save_encode_snapshot(artifacts: EncodeArtifacts, path: Path = DEFAULT_ENCODE_SNAPSHOT) -> Path:
    _save_snapshot(artifacts, path)
    return path


def save_denoise_snapshot(artifacts: DenoiseArtifacts, path: Path = DEFAULT_DENOISE_SNAPSHOT) -> Path:
    _save_snapshot(artifacts, path)
    return path


def load_encode_snapshot(path: Path = DEFAULT_ENCODE_SNAPSHOT) -> EncodeArtifacts:
    artifacts = torch.load(path, map_location="cpu")
    if not isinstance(artifacts, EncodeArtifacts):
        raise TypeError(f"Snapshot at {path} is not a EncodeArtifacts instance.")
    return artifacts


def load_denoise_snapshot(path: Path = DEFAULT_DENOISE_SNAPSHOT) -> DenoiseArtifacts:
    artifacts = torch.load(path, map_location="cpu")
    if not isinstance(artifacts, DenoiseArtifacts):
        raise TypeError(f"Snapshot at {path} is not a DenoiseArtifacts instance.")
    return artifacts
