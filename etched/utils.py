from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

MODEL_ID = "genmo/mochi-1-preview"
PIPELINE_VARIANT = "bf16"
PIPELINE_DTYPE = torch.bfloat16
DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 848
VAE_SPATIAL_SCALE_FACTOR = 8
VAE_TEMPORAL_SCALE_FACTOR = 6
TOKENIZER_MAX_LENGTH = 256
SNAPSHOT_DIR = Path("artifacts")
DEFAULT_ENCODE_SNAPSHOT = SNAPSHOT_DIR / "encode_artifacts.pt"
DEFAULT_DENOISE_SNAPSHOT = SNAPSHOT_DIR / "denoise_artifacts.pt"
DEFAULT_DENOISE_NPZ = SNAPSHOT_DIR / "denoise_artifacts.npz"
DEFAULT_VAE_WEIGHTS = SNAPSHOT_DIR / "mochi_decoder_weights.npz"


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


def _save_denoise_npz(artifacts: DenoiseArtifacts, path: Path) -> None:
    arr = artifacts.denoised_latents.detach().to(torch.float32).cpu().numpy()
    settings = artifacts.settings
    data = dict(
        denoised_latents=arr,
        prompt=np.array(settings.prompt),
        negative_prompt=np.array(settings.negative_prompt or ""),
        negative_prompt_is_none=np.array(settings.negative_prompt is None),
        num_frames=np.array(settings.num_frames, dtype=np.int32),
        num_inference_steps=np.array(settings.num_inference_steps, dtype=np.int32),
        height=np.array(settings.height, dtype=np.int32),
        width=np.array(settings.width, dtype=np.int32),
        guidance_scale=np.array(settings.guidance_scale, dtype=np.float32),
        fps=np.array(settings.fps, dtype=np.int32),
        seed=np.array(settings.seed, dtype=np.int32),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def save_denoise_snapshot(artifacts: DenoiseArtifacts, path: Path = DEFAULT_DENOISE_SNAPSHOT) -> Path:
    _save_snapshot(artifacts, path)
    npz_path = path.with_suffix(".npz")
    _save_denoise_npz(artifacts, npz_path)
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


@dataclass(frozen=True)
class DenoiseNpzPayload:
    latents: np.ndarray
    settings: GenerationSettings


def load_denoise_npz(path: Path = DEFAULT_DENOISE_NPZ) -> DenoiseNpzPayload:
    path = path.with_suffix(".npz")
    with np.load(path, allow_pickle=True) as data:
        latents = data["denoised_latents"].astype(np.float32)
        prompt = data["prompt"].item() if hasattr(data["prompt"], "item") else str(data["prompt"])
        neg_prompt_raw = data["negative_prompt"]
        negative_prompt = neg_prompt_raw.item() if hasattr(neg_prompt_raw, "item") else str(neg_prompt_raw)
        if bool(data["negative_prompt_is_none"].item() if hasattr(data["negative_prompt_is_none"], "item") else data["negative_prompt_is_none"]):
            negative_prompt = None
        settings = GenerationSettings(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=int(data["num_frames"].item() if hasattr(data["num_frames"], "item") else data["num_frames"]),
            num_inference_steps=int(
                data["num_inference_steps"].item()
                if hasattr(data["num_inference_steps"], "item")
                else data["num_inference_steps"]
            ),
            height=int(data["height"].item() if hasattr(data["height"], "item") else data["height"]),
            width=int(data["width"].item() if hasattr(data["width"], "item") else data["width"]),
            guidance_scale=float(
                data["guidance_scale"].item() if hasattr(data["guidance_scale"], "item") else data["guidance_scale"]
            ),
            fps=int(data["fps"].item() if hasattr(data["fps"], "item") else data["fps"]),
            seed=int(data["seed"].item() if hasattr(data["seed"], "item") else data["seed"]),
        )
    return DenoiseNpzPayload(latents=latents, settings=settings)
