from contextlib import nullcontext
from pathlib import Path

import torch
from diffusers.models.autoencoders.autoencoder_kl_mochi import AutoencoderKLMochi
from diffusers.utils.export_utils import export_to_video
from diffusers.video_processor import VideoProcessor

try:  # pragma: no cover - allow running as a script
    from .utils import (
        MODEL_ID,
        PIPELINE_DTYPE,
        PIPELINE_VARIANT,
        VAE_SPATIAL_SCALE_FACTOR,
        DenoiseArtifacts,
        pick_device,
    )
except ImportError:  # pragma: no cover
    from utils import (  # type: ignore
        MODEL_ID,
        PIPELINE_DTYPE,
        PIPELINE_VARIANT,
        VAE_SPATIAL_SCALE_FACTOR,
        DenoiseArtifacts,
        pick_device,
    )


def _load_vae(device: torch.device) -> AutoencoderKLMochi:
    vae = AutoencoderKLMochi.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        variant=PIPELINE_VARIANT,
        torch_dtype=PIPELINE_DTYPE,
    )
    vae.enable_tiling()
    return vae.to(device)


def run_decode(artifacts: DenoiseArtifacts, output_path: Path) -> None:
    print("=== DECODE: VAE decoding & video export ===")
    device = pick_device()
    latents = artifacts.denoised_latents.unsqueeze(0).to(device=device, dtype=PIPELINE_DTYPE)

    vae = _load_vae(device)
    video_processor = VideoProcessor(vae_scale_factor=VAE_SPATIAL_SCALE_FACTOR)

    assert hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
    assert hasattr(vae.config, "latents_std") and vae.config.latents_std is not None

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = torch.tensor(vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
    latents = latents * latents_std / vae.config.scaling_factor + latents_mean

    device_type = device.type
    autocast_context = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16, cache_enabled=False)
        if device_type == "cuda"
        else nullcontext()
    )

    with torch.no_grad(), autocast_context:
        video = vae.decode(latents, return_dict=False)[0]

    frames_list = video_processor.postprocess_video(video, output_type="pil")
    if len(frames_list) != 1:
        raise ValueError(f"Expected a single decoded video, but received {len(frames_list)} items.")
    frames = frames_list[0]
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output_path), fps=artifacts.settings.fps)
    print(f"    Exported video to {output_path}")
