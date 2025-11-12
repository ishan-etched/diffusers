from contextlib import nullcontext

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
        Part1Artifacts,
        Part2Artifacts,
        pick_device,
    )
except ImportError:  # pragma: no cover
    from utils import (  # type: ignore
        MODEL_ID,
        PIPELINE_DTYPE,
        PIPELINE_VARIANT,
        VAE_SPATIAL_SCALE_FACTOR,
        Part1Artifacts,
        Part2Artifacts,
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


def part3_decode_and_render(inputs: Part1Artifacts, outputs: Part2Artifacts) -> None:
    print("=== PART 3: VAE decoding & video export ===")
    device = pick_device()
    latents = outputs.denoised_latents.to(device=device, dtype=PIPELINE_DTYPE)

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

    frames = video_processor.postprocess_video(video, output_type="pil")[0]
    export_to_video(frames, inputs.settings.output_path, fps=inputs.settings.fps)
    print(f"    Exported video to {inputs.settings.output_path}")
