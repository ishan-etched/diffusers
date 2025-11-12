import sys
import torch
from diffusers.pipelines.mochi.pipeline_mochi import MochiPipeline

try:  # pragma: no cover - allow running as a script
    from .utils import (
        MODEL_ID,
        PIPELINE_DTYPE,
        PIPELINE_VARIANT,
        DenoiseArtifacts,
        EncodeArtifacts,
        pick_device,
    )
except ImportError:  # pragma: no cover
    from utils import (  # type: ignore
        MODEL_ID,
        PIPELINE_DTYPE,
        PIPELINE_VARIANT,
        DenoiseArtifacts,
        EncodeArtifacts,
        pick_device,
    )


def _get_pipeline_execution_device(pipe: MochiPipeline) -> torch.device:
    device = getattr(pipe, "_execution_device", None)
    return device if device is not None else pick_device()


def run_denoise(artifacts: EncodeArtifacts) -> DenoiseArtifacts:
    print("=== DENOISE: Transformer inference ===")

    pipe = MochiPipeline.from_pretrained(
        MODEL_ID,
        variant=PIPELINE_VARIANT,
        torch_dtype=PIPELINE_DTYPE,
    )

    try:
        pipe.enable_model_cpu_offload()
    except RuntimeError as error:
        print(f"Error enabling model CPU offload: {error}")
        print(
            "\nYou probably need to switch to PyTorch GPU. "
            "Run the switch_to_torch_gpu.sh script in the parent mochi1 folder."
        )
        sys.exit(1)

    pipe.enable_vae_tiling()
    device = _get_pipeline_execution_device(pipe)

    prompt_embeds = artifacts.prompt_embeds.to(device=device, dtype=PIPELINE_DTYPE)
    prompt_attention_mask = (
        artifacts.prompt_attention_mask.to(device=device) if artifacts.prompt_attention_mask is not None else None
    )
    negative_prompt_embeds = (
        artifacts.negative_prompt_embeds.to(device=device, dtype=PIPELINE_DTYPE)
        if artifacts.negative_prompt_embeds is not None
        else None
    )
    negative_prompt_attention_mask = (
        artifacts.negative_prompt_attention_mask.to(device=device)
        if artifacts.negative_prompt_attention_mask is not None
        else None
    )
    latents = artifacts.latents.unsqueeze(0).to(device=device, dtype=PIPELINE_DTYPE)

    result = pipe(
        prompt=None,
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_attention_mask=negative_prompt_attention_mask,
        height=artifacts.settings.height,
        width=artifacts.settings.width,
        num_frames=artifacts.settings.num_frames,
        num_inference_steps=artifacts.settings.num_inference_steps,
        guidance_scale=artifacts.settings.guidance_scale,
        generator=None,
        latents=latents,
        output_type="latent",
    )

    latents = result.frames
    if isinstance(latents, list):
        if len(latents) != 1:
            raise ValueError(f"Expected a single latent tensor, but received {len(latents)} items.")
        latents = latents[0]
    if latents.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 latents, but received shape {tuple(latents.shape)}")

    denoised = latents.squeeze(0).to("cpu", dtype=PIPELINE_DTYPE)

    pipe = None  # free hooks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    Finished denoising latents with shape: {tuple(denoised.shape)}")
    return DenoiseArtifacts(settings=artifacts.settings, denoised_latents=denoised)
