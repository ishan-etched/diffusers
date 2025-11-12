import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from diffusers.models.autoencoders.autoencoder_kl_mochi import AutoencoderKLMochi
from diffusers.pipelines.mochi.pipeline_mochi import MochiPipeline
from diffusers.utils.export_utils import export_to_video
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from transformers import T5EncoderModel, T5TokenizerFast

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


def _pick_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _ensure_size_is_valid(height: int, width: int) -> None:
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` must be divisible by 8 but received {height}x{width}.")


def _load_tokenizer() -> T5TokenizerFast:
    return T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")


def _load_text_encoder(device: torch.device) -> T5EncoderModel:
    model = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    return model.to(device)


def _build_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def _get_t5_prompt_embeds(
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompts: Union[str, Tuple[str, ...]],
    num_videos_per_prompt: int,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)
    tokenized = tokenizer(
        prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.bool().to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(len(prompts) * num_videos_per_prompt, seq_len, -1)

    attention_mask = attention_mask.view(len(prompts), -1)
    attention_mask = attention_mask.repeat(num_videos_per_prompt, 1)

    return prompt_embeds, attention_mask


def _encode_prompts(
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompt: str,
    negative_prompt: Optional[str],
    num_videos_per_prompt: int,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    do_classifier_free_guidance: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    prompt_embeds, prompt_attention_mask = _get_t5_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompts=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
    )

    negative_prompt_embeds = None
    negative_prompt_attention_mask = None

    if do_classifier_free_guidance:
        negative_prompt = negative_prompt or ""
        negative_prompt_embeds, negative_prompt_attention_mask = _get_t5_prompt_embeds(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompts=negative_prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )

    return (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    )


def _prepare_latents(
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    height = height // VAE_SPATIAL_SCALE_FACTOR
    width = width // VAE_SPATIAL_SCALE_FACTOR
    num_frames = (num_frames - 1) // VAE_TEMPORAL_SCALE_FACTOR + 1
    shape = (batch_size, num_channels_latents, num_frames, height, width)

    latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
    return latents.to(dtype=dtype)


def part1_prepare_inputs() -> Part1Artifacts:
    print("=== PART 1: Prompt encoding & latent sampling ===")
    settings = GenerationSettings(
        prompt="Campfire burning on a beach, Ultra high resolution 4k.",
        negative_prompt=None,
        num_frames=25,
        num_inference_steps=12,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        guidance_scale=4.5,
        fps=30,
        seed=30,
        output_path="output_video.mp4",
    )

    _ensure_size_is_valid(settings.height, settings.width)
    torch.manual_seed(settings.seed)

    device = _pick_device()
    tokenizer = _load_tokenizer()
    text_encoder = _load_text_encoder(device)

    do_cfg = settings.guidance_scale > 1.0
    (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
    ) = _encode_prompts(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=settings.prompt,
        negative_prompt=settings.negative_prompt,
        num_videos_per_prompt=NUM_VIDEOS_PER_PROMPT,
        max_sequence_length=TOKENIZER_MAX_LENGTH,
        device=device,
        dtype=PIPELINE_DTYPE,
        do_classifier_free_guidance=do_cfg,
    )

    text_encoder.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    generator = _build_generator(settings.seed, device)
    latents = _prepare_latents(
        batch_size=NUM_VIDEOS_PER_PROMPT,
        num_channels_latents=12,
        height=settings.height,
        width=settings.width,
        num_frames=settings.num_frames,
        dtype=PIPELINE_DTYPE,
        device=device,
        generator=generator,
    )

    artifacts = Part1Artifacts(
        settings=settings,
        prompt_embeds=prompt_embeds.to("cpu"),
        prompt_attention_mask=prompt_attention_mask.to("cpu") if prompt_attention_mask is not None else None,
        negative_prompt_embeds=negative_prompt_embeds.to("cpu") if negative_prompt_embeds is not None else None,
        negative_prompt_attention_mask=(
            negative_prompt_attention_mask.to("cpu") if negative_prompt_attention_mask is not None else None
        ),
        latents=latents.to("cpu"),
        num_videos_per_prompt=NUM_VIDEOS_PER_PROMPT,
        attention_kwargs=None,
    )

    del tokenizer
    del text_encoder
    return artifacts


def _get_pipeline_execution_device(pipe: MochiPipeline) -> torch.device:
    if hasattr(pipe, "_execution_device") and pipe._execution_device is not None:  # type: ignore[attr-defined]
        return pipe._execution_device  # type: ignore[attr-defined]
    return _pick_device()


def part2_run_transformer(artifacts: Part1Artifacts) -> Part2Artifacts:
    print("=== PART 2: Transformer denoising ===")

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
    latents = artifacts.latents.to(device=device, dtype=PIPELINE_DTYPE)

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
        num_videos_per_prompt=1,
        generator=None,
        latents=latents,
        attention_kwargs=artifacts.attention_kwargs,
        output_type="latent",
    )

    latents = result.frames
    if isinstance(latents, list):
        latents = latents[0]

    denoised = latents.to("cpu", dtype=PIPELINE_DTYPE)

    pipe = None  # free hooks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"    Finished denoising latents with shape: {tuple(denoised.shape)}")
    return Part2Artifacts(denoised_latents=denoised)


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
    device = _pick_device()
    latents = outputs.denoised_latents.to(device=device, dtype=PIPELINE_DTYPE)

    vae = _load_vae(device)
    video_processor = VideoProcessor(vae_scale_factor=VAE_SPATIAL_SCALE_FACTOR)

    has_latents_mean = hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None
    has_latents_std = hasattr(vae.config, "latents_std") and vae.config.latents_std is not None

    if has_latents_mean and has_latents_std:
        latents_mean = torch.tensor(vae.config.latents_mean).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(vae.config.latents_std).view(1, 12, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / vae.config.scaling_factor + latents_mean
    else:
        latents = latents / vae.config.scaling_factor

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


def main() -> None:
    try:
        part1_artifacts = part1_prepare_inputs()
        part2_artifacts = part2_run_transformer(part1_artifacts)
        part3_decode_and_render(part1_artifacts, part2_artifacts)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nGeneration interrupted by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
