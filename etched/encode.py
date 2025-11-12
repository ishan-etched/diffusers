from typing import Optional, Tuple

import torch
from diffusers.utils.torch_utils import randn_tensor
from transformers import T5EncoderModel, T5TokenizerFast

try:  # pragma: no cover - allow running as a script
    from .utils import (
        DEFAULT_HEIGHT,
        DEFAULT_WIDTH,
        MODEL_ID,
        PIPELINE_DTYPE,
        TOKENIZER_MAX_LENGTH,
        VAE_SPATIAL_SCALE_FACTOR,
        VAE_TEMPORAL_SCALE_FACTOR,
        EncodeArtifacts,
        GenerationSettings,
        pick_device,
    )
except ImportError:  # pragma: no cover
    from utils import (  # type: ignore
        DEFAULT_HEIGHT,
        DEFAULT_WIDTH,
        MODEL_ID,
        PIPELINE_DTYPE,
        TOKENIZER_MAX_LENGTH,
        VAE_SPATIAL_SCALE_FACTOR,
        VAE_TEMPORAL_SCALE_FACTOR,
        EncodeArtifacts,
        GenerationSettings,
        pick_device,
    )


def _ensure_size_is_valid(height: int, width: int) -> None:
    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` must be divisible by 8 but received {height}x{width}.")


def _load_tokenizer() -> T5TokenizerFast:
    return T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer")


def _load_text_encoder(device: torch.device) -> T5EncoderModel:
    encoder = T5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder")
    return encoder.to(device)


def _build_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)
    return generator


def _get_t5_prompt_embeds(
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompt: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenized = tokenizer(
        [prompt],
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

    return prompt_embeds, attention_mask


def _encode_prompts(
    tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompt: str,
    negative_prompt: Optional[str],
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    do_classifier_free_guidance: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    prompt_embeds, prompt_attention_mask = _get_t5_prompt_embeds(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
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
            prompt=negative_prompt,
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
    shape = (num_channels_latents, num_frames, height, width)

    latents = randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
    return latents.to(dtype=dtype)


def run_encode() -> EncodeArtifacts:
    print("=== ENCODE: Prompt encoding & latent sampling ===")
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
    )

    _ensure_size_is_valid(settings.height, settings.width)
    torch.manual_seed(settings.seed)

    device = pick_device()
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
        num_channels_latents=12,
        height=settings.height,
        width=settings.width,
        num_frames=settings.num_frames,
        dtype=PIPELINE_DTYPE,
        device=device,
        generator=generator,
    )

    artifacts = EncodeArtifacts(
        settings=settings,
        prompt_embeds=prompt_embeds.to("cpu"),
        prompt_attention_mask=prompt_attention_mask.to("cpu") if prompt_attention_mask is not None else None,
        negative_prompt_embeds=negative_prompt_embeds.to("cpu") if negative_prompt_embeds is not None else None,
        negative_prompt_attention_mask=(
            negative_prompt_attention_mask.to("cpu") if negative_prompt_attention_mask is not None else None
        ),
        latents=latents.to("cpu"),
    )

    del tokenizer
    del text_encoder
    return artifacts
