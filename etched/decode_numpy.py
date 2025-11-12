from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from diffusers.utils.export_utils import export_to_video
from jaxtyping import Array, Float
from PIL import Image

try:  # pragma: no cover - allow running as a script
    from .utils import (
        DEFAULT_DENOISE_NPZ,
        DEFAULT_VAE_WEIGHTS,
        DenoiseNpzPayload,
        load_denoise_npz,
    )
except ImportError:  # pragma: no cover
    from utils import (  # type: ignore
        DEFAULT_DENOISE_NPZ,
        DEFAULT_VAE_WEIGHTS,
        DenoiseNpzPayload,
        load_denoise_npz,
    )

Array5D = Float[Array, "batch channels frames height width"]


@dataclass(frozen=True)
class Conv3DParams:
    weight: Array  # (out_channels, in_channels, kt, kh, kw)
    bias: Array
    kernel_size: Tuple[int, int, int]
    stride: Tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True)
class LinearParams:
    weight: Array  # (out_dim, in_dim)
    bias: Array


@dataclass(frozen=True)
class ResnetParams:
    norm1_weight: Array
    norm1_bias: Array
    norm2_weight: Array
    norm2_bias: Array
    conv1: Conv3DParams
    conv2: Conv3DParams


@dataclass(frozen=True)
class UpBlockParams:
    resnets: Sequence[ResnetParams]
    proj: LinearParams
    temporal_expansion: int
    spatial_expansion: int


@dataclass(frozen=True)
class DecoderParams:
    conv_in: Conv3DParams
    block_in: Sequence[ResnetParams]
    up_blocks: Sequence[UpBlockParams]
    block_out: Sequence[ResnetParams]
    proj_out: LinearParams
    latents_mean: Array
    latents_std: Array
    scaling_factor: float


def _to_jax(array: np.ndarray) -> Array:
    return jnp.asarray(array.astype(np.float32))


def _load_conv(weight: np.ndarray, bias: np.ndarray, kernel_size: Tuple[int, int, int]) -> Conv3DParams:
    return Conv3DParams(weight=_to_jax(weight), bias=_to_jax(bias), kernel_size=kernel_size)


def _load_resnet(prefix: str, tensors: dict[str, np.ndarray]) -> ResnetParams:
    conv_kwargs = dict(kernel_size=(3, 3, 3))
    return ResnetParams(
        norm1_weight=_to_jax(tensors[f"{prefix}.norm1.norm_layer.weight"]),
        norm1_bias=_to_jax(tensors[f"{prefix}.norm1.norm_layer.bias"]),
        norm2_weight=_to_jax(tensors[f"{prefix}.norm2.norm_layer.weight"]),
        norm2_bias=_to_jax(tensors[f"{prefix}.norm2.norm_layer.bias"]),
        conv1=_load_conv(tensors[f"{prefix}.conv1.conv.weight"], tensors[f"{prefix}.conv1.conv.bias"], **conv_kwargs),
        conv2=_load_conv(tensors[f"{prefix}.conv2.conv.weight"], tensors[f"{prefix}.conv2.conv.bias"], **conv_kwargs),
    )


def load_decoder_params(path: Path) -> DecoderParams:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        tensors = {key: data[key] for key in data.files}

    conv_in = _load_conv(
        tensors["decoder.conv_in.weight"],
        tensors["decoder.conv_in.bias"],
        kernel_size=(1, 1, 1),
    )

    block_in = [_load_resnet(f"decoder.block_in.resnets.{i}", tensors) for i in range(3)]

    up_block_specs = [
        (0, 6, 3, 2),
        (1, 4, 2, 2),
        (2, 3, 1, 2),
    ]
    up_blocks: List[UpBlockParams] = []
    for block_idx, num_res, t_exp, s_exp in up_block_specs:
        resnets = [_load_resnet(f"decoder.up_blocks.{block_idx}.resnets.{i}", tensors) for i in range(num_res)]
        proj = LinearParams(
            weight=_to_jax(tensors[f"decoder.up_blocks.{block_idx}.proj.weight"]),
            bias=_to_jax(tensors[f"decoder.up_blocks.{block_idx}.proj.bias"]),
        )
        up_blocks.append(
            UpBlockParams(
                resnets=resnets,
                proj=proj,
                temporal_expansion=t_exp,
                spatial_expansion=s_exp,
            )
        )

    block_out = [_load_resnet(f"decoder.block_out.resnets.{i}", tensors) for i in range(3)]
    proj_out = LinearParams(
        weight=_to_jax(tensors["decoder.proj_out.weight"]),
        bias=_to_jax(tensors["decoder.proj_out.bias"]),
    )

    latents_mean = _to_jax(tensors["latents_mean"]).reshape(1, -1, 1, 1, 1)
    latents_std = _to_jax(tensors["latents_std"]).reshape(1, -1, 1, 1, 1)
    scaling_factor = float(tensors["scaling_factor"][0])

    return DecoderParams(
        conv_in=conv_in,
        block_in=block_in,
        up_blocks=up_blocks,
        block_out=block_out,
        proj_out=proj_out,
        latents_mean=latents_mean,
        latents_std=latents_std,
        scaling_factor=scaling_factor,
    )


def _replicate_pad(x: Array5D, kernel_size: Tuple[int, int, int]) -> Array5D:
    kt, kh, kw = kernel_size
    if kt == kh == kw == 1:
        return x
    pad_t = kt - 1
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    pad_config = ((0, 0), (0, 0), (pad_t, 0), (pad_h, pad_h), (pad_w, pad_w))
    return jnp.pad(x, pad_config, mode="edge")


def _conv3d(x: Array5D, params: Conv3DParams) -> Array5D:
    x = _replicate_pad(x, params.kernel_size)
    kernel = jnp.transpose(params.weight, (2, 3, 4, 1, 0))
    x_t = jnp.transpose(x, (0, 2, 3, 4, 1))
    strides = params.stride
    y = jax.lax.conv_general_dilated(
        x_t,
        kernel,
        window_strides=strides,
        padding="VALID",
        dimension_numbers=("NTHWC", "THWIO", "NTHWC"),
    )
    y = jnp.transpose(y, (0, 4, 1, 2, 3))
    if params.bias is not None:
        y = y + params.bias.reshape(1, -1, 1, 1, 1)
    return y


def _group_norm(x: Array5D, weight: Array, bias: Array, groups: int = 32, eps: float = 1e-5) -> Array5D:
    b, c, t, h, w = x.shape
    x_group = x.reshape(b, groups, c // groups, t, h, w)
    mean = jnp.mean(x_group, axis=(2, 3, 4, 5), keepdims=True)
    var = jnp.var(x_group, axis=(2, 3, 4, 5), keepdims=True)
    x_group = (x_group - mean) / jnp.sqrt(var + eps)
    normalized = x_group.reshape(b, c, t, h, w)
    return normalized * weight.reshape(1, -1, 1, 1, 1) + bias.reshape(1, -1, 1, 1, 1)


def _swish(x: Array) -> Array:
    return x * jax.nn.sigmoid(x)


def _resnet_block(x: Array5D, params: ResnetParams) -> Array5D:
    hidden = _group_norm(x, params.norm1_weight, params.norm1_bias)
    hidden = _swish(hidden)
    hidden = _conv3d(hidden, params.conv1)

    hidden = _group_norm(hidden, params.norm2_weight, params.norm2_bias)
    hidden = _swish(hidden)
    hidden = _conv3d(hidden, params.conv2)
    return hidden + x


def _mid_block(x: Array5D, resnets: Iterable[ResnetParams]) -> Array5D:
    for res in resnets:
        x = _resnet_block(x, res)
    return x


def _linear(x: Array, params: LinearParams) -> Array:
    y = jnp.tensordot(x, jnp.transpose(params.weight), axes=1)
    return y + params.bias


def _upsample_volume(x: Array5D, t_expansion: int, s_expansion: int) -> Array5D:
    b, c, t, h, w = x.shape
    factor = t_expansion * s_expansion * s_expansion
    out_channels = c // factor
    x = x.reshape(b, out_channels, t_expansion, s_expansion, s_expansion, t, h, w)
    x = jnp.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
    return x.reshape(b, out_channels, t * t_expansion, h * s_expansion, w * s_expansion)


def _up_block(x: Array5D, params: UpBlockParams) -> Array5D:
    x = _mid_block(x, params.resnets)
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    x = _linear(x, params.proj)
    x = jnp.transpose(x, (0, 4, 1, 2, 3))
    return _upsample_volume(x, params.temporal_expansion, params.spatial_expansion)


def decoder_forward(latents: Array5D, params: DecoderParams) -> Array5D:
    x = _conv3d(latents, params.conv_in)
    x = _mid_block(x, params.block_in)
    for block in params.up_blocks:
        x = _up_block(x, block)
    x = _mid_block(x, params.block_out)
    x = _swish(x)
    x = jnp.transpose(x, (0, 2, 3, 4, 1))
    x = _linear(x, params.proj_out)
    return jnp.transpose(x, (0, 4, 1, 2, 3))


def _video_to_pil_frames(video: Array5D) -> List[Image.Image]:
    np_video = np.array(video)
    np_video = np.clip(np_video, -1.0, 1.0)
    np_video = (np_video + 1.0) / 2.0
    np_video = (np_video * 255).round().astype(np.uint8)

    frames: List[Image.Image] = []
    batch, _, num_frames, _, _ = np_video.shape
    assert batch == 1, "Only batch size 1 is supported in numpy decoder."

    for frame_idx in range(num_frames):
        frame = np_video[0, :, frame_idx]
        frame = np.transpose(frame, (1, 2, 0))
        frames.append(Image.fromarray(frame))
    return frames


def run_decode_numpy(
    denoise_npz_path: Path | None = None,
    decoder_weights_path: Path | None = None,
    output_path: Path | None = None,
) -> None:
    npz_path = Path(denoise_npz_path or DEFAULT_DENOISE_NPZ)
    weights_path = Path(decoder_weights_path or DEFAULT_VAE_WEIGHTS)
    out_path = Path(output_path or (Path(DEFAULT_DENOISE_NPZ).parent / "output_video_np.mp4"))

    payload: DenoiseNpzPayload = load_denoise_npz(npz_path)
    params = load_decoder_params(weights_path)

    latents = jnp.asarray(payload.latents)
    latents = latents * params.latents_std / params.scaling_factor + params.latents_mean

    video = decoder_forward(latents, params)
    frames = _video_to_pil_frames(video)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=payload.settings.fps)
    print(f"    [numpy] Exported video to {out_path}")
