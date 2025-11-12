from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl_mochi import AutoencoderKLMochi

from .utils import MODEL_ID, PIPELINE_DTYPE, PIPELINE_VARIANT, DEFAULT_VAE_WEIGHTS


def export_vae_decoder_weights(output_path: Path) -> Path:
    vae = AutoencoderKLMochi.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        variant=PIPELINE_VARIANT,
        torch_dtype=PIPELINE_DTYPE,
    )
    state_dict = vae.state_dict()

    arrays: dict[str, np.ndarray] = {}
    for name, tensor in state_dict.items():
        if not name.startswith("decoder."):
            continue
        arrays[name] = tensor.detach().cpu().float().numpy()

    arrays["latents_mean"] = np.array(vae.config.latents_mean, dtype=np.float32)
    arrays["latents_std"] = np.array(vae.config.latents_std, dtype=np.float32)
    arrays["scaling_factor"] = np.array([vae.config.scaling_factor], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Mochi VAE decoder weights to a numpy archive.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VAE_WEIGHTS,
        help=f"Path to store the compressed weight archive (default: {DEFAULT_VAE_WEIGHTS}).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = export_vae_decoder_weights(Path(args.output))
    print(f"Saved decoder weights to {path}")


if __name__ == "__main__":
    main()
