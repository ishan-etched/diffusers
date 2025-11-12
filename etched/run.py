import sys

import torch
from diffusers.pipelines.mochi.pipeline_mochi import MochiPipeline
from diffusers.utils.export_utils import export_to_video


def main() -> None:
    pipe = MochiPipeline.from_pretrained(
        "genmo/mochi-1-preview", variant="bf16", torch_dtype=torch.bfloat16
    )

    try:
        pipe.enable_model_cpu_offload()
    except RuntimeError as e:
        print(f"Error enabling model CPU offload: {e}")
        print(
            "\nYou probably need to switch to PyTorch GPU. Run the switch_to_torch_gpu.sh script in the parent mochi1 folder."
        )
        sys.exit(1)

    pipe.enable_vae_tiling()

    torch.manual_seed(30)
    prompt = "Campfire burning on a beach, Ultra high resolution 4k."
    frames = pipe(prompt, num_frames=25, num_inference_steps=12).frames[0]

    export_to_video(frames, "output_video.mp4", fps=30)


if __name__ == "__main__":
    main()
