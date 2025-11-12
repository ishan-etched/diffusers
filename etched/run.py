import argparse
import sys
from pathlib import Path

try:  # pragma: no cover - allow running as a script
    from .decode import run_decode
    from .decode_numpy import run_decode_numpy
    from .denoise import run_denoise
    from .encode import run_encode
    from .utils import (
        DEFAULT_DENOISE_SNAPSHOT,
        DEFAULT_ENCODE_SNAPSHOT,
        DEFAULT_VAE_WEIGHTS,
        load_denoise_snapshot,
        load_encode_snapshot,
        save_denoise_snapshot,
        save_encode_snapshot,
    )
except ImportError:  # pragma: no cover
    from decode import run_decode  # type: ignore
    from decode_numpy import run_decode_numpy  # type: ignore
    from denoise import run_denoise  # type: ignore
    from encode import run_encode  # type: ignore
    from utils import (  # type: ignore
        DEFAULT_DENOISE_SNAPSHOT,
        DEFAULT_ENCODE_SNAPSHOT,
        DEFAULT_VAE_WEIGHTS,
        load_denoise_snapshot,
        load_encode_snapshot,
        save_denoise_snapshot,
        save_encode_snapshot,
    )


def _as_path(value: Path | str) -> Path:
    return value if isinstance(value, Path) else Path(value)


DEFAULT_OUTPUT_FILENAME = "output_video.mp4"


def _resolve_paths(base_dir: Path | None) -> tuple[Path, Path, Path, Path]:
    base_dir = Path(base_dir or "artifacts").expanduser()
    base_dir.mkdir(parents=True, exist_ok=True)
    encode_snapshot = base_dir / DEFAULT_ENCODE_SNAPSHOT.name
    denoise_snapshot = base_dir / DEFAULT_DENOISE_SNAPSHOT.name
    video_path = base_dir / DEFAULT_OUTPUT_FILENAME
    return base_dir, encode_snapshot, denoise_snapshot, video_path


def run_encode_command(encode_snapshot: Path) -> None:
    artifacts = run_encode()
    path = save_encode_snapshot(artifacts, encode_snapshot)
    print(f"Saved encode snapshot to {path}")


def run_denoise_command(encode_snapshot: Path, denoise_snapshot: Path) -> None:
    encode_artifacts = load_encode_snapshot(encode_snapshot)
    denoise_artifacts = run_denoise(encode_artifacts)
    path = save_denoise_snapshot(denoise_artifacts, denoise_snapshot)
    print(f"Saved denoise snapshot to {path}")


def run_decode_torch(denoise_snapshot: Path, video_path: Path) -> None:
    denoise_artifacts = load_denoise_snapshot(denoise_snapshot)
    run_decode(denoise_artifacts, video_path)


def run_decode_numpy_backend(denoise_snapshot: Path, video_path: Path, vae_weights: Path) -> None:
    npz_path = denoise_snapshot.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Expected numpy snapshot at {npz_path}.")
    if not Path(vae_weights).exists():
        raise FileNotFoundError(f"Decoder weight archive not found at {vae_weights}.")
    run_decode_numpy(npz_path, vae_weights, video_path)


def run_all(encode_snapshot: Path, denoise_snapshot: Path, video_path: Path, backend: str, vae_weights: Path) -> None:
    encode_artifacts = run_encode()
    save_encode_snapshot(encode_artifacts, encode_snapshot)

    denoise_artifacts = run_denoise(encode_artifacts)
    save_denoise_snapshot(denoise_artifacts, denoise_snapshot)

    if backend == "numpy":
        run_decode_numpy_backend(denoise_snapshot, video_path, vae_weights)
    else:
        run_decode(denoise_artifacts, video_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Mochi1 pipeline stages with snapshot IO.")
    parser.add_argument(
        "--dir",
        type=_as_path,
        default=Path("artifacts"),
        help="Directory for snapshots and final video (default: artifacts/).",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "numpy"),
        default="torch",
        help="Decoding backend to use (default: torch).",
    )

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("encode", help="Run only the encode stage.")
    subparsers.add_parser("denoise", help="Run only the denoise stage.")
    subparsers.add_parser("decode", help="Run only the decode stage.")
    subparsers.add_parser("all", help="Run encode→denoise→decode sequentially.")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "all"

    base_dir, encode_snapshot, denoise_snapshot, video_path = _resolve_paths(args.dir)
    vae_weights = base_dir / DEFAULT_VAE_WEIGHTS.name

    try:
        if command == "encode":
            run_encode_command(encode_snapshot)
        elif command == "denoise":
            run_denoise_command(encode_snapshot, denoise_snapshot)
        elif command == "decode":
            if args.backend == "numpy":
                run_decode_numpy_backend(denoise_snapshot, video_path, vae_weights)
            else:
                run_decode_torch(denoise_snapshot, video_path)
        elif command == "all":
            run_all(encode_snapshot, denoise_snapshot, video_path, args.backend, vae_weights)
        else:
            parser.error(f"Unknown command: {command}")
    except KeyboardInterrupt:  # pragma: no cover
        print("\nGeneration interrupted by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
