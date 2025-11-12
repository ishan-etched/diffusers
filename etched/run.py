import sys

try:  # pragma: no cover - allow running as a script
    from .part1 import part1_prepare_inputs
    from .part2 import part2_run_transformer
    from .part3 import part3_decode_and_render
except ImportError:  # pragma: no cover
    from part1 import part1_prepare_inputs  # type: ignore
    from part2 import part2_run_transformer  # type: ignore
    from part3 import part3_decode_and_render  # type: ignore


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
