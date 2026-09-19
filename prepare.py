"""Generate multiscale DNABERT-2 path-complex features."""

from src.prepare_data import build_arg_parser, prepare_features


def main():
    args = build_arg_parser().parse_args()
    prepare_features(args)


if __name__ == "__main__":
    main()