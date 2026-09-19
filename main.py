"""Run multiscale path-complex viral-classification tasks."""

import argparse

from src.tasks.deep_viral_classification import (
    build_arg_parser as train_parser,
    train_deep_viral_classification,
)
from src.tasks.external_viral_classification import (
    build_arg_parser as external_parser,
    train_external_viral_classification,
)
from src.tasks.internal_cross_validation import (
    build_arg_parser as internal_cv_parser,
    train_internal_cross_validation,
)

TASKS = {
    "deep_viral_classification": (
        train_parser,
        train_deep_viral_classification,
    ),
    "external_viral_classification": (
        external_parser,
        train_external_viral_classification,
    ),
    "internal_cross_validation": (
        internal_cv_parser,
        train_internal_cross_validation,
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Multiscale path-complex viral-classification pipeline."
    )
    parser.add_argument("task", choices=sorted(TASKS))

    selected, remaining = parser.parse_known_args()
    build_parser, runner = TASKS[selected.task]
    runner(build_parser().parse_args(remaining))


if __name__ == "__main__":
    main()
