"""Run multiscale path-complex viral-classification tasks."""

import argparse

from src.evaluation.external_cv_summary import (
    build_arg_parser as cv_summary_parser,
    evaluate_external_cv_summary,
)
from src.evaluation.external_ensemble import (
    build_arg_parser as ensemble_parser,
    evaluate_external_ensemble,
)
from src.evaluation.internal_cv_summary import (
    build_arg_parser as internal_cv_summary_parser,
    evaluate_internal_cv_summary,
)
from src.evaluation.knn_classification import (
    build_arg_parser as knn_parser,
    evaluate_knn,
)
from src.tasks.deep_viral_classification import (
    build_arg_parser as train_parser,
    train_deep_viral_classification,
)
from src.tasks.external_cross_validation import (
    build_arg_parser as external_cv_parser,
    train_external_cross_validation,
)
from src.tasks.external_viral_classification import (
    build_arg_parser as external_parser,
    train_external_viral_classification,
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
    "external_cross_validation": (
        external_cv_parser,
        train_external_cross_validation,
    ),
    "external_ensemble": (
        ensemble_parser,
        evaluate_external_ensemble,
    ),
    "external_cv_summary": (
        cv_summary_parser,
        evaluate_external_cv_summary,
    ),
    "internal_cv_summary": (
        internal_cv_summary_parser,
        evaluate_internal_cv_summary,
    ),
    "knn_classification": (
        knn_parser,
        evaluate_knn,
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