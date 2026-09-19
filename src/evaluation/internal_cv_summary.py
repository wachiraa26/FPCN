"""Summarize direct PCNN metrics from five internal cross-validation folds."""

import argparse
import json
from pathlib import Path

import numpy as np


METRIC_NAMES = {
    "accuracy": ("accuracy", "acc", "test_accuracy", "test_acc"),
    "balanced_accuracy": (
        "balanced_accuracy",
        "ba",
        "test_balanced_accuracy",
        "test_ba",
    ),
    "macro_f1": ("macro_f1", "f1", "test_macro_f1", "test_f1"),
    "macro_recall": (
        "macro_recall",
        "recall",
        "test_macro_recall",
        "test_recall",
    ),
    "macro_precision": (
        "macro_precision",
        "precision",
        "test_macro_precision",
        "test_precision",
    ),
}


def _normalise_key(key):
    return str(key).lower().replace("-", "_").replace(" ", "_")


def _metric_values(record):
    values = {}

    for metric, aliases in METRIC_NAMES.items():
        for alias in aliases:
            if alias in record:
                values[metric] = float(record[alias])
                break

    return values


def _find_test_metrics(payload):
    candidates = []

    def visit(value, path=""):
        if not isinstance(value, dict):
            return

        metrics = _metric_values(
            {_normalise_key(key): item for key, item in value.items()}
        )
        if len(metrics) >= 3:
            priority = 1 if "test" in path.lower() else 0
            candidates.append((priority, len(metrics), metrics))

        for key, item in value.items():
            visit(item, f"{path}.{key}")

    visit(payload)

    if not candidates:
        raise ValueError("Could not find test metrics in the fold metrics file.")

    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _fold_number(path):
    name = path.parent.name
    try:
        return int(name.split("-")[1])
    except (IndexError, ValueError):
        return name


def evaluate_internal_cv_summary(args):
    results_dir = Path(args.results_dir)
    files = sorted(
        results_dir.glob("fold-*-of-*/*metrics.json"),
        key=_fold_number,
    )

    if len(files) != args.expected_folds:
        raise ValueError(
            f"Expected {args.expected_folds} fold metrics files, found {len(files)}."
        )

    fold_results = []

    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = _find_test_metrics(payload)
        fold_results.append(
            {
                "fold": path.parent.name,
                "metrics_file": str(path),
                **metrics,
            }
        )

    summary = {
        "fold_count": len(fold_results),
        "folds": fold_results,
        "mean": {},
        "standard_deviation": {},
    }

    for metric in METRIC_NAMES:
        scores = [fold[metric] for fold in fold_results]

        if len(scores) != len(fold_results):
            raise ValueError(f"Metric '{metric}' was missing from one or more folds.")

        summary["mean"][metric] = float(np.mean(scores))
        summary["standard_deviation"][metric] = float(np.std(scores, ddof=1))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "internal_cv5_summary.json"
    output_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mean = summary["mean"]
    std = summary["standard_deviation"]

    print("Internal five-fold direct-PCNN cross-validation:")
    print(
        "ACC={:.4f}±{:.4f} BA={:.4f}±{:.4f} "
        "Macro-F1={:.4f}±{:.4f} Macro-Recall={:.4f}±{:.4f} "
        "Macro-Precision={:.4f}±{:.4f}".format(
            mean["accuracy"],
            std["accuracy"],
            mean["balanced_accuracy"],
            std["balanced_accuracy"],
            mean["macro_f1"],
            std["macro_f1"],
            mean["macro_recall"],
            std["macro_recall"],
            mean["macro_precision"],
            std["macro_precision"],
        )
    )
    print(f"Saved summary: {output_file}")

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Summarize five-fold direct-PCNN internal cross-validation."
    )
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-folds", type=int, default=5)
    return parser