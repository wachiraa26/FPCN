"""Ensemble closed-set external predictions from independently trained PCNN models."""

import argparse
import json
from pathlib import Path

import numpy as np

from src.tasks.external_viral_classification import metric_summary


def evaluate_external_ensemble(args):
    """Average prediction probabilities across independently trained models."""
    prediction_dir = Path(args.prediction_dir)
    files = sorted(prediction_dir.glob("*_closed_external_predictions.npz"))

    if len(files) != args.expected_seeds:
        raise ValueError(f"Expected {args.expected_seeds} prediction files, found {len(files)}.")

    archives = [np.load(path, allow_pickle=True) for path in files]
    first = archives[0]

    ids = first["ids"].astype(str)
    labels = first["labels"].astype(str)
    classes = first["label_classes"].astype(str)

    probabilities = []
    seeds = []

    for path, archive in zip(files, archives):
        archive_ids = archive["ids"].astype(str)
        archive_labels = archive["labels"].astype(str)
        archive_classes = archive["label_classes"].astype(str)

        if not np.array_equal(ids, archive_ids):
            raise ValueError(f"External IDs differ in {path.name}.")

        if not np.array_equal(labels, archive_labels):
            raise ValueError(f"External labels differ in {path.name}.")

        if not np.array_equal(classes, archive_classes):
            raise ValueError(f"Class definitions differ in {path.name}.")

        probabilities.append(archive["probabilities"])
        seeds.append(int(archive["model_seed"]))

    mean_probabilities = np.mean(probabilities, axis=0)

    class_lookup = {label: index for index, label in enumerate(classes)}
    y_true = np.asarray([class_lookup[label] for label in labels])

    metrics = metric_summary(y_true, mean_probabilities, len(classes))
    predictions = classes[mean_probabilities.argmax(axis=1)]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = str(first["source_dataset"])
    external_dataset = str(first["external_test_dataset"])
    stem = f"external_ensemble_{source_dataset}_to_{external_dataset}"

    np.savez_compressed(output_dir / f"{stem}_predictions.npz",ids=ids,labels=labels,predictions=predictions,probabilities=mean_probabilities.astype(np.float32),label_classes=classes,model_seeds=np.asarray(seeds, dtype=np.int32),)

    report = {
        "source_dataset": source_dataset,
        "external_test_dataset": external_dataset,
        "model_seeds": seeds,
        "ensemble_size": len(seeds),
        "n_closed_external": len(ids),
        **metrics,
    }

    metrics_path = output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    print(f"{len(seeds)}-seed closed external ensemble: {metric_text}")
    print(f"Saved metrics: {metrics_path}")

    return report


def build_arg_parser():
    """Build command-line arguments for external prediction ensembling."""
    parser = argparse.ArgumentParser(
        description="Evaluate an external PCNN probability ensemble."
    )
    parser.add_argument(
        "--prediction-dir",
        required=True,
        help="Directory containing external prediction .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/external_viral_classification",
        help="Directory for ensemble predictions and metrics.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=int,
        default=5,
        help="Expected number of independently trained model predictions.",
    )
    return parser
if __name__ == "__main__":
    evaluate_external_ensemble(build_arg_parser().parse_args())