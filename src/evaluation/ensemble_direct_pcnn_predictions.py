"""Average direct PCNN classifier predictions across independently trained seeds."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def load_prediction_file(path: Path):
    archive = np.load(path, allow_pickle=True)

    required = {"ids", "labels", "probabilities", "label_classes"}
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"{path.name} is missing keys: {sorted(missing)}")

    model_seed = int(archive["model_seed"]) if "model_seed" in archive else -1

    return {"path": path,"ids": archive["ids"].astype(str),"labels": archive["labels"].astype(str),"probabilities": archive["probabilities"].astype(np.float64),"classes": archive["label_classes"].astype(str),"model_seed": model_seed,}


def metric_summary(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(
            recall_score(labels, predictions, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def find_prediction_files(prediction_dir: Path):
    files = sorted(prediction_dir.glob("**/*_test_predictions.npz"))
    files = [path for path in files if "ensemble" not in path.name.lower()]

    if not files:
        raise FileNotFoundError(
            f"No '*_test_predictions.npz' files found in {prediction_dir}"
        )

    return files


def evaluate_ensemble(args):
    prediction_dir = Path(args.prediction_dir or args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_prediction_files(prediction_dir)

    if args.expected_seeds is not None and len(files) != args.expected_seeds:
        raise ValueError(
            f"Expected {args.expected_seeds} prediction files, found {len(files)}."
        )

    predictions = [load_prediction_file(path) for path in files]
    first = predictions[0]

    ids = first["ids"]
    labels = first["labels"]
    classes = first["classes"]

    probabilities = []
    seeds = []

    for item in predictions:
        if not np.array_equal(ids, item["ids"]):
            raise ValueError(f"Test IDs differ in {item['path']}")
        if not np.array_equal(labels, item["labels"]):
            raise ValueError(f"Test labels differ in {item['path']}")
        if not np.array_equal(classes, item["classes"]):
            raise ValueError(f"Label classes differ in {item['path']}")

        probabilities.append(item["probabilities"])
        seeds.append(item["model_seed"])

    mean_probabilities = np.mean(probabilities, axis=0).astype(np.float32)
    ensemble_predictions = classes[mean_probabilities.argmax(axis=1)]
    metrics = metric_summary(labels, ensemble_predictions)

    stem = args.output_stem or "direct_pcnn_ensemble"

    np.savez_compressed(
        output_dir / f"{stem}_predictions.npz",
        ids=ids,
        labels=labels,
        predictions=ensemble_predictions,
        probabilities=mean_probabilities,
        label_classes=classes,
        model_seeds=np.asarray(seeds, dtype=np.int64),
    )

    report = {
        "prediction_dir": str(prediction_dir),
        "n_models": len(files),
        "model_seeds": seeds,
        "n_test": int(len(ids)),
        "n_classes": int(len(classes)),
        **metrics,
    }

    (output_dir / f"{stem}_metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        f"{len(files)}-seed direct PCNN ensemble: "
        + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
    )
    print(f"Saved predictions: {output_dir / f'{stem}_predictions.npz'}")
    print(f"Saved metrics: {output_dir / f'{stem}_metrics.json'}")

    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Average direct PCNN *_test_predictions.npz files across seeds."
    )
    parser.add_argument(
        "--prediction-dir",
        default=None,
        help="Directory containing exported *_test_predictions.npz files.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Alias for --prediction-dir.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-seeds", type=int, default=None)
    parser.add_argument("--output-stem", default=None)
    return parser


def main():
    args = build_arg_parser().parse_args()

    if args.prediction_dir is None and args.results_dir is None:
        raise ValueError("Please provide --prediction-dir or --results-dir.")

    evaluate_ensemble(args)


if __name__ == "__main__":
    main()
