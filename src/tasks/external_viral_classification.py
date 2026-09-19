"""Train EXP3_C on an earlier release and evaluate a later release."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,)
from torch import nn

from .deep_viral_classification import (DNAFeatureDataset, make_data_loader, move_samples, set_seed, sha256_file,)
from ..features import normalize_label, read_metadata
from ..model import DNATopoFusionModel


def split_source_by_family(dataset, validation_fraction, split_seed):
    """Reserve validation genomes while retaining at least seven per family for training."""
    train_indices = []
    validation_indices = []
    rng = np.random.default_rng(split_seed)

    for label in np.unique(dataset.y):
        indices = rng.permutation(np.flatnonzero(dataset.y == label))

        if len(indices) < 8:
            raise ValueError("Source data must contain at least eight genomes per family.")

        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        validation_count = min(validation_count, len(indices) - 7)

        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    return dataset.ids[train_indices], dataset.ids[validation_indices]


def external_family_sets(args, source_classes):
    """Separate later-release accessions into closed-set and novel-family sets."""
    rows, accession_column, fieldnames = read_metadata(
        args.external_test_dataset,
        data_dir=args.data_dir,
        accession_column=args.accession_column,
    )

    if args.label_column not in fieldnames:
        raise ValueError(f"External metadata is missing label column {args.label_column}.")

    source_classes = set(map(str, source_classes))

    closed_ids = []
    novel = []
    skipped = []
    seen = set()

    for row in rows:
        sid = str(row.get(accession_column, "")).strip()
        family = normalize_label(row.get(args.label_column))

        if not sid or family is None:
            skipped.append({"id": sid, "reason": "missing_accession_or_label",})
            continue

        if sid in seen:
            raise ValueError(f"Duplicate external accession: {sid}")

        seen.add(sid)

        if family in source_classes:
            closed_ids.append(sid)
        else:
            novel.append({"id": sid, "family": family,})

    return closed_ids, novel, skipped


def evaluate_probabilities(model, loader, device):
    model.eval()

    ids = []
    labels = []
    probabilities = []

    with torch.no_grad():
        for samples, y in loader:
            logits = model(move_samples(samples, device))
            probs = torch.softmax(logits, dim=-1)

            probabilities.append(probs.cpu().numpy())
            labels.append(y.numpy())
            ids.extend(sample["sid"] for sample in samples)

    if not probabilities:
        raise ValueError("The evaluation loader produced no samples.")

    return (np.asarray(ids, dtype=object), np.concatenate(labels), np.vstack(probabilities).astype(np.float32),)


def metric_summary(y_true, probabilities, class_count=None):
    """Compute metrics over classes present in the evaluation set.

    This is important for closed-set external testing, where the model may have
    more source-training classes than the external test set contains.
    """
    y_pred = probabilities.argmax(axis=1)
    present_labels = np.unique(y_true)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true,y_pred,labels=present_labels,average="macro",zero_division=0,)),
        "macro_recall": float(recall_score(y_true,y_pred,labels=present_labels,average="macro",zero_division=0,)),
        "macro_precision": float(precision_score(y_true,y_pred,labels=present_labels,average="macro",zero_division=0,)),
        "n_eval_classes": int(len(present_labels)),
        "n_model_classes": int(class_count) if class_count is not None else None,
    }


def train_external_viral_classification(args):
    set_seed(args.model_seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    source = DNAFeatureDataset(args.dataset,args.data_dir,args.feature_root,args.label_column,args.accession_column,min_class_count=10,)

    if min(Counter(source.labels).values()) < 10:
        raise ValueError("The source data must already be filtered to at least 10 genomes per family.")

    train_ids, validation_ids = split_source_by_family(source, args.validation_fraction, args.split_seed,)

    train = DNAFeatureDataset(args.dataset,args.data_dir,args.feature_root,args.label_column,args.accession_column,train_ids,source.label_encoder,)

    validation = DNAFeatureDataset(args.dataset,args.data_dir,args.feature_root,args.label_column,args.accession_column,validation_ids,source.label_encoder,)

    closed_ids, novel, external_skipped = external_family_sets(args, source.label_encoder.classes_,)

    external = DNAFeatureDataset(args.external_test_dataset,args.data_dir,args.external_test_feature_root,args.label_column,args.accession_column,closed_ids,source.label_encoder,)

    if set(map(str, source.ids)).intersection(map(str, external.ids)):
        raise ValueError("Source and external-test accession IDs overlap.")

    if not len(external):
        raise ValueError("No closed-set external genomes remain for evaluation.")

    fragment_counts = list(train[0][0]["scale_counts"])

    if fragment_counts != list(external[0][0]["scale_counts"]):
        raise ValueError("Source and external feature fragment-count scales differ.")

    dims = train.infer_dimensions()

    if dims != external.infer_dimensions():
        raise ValueError("Source and external feature dimensions differ.")

    train_loader = make_data_loader(train, args, shuffle=True, seed_offset=0,)

    validation_loader = make_data_loader(validation, args, shuffle=False, seed_offset=1, persistent=False,)

    external_loader = make_data_loader(external, args, shuffle=False, seed_offset=2, persistent=False,)

    model = DNATopoFusionModel(num_classes=len(source.label_encoder.classes_),path_input_dim=dims["path_input_dim"],hidden_dim=args.hidden_dim,pcnn_layers=args.pcnn_layers,pcnn_heads=args.pcnn_heads,dropout=args.dropout,pcnn_direction=args.pcnn_direction,).to(device)

    optimizer = {"adam": torch.optim.Adam,"adamw": torch.optim.AdamW,}[args.optimizer](model.parameters(),lr=args.lr,weight_decay=args.weight_decay,)

    scheduler = None
    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=args.lr,epochs=args.epochs,steps_per_epoch=max(len(train_loader), 1),pct_start=args.onecycle_pct_start,div_factor=args.onecycle_div_factor,final_div_factor=args.onecycle_final_div_factor,)

    criterion = nn.CrossEntropyLoss()
    best_score = -np.inf
    best_epoch = None
    best_state = None
    history = []

    print(f"Source: {args.dataset} | external test: {args.external_test_dataset}")
    print(f"Source split: train={len(train)} validation={len(validation)}")
    print(f"External sets: closed={len(external)} " f"novel={len(novel)} skipped={len(external_skipped)}")
    print(f"Fragment-count scales: {fragment_counts}")

    for epoch in range(1, args.epochs + 1):
        model.train()

        loss_sum = 0.0
        total = 0

        for samples, labels in train_loader:
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(move_samples(samples, device))
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            loss_sum += float(loss.item()) * len(labels)
            total += len(labels)

        _, validation_y, validation_probabilities = evaluate_probabilities(model, validation_loader, device,)

        validation_metrics = metric_summary(validation_y,validation_probabilities,len(source.label_encoder.classes_))

        history.append({"epoch": epoch, "train_loss": loss_sum / max(total, 1), **validation_metrics,})

        print(f"Epoch {epoch}/{args.epochs} "f"loss={loss_sum / max(total, 1):.4f} "f"val_acc={validation_metrics['accuracy']:.4f} "f"val_macro_f1={validation_metrics['macro_f1']:.4f}")

        if validation_metrics[args.checkpoint_metric] > best_score:
            best_score = validation_metrics[args.checkpoint_metric]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)

    output_dir = Path(args.output_dir) / args.dataset
    checkpoint_dir = Path(args.checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stem = f"external_{args.dataset}_to_{args.external_test_dataset}_model-{args.model_seed}"

    checkpoint_path = checkpoint_dir / f"{stem}-best.pt"
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    external_ids, external_y, external_probabilities = evaluate_probabilities(model, external_loader, device,)

    external_pred = external_probabilities.argmax(axis=1)
    external_labels = source.label_encoder.inverse_transform(external_y)
    external_predictions = source.label_encoder.inverse_transform(external_pred)

    external_metrics = metric_summary(external_y, external_probabilities, len(source.label_encoder.classes_),)

    prediction_path = output_dir / f"{stem}_closed_external_predictions.npz"

    np.savez_compressed(
        prediction_path,
        ids=external_ids,
        y_true=external_y.astype(np.int64),
        y_pred=external_pred.astype(np.int64),
        labels=np.asarray(external_labels, dtype=object),
        predictions=np.asarray(external_predictions, dtype=object),
        probabilities=external_probabilities.astype(np.float32),
        label_classes=np.asarray(source.label_encoder.classes_, dtype=object),
        source_dataset=np.asarray(args.dataset),
        external_test_dataset=np.asarray(args.external_test_dataset),
        model_seed=np.asarray(args.model_seed, dtype=np.int64),
        split_seed=np.asarray(args.split_seed, dtype=np.int64),
        checkpoint=np.asarray(str(checkpoint_path)),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        fragment_counts=np.asarray(fragment_counts, dtype=np.int32),
    )

    report = {
        "source_dataset": args.dataset,
        "external_test_dataset": args.external_test_dataset,
        "source_feature_root": str(args.feature_root),
        "external_test_feature_root": str(args.external_test_feature_root),
        "model_seed": args.model_seed,
        "split_seed": args.split_seed,
        "validation_fraction": args.validation_fraction,
        "n_source": len(source),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_closed_external": len(external),
        "n_novel_external": len(novel),
        "n_external_skipped": len(external_skipped),
        "fragment_counts": fragment_counts,
        "best_epoch": best_epoch,
        "checkpoint_metric": args.checkpoint_metric,
        "checkpoint_score": best_score,
        "external_closed_metrics": external_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "prediction_file": str(prediction_path),
        "novel_external_examples": novel[:20],
        "external_skipped_examples": external_skipped[:20],
        "history": history,
    }

    metrics_path = output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8",)

    print(
        "Closed external metrics: "
        + " ".join(
            f"{key}={value:.4f}"
            for key, value in external_metrics.items()
            if isinstance(value, float)
        )
    )

    print(f"Evaluation classes: {external_metrics['n_eval_classes']}")
    print(f"Model classes: {external_metrics['n_model_classes']}")
    print(f"Saved predictions: {prediction_path}")
    print(f"Saved metrics: {metrics_path}")

    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train EXP3_C on an earlier release and test a later release.")

    parser.add_argument("--dataset", required=True, help="Filtered earlier-release source dataset.",)

    parser.add_argument(
        "--external-test-dataset",
        required=True,
        help="Later-release dataset used only for final evaluation.",
    )

    parser.set_defaults(data_dir="./src/data")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--external-test-feature-root", required=True)
    parser.add_argument("--output-dir", default="./results/external_viral_classification")
    parser.add_argument("--checkpoint-dir", default="./checkpoints/external_viral_classification",)
    parser.add_argument("--label-column", default="Family")
    parser.add_argument("--accession-column", default=None)

    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--model-seed", type=int, default=42)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--pcnn-layers", type=int, default=3)
    parser.add_argument("--pcnn-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pcnn-direction", choices=["bidirectional", "upward"], default="bidirectional",)

    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw",)

    parser.add_argument("--scheduler", choices=["none", "onecycle"], default="onecycle",)

    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10000.0)

    parser.add_argument("--checkpoint-metric", choices=["accuracy", "macro_f1"], default="macro_f1",)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)

    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True,)

    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True,)

    parser.add_argument("--device", default="cuda")

    return parser


def main():
    train_external_viral_classification(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
