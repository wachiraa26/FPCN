import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, f1_score, pairwise_distances, precision_score, recall_score, )
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
try:
    from ..features import (feature_path, normalize_label, read_metadata, )
    from ..model import DNATopoFusionModel
except ImportError:
    from src.features import (feature_path, normalize_label, read_metadata, )
    from src.model import DNATopoFusionModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True, )


def seed_worker(worker_id):
    """Give each data-loading worker deterministic NumPy/Python RNG state."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_data_loader(dataset, args, shuffle=False, seed_offset=0, persistent=None):
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if args.prefetch_factor < 1:
        raise ValueError("prefetch_factor must be at least 1.")
    generator = torch.Generator()
    generator.manual_seed(int(args.model_seed) + int(seed_offset))
    kwargs = {"dataset": dataset, "batch_size": args.batch_size, "shuffle": shuffle, "collate_fn": collate_samples,
              "num_workers": args.num_workers, "pin_memory": args.pin_memory, "worker_init_fn": seed_worker, "generator": generator, }
    if args.num_workers > 0:
        keep_workers = args.persistent_workers if persistent is None else persistent
        kwargs["persistent_workers"] = bool(keep_workers)
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def tensor_from_h5(handle, key, dtype=torch.float32, ):
    return torch.as_tensor(handle[key][:], dtype=dtype, )


def path_sample_from_h5(handle, sid, scale):
    source = handle["scales"][str(scale)]
    return {"sid": sid, "scale": int(scale), "path0": tensor_from_h5(source, "path0"), "path1": tensor_from_h5(source, "path1"), "path2": tensor_from_h5(source, "path2"), "path0_link": tensor_from_h5(source, "path0_link", dtype=torch.long), "path1_link": tensor_from_h5(source, "path1_link", dtype=torch.long), }


class DNAFeatureDataset(Dataset):
    def __init__(self, dataset, data_dir, feature_root, label_column, accession_column=None, ids=None, label_encoder=None, min_class_count=1, ):
        if min_class_count < 1:
            raise ValueError("min_class_count must be at least 1.")
        rows, accession_column, fieldnames = read_metadata(dataset, data_dir=data_dir, accession_column=accession_column, )
        if label_column not in fieldnames:
            raise ValueError(f"Metadata is missing label column " f"{label_column}")
        self.dataset = dataset
        self.feature_root = feature_root
        self.accession_column = accession_column
        self.label_column = label_column
        self.min_class_count = min_class_count
        wanted = ({str(sid).strip() for sid in ids} if ids is not None else None)
        records = []
        labels = []
        skipped = []
        seen_ids = set()
        for row in rows:
            sid = str(row.get(accession_column, "")).strip()
            if wanted is not None and sid not in wanted:
                continue
            if not sid:
                skipped.append((sid, "missing_accession"))
                continue
            if sid in seen_ids:
                raise ValueError(f"Duplicate accession found in " f"metadata: {sid}")
            seen_ids.add(sid)
            label = normalize_label(row.get(label_column))
            if label is None:
                skipped.append((sid, "missing_label"))
                continue
            path = feature_path(feature_root, dataset, sid, )
            if not path.exists():
                skipped.append((sid, "missing_feature"))
                continue
            records.append(sid)
            labels.append(label)
        # Counts are calculated only after excluding
        # missing labels and missing feature files.
        class_counts_before = Counter(labels)
        if min_class_count > 1:
            keep = [class_counts_before[label] >= min_class_count for label in labels]
            for sid, label, retain in zip(records, labels, keep, ):
                if not retain:
                    skipped.append((sid, ("class_below_min_count:" f"{label}"), ))
            records = [sid for sid, retain in zip(records, keep, ) if retain]
            labels = [label for label, retain in zip(labels, keep, ) if retain]
        if not records:
            raise ValueError("No labeled records with feature files " "remain after filtering.")
        self.ids = np.asarray(records, dtype=object, )
        self.labels = np.asarray(labels, dtype=object, )
        self.skipped = skipped
        self.class_counts_before = dict(sorted(class_counts_before.items()))
        self.class_counts_after = dict(sorted(Counter(labels).items()))
        if label_encoder is None:
            label_encoder = LabelEncoder().fit(labels)
        self.label_encoder = label_encoder
        self.y = label_encoder.transform(labels)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = str(self.ids[idx])
        path = feature_path(self.feature_root, self.dataset, sid, )
        with h5py.File(path, "r") as handle:
            counts = [int(count) for count in handle.attrs.get("multiscale_fixed_fragment_counts", [])]
            if not counts or "scales" not in handle:
                raise ValueError(f"{path} is not an EXP3_C multiscale path-complex archive.")
            if len(counts) != len(set(counts)) or any(count < 1 for count in counts):
                raise ValueError(f"Invalid fragment-count scales in {path}: {counts}")
            missing_scales = [count for count in counts if str(count) not in handle["scales"]]
            if missing_scales:
                raise ValueError(f"Missing scale groups in {path}: {missing_scales}")
            sample = {"sid": sid, "scales": [path_sample_from_h5(handle, sid, scale=count) for count in counts], "scale_counts": counts, }
            sample["path_definition"] = handle.attrs.get("path_definition", "multiscale_fragment", )
        return sample, int(self.y[idx])

    def infer_dimensions(self):
        sample, _ = self[0]
        sample = sample["scales"][0]
        path_dim = int(sample["path0"].shape[-1])
        return {"path_input_dim": path_dim}


def collate_samples(batch):
    samples, labels = zip(*batch)
    return (list(samples), torch.as_tensor(labels, dtype=torch.long, ), )


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def move_samples(samples, device):
    return [move_to_device(sample, device) for sample in samples]


def make_loaders(args):
    if not 0.0 < args.test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0.0 < args.val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")
    if args.test_size + args.val_size >= 1.0:
        raise ValueError("test_size + val_size must be less than 1.")
    # Apply the minimum-family filter to the complete
    # usable cohort before creating any splits.
    full = DNAFeatureDataset(dataset=args.dataset, data_dir=args.data_dir, feature_root=args.feature_root, label_column=args.label_column,
                             accession_column=args.accession_column, min_class_count=args.min_class_count, )
    print("Filtered cohort: " f"{len(full)} genomes, " f"{len(full.label_encoder.classes_)} families, " f"minimum family count={args.min_class_count}")
    print("Skipped records: " f"{len(full.skipped)}")
    y = full.y
    stratify = (y if min(np.bincount(y)) >= 2 else None)
    (train_val_ids, test_ids, train_val_y, _, ) = train_test_split(full.ids, y, test_size=args.test_size, random_state=args.split_seed, stratify=stratify, )
    val_fraction = (args.val_size / (1.0 - args.test_size))
    train_val_counts = np.bincount(train_val_y, minlength=len(full.label_encoder.classes_), )
    nonzero_counts = train_val_counts[train_val_counts > 0]
    val_stratify = (train_val_y if (len(nonzero_counts) and nonzero_counts.min() >= 2) else None)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_fraction, random_state=args.split_seed, stratify=val_stratify, )
    # Do not recalculate the minimum count separately
    # inside train, validation, and test.
    train = DNAFeatureDataset(args.dataset, args.data_dir, args.feature_root, args.label_column, args.accession_column,
                              ids=train_ids, label_encoder=full.label_encoder, min_class_count=1, )
    val = DNAFeatureDataset(args.dataset, args.data_dir, args.feature_root, args.label_column, args.accession_column,
                            ids=val_ids, label_encoder=full.label_encoder, min_class_count=1, )
    test = DNAFeatureDataset(args.dataset, args.data_dir, args.feature_root, args.label_column, args.accession_column,
                             ids=test_ids, label_encoder=full.label_encoder, min_class_count=1, )
    train_loader = make_data_loader(train, args, shuffle=True, seed_offset=0, )
    val_loader = make_data_loader(val, args, shuffle=False, seed_offset=1, persistent=False, )
    test_loader = make_data_loader(test, args, shuffle=False, seed_offset=2, persistent=False, )
    return (full, train, val, test, train_loader, val_loader, test_loader, )


def evaluate(model, loader, device):
    model.eval()
    all_y = []
    all_pred = []
    with torch.no_grad():
        for samples, labels in loader:
            samples = move_samples(samples, device, )
            labels = labels.to(device)
            logits = model(samples)
            pred = logits.argmax(dim=-1)
            all_y.append(labels.cpu().numpy())
            all_pred.append(pred.cpu().numpy())
    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    return y, pred


def collect_probabilities(model, loader, device):
    model.eval()
    ids, labels, predictions, probabilities = [], [], [], []
    with torch.no_grad():
        for samples, y in loader:
            moved = move_samples(samples, device)
            logits = model(moved)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            ids.extend(sample["sid"] for sample in samples)
            labels.append(y.numpy())
            predictions.append(pred.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
    if not probabilities:
        raise ValueError("The prediction loader produced no samples.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate accession IDs were found while collecting predictions.")
    return (np.asarray(ids, dtype=object), np.concatenate(labels), np.concatenate(predictions), np.vstack(probabilities).astype(np.float32), )


def collect_embeddings(model, loader, device):
    """Return accession IDs, integer labels, and embeddings in loader order."""
    model.eval()
    ids, labels, embeddings = [], [], []
    with torch.no_grad():
        for samples, y in loader:
            moved = move_samples(samples, device)
            embeddings.append(model.embed(moved).cpu().numpy())
            labels.append(y.numpy())
            ids.extend(sample["sid"] for sample in samples)
    if not embeddings:
        raise ValueError("The embedding loader produced no samples.")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate accession IDs were found while collecting embeddings.")
    return (np.asarray(ids, dtype=object), np.concatenate(labels), np.vstack(embeddings).astype(np.float32), )


def validation_1nn_accuracy(model, train_loader, val_loader, device, metric, n_jobs):
    """Classify validation genomes using training genomes as neighbors."""
    train_ids, train_y, train_z = collect_embeddings(model, train_loader, device)
    val_ids, val_y, val_z = collect_embeddings(model, val_loader, device)
    if set(map(str, train_ids)).intersection(map(str, val_ids)):
        raise ValueError("Training and validation IDs overlap during 1-NN checkpointing.")
    distances = pairwise_distances(val_z, train_z, metric=metric, n_jobs=n_jobs)
    predictions = train_y[np.argmin(distances, axis=1)]
    return float(accuracy_score(val_y, predictions))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_embeddings(model, dataset, split_name, loader, label_encoder, output_path, device, min_class_count, model_mode, model_seed, split_seed, checkpoint_path, checkpoint_sha256, fragment_counts, ):
    ids, encoded_labels, embeddings = collect_embeddings(model, loader, device)
    labels = label_encoder.inverse_transform(encoded_labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, ids=np.asarray(ids, dtype=object), labels=np.asarray(labels, dtype=object), embeddings=embeddings, dataset=np.asarray(dataset), split=np.asarray(split_name), model_mode=np.asarray(model_mode), seed=np.asarray(model_seed, dtype=np.int64), model_seed=np.asarray(model_seed, dtype=np.int64), split_seed=np.asarray(
        split_seed, dtype=np.int64), checkpoint=np.asarray(str(checkpoint_path)), checkpoint_sha256=np.asarray(checkpoint_sha256), label_classes=np.asarray(label_encoder.classes_, dtype=object), min_class_count=np.asarray(min_class_count, dtype=np.int32), fragment_counts=np.asarray(fragment_counts, dtype=np.int32), )
    print(f"Saved {split_name} embeddings: {output_path}")


def train_deep_viral_classification(args):
    args.model_mode = "path_only"
    if args.min_class_count is None:
        args.min_class_count = 3 if args.protocol == "1nn" else 15
    if args.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative.")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative.")
    if args.checkpoint_eval_interval < 1:
        raise ValueError("checkpoint_eval_interval must be at least 1.")
    if args.checkpoint_distance_jobs == 0:
        raise ValueError("checkpoint_distance_jobs cannot be 0.")
    if args.gradient_clip_norm < 0:
        raise ValueError("gradient_clip_norm must be non-negative.")
    set_seed(args.model_seed)
    device = torch.device(args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu")
    (full, train, val, test, train_loader, val_loader, test_loader, ) = make_loaders(args)
    split_sets = {"train": set(map(str, train.ids)), "validation": set(map(str, val.ids)), "test": set(map(str, test.ids)), }
    if split_sets["train"] & split_sets["validation"]:
        raise ValueError("Training and validation IDs overlap.")
    if split_sets["train"] & split_sets["test"]:
        raise ValueError("Training and test IDs overlap.")
    if split_sets["validation"] & split_sets["test"]:
        raise ValueError("Validation and test IDs overlap.")
    checkpoint_train_loader = None
    if args.checkpoint_metric == "val_1nn_accuracy":
        checkpoint_train_loader = make_data_loader(train, args, shuffle=False, seed_offset=4, persistent=False, )
    print(f"Data loading: workers={args.num_workers} " f"persistent={args.persistent_workers and args.num_workers > 0} " f"pin_memory={args.pin_memory}")
    print(f"Split/model seeds: split={args.split_seed} model={args.model_seed}")
    print(f"Checkpoint selection: metric={args.checkpoint_metric} " f"interval={args.checkpoint_eval_interval} " f"distance={args.checkpoint_knn_metric}")
    print(f"Optimizer: {args.optimizer} " f"lr={args.lr:g} weight_decay={args.weight_decay:g}")
    print(f"PCNN direction: {args.pcnn_direction}")
    print(f"Split sizes: " f"train={len(train)} " f"val={len(val)} " f"test={len(test)}")
    first_sample, _ = train[0]
    fragment_counts = list(first_sample["scale_counts"])
    print(f"Fragment-count scales: {fragment_counts}")
    dims = train.infer_dimensions()
    model = DNATopoFusionModel(num_classes=len(full.label_encoder.classes_), path_input_dim=dims["path_input_dim"], hidden_dim=args.hidden_dim, pcnn_layers=(
        args.pcnn_layers), pcnn_heads=(args.pcnn_heads), dropout=args.dropout, pcnn_direction=args.pcnn_direction, ).to(device)
    optimizer_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, }[args.optimizer]
    optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, )
    scheduler = None
    if args.scheduler == "onecycle":
        scheduler = (torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=max(len(train_loader), 1, ),
                     pct_start=(args.onecycle_pct_start), div_factor=(args.onecycle_div_factor), final_div_factor=(args.onecycle_final_div_factor), ))
    class_weights = None
    if args.class_weighted_loss:
        counts = np.bincount(train.y, minlength=len(full.label_encoder.classes_), ).astype(np.float64)
        class_weights = counts.sum() / np.maximum(counts, 1.0)
        class_weights /= class_weights.mean()
        class_weights = torch.as_tensor(class_weights, dtype=torch.float32, device=device, )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_score = -np.inf
    best_epoch = None
    best_state = None
    checks_without_improvement = 0
    stopped_early = False
    history = []
    for epoch in range(1, args.epochs + 1, ):
        model.train()
        total_loss = 0.0
        total = 0
        for samples, labels in train_loader:
            samples = move_samples(samples, device, )
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(samples)
            loss = criterion(logits, labels)
            loss.backward()
            if args.gradient_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm, )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total_loss += (float(loss.item()) * len(labels))
            total += len(labels)
        y_true, y_pred = evaluate(model, val_loader, device, )
        label_ids = np.arange(len(full.label_encoder.classes_))
        acc = accuracy_score(y_true, y_pred, )
        ba = balanced_accuracy_score(y_true, y_pred, )
        macro = f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0, )
        recall = recall_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0, )
        precision = precision_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0, )
        checkpoint_due = (args.checkpoint_metric == "val_macro_f1" or epoch == 1 or epoch % args.checkpoint_eval_interval == 0 or epoch == args.epochs)
        val_1nn = None
        if args.checkpoint_metric == "val_1nn_accuracy" and checkpoint_due:
            val_1nn = validation_1nn_accuracy(model, checkpoint_train_loader, val_loader, device, args.checkpoint_knn_metric, args.checkpoint_distance_jobs, )
        row = {"epoch": epoch, "train_loss": (total_loss / max(total, 1)), "val_accuracy": float(acc), "val_balanced_accuracy": float(ba), "val_macro_f1": float(
            macro), "val_macro_recall": float(recall), "val_macro_precision": float(precision), "val_1nn_accuracy": val_1nn, "lr": float(optimizer.param_groups[0]["lr"]), }
        history.append(row)
        log_line = (
            f"Epoch {epoch}/{args.epochs} " f"loss={row['train_loss']:.4f} " f"val_acc={acc:.4f} " f"val_ba={ba:.4f} " f"val_macro_f1={macro:.4f} " f"val_recall={recall:.4f} " f"val_precision={precision:.4f}")
        if val_1nn is not None:
            log_line += f" val_1nn_acc={val_1nn:.4f}"
        print(log_line)
        if checkpoint_due:
            checkpoint_score = (float(macro) if args.checkpoint_metric == "val_macro_f1" else float(val_1nn))
            if best_state is None or checkpoint_score > best_score + args.early_stopping_min_delta:
                best_score = checkpoint_score
                best_epoch = epoch
                checks_without_improvement = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                checks_without_improvement += 1
            if (args.early_stopping_patience > 0 and checks_without_improvement >= args.early_stopping_patience):
                stopped_early = True
                print(
                    f"Early stopping at epoch {epoch}: {args.checkpoint_metric} " f"did not improve by more than {args.early_stopping_min_delta:g} " f"for {args.early_stopping_patience} checkpoint evaluations.")
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    y_true, y_pred = evaluate(model, test_loader, device, )
    label_ids = np.arange(len(full.label_encoder.classes_))
    report = classification_report(y_true, y_pred, labels=label_ids, target_names=(full.label_encoder.classes_), output_dict=True, zero_division=0, )
    best_row = next(row for row in history if row["epoch"] == best_epoch)
    metrics = {"dataset": args.dataset, "label_column": args.label_column, "model_mode": args.model_mode, "protocol": args.protocol, "fragment_counts": fragment_counts, "class_weighted_loss": bool(args.class_weighted_loss), "gradient_clip_norm": float(args.gradient_clip_norm), "optimizer": args.optimizer, "pcnn_direction": args.pcnn_direction, "seed": int(args.model_seed), "model_seed": int(args.model_seed), "split_seed": int(args.split_seed), "early_stopping_patience": int(args.early_stopping_patience), "early_stopping_min_delta": float(args.early_stopping_min_delta), "stopped_early": bool(stopped_early), "epochs_completed": int(len(history)), "min_class_count": int(args.min_class_count), "n_retained": int(len(full)), "n_classes_retained": int(len(full.label_encoder.classes_)), "class_counts_before": (full.class_counts_before), "class_counts_after": (full.class_counts_after), "skipped_count": int(len(full.skipped)), "skipped_examples": (full.skipped[:10]), "classes": (full.label_encoder .classes_ .tolist()), "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)), "best_epoch": int(best_epoch), "checkpoint_metric": args.checkpoint_metric, "checkpoint_metric_score": float(
        best_score), "checkpoint_knn_metric": (args.checkpoint_knn_metric if args.checkpoint_metric == "val_1nn_accuracy" else None), "checkpoint_eval_interval": int(args.checkpoint_eval_interval), "checkpoint_distance_jobs": int(args.checkpoint_distance_jobs), "best_val_macro_f1": float(best_row["val_macro_f1"]), "best_val_1nn_accuracy": best_row["val_1nn_accuracy"], "accuracy": float(accuracy_score(y_true, y_pred, )), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred, )), "macro_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0, )), "macro_recall": float(recall_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0, )), "macro_precision": float(precision_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0, )), "weighted_f1": float(f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0, )), "weighted_recall": float(recall_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0, )), "weighted_precision": float(precision_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0, )), "history": history, "classification_report": report, }
    output_dir = (Path(args.output_dir) / args.dataset)
    output_dir.mkdir(parents=True, exist_ok=True, )
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True, )
    stem = (f"{args.model_mode}_{args.label_column}_" f"split-{args.split_seed}_model-{args.model_seed}")
    checkpoint_name = (f"{args.dataset}-{args.label_column}-{args.model_mode}-" f"split-{args.split_seed}-model-{args.model_seed}-best.pt")
    checkpoint_path = checkpoint_dir / checkpoint_name
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint_digest = sha256_file(checkpoint_path)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_sha256"] = checkpoint_digest
    test_ids, test_y, test_pred, test_probs = collect_probabilities(model, test_loader, device)
    test_labels = full.label_encoder.inverse_transform(test_y)
    pred_labels = full.label_encoder.inverse_transform(test_pred)
    prediction_path = output_dir / f"{stem}_test_predictions.npz"
    np.savez_compressed(prediction_path, ids=np.asarray(test_ids, dtype=object), y_true=np.asarray(test_y, dtype=np.int64), y_pred=np.asarray(test_pred, dtype=np.int64), labels=np.asarray(test_labels, dtype=object), predicted_labels=np.asarray(pred_labels, dtype=object), probabilities=test_probs, dataset=np.asarray(args.dataset), model_mode=np.asarray(args.model_mode), seed=np.asarray(
        args.model_seed, dtype=np.int64), model_seed=np.asarray(args.model_seed, dtype=np.int64), split_seed=np.asarray(args.split_seed, dtype=np.int64), checkpoint=np.asarray(str(checkpoint_path)), checkpoint_sha256=np.asarray(checkpoint_digest), label_classes=np.asarray(full.label_encoder.classes_, dtype=object), fragment_counts=np.asarray(fragment_counts, dtype=np.int32), )
    metrics["prediction_export"] = str(prediction_path)
    print(f"Saved test predictions: {prediction_path}")
    torch.save({"model_state": model.state_dict(), "label_classes": full.label_encoder.classes_.tolist(), "dims": dims, "args": vars(args), "checkpoint_path": str(
        checkpoint_path), "checkpoint_sha256": checkpoint_digest, "min_class_count": int(args.min_class_count), "class_counts_after": full.class_counts_after, }, output_dir / f"{stem}.pt", )
    embedding_exports = {}
    if args.export_embeddings:
        train_export_loader = make_data_loader(train, args, shuffle=False, seed_offset=3, persistent=False, )
        split_loaders = (("train", train_export_loader), ("validation", val_loader), ("test", test_loader), )
        for split_name, split_loader in split_loaders:
            embedding_path = output_dir / f"{stem}_{split_name}_embeddings.npz"
            export_embeddings(model=model, dataset=args.dataset, split_name=split_name, loader=split_loader, label_encoder=full.label_encoder, output_path=embedding_path, device=device, min_class_count=args.min_class_count,
                              model_mode=args.model_mode, model_seed=args.model_seed, split_seed=args.split_seed, checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_digest, fragment_counts=fragment_counts, )
            embedding_exports[split_name] = str(embedding_path)
    metrics["embedding_exports"] = embedding_exports
    metrics_path = output_dir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8", )
    print(f"Best checkpoint saved to " f"{checkpoint_path}")
    print(f"Best/exported model outputs " f"saved to {output_dir}")
    print("Final metrics: " f"ACC={metrics['accuracy']:.4f} " f"BA={metrics['balanced_accuracy']:.4f} " f"F1={metrics['macro_f1']:.4f} " f"Recall={metrics['macro_recall']:.4f} " f"Precision=" f"{metrics['macro_precision']:.4f}")
    return metrics


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train the EXP3_C multiscale path-complex classifier.")
    parser.add_argument("--dataset", required=True)
    parser.set_defaults(data_dir="./src/data")
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-dir", default="./results/deep_viral_classification")
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    parser.add_argument("--label-column", default="Family")
    parser.add_argument("--accession-column", default=None)
    parser.add_argument("--protocol", choices=["1nn", "5nn"], default="1nn", help="CAKR comparison protocol: 1nn uses minimum class count 3; 5nn uses 15.", )
    parser.add_argument("--min-class-count", type=int, default=None, help="Override the protocol-specific minimum family count.", )
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--pcnn-layers", type=int, default=3)
    parser.add_argument("--pcnn-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pcnn-direction", choices=["bidirectional", "upward"], default="bidirectional", )
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adamw")
    parser.add_argument("--scheduler", choices=["none", "onecycle"], default="onecycle")
    parser.add_argument("--onecycle-pct-start", type=float, default=0.3)
    parser.add_argument("--onecycle-div-factor", type=float, default=25.0)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=10000.0)
    parser.add_argument("--class-weighted-loss", action="store_true")
    parser.add_argument("--gradient-clip-norm", type=float, default=0.0)
    parser.add_argument("--checkpoint-metric", choices=["val_macro_f1", "val_1nn_accuracy"], default="val_macro_f1", )
    parser.add_argument("--checkpoint-knn-metric", choices=["cosine", "euclidean", "manhattan"], default="cosine", )
    parser.add_argument("--checkpoint-eval-interval", type=int, default=5)
    parser.add_argument("--checkpoint-distance-jobs", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True, )
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True, )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export-embeddings", action=argparse.BooleanOptionalAction, default=True, )
    return parser


def main():
    args = build_arg_parser().parse_args()
    train_deep_viral_classification(args)


if __name__ == "__main__":
    main()
