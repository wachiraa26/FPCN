import argparse
import json
from pathlib import Path

import numpy as np

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    from . import deep_viral_classification as deep
except ImportError:
    from src.tasks import deep_viral_classification as deep


def build_arg_parser():
    parser = deep.build_arg_parser()
    parser.description = "Run one fold of repeated internal cross-validation."
    parser.add_argument(
        "--cv-repeats", type=int, default=1,
        help="Total repeats planned; this process trains only one fold.",
    )
    parser.add_argument(
        "--repeat-index", type=int, default=0,
        help="Zero-based repeat index.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds.",
    )
    parser.add_argument(
        "--cv-fold-index",
        type=int,
        required=True,
        help="Zero-based held-out fold index.",
    )
    parser.add_argument(
        "--cv-validation-size",
        type=float,
        default=0.20,
        help="Validation fraction drawn from the non-test portion of each fold.",
    )
    return parser


def make_fold_loaders(args):
    if args.cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if not 0 <= args.cv_fold_index < args.cv_folds:
        raise ValueError("cv_fold_index must be in [0, cv_folds).")
    if not 0 < args.cv_validation_size < 1:
        raise ValueError("cv_validation_size must lie strictly between 0 and 1.")
    if args.min_class_count < args.cv_folds:
        raise ValueError(
            "min_class_count must be at least cv_folds for stratified cross-validation."
        )

    full = deep.DNAFeatureDataset(dataset=args.dataset,data_dir=args.data_dir,feature_root=args.feature_root,label_column=args.label_column,accession_column=args.accession_column,min_class_count=args.min_class_count,)
    if len(set(map(str, full.ids))) != len(full.ids):
        raise ValueError("The retained dataset contains duplicate accession IDs.")
    splitter = StratifiedKFold(n_splits=args.cv_folds,shuffle=True,random_state=args.split_seed,)
    train_val_index, test_index = list(splitter.split(full.ids, full.y))[args.cv_fold_index]

    train_val_ids = full.ids[train_val_index]
    train_val_y = full.y[train_val_index]
    validation_splitter = StratifiedShuffleSplit(n_splits=1,test_size=args.cv_validation_size,random_state=args.split_seed + args.cv_fold_index,)
    train_index, validation_index = next(validation_splitter.split(train_val_ids, train_val_y))

    def subset(ids):
        return deep.DNAFeatureDataset(args.dataset,args.data_dir,args.feature_root,args.label_column,args.accession_column,ids=ids,label_encoder=full.label_encoder,min_class_count=1,)

    train = subset(train_val_ids[train_index])
    validation = subset(train_val_ids[validation_index])
    test = subset(full.ids[test_index])
    split_sets = [set(map(str, dataset.ids)) for dataset in (train, validation, test)]
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Training, validation and test accession IDs overlap.")
    if set.union(*split_sets) != set(map(str, full.ids)):
        raise ValueError("The fold subsets do not cover the retained source dataset.")
    loaders = (
        deep.make_data_loader(train, args, shuffle=True, seed_offset=0),
        deep.make_data_loader(validation, args, shuffle=False, seed_offset=1, persistent=False),
        deep.make_data_loader(test, args, shuffle=False, seed_offset=2, persistent=False),
    )
    return full, train, validation, test, *loaders


def train_internal_cross_validation(args):
    # Work on a copy so repeated programmatic calls do not nest output paths.
    args = argparse.Namespace(**vars(args))
    args.cv_repeats = getattr(args, "cv_repeats", 1)
    args.repeat_index = getattr(args, "repeat_index", 0)
    if args.cv_repeats < 1 or not 0 <= args.repeat_index < args.cv_repeats:
        raise ValueError("repeat_index must be in [0, cv_repeats).")

    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    args.model_seed = args.seed if args.model_seed is None else args.model_seed

    repeat_name = f"repeat_{args.repeat_index + 1:02d}"
    fold_name = f"fold-{args.cv_fold_index + 1}-of-{args.cv_folds}"
    output_dir = Path(args.output_dir) / repeat_name / fold_name
    checkpoint_dir = Path(args.checkpoint_dir) / repeat_name / fold_name
    if output_dir.resolve() == checkpoint_dir.resolve():
        raise ValueError("Output and checkpoint directories must be different.")

    # Do not silently overwrite completed or partially written training runs.
    for directory in (output_dir, checkpoint_dir):
        if directory.exists():
            raise FileExistsError(
                f"Run directory already exists: {directory}. "
                "Use a new run name/output root for a deliberate rerun."
            )

    args.output_dir = str(output_dir)
    args.checkpoint_dir = str(checkpoint_dir)
    datasets = make_fold_loaders(args)
    full, train, validation, test, *_ = datasets

    # Atomic directory creation also protects against duplicate submissions.
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)

    split_path = output_dir / f"{fold_name}_split_ids.npz"
    np.savez_compressed(
        split_path,
        retained_ids=np.asarray(full.ids).astype(str),
        train_ids=np.asarray(train.ids).astype(str),
        validation_ids=np.asarray(validation.ids).astype(str),
        test_ids=np.asarray(test.ids).astype(str),
        repeat_index=np.asarray(args.repeat_index),
        cv_repeats=np.asarray(args.cv_repeats),
        cv_folds=np.asarray(args.cv_folds),
        cv_fold_index=np.asarray(args.cv_fold_index),
        split_seed=np.asarray(args.split_seed),
        validation_split_seed=np.asarray(args.split_seed + args.cv_fold_index),
        model_seed=np.asarray(args.model_seed),
        cv_validation_size=np.asarray(args.cv_validation_size),
    )

    run_metadata = {
        "evaluation_protocol": "repeated_internal_stratified_cross_validation",
        "cv_repeats": int(args.cv_repeats),
        "repeat_index": int(args.repeat_index),
        "cv_folds": int(args.cv_folds),
        "cv_fold_index": int(args.cv_fold_index),
        "split_seed": int(args.split_seed),
        "validation_split_seed": int(args.split_seed + args.cv_fold_index),
        "model_seed": int(args.model_seed),
        "cv_validation_size": float(args.cv_validation_size),
        "n_retained": len(full),
        "n_train": len(train),
        "n_validation": len(validation),
        "n_test": len(test),
        "split_ids_file": str(split_path),
        "arguments": json.loads(json.dumps(vars(args), default=str)),
    }
    metadata_path = output_dir / f"{fold_name}_run_config.json"
    metadata_path.write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"Internal CV repeat {args.repeat_index + 1}/{args.cv_repeats}, "
        f"fold {args.cv_fold_index + 1}/{args.cv_folds}: "
        f"train={len(train)} validation={len(validation)} test={len(test)}"
    )
    print(f"Split seed={args.split_seed}; model seed={args.model_seed}")

    original_make_loaders = deep.make_loaders
    deep.make_loaders = lambda _: datasets
    try:
        metrics = deep.train_deep_viral_classification(args)
    finally:
        deep.make_loaders = original_make_loaders

    metrics.update(run_metadata)
    summary_path = output_dir / f"{fold_name}_metrics.json"
    summary_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Saved repeat/fold metrics: {summary_path}")
    return metrics


def main():
    train_internal_cross_validation(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
