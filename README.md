# Fragment Path Complex Networks (FPCN)

FPCN is a multiscale framework for viral family classification. 

## Repository structure

```text
FPCN/
├── main.py                         # Training and evaluation entry point
├── prepare.py                      # Multiscale feature generation
└── src/
    ├── data/                       # Dataset metadata and FASTA files
    ├── evaluation/                 # Ensemble and summary utilities
    ├── model/                      # FPCN model
    ├── tasks/                      # Benchmark, internal-CV and temporal tasks
    ├── topo/                       # Ordered path-complex construction
    ├── prepare_data.py
    ├── prepare_raw.py
    └── utils.py
```

## Requirements

FPCN requires Python, PyTorch, NumPy, h5py, scikit-learn and MultiMolecule. Feature preparation downloads the frozen `multimolecule/dnabert2` model from the Hugging Face Hub. A CUDA-capable GPU is recommended for feature generation and training.

Install the core Python dependencies in your environment:

```bash
pip install torch numpy h5py scikit-learn multimolecule
```

## Data preparation

Place each dataset's metadata and FASTA files in `src/data/` using the filenames defined in `src/prepare_data.py`. For example:

```text
src/data/
├── NCBI_record_valid_nucleotide.csv
└── NCBI_record_valid_nucleotide.fasta
```

The accession identifiers in the metadata and FASTA files must match. The default label column used for classification is `Family`.

Generate the four-scale features used by FPCN with:

```bash
python -u prepare.py \
  --dataset NCBI_record_valid_nucleotide \
  --output-dir ./features_2024_multi_15_17_21_23 \
  --multiscale-fixed-fragment-counts "15,17,21,23" \
  --batch-size 4 \
  --device cuda
```

Fragment counts must be comma-separated. Feature files are written to:

```text
features_2024_multi_15_17_21_23/NCBI_record_valid_nucleotide/
```

For array-based preparation, `START_IDX` and `END_IDX` can be used to process a subset of genomes:

```bash
START_IDX=0 END_IDX=50 python -u prepare.py \
  --dataset NCBI_record_valid_nucleotide \
  --output-dir ./features_2024_multi_15_17_21_23 \
  --multiscale-fixed-fragment-counts "15,17,21,23" \
  --batch-size 4 \
  --device cuda
```

## Model training

Train FPCN on one benchmark dataset with:

```bash
python -u main.py deep_viral_classification \
  --dataset NCBI_record_valid_nucleotide \
  --feature-root ./features_2024_multi_15_17_21_23 \
  --output-dir ./results/deep_viral_classification \
  --checkpoint-dir ./checkpoints \
  --model-seed 42 \
  --split-seed 42
```

The current training configuration uses validation macro-F1 for checkpoint selection. Fixed architecture and optimization settings are defined in `src/tasks/deep_viral_classification.py`.

## Internal cross-validation

Run one fold of repeated five-fold cross-validation with:

```bash
python -u main.py internal_cross_validation \
  --dataset NCBI2026_record_valid_count \
  --feature-root ./features_EXP3_C_2026_multi_15_17_21_23 \
  --output-dir ./results_internal_cv/fpcn_cv \
  --checkpoint-dir ./checkpoints_internal_cv/fpcn_cv \
  --min-class-count 10 \
  --cv-repeats 30 \
  --repeat-index 0 \
  --cv-folds 5 \
  --cv-fold-index 0 \
  --model-seed 42 \
  --split-seed 42
```

`repeat-index` ranges from `0` to `29`, and `cv-fold-index` ranges from `0` to `4`. Use distinct model and split seeds for the planned repetitions. These jobs can be distributed with a SLURM array.

## Temporal closed-set evaluation

Train on an earlier NCBI release and evaluate newly deposited genomes from families represented during training:

```bash
python -u main.py external_viral_classification \
  --dataset NCBI_record_valid_count \
  --external-test-dataset NCBI2026_test_valid_count \
  --feature-root ./features_EXP3_C_2024_multi_15_17_21_23 \
  --external-test-feature-root ./features_EXP3_C_2026_test_multi_15_17_21_23 \
  --output-dir ./results_external/fpcn_temporal \
  --checkpoint-dir ./checkpoints_external/fpcn_temporal \
  --model-seed 42 \
  --split-seed 42
```

Source and temporal test features must use the same fragment-count scales.

## Outputs

Training produces:

- model checkpoints (`.pt`);
- test predictions and class probabilities (`.npz`);
- train, validation and test embeddings (`.npz`, for benchmark and internal-CV tasks);
- performance metrics and per-class reports (`.json`).

The exported metrics include accuracy, balanced accuracy, macro-F1, macro-recall and macro-precision.

## Data availability

The original viral genome records were obtained from the [NCBI Virus database](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/). Preprocessed NCBI benchmark datasets are available from [Zenodo record 19655917](https://zenodo.org/records/19655917). Processed NCBI 2026 datasets are available from [Zenodo record 22849345](https://zenodo.org/records/22849345).

## Citation

If you use FPCN, please cite the associated manuscript. Citation details will be added when the paper becomes available.

