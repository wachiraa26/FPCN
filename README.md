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

## Feature generation
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

## Model training

Train FPCN on one benchmark dataset with:
### Benchmark datasets
```bash
python -u main.py deep_viral_classification \
  --dataset NCBI_record_valid_nucleotide \
  --feature-root ./features_2024_multi_15_17_21_23 \
  --output-dir ./results/deep_viral_classification \
  --checkpoint-dir ./checkpoints \
  --model-seed 42 \
  --split-seed 42
```

## Data availability

The original viral genome records were obtained from the [NCBI Virus database](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/). Preprocessed NCBI benchmark datasets are available from [Zenodo record 19655917](https://zenodo.org/records/19655917). Processed NCBI 2026 datasets are available from [Zenodo record 22849345](https://zenodo.org/records/22849345).

## Citation

If you use FPCN, please cite the associated manuscript. Citation details will be added when the paper becomes available.

