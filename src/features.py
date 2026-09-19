import csv
from pathlib import Path

from .prepare_data import DATASETS
from .utils import safe_id

MISSING_LABEL_VALUES = {"", "na", "n/a", "nan", "none", "null"}
ACCESSION_COLUMNS = ("Accession (version)", "accession", "Accession", "accession_version", "id", "ID")


def normalize_label(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in MISSING_LABEL_VALUES else value


def dataset_csv_path(dataset, data_dir):
    """Return metadata path for a registered or filtered dataset name."""
    csv_name = DATASETS.get(dataset, (f"{dataset}.csv", None))[0]
    path = Path(data_dir) / csv_name
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    return path


def read_metadata(dataset, data_dir, accession_column=None):
    path = dataset_csv_path(dataset, data_dir)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = reader.fieldnames or []
    accession_column = accession_column or next((name for name in ACCESSION_COLUMNS if name in fields), fields[0] if fields else None)
    if accession_column not in fields:
        raise ValueError(f"{path} is missing accession column {accession_column}.")
    return rows, accession_column, fields


def feature_path(feature_root, dataset, sequence_id):
    return Path(feature_root) / dataset / f"{safe_id(sequence_id)}.h5"
