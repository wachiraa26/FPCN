import argparse
import csv
import os
from pathlib import Path

from .prepare_raw import DNABert2Encoder, clean_dna, encode_segments
from .topo import build_multiscale_path_complex
from .utils import safe_id, save_multiscale_path_complex

DATASETS = {
    "Yau2020_record_processed": ("Yau2020_record_processed.csv", "Yau2020_record_processed.fasta"),
    "Yau2022_record_processed": ("Yau2022_record_processed.csv", "Yau2022_record_processed.fasta"),
    "NCBI_record_valid_nucleotide": ("NCBI_record_valid_nucleotide.csv", "NCBI_record_valid_nucleotide.fasta"),
    "NCBI_record_valid_count": ("NCBI_record_valid_count.csv", "NCBI_record_valid_count.fasta"),
    "NCBI2026_record_valid_nucleotide": ("NCBI2026_record_valid_nucleotide.csv", "NCBI2026_record_valid_nucleotide.fasta"),
    "NCBI2026_record_valid_count": ("NCBI2026_record_valid_count.csv", "NCBI2026_record_valid_count.fasta"),
    "NCBI2026_test_valid_nucleotide": ("NCBI2026_test_valid_nucleotide.csv", "NCBI2026_test_valid_nucleotide.fasta"),
    "NCBI2026_test_valid_count": ("NCBI2026_test_valid_count.csv", "NCBI2026_test_valid_count.fasta"),
}
ACCESSION_COLUMNS = ("Accession (version)", "accession", "Accession", "accession_version", "id", "ID")


def parse_counts(value):
    try:
        counts = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Fragment counts must be comma-separated integers.") from exc
    if not counts or any(count < 1 for count in counts) or len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("Fragment counts must be unique positive integers.")
    return counts


def read_fasta(path):
    sequences, accession, pieces = {}, None, []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if accession is not None:
                    sequences[accession] = clean_dna("".join(pieces))
                accession, pieces = line[1:].split()[0], []
            else:
                pieces.append(line)
    if accession is not None:
        sequences[accession] = clean_dna("".join(pieces))
    return sequences


def metadata_ids(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        column = next((name for name in ACCESSION_COLUMNS if name in fields), fields[0] if fields else None)
        if column is None:
            raise ValueError(f"No accession column found in {path}.")
        return [str(row[column]).strip() for row in reader]


def load_dataset(dataset, data_dir):
    csv_name, fasta_name = DATASETS[dataset]
    ids = metadata_ids(Path(data_dir) / csv_name)
    sequences = read_fasta(Path(data_dir) / fasta_name)
    missing = [sequence_id for sequence_id in ids if sequence_id not in sequences]
    if missing:
        raise ValueError(f"{len(missing)} metadata accessions are absent from FASTA; examples: {missing[:5]}")
    return [(sequence_id, sequences[sequence_id]) for sequence_id in ids]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate multiscale DNABERT-2 path-complex features.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.set_defaults(data_dir="./src/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--multiscale-fixed-fragment-counts", type=parse_counts, required=True, metavar="COUNTS")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-token-length", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--window-size", type=int, default=512, help="DNABERT-2 pooling-window size in bases.")
    return parser


def prepare_features(args):
    if args.max_token_length != 512:
        raise ValueError("This reproducible pipeline uses max-token-length=512.")
    if args.window_size != 512:
        raise ValueError("This reproducible pipeline uses 512-base pooling windows.")

    data = load_dataset(args.dataset, args.data_dir)
    start = int(os.environ.get("START_IDX", "0"))
    end = min(int(os.environ.get("END_IDX", str(len(data)))), len(data))
    selected = data[start:end]
    output_dir = Path(args.output_dir) / args.dataset
    encoder = DNABert2Encoder(device=args.device, max_token_length=args.max_token_length)

    print(f"Dataset: {args.dataset}")
    print(f"Total genomes: {len(data)}")
    print(f"Processing indices: [{start}, {end})")
    print(f"Fragment counts: {args.multiscale_fixed_fragment_counts}")
    for index, (sequence_id, sequence) in enumerate(selected, start=start):
        embeddings = {
            count: encode_segments(sequence, encoder, count, args.window_size, args.batch_size)
            for count in args.multiscale_fixed_fragment_counts
        }
        complexes = build_multiscale_path_complex(embeddings, args.multiscale_fixed_fragment_counts)
        save_multiscale_path_complex(output_dir, sequence_id, complexes, args.multiscale_fixed_fragment_counts, encoder.model_name)
        print(f"{index + 1}/{len(data)} {sequence_id} complete")
    print(f"Processed {len(selected)} genomes.")