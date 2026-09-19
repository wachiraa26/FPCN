import numpy as np

DNA_ALPHABET = frozenset("ACGT")
IUPAC_DNA_ALPHABET = frozenset("ACGTRYSWKMBDHVN")


def clean_dna(sequence):
    """Normalize DNA while preserving valid IUPAC ambiguity symbols and positions."""
    sequence = "".join(str(sequence).upper().split())
    if not sequence:
        raise ValueError("DNA sequence is empty.")
    invalid = sorted(set(sequence) - IUPAC_DNA_ALPHABET)
    if invalid:
        raise ValueError("Unsupported DNA symbols: " + ", ".join(invalid))
    return sequence


def canonical_runs(sequence):
    """Return maximal canonical A/C/G/T runs without changing coordinates."""
    sequence = clean_dna(sequence)
    runs, start = [], None
    for index, base in enumerate(sequence):
        if base in DNA_ALPHABET:
            start = index if start is None else start
        elif start is not None:
            runs.append(sequence[start:index])
            start = None
    if start is not None:
        runs.append(sequence[start:])
    return runs


def equal_coordinate_segments(sequence, count):
    """Divide a genome into ``count`` ordered, non-empty equal-coordinate segments."""
    sequence = clean_dna(sequence)
    count = int(count)
    if count < 1 or count > len(sequence):
        raise ValueError(f"segment count must be between 1 and {len(sequence)}, got {count}.")
    boundaries = np.floor(np.linspace(0, len(sequence), count + 1)).astype(int)
    return [sequence[start:end] for start, end in zip(boundaries[:-1], boundaries[1:])]


def split_windows(sequence, window_size):
    """Split a segment into non-overlapping windows for DNABERT-2 pooling."""
    return [sequence[start : start + window_size] for start in range(0, len(sequence), window_size)]


class DNABert2Encoder:
    """Frozen multimolecule DNABERT-2 encoder with canonical-token mean pooling."""

    def __init__(self, device=None, max_token_length=512):
        try:
            import torch
            from multimolecule import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("DNABERT-2 feature generation requires torch and multimolecule.") from exc

        self.torch = torch
        self.max_token_length = int(max_token_length)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "multimolecule/dnabert2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.last_valid_mask = None

    def hidden_size(self):
        return int(self.model.config.hidden_size)

    def encode(self, fragments, batch_size):
        vectors, valid_masks = [], []
        for start in range(0, len(fragments), batch_size):
            batch = [clean_dna(fragment) for fragment in fragments[start : start + batch_size]]
            runs, owners = [], []
            for owner, fragment in enumerate(batch):
                for run in canonical_runs(fragment):
                    runs.append(run)
                    owners.append(owner)

            if not runs:
                vectors.append(np.zeros((len(batch), self.hidden_size()), dtype=np.float32))
                valid_masks.append(np.zeros(len(batch), dtype=bool))
                continue

            tokens = self.tokenizer(
                runs,
                padding=True,
                truncation=True,
                max_length=self.max_token_length,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            offsets = tokens.pop("offset_mapping")
            special_tokens = tokens.pop("special_tokens_mask")
            attention_mask = tokens["attention_mask"].bool()
            canonical_mask = self.torch.zeros_like(attention_mask)
            for row, (run, row_offsets) in enumerate(zip(runs, offsets.tolist())):
                for column, (left, right) in enumerate(row_offsets):
                    if right > left and all(base in DNA_ALPHABET for base in run[left:right]):
                        canonical_mask[row, column] = True

            pool_mask = attention_mask & canonical_mask & ~special_tokens.bool()
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with self.torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state

            mask = pool_mask.to(self.device).unsqueeze(-1).to(hidden.dtype)
            run_sums = (hidden * mask).sum(dim=1)
            run_counts = mask.sum(dim=1)
            owner_index = self.torch.as_tensor(owners, dtype=self.torch.long, device=self.device)
            sums = self.torch.zeros((len(batch), hidden.shape[-1]), dtype=hidden.dtype, device=self.device)
            counts = self.torch.zeros((len(batch), 1), dtype=hidden.dtype, device=self.device)
            sums.index_add_(0, owner_index, run_sums)
            counts.index_add_(0, owner_index, run_counts)
            valid = counts[:, 0] > 0
            pooled = sums / counts.clamp(min=1)
            pooled[~valid] = 0
            vectors.append(pooled.cpu().numpy())
            valid_masks.append(valid.cpu().numpy())

        self.last_valid_mask = np.concatenate(valid_masks)
        return np.concatenate(vectors).astype(np.float32)


def encode_segments(sequence, encoder, segment_count, window_size, batch_size):
    """Encode each equal-coordinate segment by mean-pooling its 512-nt windows."""
    segments = equal_coordinate_segments(sequence, segment_count)
    windows, owners = [], []
    for segment_index, segment in enumerate(segments):
        segment_windows = split_windows(segment, window_size)
        windows.extend(segment_windows)
        owners.extend([segment_index] * len(segment_windows))

    window_embeddings = encoder.encode(windows, batch_size=batch_size)
    output = np.zeros((len(segments), window_embeddings.shape[1]), dtype=np.float32)
    counts = np.zeros(len(segments), dtype=np.int64)
    for embedding, owner, valid in zip(window_embeddings, owners, encoder.last_valid_mask):
        if valid:
            output[owner] += embedding
            counts[owner] += 1
    nonempty = counts > 0
    output[nonempty] /= counts[nonempty, None]
    return output