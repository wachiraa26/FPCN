import numpy as np


class PathComplex:
    """A path complex derived from an ordered sequence of fragment embeddings."""

    def __init__(self, embeddings):
        path0 = np.asarray(embeddings, dtype=np.float32)
        if path0.ndim != 2 or path0.shape[0] == 0:
            raise ValueError("Fragment embeddings must be a non-empty two-dimensional matrix.")
        self.path0 = path0
        count, dimension = path0.shape
        self.path1 = 0.5 * (path0[:-1] + path0[1:]) if count >= 2 else np.empty((0, dimension), dtype=np.float32)
        self.path2 = (path0[:-2] + path0[1:-1] + path0[2:]) / 3.0 if count >= 3 else np.empty((0, dimension), dtype=np.float32)
        self.path0_link = np.vstack((np.arange(count - 1), np.arange(1, count))).astype(np.int64) if count >= 2 else np.empty((2, 0), dtype=np.int64)
        self.path1_link = np.vstack((np.arange(count - 2), np.arange(1, count - 1))).astype(np.int64) if count >= 3 else np.empty((2, 0), dtype=np.int64)


def build_multiscale_path_complex(embeddings_by_count, counts):
    """Build one ordered path complex for every requested fragment count."""
    counts = [int(count) for count in counts]
    return {count: PathComplex(embeddings_by_count[count]) for count in counts}