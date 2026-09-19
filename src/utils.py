"""HDF5 storage for multiscale path-complex features."""

from pathlib import Path

import h5py
import numpy as np


def safe_id(sequence_id):
    return str(sequence_id).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _write_array(group, name, value):
    value = np.asarray(value)
    kwargs = {} if 0 in value.shape else {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    group.create_dataset(name, data=value, **kwargs)


def save_multiscale_path_complex(output_dir, sequence_id, complexes, counts, encoder_name):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"{safe_id(sequence_id)}.h5"
    with h5py.File(filename, "w") as handle:
        scales = handle.create_group("scales")
        for count in counts:
            complex_ = complexes[int(count)]
            group = scales.create_group(str(count))
            _write_array(group, "path0", complex_.path0)
            _write_array(group, "path1", complex_.path1)
            _write_array(group, "path2", complex_.path2)
            _write_array(group, "path0_link", complex_.path0_link)
            _write_array(group, "path1_link", complex_.path1_link)

        handle.attrs["sequence_id"] = str(sequence_id)
        handle.attrs["feature_mode"] = "path_only"
        handle.attrs["path_definition"] = "multiscale_fragment"
        handle.attrs["multiscale_fixed_fragment_counts"] = np.asarray(counts, dtype=np.int64)
        handle.attrs["encoder_model"] = encoder_name
        handle.attrs["window_size"] = 512