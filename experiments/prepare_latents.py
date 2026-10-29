"""Preserve FP32 source latents in a contiguous mmap; no VAE inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def values(array):
    if isinstance(array, pa.ExtensionArray):
        array = array.storage
    while pa.types.is_list(array.type) or pa.types.is_large_list(array.type) or pa.types.is_fixed_size_list(array.type):
        array = array.flatten()
    return array.to_numpy(zero_copy_only=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    root = Path(args.source)
    files = sorted(root.glob("train-*.parquet"))
    assert (root / "download_complete.json").exists()
    counts = [pq.ParquetFile(p).metadata.num_rows for p in files]
    n = sum(counts)
    print(json.dumps(dict(shards=len(files), samples=n)), flush=True)
    target = root / "mmap"
    target.mkdir(exist_ok=True)
    data = np.memmap(target / "latents.f32", dtype="float32", mode="w+", shape=(n, 2, 4, 32, 32))
    labels = np.memmap(target / "labels.i64", dtype="int64", mode="w+", shape=(n,))
    cursor = 0
    for path in files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["latent_mean", "latent_mean_flip", "label"]):
            end = cursor + batch.num_rows
            data[cursor:end, 0] = values(batch.column(0)).reshape(-1, 4, 32, 32)
            data[cursor:end, 1] = values(batch.column(1)).reshape(-1, 4, 32, 32)
            labels[cursor:end] = values(batch.column(2))
            cursor = end
        print(json.dumps(dict(shard=path.name, samples=cursor, total=n)), flush=True)
    data.flush(); labels.flush()
    (target / "metadata.json").write_text(json.dumps(dict(samples=n, shape=[n, 2, 4, 32, 32],
        dtype="float32", scale_factor=.18215, posterior="mean", source_revision=json.loads(
            (root / "download_complete.json").read_text())["revision"]), indent=2))


if __name__ == "__main__":
    main()
