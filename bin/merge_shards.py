#!/usr/bin/env python3
"""Merge one model's per-shard EMBED outputs into a single embeddings.npz + meta.json.

Shards are concatenated in name order, which is FASTA order. Each sampled per-residue
embedding goes to its row of residue_sample.tsv, so both models' per_residue arrays line up
row for row. Writes <outdir>/embeddings.npz (ids, lengths, mean, residue_ids,
residue_positions, per_residue) and <outdir>/meta.json, where load_seconds is the median time
of one model load (one per shard) and embed_seconds the total across shards.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

# Keys that describe one shard's task; everything else must be identical across shards.
PER_SHARD = {"shard", "n_sequences", "n_residues", "n_sampled_residues", "n_batches", "load_seconds",
             "embed_seconds", "residues_per_second", "peak_gpu_memory_gb", "padding_check", "host", "gpu"}


def fail(msg):
    sys.exit(f"merge: {msg}")


def read_residue_sample(path):
    ids, positions = [], []
    with open(path) as fh:
        next(fh)  # header
        for line in fh:
            sid, pos, _ = line.rstrip("\n").split("\t")
            ids.append(sid)
            positions.append(int(pos))
    return np.array(ids), np.array(positions, dtype=np.int64)


def merge_meta(shards, metas, n_sequences, n_residues, n_sampled):
    static = {key: value for key, value in metas[0].items() if key not in PER_SHARD}
    for shard, meta in zip(shards, metas):
        for key, value in static.items():
            if meta.get(key) != value:
                fail(f"{shard.name}: {key} = {meta.get(key)!r}, but {shards[0].name} has {value!r}")
    embed_s = sum(meta["embed_seconds"] for meta in metas)
    gpus = sorted({meta["gpu"] for meta in metas if meta["gpu"]})
    peaks = [meta["peak_gpu_memory_gb"] for meta in metas if meta["peak_gpu_memory_gb"] is not None]
    checks = [meta["padding_check"] for meta in metas if "min_cosine" in meta["padding_check"]]
    return {
        **static,
        "n_shards": len(shards),
        "n_sequences": n_sequences,
        "n_residues": n_residues,
        "n_sampled_residues": n_sampled,
        # A resubmitted run can land on another node or GPU type; list them all.
        "gpu": ", ".join(gpus) or None,
        "host": ", ".join(sorted({meta["host"] for meta in metas})),
        "load_seconds": round(statistics.median(meta["load_seconds"] for meta in metas), 3),
        "embed_seconds": round(embed_s, 3),
        "residues_per_second": round(n_residues / embed_s, 1),
        "peak_gpu_memory_gb": max(peaks) if peaks else None,
        "padding_check": min(checks, key=lambda c: c["min_cosine"]) if checks else {"skipped": "no sequence was padded"},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--residue_sample", required=True, help="residue_sample.tsv from the reference build")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("shards", nargs="+", help="this model's EMBED output dirs")
    args = ap.parse_args()

    shards = sorted((Path(p) for p in args.shards), key=lambda p: p.name)
    metas = [json.loads((shard / "meta.json").read_text()) for shard in shards]
    if {meta["model"] for meta in metas} != {args.model}:
        fail(f"expected only {args.model} shards, got {sorted({meta['model'] for meta in metas})}")
    dim = metas[0]["embedding_dim"]
    n = sum(meta["n_sequences"] for meta in metas)
    residue_ids, residue_positions = read_residue_sample(args.residue_sample)

    # Preallocated and filled shard by shard: the merged mean array is several GB at full scale.
    ids, lengths = [], []
    mean = np.empty((n, dim), dtype=np.float32)
    per_residue = np.empty((len(residue_ids), dim), dtype=np.float32)
    filled = np.zeros(len(residue_ids), dtype=bool)
    start = 0
    for shard in shards:
        with np.load(shard / "embeddings.npz") as z:
            shard_ids, rows = z["ids"], z["residue_rows"]
            ids.append(shard_ids)
            lengths.append(z["lengths"])
            mean[start : start + len(shard_ids)] = z["mean"]
            if len(np.unique(rows)) != len(rows) or filled[rows].any():
                fail(f"{shard.name}: residue sample rows stored twice")
            per_residue[rows] = z["per_residue"]
            filled[rows] = True
        start += len(shard_ids)
    ids, lengths = np.concatenate(ids), np.concatenate(lengths)
    if start != n or len(set(ids.tolist())) != n:
        fail("shards overlap or disagree with their meta.json sequence counts")
    if not filled.all():
        fail(f"{int((~filled).sum())} sampled residues are missing from the shards")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / "embeddings.npz",
        ids=ids,
        lengths=lengths,
        mean=mean,
        residue_ids=residue_ids,
        residue_positions=residue_positions,
        per_residue=per_residue,
    )
    meta = merge_meta(shards, metas, n, int(lengths.sum()), len(residue_ids))
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"merge: {args.model}: {len(shards)} shard(s), {n} sequences, {len(residue_ids)} sampled residues",
          file=sys.stderr)


if __name__ == "__main__":
    main()
