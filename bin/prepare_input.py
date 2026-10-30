#!/usr/bin/env python3
"""Reference build: validate the benchmark CSV and write the pipeline's inputs.

The CSV needs a sequence_id and a sequence column, in any position; other columns are ignored.
Stops with the offending line on: a missing or duplicated required column, a row whose column
count differs from the header, an empty/duplicate/whitespace-containing ID, a residue outside
the 20 canonical amino acids, or a sequence longer than ESM-2's limit.

Writes to --outdir:
  sequences.fasta       all sequences, in input order
  manifest.tsv          sequence_id, length and amino-acid composition per sequence
  shards/shard_*.fasta  the same sequences in chunks of SHARD_SIZE. Each shard is one EMBED task
                        per model, so -resume keeps finished shards when a long run is interrupted
  residue_sample.tsv    residues whose per-residue embeddings are kept (sequence_id, 0-based
                        position, amino_acid): all of them for small inputs, otherwise a fixed-seed
                        random sample, so both models store exactly the same residues
A file whose content is unchanged is not rewritten, so its timestamp, which Nextflow's -resume
cache checks, only changes when its content does.
With --check the input is validated and nothing is written.
"""

import argparse
import bisect
import csv
import os
import random
import statistics
import sys
from pathlib import Path

REQUIRED_COLUMNS = ["sequence_id", "sequence"]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MAX_LEN = 1022  # ESM-2 has 1024 positions, two of which hold BOS/EOS
MIN_SEQUENCES = 12  # compare_embeddings.py: kNN overlap at k = 10 needs k + 2 sequences
SHARD_SIZE = 25_000  # sequences per EMBED task: an interrupted run loses at most one shard per model
RESIDUE_SAMPLE = 200_000  # per-residue embeddings kept per model (~1 GB in fp32)
SEED = 0


def fail(msg):
    sys.exit(f"prepare_input: {msg}")


def read_records(path):
    records, first_seen = [], {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = csv.reader(fh)
        header = [cell.strip() for cell in next(rows, [])]
        for column in REQUIRED_COLUMNS:
            if header.count(column) != 1:
                problem = "missing" if column not in header else "appears more than once"
                fail(f"{path}: line 1: column {column!r} {problem}; header is {','.join(header)!r}")
        id_col, seq_col = header.index("sequence_id"), header.index("sequence")
        for row in rows:
            line = rows.line_num
            if not any(cell.strip() for cell in row):
                continue
            # A count mismatch usually means an unquoted comma, which would shift the columns.
            if len(row) != len(header):
                fail(f"line {line}: expected {len(header)} columns like the header, got {len(row)}")
            seq_id, seq = row[id_col].strip(), row[seq_col].strip()
            if not seq_id or len(seq_id.split()) != 1:
                fail(f"line {line}: sequence_id must be non-empty and contain no whitespace, got {seq_id!r}")
            if seq_id in first_seen:
                fail(f"line {line}: duplicate sequence_id {seq_id!r} (first seen on line {first_seen[seq_id]})")
            if not seq:
                fail(f"line {line}: {seq_id}: empty sequence")
            bad = sorted(set(seq) - set(AMINO_ACIDS))
            if bad:
                fail(f"line {line}: {seq_id}: residues outside the 20 canonical amino acids: {''.join(bad)!r}")
            if len(seq) > MAX_LEN:
                fail(f"line {line}: {seq_id}: length {len(seq)} exceeds the ESM-2 limit of {MAX_LEN}")
            first_seen[seq_id] = line
            records.append((seq_id, seq))
    if len(records) < MIN_SEQUENCES:
        fail(f"{path}: {len(records)} sequences; the comparison needs at least {MIN_SEQUENCES}")
    return records


def sample_residues(records):
    """(sequence index, position) pairs in sequence order: every residue, or RESIDUE_SAMPLE of them."""
    offsets = [0]
    for _, seq in records:
        offsets.append(offsets[-1] + len(seq))
    total = offsets[-1]
    picks = range(total) if total <= RESIDUE_SAMPLE else sorted(random.Random(SEED).sample(range(total), RESIDUE_SAMPLE))
    pairs = []
    for flat in picks:
        i = bisect.bisect_right(offsets, flat) - 1
        pairs.append((i, flat - offsets[i]))
    return pairs


def fasta(records):
    return "".join(f">{sid}\n{seq}\n" for sid, seq in records)


def write_if_changed(path, text):
    """Atomic write that leaves an identical existing file, and its timestamp, untouched."""
    if path.exists() and path.read_text() == text:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="input CSV with sequence_id and sequence columns")
    ap.add_argument("--outdir", help="where to write the outputs listed above")
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()
    if not args.check and not args.outdir:
        ap.error("--outdir is required unless --check is given")

    records = read_records(args.csv)
    lengths = [len(seq) for _, seq in records]
    print(
        f"prepare_input: {len(records)} valid sequences, {sum(lengths)} residues, length "
        f"{min(lengths)}-{max(lengths)} (median {statistics.median(lengths):g})",
        file=sys.stderr,
    )
    if args.check:
        return

    outdir = Path(args.outdir)
    shard_dir = outdir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    write_if_changed(outdir / "sequences.fasta", fasta(records))

    lines = ["\t".join(["sequence_id", "length"] + [f"frac_{aa}" for aa in AMINO_ACIDS])]
    for sid, seq in records:
        fracs = [f"{seq.count(aa) / len(seq):.6f}" for aa in AMINO_ACIDS]
        lines.append("\t".join([sid, str(len(seq))] + fracs))
    write_if_changed(outdir / "manifest.tsv", "\n".join(lines) + "\n")

    shard_names = set()
    for k, start in enumerate(range(0, len(records), SHARD_SIZE)):
        name = f"shard_{k:05d}.fasta"
        shard_names.add(name)
        write_if_changed(shard_dir / name, fasta(records[start : start + SHARD_SIZE]))
    for stale in shard_dir.glob("shard_*.fasta"):
        if stale.name not in shard_names:
            stale.unlink()

    pairs = sample_residues(records)
    lines = ["sequence_id\tposition\tamino_acid"]
    lines += [f"{records[i][0]}\t{pos}\t{records[i][1][pos]}" for i, pos in pairs]
    write_if_changed(outdir / "residue_sample.tsv", "\n".join(lines) + "\n")
    print(
        f"prepare_input: {len(shard_names)} shard(s) of up to {SHARD_SIZE} sequences; "
        f"per-residue embeddings kept for {len(pairs)} of {sum(lengths)} residues",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
