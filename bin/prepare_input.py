#!/usr/bin/env python3
"""Reference build: validate the benchmark CSV and write sequences.fasta + manifest.tsv.

The CSV needs a sequence_id and a sequence column, in any position; other columns are ignored.
Stops with the offending line on: a missing or duplicated required column, a row whose column
count differs from the header, an empty/duplicate/whitespace-containing ID, a residue outside
the 20 canonical amino acids, or a sequence longer than ESM-2's limit.
With --check the input is validated and nothing is written.
"""

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

REQUIRED_COLUMNS = ["sequence_id", "sequence"]
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
MAX_LEN = 1022  # ESM-2 has 1024 positions, two of which hold BOS/EOS
MIN_SEQUENCES = 12  # compare_embeddings.py: kNN overlap at k = 10 needs k + 2 sequences


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


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="input CSV with sequence_id and sequence columns")
    ap.add_argument("--outdir", help="where to write sequences.fasta and manifest.tsv")
    ap.add_argument("--check", action="store_true", help="validate only, write nothing")
    args = ap.parse_args()
    if not args.check and not args.outdir:
        ap.error("--outdir is required unless --check is given")

    records = read_records(args.csv)
    lengths = [len(seq) for _, seq in records]
    print(
        f"prepare_input: {len(records)} valid sequences, length {min(lengths)}-{max(lengths)} "
        f"(median {statistics.median(lengths):g})",
        file=sys.stderr,
    )
    if args.check:
        return

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_atomic(outdir / "sequences.fasta", "".join(f">{sid}\n{seq}\n" for sid, seq in records))

    lines = ["\t".join(["sequence_id", "length"] + [f"frac_{aa}" for aa in AMINO_ACIDS])]
    for sid, seq in records:
        fracs = [f"{seq.count(aa) / len(seq):.6f}" for aa in AMINO_ACIDS]
        lines.append("\t".join([sid, str(len(seq))] + fracs))
    write_atomic(outdir / "manifest.tsv", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
