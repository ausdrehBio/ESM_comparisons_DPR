#!/usr/bin/env python3
"""Embed every sequence of a FASTA file (one shard) with ESM-2 650M or ESM-C 600M.

Both models share batching, pooling, checks and output format; only model loading and
tokenisation differ, so each adapter imports its own package lazily and the script runs in
either model env. Embeddings are the final-layer, post-LayerNorm hidden states of the
residue positions (BOS/EOS/padding excluded); the sequence embedding is their mean.
Per-residue embeddings are kept only for the residues listed in --residue_sample.

Writes <outdir>/embeddings.npz (ids, lengths, mean, residue_rows, per_residue) and
<outdir>/meta.json; residue_rows are the rows of residue_sample.tsv held in per_residue.
"""

import argparse
import json
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

ESMC_REPO = "biohub/ESMC-600M"
EXPECTED_DIM = {"esm2_650m": 1280, "esmc_600m": 1152}
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
# Minimum per-residue cosine between a sequence embedded inside a padded batch and alone.
PADDING_MIN_COSINE = {"fp32": 0.9999, "bf16": 0.99}


class Esm2:
    """ESM-2 650M through fair-esm."""

    def __init__(self, device, dtype):
        import esm

        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.model = model.eval().to(device=device, dtype=dtype)
        self.converter = alphabet.get_batch_converter()
        self.bos, self.eos = alphabet.cls_idx, alphabet.eos_idx
        self.layer = model.num_layers
        self.device = device
        self.info = {
            "package": "fair-esm",
            "package_version": esm.__version__,
            "weights": "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt",
            "layer": self.layer,
        }

    def tokenize(self, seqs):
        _, _, tokens = self.converter([(str(i), seq) for i, seq in enumerate(seqs)])
        return tokens

    def hidden(self, tokens):
        # The last entry of repr_layers is taken after emb_layer_norm_after.
        return self.model(tokens.to(self.device), repr_layers=[self.layer])["representations"][self.layer]


class Esmc:
    """ESM-C 600M through EvolutionaryScale's esm package."""

    def __init__(self, device, dtype):
        import esm
        from esm.models.esmc import EsmcModel, EsmcTokenizer, kernels
        from esm.models.hub import resolve_model_dir

        weights_dir = resolve_model_dir(ESMC_REPO, local_files_only=True)
        # dtype must be explicit: without it the checkpoint's stored dtype is kept.
        self.model = EsmcModel.from_pretrained(weights_dir, device=device, dtype=dtype)
        self.tokenizer = EsmcTokenizer()
        self.bos, self.eos = self.tokenizer.cls_token_id, self.tokenizer.eos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.layer = self.model.config.num_hidden_layers
        self.device = device
        self.info = {
            "package": "esm",
            "package_version": esm.__version__,
            "weights": ESMC_REPO,
            "weights_revision": Path(weights_dir).name,
            "layer": self.layer,
            "attn_implementation": self.model.config.attn_implementation,
            "fused_kernels_installed": {
                "transformer_engine": kernels.TE_INSTALLED,
                "xformers": kernels.XFORMERS_INSTALLED,
                "flash_attn": kernels.FLASH_ATTN_INSTALLED,
            },
        }

    def tokenize(self, seqs):
        return self.tokenizer(seqs, return_tensors="pt", padding=True)["input_ids"]

    def hidden(self, tokens):
        tokens = tokens.to(self.device)
        return self.model(input_ids=tokens, attention_mask=tokens != self.pad).last_hidden_state


ADAPTERS = {"esm2_650m": Esm2, "esmc_600m": Esmc}


def read_fasta(path):
    records, name, chunks = [], None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name is not None:
                records.append((name, "".join(chunks)))
            name, chunks = line[1:].split()[0], []
        else:
            chunks.append(line)
    if name is not None:
        records.append((name, "".join(chunks)))
    return records


def read_residue_sample(path, records):
    """The residue-sample rows that fall in these records: {sequence_id: [(row, position), ...]}."""
    seq_of = dict(records)
    picks = {}
    with open(path) as fh:
        next(fh)  # header
        for row, line in enumerate(fh):
            sid, pos, aa = line.rstrip("\n").split("\t")
            if sid not in seq_of:
                continue
            pos = int(pos)
            if pos >= len(seq_of[sid]) or seq_of[sid][pos] != aa:
                sys.exit(f"embed: residue sample row {row} ({sid} position {pos} = {aa}) does not match the FASTA")
            picks.setdefault(sid, []).append((row, pos))
    return picks


def make_batches(records, max_tokens):
    """Group record indices longest-first so each padded batch (incl. BOS/EOS) fits max_tokens."""
    order = sorted(range(len(records)), key=lambda i: len(records[i][1]), reverse=True)
    batches, current = [], []
    for i in order:
        if current and (len(current) + 1) * (len(records[current[0]][1]) + 2) > max_tokens:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


def residue_states(adapter, seqs):
    """Per-residue final hidden states, one float32 [L, d] array per sequence."""
    tokens = adapter.tokenize(seqs)
    for row, seq in enumerate(seqs):
        if tokens[row, 0] != adapter.bos or tokens[row, len(seq) + 1] != adapter.eos:
            sys.exit(f"embed: unexpected token layout for a length-{len(seq)} sequence; expected BOS seq EOS pad")
    with torch.inference_mode():
        hidden = adapter.hidden(tokens).float().cpu().numpy()
    return [hidden[row, 1 : len(seq) + 1] for row, seq in enumerate(seqs)]


def most_padded(records, batches):
    """(index, padding, padded length) of the sequence with the most padding in its batch."""
    worst = (None, 0, 0)
    for batch in batches:
        width = max(len(records[i][1]) for i in batch)
        for i in batch:
            if width - len(records[i][1]) > worst[1]:
                worst = (i, width - len(records[i][1]), width)
    return worst


def padding_check(adapter, records, worst, batched, min_cosine):
    """Re-embed the most-padded sequence alone; padding must not change its embedding."""
    i, padding, width = worst
    if i is None:
        return {"skipped": "no sequence was padded"}

    alone = residue_states(adapter, [records[i][1]])[0]
    cosine = (alone * batched).sum(1) / (np.linalg.norm(alone, axis=1) * np.linalg.norm(batched, axis=1))
    result = {
        "sequence_id": records[i][0],
        "length": len(records[i][1]),
        "padded_to": width,
        "min_cosine": float(cosine.min()),
        "max_abs_diff": float(np.abs(alone - batched).max()),
    }
    if result["min_cosine"] < min_cosine:
        sys.exit(f"embed: padding changes embeddings (min cosine < {min_cosine}): {result}")
    return result


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--residue_sample", required=True, help="residue_sample.tsv from the reference build")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--precision", default="fp32", choices=sorted(DTYPES))
    ap.add_argument("--max_tokens", type=int, default=8192, help="padded tokens per batch")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("embed: --device cuda requested but torch.cuda.is_available() is False")
    torch.manual_seed(0)
    # Plain fp32 matmuls for both models; TF32 would blur the comparison.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    records = read_fasta(args.fasta)
    if not records:
        sys.exit(f"embed: no sequences in {args.fasta}")
    picks = read_residue_sample(args.residue_sample, records)
    slot, residue_rows = {}, []  # where each sequence's sampled residues go in per_residue
    for sid, _ in records:
        if sid in picks:
            slot[sid] = len(residue_rows)
            residue_rows += [row for row, _ in picks[sid]]
    batches = make_batches(records, args.max_tokens)
    worst = most_padded(records, batches)

    t0 = time.perf_counter()
    adapter = ADAPTERS[args.model](args.device, DTYPES[args.precision])
    sync(args.device)
    load_s = time.perf_counter() - t0

    # Filled in place: a shard's outputs are held once, never as a list plus a concatenated copy.
    dim = EXPECTED_DIM[args.model]
    mean = np.empty((len(records), dim), dtype=np.float32)
    per_residue = np.empty((len(residue_rows), dim), dtype=np.float32)
    worst_state = None
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    report_every = max(1, len(batches) // 20)
    t0 = time.perf_counter()
    for n, batch in enumerate(batches, 1):
        states = residue_states(adapter, [records[i][1] for i in batch])
        for i, state in zip(batch, states):
            if state.shape[1] != dim:
                sys.exit(f"embed: {args.model} produced dim {state.shape[1]}, expected {dim}")
            mean[i] = state.mean(0)
            sid = records[i][0]
            if sid in picks:
                positions = [pos for _, pos in picks[sid]]
                per_residue[slot[sid] : slot[sid] + len(positions)] = state[positions]
            if i == worst[0]:
                worst_state = state.copy()
        if n % report_every == 0 or n == len(batches):
            print(f"embed: {args.model}: batch {n}/{len(batches)}", file=sys.stderr)
    sync(args.device)
    embed_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else None

    padding = padding_check(adapter, records, worst, worst_state, PADDING_MIN_COSINE[args.precision])
    if not (np.isfinite(mean).all() and np.isfinite(per_residue).all()):
        sys.exit("embed: non-finite embeddings")

    lengths = np.array([len(seq) for _, seq in records], dtype=np.int64)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / "embeddings.npz",
        ids=np.array([sid for sid, _ in records]),
        lengths=lengths,
        mean=mean,
        residue_rows=np.array(residue_rows, dtype=np.int64),
        per_residue=per_residue,
    )
    meta = {
        "model": args.model,
        **adapter.info,
        "embedding_dim": dim,
        "shard": Path(args.fasta).name,
        "n_sequences": len(records),
        "n_residues": int(lengths.sum()),
        "n_sampled_residues": len(residue_rows),
        "precision": args.precision,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python_version": platform.python_version(),
        "host": socket.gethostname(),
        "max_tokens": args.max_tokens,
        "n_batches": len(batches),
        "load_seconds": round(load_s, 3),
        "embed_seconds": round(embed_s, 3),
        "residues_per_second": round(int(lengths.sum()) / embed_s, 1),
        "peak_gpu_memory_gb": round(peak_gb, 3) if peak_gb is not None else None,
        "padding_check": padding,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"embed: {args.model} {meta['shard']}: {len(records)} sequences, d={dim}, "
        f"load {load_s:.1f}s, embed {embed_s:.1f}s, padding check {padding}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
