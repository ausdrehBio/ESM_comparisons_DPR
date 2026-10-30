#!/usr/bin/env python3
"""Embed every sequence of a FASTA file with ESM-2 650M or ESM-C 600M.

Both models share batching, pooling, checks and output format; only model loading and
tokenisation differ, so each adapter imports its own package lazily and the script runs in
either model env. Embeddings are the final-layer, post-LayerNorm hidden states of the
residue positions (BOS/EOS/padding excluded); the sequence embedding is their mean.

Writes <outdir>/embeddings.npz (ids, lengths, offsets, mean, per_residue) and <outdir>/meta.json.
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
        hidden = adapter.hidden(tokens)
    return [hidden[row, 1 : len(seq) + 1].float().cpu().numpy() for row, seq in enumerate(seqs)]


def padding_check(adapter, records, batches, per_residue, min_cosine):
    """Re-embed the most-padded sequence alone; padding must not change its embedding."""
    worst = (None, 0, 0)  # index, padding, padded length
    for batch in batches:
        width = max(len(records[i][1]) for i in batch)
        for i in batch:
            if width - len(records[i][1]) > worst[1]:
                worst = (i, width - len(records[i][1]), width)
    i, padding, width = worst
    if i is None:
        return {"skipped": "no sequence was padded"}

    alone = residue_states(adapter, [records[i][1]])[0]
    batched = per_residue[i]
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
    batches = make_batches(records, args.max_tokens)

    t0 = time.perf_counter()
    adapter = ADAPTERS[args.model](args.device, DTYPES[args.precision])
    sync(args.device)
    load_s = time.perf_counter() - t0

    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    per_residue = [None] * len(records)
    t0 = time.perf_counter()
    for n, batch in enumerate(batches, 1):
        states = residue_states(adapter, [records[i][1] for i in batch])
        for i, state in zip(batch, states):
            per_residue[i] = state
        print(f"embed: {args.model}: batch {n}/{len(batches)} ({len(batch)} sequences)", file=sys.stderr)
    sync(args.device)
    embed_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if args.device == "cuda" else None

    padding = padding_check(adapter, records, batches, per_residue, PADDING_MIN_COSINE[args.precision])

    lengths = np.array([len(seq) for _, seq in records], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    residues = np.concatenate(per_residue).astype(np.float32)
    mean = np.stack([state.mean(0) for state in per_residue]).astype(np.float32)
    if mean.shape[1] != EXPECTED_DIM[args.model]:
        sys.exit(f"embed: {args.model} produced dim {mean.shape[1]}, expected {EXPECTED_DIM[args.model]}")
    if residues.shape[0] != offsets[-1] or not np.isfinite(residues).all():
        sys.exit("embed: per-residue embeddings are inconsistent or non-finite")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        outdir / "embeddings.npz",
        ids=np.array([sid for sid, _ in records]),
        lengths=lengths,
        offsets=offsets,
        mean=mean,
        per_residue=residues,
    )
    meta = {
        "model": args.model,
        **adapter.info,
        "embedding_dim": int(mean.shape[1]),
        "n_sequences": len(records),
        "n_residues": int(offsets[-1]),
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
        "residues_per_second": round(int(offsets[-1]) / embed_s, 1),
        "peak_gpu_memory_gb": round(peak_gb, 3) if peak_gb is not None else None,
        "padding_check": padding,
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"embed: {args.model}: {len(records)} sequences, d={mean.shape[1]}, "
        f"load {load_s:.1f}s, embed {embed_s:.1f}s, padding check {padding}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
