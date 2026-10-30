#!/usr/bin/env python3
"""Compare ESM-2 650M and ESM-C 600M embeddings of the same random sequences.

Random sequences differ only in length and amino-acid composition, so every agreement
metric is reported next to a chance baseline (permutation null) and next to how much of it
length and composition alone explain.

Writes <outdir>/summary.md, metrics.json, per_sequence.tsv, pairwise_distances.tsv, figures/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MODELS = {"esm2_650m": "ESM-2 650M", "esmc_600m": "ESM-C 600M"}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
KNN_K = (5, 10)
RESIDUE_NULL_PERMS = 20  # each residue-level permutation costs a (n_res x d)^T (n_res x d) product

# Reference data-viz palette: categorical slots 1-2 for the two models, blue ramp for magnitude.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"esm2_650m": "#2a78d6", "esmc_600m": "#eb6834"}
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
SEQUENTIAL_MARKS = LinearSegmentedColormap.from_list("blue_marks", BLUE_RAMP[3:])  # lightest marks stay visible


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def linear_cka(x, y):
    """Linear CKA from features (Kornblith et al. 2019); cheap when n >> d."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    return float(np.linalg.norm(y.T @ x) ** 2 / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def centered_gram(x):
    gram = x @ x.T
    return gram - gram.mean(0) - gram.mean(1)[:, None] + gram.mean()


def gram_cka(kc, lc):
    return float((kc * lc).sum() / (np.linalg.norm(kc) * np.linalg.norm(lc)))


def unbiased_hsic(k, l):
    """Unbiased HSIC estimator (Song et al. 2012) on uncentered Gram matrices."""
    n = k.shape[0]
    k, l = k.copy(), l.copy()
    np.fill_diagonal(k, 0)
    np.fill_diagonal(l, 0)
    kl = k @ l
    return (np.trace(kl) + k.sum() * l.sum() / ((n - 1) * (n - 2)) - 2 * kl.sum() / (n - 2)) / (n * (n - 3))


def debiased_cka(k, l):
    """CKA built on unbiased HSIC; removes the upward bias of linear CKA when n < d."""
    return float(unbiased_hsic(k, l) / np.sqrt(unbiased_hsic(k, k) * unbiased_hsic(l, l)))


def cosine_distances(x):
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    d = 1 - unit @ unit.T
    np.fill_diagonal(d, 0)
    return np.clip(d, 0, None)


def upper(m):
    i, j = np.triu_indices(len(m), 1)
    return m[i, j]


def rank_matrix(d):
    """Symmetric matrix whose off-diagonal entries are the ranks of d's upper triangle."""
    n = len(d)
    i, j = np.triu_indices(n, 1)
    r = np.zeros_like(d, dtype=np.float64)
    r[i, j] = stats.rankdata(d[i, j])
    r[j, i] = r[i, j]
    return r


def zscore(v):
    return (v - v.mean()) / v.std()


def p_value(null, observed):
    """One-sided permutation p-value, counting the observed statistic as one draw."""
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def null_summary(null, observed):
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p": p_value(null, observed),
        "permutations": len(null),
    }


def mantel(dx, dy, perms):
    """Spearman correlation of two distance matrices with a row/column permutation null."""
    i, j = np.triu_indices(len(dx), 1)
    a = zscore(rank_matrix(dx)[i, j])
    ry = rank_matrix(dy)
    observed = float(np.mean(a * zscore(ry[i, j])))
    # Permuting rows and columns only reorders the off-diagonal ranks, so they stay valid ranks.
    null = np.array([np.mean(a * zscore(ry[np.ix_(p, p)][i, j])) for p in perms])
    return null_summary(null, observed)


def partial_rsa(dx, dy, controls):
    """Spearman correlation of two distance matrices after regressing out control distances."""
    design = np.column_stack([np.ones(len(upper(dx)))] + [stats.rankdata(upper(c)) for c in controls])

    def residual(d):
        v = stats.rankdata(upper(d))
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    return float(stats.pearsonr(residual(dx), residual(dy))[0])


def knn_overlap(dx, dy, k):
    """Per-sequence fraction of k nearest neighbours shared by the two embedding spaces."""
    def neighbours(d):
        d = d.copy()
        np.fill_diagonal(d, np.inf)
        return np.argsort(d, axis=1, kind="stable")[:, :k]

    nx, ny = neighbours(dx), neighbours(dy)
    return np.array([len(set(a) & set(b)) / k for a, b in zip(nx, ny)])


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def self_test():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 8))
    q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    assert abs(linear_cka(x, 3 * x @ q) - 1) < 1e-9, "linear CKA must be 1 under rotation + scaling"
    assert abs(gram_cka(centered_gram(x), centered_gram(x @ q)) - 1) < 1e-9, "Gram CKA must match"
    assert abs(debiased_cka(x @ x.T, x @ x.T) - 1) < 1e-9, "debiased CKA(X, X) must be 1"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_embeddings(directory, ids):
    directory = Path(directory)
    with np.load(directory / "embeddings.npz") as z:
        data = {key: z[key] for key in z.files}
    meta = json.loads((directory / "meta.json").read_text())
    position = {sid: k for k, sid in enumerate(data["ids"])}
    if set(position) != set(ids):
        sys.exit(f"compare: sequence IDs in {directory} do not match the manifest")
    order = [position[sid] for sid in ids]
    offsets = data["offsets"]
    residues = np.concatenate([data["per_residue"][offsets[k]:offsets[k + 1]] for k in order])
    return data["mean"][order].astype(np.float64), data["lengths"][order], residues.astype(np.float64), meta


def read_fasta(path):
    seqs, name = {}, None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = ""
        elif line:
            seqs[name] += line
    return seqs


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 9,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.labelcolor": INK_2, "axes.labelsize": 9, "axes.edgecolor": AXIS,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8,
    })


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_distance_agreement(dx, dy, length_diff, rho, path):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    points = ax.scatter(upper(dx), upper(dy), c=upper(length_diff), cmap=SEQUENTIAL_MARKS,
                        s=9, alpha=0.75, linewidths=0, rasterized=True)
    bar = fig.colorbar(points, ax=ax, shrink=0.85)
    bar.set_label("|Δ length| (residues)", color=INK_2)
    bar.outline.set_visible(False)
    ax.set_xlabel(f"{MODELS['esm2_650m']} cosine distance")
    ax.set_ylabel(f"{MODELS['esmc_600m']} cosine distance")
    ax.set_title(f"Pairwise distances, all {len(upper(dx)):,} pairs: Spearman ρ = {rho:.2f}")
    save(fig, path)


def plot_distance_matrices(distances, lengths, path):
    order = np.argsort(lengths, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, (model, d) in zip(axes, distances.items()):
        image = ax.imshow(d[np.ix_(order, order)], cmap=SEQUENTIAL, interpolation="nearest")
        bar = fig.colorbar(image, ax=ax, shrink=0.8)
        bar.set_label("cosine distance", color=INK_2)
        bar.outline.set_visible(False)
        ax.set_title(MODELS[model])
        ax.set_xlabel("sequences, shortest → longest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, path)


def plot_pca(means, lengths, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, (model, x) in zip(axes, means.items()):
        centered = x - x.mean(0)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        explained = s ** 2 / np.sum(s ** 2)
        points = ax.scatter(u[:, 0] * s[0], u[:, 1] * s[1], c=lengths, cmap=SEQUENTIAL_MARKS,
                            s=36, edgecolors=SURFACE, linewidths=0.8)
        ax.set_title(MODELS[model])
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)")
    bar = fig.colorbar(points, ax=axes, shrink=0.85)
    bar.set_label("sequence length (residues)", color=INK_2)
    bar.outline.set_visible(False)
    save(fig, path)


def plot_cka_null(nulls, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), constrained_layout=True)
    for ax, (label, (null, observed)) in zip(axes, nulls.items()):
        lo, hi = min(null.min(), observed), max(null.max(), observed)
        pad = 0.05 * (hi - lo)
        ax.hist(null, bins=40, range=(lo - pad, hi + pad), color=AXIS, edgecolor=SURFACE, linewidth=0.5)
        ax.axvline(observed, color=INK, linewidth=2)
        ax.annotate(f"observed {observed:.3f}", xy=(observed, 0.95), xycoords=("data", "axes fraction"),
                    xytext=(-6, 0), textcoords="offset points", ha="right", va="top", color=INK)
        ax.set_title(label)
        ax.set_xlabel(f"CKA under shuffled sequence correspondence ({len(null):,} permutations)")
        ax.set_ylabel("permutations")
        ax.grid(axis="x", visible=False)
    save(fig, path)


def plot_norm_vs_length(means, lengths, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    for ax, (model, x) in zip(axes, means.items()):
        norms = np.linalg.norm(x, axis=1)
        ax.scatter(lengths, norms, color=SERIES[model], s=36, edgecolors=SURFACE, linewidths=0.8)
        ax.set_title(f"{MODELS[model]}: Spearman ρ = {spearman(norms, lengths):.2f}")
        ax.set_xlabel("sequence length (residues)")
        ax.set_ylabel("‖mean embedding‖")
    save(fig, path)


def plot_confounds(confounds, path):
    labels = ["|Δ length|", "composition L1"]
    keys = ["distance_vs_length_diff", "distance_vs_composition_l1"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    width = 0.36
    for offset, model in zip((-width / 2, width / 2), MODELS):
        values = [confounds[model][key] for key in keys]
        # Thin gap between the paired bars keeps them separable.
        bars = ax.bar(np.arange(len(keys)) + offset, values, width - 0.02, color=SERIES[model], label=MODELS[model])
        ax.bar_label(bars, fmt="%.2f", padding=2, color=INK_2, fontsize=8)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_xticks(np.arange(len(keys)), labels)
    ax.set_ylabel("Spearman ρ with embedding distance")
    ax.set_title("What drives distances between random sequences")
    ax.grid(axis="x", visible=False)
    # Below the axis: correlations can sit anywhere in [-1, 1], so no corner inside is reliably empty.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    save(fig, path)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def fmt_p(p, permutations):
    floor = 1 / (1 + permutations)  # smallest p a permutation test of this size can return
    return f"≤ {floor:.2g}" if p <= floor else f"{p:.3g}"


def write_summary(path, m):
    meta, agree, enc = m["models"], m["agreement"], m["per_model"]
    e2, ec = "esm2_650m", "esmc_600m"

    def row(label, value):
        return f"| {label} | {value(meta[e2])} | {value(meta[ec])} |"

    def null_row(label, s):
        return (f"| {label} | {s['observed']:.3f} | null mean {s['null_mean']:.3f}, 95th pct {s['null_p95']:.3f} "
                f"| {fmt_p(s['p'], s['permutations'])} |")

    knn_rows = [f"| kNN overlap, k = {k} | {agree[f'knn{k}']['mean']:.3f} | chance {agree[f'knn{k}']['chance']:.3f} | – |"
                for k in KNN_K]
    lines = [
        "# ESM-2 650M vs ESM-C 600M: embedding agreement on random sequences",
        "",
        f"{m['input']['n_sequences']} sequences, {m['input']['n_residues']:,} residues, "
        f"length {m['input']['length_min']}–{m['input']['length_max']}.",
        "",
        "| | ESM-2 650M | ESM-C 600M |",
        "|---|---|---|",
        row("package", lambda d: f"{d['package']} {d['package_version']}"),
        row("torch", lambda d: d["torch_version"]),
        row("embedding dim", lambda d: d["embedding_dim"]),
        row("layer (final, post-LayerNorm)", lambda d: d["layer"]),
        row("precision / device", lambda d: f"{d['precision']} / {d['device']}"),
        row("GPU", lambda d: d["gpu"] or "–"),
        row("load / embedding time (s)", lambda d: f"{d['load_seconds']} / {d['embed_seconds']}"),
        row("padding check, min cosine",
            lambda d: "skipped" if d["padding_min_cosine"] is None else f"{d['padding_min_cosine']:.6f}"),
        "",
        "## Agreement between the two embedding spaces",
        "",
        "| metric | value | chance baseline | p |",
        "|---|---|---|---|",
        null_row("linear CKA, mean-pooled (n = sequences)", agree["linear_cka"]),
        null_row("debiased CKA, mean-pooled", agree["debiased_cka"]),
        null_row("RSA: Spearman of pairwise cosine distances (Mantel)", agree["rsa"]),
        f"| partial RSA, length and composition removed | {agree['partial_rsa']:.3f} | 0 | – |",
        *knn_rows,
        null_row("linear CKA, per residue (n = residues)", agree["residue_linear_cka"]),
        "",
        "## What each model's embeddings track",
        "",
        "| | ESM-2 650M | ESM-C 600M |",
        "|---|---|---|",
        *[f"| {label} | {enc[e2][key]:.3f} | {enc[ec][key]:.3f} |" for label, key in [
            (r"Spearman(distance, \|Δ length\|)", "distance_vs_length_diff"),
            ("Spearman(distance, composition L1)", "distance_vs_composition_l1"),
            ("Spearman(‖mean embedding‖, length)", "norm_vs_length"),
            ("per-residue linear CKA with one-hot amino acid", "residue_cka_vs_amino_acid"),
        ]],
        "",
        "## How to read this",
        "",
        "- CKA and RSA compare geometry, so they work across the different widths (1280 vs 1152). "
        "1 means the same geometry up to rotation and scale.",
        "- Linear CKA on 100 mean-pooled vectors is biased upward because n < d. Read it against its "
        "permutation null, or use the debiased value.",
        "- Random sequences carry no evolutionary signal: shared structure comes from length, composition "
        "and residue identity. The partial RSA shows how much agreement is left after removing length "
        "and composition.",
        "",
        "Figures: `figures/distance_agreement.png`, `distance_matrices.png`, `pca_by_length.png`, "
        "`cka_null.png`, `norm_vs_length.png`, `confounds.png`.",
        "",
    ]
    Path(path).write_text("\n".join(lines))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--esm2", required=True, help="EMBED output dir for esm2_650m")
    ap.add_argument("--esmc", required=True, help="EMBED output dir for esmc_600m")
    ap.add_argument("--manifest", required=True, help="manifest.tsv from the reference build")
    ap.add_argument("--fasta", required=True, help="sequences.fasta from the reference build")
    ap.add_argument("--perms", type=int, default=9999, help="permutations for sequence-level nulls")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    self_test()
    manifest = pd.read_csv(args.manifest, sep="\t")
    ids = manifest["sequence_id"].tolist()
    seqs = read_fasta(args.fasta)
    lengths = manifest["length"].to_numpy()
    composition = manifest[[f"frac_{aa}" for aa in AMINO_ACIDS]].to_numpy()

    means, residues, metas = {}, {}, {}
    for model, directory in (("esm2_650m", args.esm2), ("esmc_600m", args.esmc)):
        mean, model_lengths, res, meta = load_embeddings(directory, ids)
        if not np.array_equal(model_lengths, lengths):
            sys.exit(f"compare: {model} sequence lengths do not match the manifest")
        means[model], residues[model], metas[model] = mean, res, meta
    e2, ec = "esm2_650m", "esmc_600m"
    n = len(ids)

    rng = np.random.default_rng(0)
    perms = [rng.permutation(n) for _ in range(args.perms)]
    print(f"compare: {n} sequences, {args.perms} permutations", file=sys.stderr)

    # Sequence level: CKA with permutation nulls.
    k_x, k_y = means[e2] @ means[e2].T, means[ec] @ means[ec].T
    kc_x, kc_y = centered_gram(means[e2]), centered_gram(means[ec])
    linear_null = np.array([gram_cka(kc_x, kc_y[np.ix_(p, p)]) for p in perms])
    debiased_null = np.array([debiased_cka(k_x, k_y[np.ix_(p, p)]) for p in perms])
    linear = null_summary(linear_null, gram_cka(kc_x, kc_y))
    debiased = null_summary(debiased_null, debiased_cka(k_x, k_y))

    # Sequence level: distance geometry.
    dist = {model: cosine_distances(x) for model, x in means.items()}
    length_diff = np.abs(lengths[:, None] - lengths[None, :]).astype(np.float64)
    composition_l1 = np.abs(composition[:, None, :] - composition[None, :, :]).sum(-1)
    rsa = mantel(dist[e2], dist[ec], perms)
    knn = {k: knn_overlap(dist[e2], dist[ec], k) for k in KNN_K}

    # Residue level: CKA between models (small null) and with amino-acid identity.
    print("compare: residue-level CKA", file=sys.stderr)
    residue_seq = "".join(seqs[sid] for sid in ids)
    one_hot = (np.frombuffer(residue_seq.encode(), dtype=np.uint8)[:, None]
               == np.frombuffer(AMINO_ACIDS.encode(), dtype=np.uint8)[None, :]).astype(np.float64)
    residue_cka = linear_cka(residues[e2], residues[ec])
    residue_null = np.array([linear_cka(residues[e2], residues[ec][rng.permutation(len(residue_seq))])
                             for _ in range(RESIDUE_NULL_PERMS)])

    per_model = {
        model: {
            "distance_vs_length_diff": spearman(upper(dist[model]), upper(length_diff)),
            "distance_vs_composition_l1": spearman(upper(dist[model]), upper(composition_l1)),
            "norm_vs_length": spearman(np.linalg.norm(means[model], axis=1), lengths),
            "residue_cka_vs_amino_acid": linear_cka(residues[model], one_hot),
        }
        for model in MODELS
    }
    metrics = {
        "input": {"n_sequences": n, "n_residues": len(residue_seq),
                  "length_min": int(lengths.min()), "length_max": int(lengths.max())},
        "models": {
            model: {**{key: metas[model].get(key) for key in (
                "package", "package_version", "embedding_dim", "layer", "precision", "device", "gpu",
                "load_seconds", "embed_seconds", "torch_version")},
                "padding_min_cosine": metas[model].get("padding_check", {}).get("min_cosine")}
            for model in MODELS
        },
        "agreement": {
            "linear_cka": linear,
            "debiased_cka": debiased,
            "rsa": rsa,
            "partial_rsa": partial_rsa(dist[e2], dist[ec], [length_diff, composition_l1]),
            **{f"knn{k}": {"mean": float(v.mean()), "chance": k / (n - 1)} for k, v in knn.items()},
            "residue_linear_cka": null_summary(residue_null, residue_cka),
        },
        "per_model": per_model,
    }

    outdir = Path(args.outdir)
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_summary(outdir / "summary.md", metrics)

    pd.DataFrame({
        "sequence_id": ids,
        "length": lengths,
        "norm_esm2_650m": np.linalg.norm(means[e2], axis=1),
        "norm_esmc_600m": np.linalg.norm(means[ec], axis=1),
        "mean_distance_esm2_650m": dist[e2].sum(1) / (n - 1),
        "mean_distance_esmc_600m": dist[ec].sum(1) / (n - 1),
        **{f"knn{k}_overlap": v for k, v in knn.items()},
    }).to_csv(outdir / "per_sequence.tsv", sep="\t", index=False, float_format="%.6g")
    i, j = np.triu_indices(n, 1)
    pd.DataFrame({
        "sequence_a": np.array(ids)[i],
        "sequence_b": np.array(ids)[j],
        "abs_length_diff": length_diff[i, j].astype(int),
        "composition_l1": composition_l1[i, j],
        "cosine_distance_esm2_650m": dist[e2][i, j],
        "cosine_distance_esmc_600m": dist[ec][i, j],
    }).to_csv(outdir / "pairwise_distances.tsv", sep="\t", index=False, float_format="%.6g")

    style()
    plot_distance_agreement(dist[e2], dist[ec], length_diff, rsa["observed"], figures / "distance_agreement.png")
    plot_distance_matrices(dist, lengths, figures / "distance_matrices.png")
    plot_pca(means, lengths, figures / "pca_by_length.png")
    plot_cka_null({"Linear CKA, mean-pooled": (linear_null, linear["observed"]),
                   "Debiased CKA, mean-pooled": (debiased_null, debiased["observed"])},
                  figures / "cka_null.png")
    plot_norm_vs_length(means, lengths, figures / "norm_vs_length.png")
    plot_confounds(per_model, figures / "confounds.png")
    print(f"compare: wrote {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
