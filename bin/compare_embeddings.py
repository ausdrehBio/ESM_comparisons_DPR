#!/usr/bin/env python3
"""Compare ESM-2 650M and ESM-C 600M embeddings of the same random sequences.

Random sequences differ only in length and amino-acid composition, so every agreement
metric is reported next to a chance baseline (permutation null) and next to how much of it
length and composition alone explain.

Built for inputs of hundreds of thousands of sequences. Metrics that need all-vs-all matrices
(Gram-based CKA, RSA, distance confounds) use a fixed random sample of --pair_sample sequences
(all of them for small inputs). Linear CKA, nearest-neighbour search, norms and PCA stream
over every sequence. Per-residue metrics use the residue sample from the reference build.

Writes <outdir>/summary.md, metrics.json, per_sequence.tsv, pairwise_distances.tsv, figures/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MODELS = {"esm2_650m": "ESM-2 650M", "esmc_600m": "ESM-C 600M"}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
KNN_K = (5, 10)
LARGE_NULL_PERMS = 20  # nulls of feature-space CKA: every permutation is a full pass over the rows
MIN_PAIR_SAMPLE = 12
CHUNK = 65_536  # rows per block when streaming over large float32 arrays
PLOT_POINTS = 20_000  # scatter plots draw at most this many points (a fixed random subset)

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
def row_blocks(n, size=CHUNK):
    return (slice(start, min(start + size, n)) for start in range(0, n, size))


def centered_cross(x, y, rows=None, chunk=CHUNK):
    """(x - mean x)ᵀ (y[rows] - mean y) in float64, streamed over row blocks of (float32) input."""
    mx, my = x.mean(0, dtype=np.float64), y.mean(0, dtype=np.float64)
    out = np.zeros((x.shape[1], y.shape[1]))
    for block in row_blocks(len(x), chunk):
        y_block = y[block] if rows is None else y[rows[block]]
        out += (x[block] - mx).T @ (y_block - my)
    return out


def linear_cka(x, y, chunk=CHUNK):
    """Linear CKA from features (Kornblith et al. 2019); cheap when n >> d."""
    xy, xx, yy = centered_cross(x, y, chunk=chunk), centered_cross(x, x, chunk=chunk), centered_cross(y, y, chunk=chunk)
    return float(np.linalg.norm(xy) ** 2 / (np.linalg.norm(xx) * np.linalg.norm(yy)))


def linear_cka_null(x, y, perms):
    """Linear CKA with the rows of y shuffled; the self-similarity terms do not depend on the order."""
    scale = np.linalg.norm(centered_cross(x, x)) * np.linalg.norm(centered_cross(y, y))
    return np.array([np.linalg.norm(centered_cross(x, y, rows=p)) ** 2 / scale for p in perms])


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
    # tr(KL) and 1ᵀKL1 without forming KL: O(n²) instead of O(n³) per permutation.
    trace_kl = np.sum(k * l.T)
    sum_kl = k.sum(0) @ l.sum(1)
    return (trace_kl + k.sum() * l.sum() / ((n - 1) * (n - 2)) - 2 * sum_kl / (n - 2)) / (n * (n - 3))


def debiased_cka(k, l):
    """CKA built on unbiased HSIC; removes the upward bias of linear CKA when n < d."""
    return float(unbiased_hsic(k, l) / np.sqrt(unbiased_hsic(k, k) * unbiased_hsic(l, l)))


def debiased_cka_null(k, l, perms):
    """Debiased CKA with l's rows and columns shuffled; HSIC(K, K) and HSIC(L, L) are unchanged by it."""
    scale = np.sqrt(unbiased_hsic(k, k) * unbiased_hsic(l, l))
    return np.array([unbiased_hsic(k, l[np.ix_(p, p)]) / scale for p in perms])


def cosine_distances(x):
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    d = 1 - unit @ unit.T
    np.fill_diagonal(d, 0)
    return np.clip(d, 0, None)


def row_norms(x):
    return np.concatenate([np.linalg.norm(x[block].astype(np.float64), axis=1) for block in row_blocks(len(x))])


def unit_rows_inplace(x):
    """Scale each row to unit length in place, block by block (no full-size temporary)."""
    for block in row_blocks(len(x)):
        x[block] /= np.linalg.norm(x[block], axis=1, keepdims=True)


def mean_cosine_distance(unit):
    """Each row's mean cosine distance to all other rows, from the column sum (no n x n matrix)."""
    n = len(unit)
    total = unit.sum(0, dtype=np.float64)
    return np.concatenate([1 - (unit[block] @ total - 1) / (n - 1) for block in row_blocks(n)])


def nearest_neighbours(unit, queries, k, block=128):
    """Indices of each query's k most cosine-similar other rows (among all rows), most similar first."""
    out = np.empty((len(queries), k), dtype=np.int64)
    for b in row_blocks(len(queries), block):
        q = queries[b]
        sim = unit[q] @ unit.T
        sim[np.arange(len(q)), q] = -np.inf  # a sequence is not its own neighbour
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(sim, top, axis=1), axis=1, kind="stable")
        out[b] = np.take_along_axis(top, order, axis=1)
    return out


def knn_overlap(nx, ny, k):
    """Per-query fraction of the k nearest neighbours shared by the two embedding spaces."""
    return np.array([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(nx, ny)])


def pca_2d(x):
    """Projections of all rows onto the first two principal components, and their variance fractions."""
    values, vectors = np.linalg.eigh(centered_cross(x, x))
    top = vectors[:, ::-1][:, :2]
    center = x.mean(0, dtype=np.float64)
    projection = np.concatenate([(x[block] - center) @ top for block in row_blocks(len(x))])
    return projection, values[::-1][:2] / values.sum()


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
    # Permuting rows and columns only reorders the off-diagonal ranks: they stay valid ranks with the
    # same mean and spread, so standardise once and gather each permuted upper triangle directly.
    ry = (ry - ry[i, j].mean()) / ry[i, j].std()
    null = np.array([np.mean(a * ry[p[i], p[j]]) for p in perms])
    return null_summary(null, observed)


def partial_rsa(dx, dy, controls):
    """Spearman correlation of two distance matrices after regressing out control distances."""
    design = np.column_stack([np.ones(len(upper(dx)))] + [stats.rankdata(upper(c)) for c in controls])

    def residual(d):
        v = stats.rankdata(upper(d))
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    return float(stats.pearsonr(residual(dx), residual(dy))[0])


def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def self_test():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((60, 8))
    y = rng.standard_normal((60, 5))
    q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    assert abs(linear_cka(x, 3 * x @ q) - 1) < 1e-9, "linear CKA must be 1 under rotation + scaling"
    assert abs(linear_cka(x, y, chunk=7) - linear_cka(x, y)) < 1e-12, "block-wise CKA must match"
    assert abs(gram_cka(centered_gram(x), centered_gram(y)) - linear_cka(x, y)) < 1e-9, "Gram CKA must match"
    perms = [rng.permutation(60) for _ in range(3)]
    assert np.allclose(linear_cka_null(x, y, perms), [linear_cka(x, y[p]) for p in perms]), "CKA null must match"

    k, l = x @ x.T, y @ y.T
    k0, l0, n = k - np.diag(np.diag(k)), l - np.diag(np.diag(l)), len(x)
    kl = k0 @ l0
    reference = (np.trace(kl) + k0.sum() * l0.sum() / ((n - 1) * (n - 2)) - 2 * kl.sum() / (n - 2)) / (n * (n - 3))
    assert abs(unbiased_hsic(k, l) - reference) < 1e-9 * abs(reference), "O(n²) HSIC must match KL form"
    assert abs(debiased_cka(k, k) - 1) < 1e-9, "debiased CKA(X, X) must be 1"
    assert np.allclose(debiased_cka_null(k, l, perms), [debiased_cka(k, l[np.ix_(p, p)]) for p in perms]), \
        "debiased CKA null must match"
    dx, dy = cosine_distances(x), cosine_distances(y)
    naive = np.mean([spearman(upper(dx), upper(dy[np.ix_(p, p)])) for p in perms])
    assert abs(mantel(dx, dy, perms)["null_mean"] - naive) < 1e-9, "Mantel null must match Spearman of shuffles"

    u = rng.standard_normal((30, 4)).astype(np.float32)
    unit_rows_inplace(u)
    d = 1 - u.astype(np.float64) @ u.T.astype(np.float64)
    assert np.allclose(mean_cosine_distance(u), (d.sum(1) - np.diag(d)) / 29, atol=1e-5), "mean distance must match"
    np.fill_diagonal(d, np.inf)
    brute = np.argsort(d, axis=1, kind="stable")[:, :3]
    assert (np.sort(nearest_neighbours(u, np.arange(30), 3, block=7), 1) == np.sort(brute, 1)).all(), "kNN must match"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_embeddings(directory, ids):
    """MERGE output, with the per-sequence arrays in manifest order."""
    directory = Path(directory)
    with np.load(directory / "embeddings.npz") as z:
        data = {key: z[key] for key in z.files}
    meta = json.loads((directory / "meta.json").read_text())
    if len(data["ids"]) != len(ids) or set(data["ids"].tolist()) != set(ids):
        sys.exit(f"compare: sequence IDs in {directory} do not match the manifest")
    if not np.array_equal(data["ids"], np.asarray(ids)):  # reorder only when needed: mean can be GBs
        position = {sid: k for k, sid in enumerate(data["ids"].tolist())}
        order = np.array([position[sid] for sid in ids])
        data["mean"], data["lengths"] = data["mean"][order], data["lengths"][order]
    return data, meta


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


def plot_subset(n, rng):
    """Indices of the points to draw: all of them, or a fixed random PLOT_POINTS of them."""
    return np.arange(n) if n <= PLOT_POINTS else np.sort(rng.choice(n, PLOT_POINTS, replace=False))


def dots(count):
    """Outlined dots for small inputs; small rasterised dots when thousands are drawn."""
    if count <= 500:
        return {"s": 36, "edgecolors": SURFACE, "linewidths": 0.8}
    return {"s": 4, "linewidths": 0, "alpha": 0.6, "rasterized": True}


def shown_note(ax, shown, total, what):
    if shown < total:
        ax.text(0.02, 0.98, f"{shown:,} of {total:,} {what} shown", transform=ax.transAxes,
                ha="left", va="top", color=MUTED, fontsize=8)


def plot_distance_agreement(dx, dy, length_diff, rho, rng, path):
    x, y, c = upper(dx), upper(dy), upper(length_diff)
    pick = plot_subset(len(x), rng)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    points = ax.scatter(x[pick], y[pick], c=c[pick], cmap=SEQUENTIAL_MARKS,
                        s=9 if len(pick) <= 5000 else 3, alpha=0.75, linewidths=0, rasterized=True)
    bar = fig.colorbar(points, ax=ax, shrink=0.85)
    bar.set_label("|Δ length| (residues)", color=INK_2)
    bar.outline.set_visible(False)
    ax.set_xlabel(f"{MODELS['esm2_650m']} cosine distance")
    ax.set_ylabel(f"{MODELS['esmc_600m']} cosine distance")
    ax.set_title(f"Pairwise distances, {len(x):,} pairs: Spearman ρ = {rho:.2f}")
    shown_note(ax, len(pick), len(x), "pairs")
    save(fig, path)


def plot_distance_matrices(distances, lengths, path):
    order = np.argsort(lengths, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    for ax, (model, d) in zip(axes, distances.items()):
        image = ax.imshow(d[np.ix_(order, order)], cmap=SEQUENTIAL, interpolation="antialiased")
        bar = fig.colorbar(image, ax=ax, shrink=0.8)
        bar.set_label("cosine distance", color=INK_2)
        bar.outline.set_visible(False)
        ax.set_title(MODELS[model])
        ax.set_xlabel(f"{len(d):,} sequences, shortest → longest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, path)


def plot_pca(pca, lengths, rng, path):
    pick = plot_subset(len(lengths), rng)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for ax, (model, (projection, explained)) in zip(axes, pca.items()):
        points = ax.scatter(projection[pick, 0], projection[pick, 1], c=lengths[pick], cmap=SEQUENTIAL_MARKS,
                            **dots(len(pick)))
        ax.set_title(MODELS[model])
        ax.set_xlabel(f"PC1 ({explained[0]:.0%} of variance)")
        ax.set_ylabel(f"PC2 ({explained[1]:.0%} of variance)")
        shown_note(ax, len(pick), len(lengths), "sequences")
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


def plot_norm_vs_length(norms, lengths, rng, path):
    pick = plot_subset(len(lengths), rng)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    for ax, (model, norm) in zip(axes, norms.items()):
        ax.scatter(lengths[pick], norm[pick], color=SERIES[model], **dots(len(pick)))
        ax.set_title(f"{MODELS[model]}: Spearman ρ = {spearman(norm, lengths):.2f}")
        ax.set_xlabel("sequence length (residues)")
        ax.set_ylabel("‖mean embedding‖")
        shown_note(ax, len(pick), len(lengths), "sequences")
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
    meta, agree, enc, inp = m["models"], m["agreement"], m["per_model"], m["input"]
    e2, ec = "esm2_650m", "esmc_600m"
    n, pairs, residues = inp["n_sequences"], inp["pair_sample"], inp["residue_sample"]
    on_sample = f", {pairs:,}-sequence sample" if pairs < n else ""

    def row(label, value):
        return f"| {label} | {value(meta[e2])} | {value(meta[ec])} |"

    def null_row(label, s):
        return (f"| {label} | {s['observed']:.3f} | null mean {s['null_mean']:.3f}, 95th pct {s['null_p95']:.3f} "
                f"| {fmt_p(s['p'], s['permutations'])} |")

    knn_scope = f" ({pairs:,} queries, neighbours from all {n:,})" if pairs < n else ""
    knn_rows = [f"| kNN overlap, k = {k}{knn_scope} | {agree[f'knn{k}']['mean']:.3f} "
                f"| chance {agree[f'knn{k}']['chance']:.3g} | – |" for k in KNN_K]
    sampling = []
    if pairs < n:
        sampling.append(f"All-vs-all metrics (Gram-based CKA, RSA, distance confounds) use a fixed random sample of "
                        f"{pairs:,} sequences. Linear CKA, nearest neighbours, norms and PCA use all {n:,}.")
    if residues < inp["n_residues"]:
        sampling.append(f"Per-residue metrics use a fixed random sample of {residues:,} residues, the same for both "
                        "models.")
    lines = [
        "# ESM-2 650M vs ESM-C 600M: embedding agreement on random sequences",
        "",
        f"{n:,} sequences, {inp['n_residues']:,} residues, length {inp['length_min']}–{inp['length_max']}.",
        *(["", *sampling] if sampling else []),
        "",
        "| | ESM-2 650M | ESM-C 600M |",
        "|---|---|---|",
        row("package", lambda d: f"{d['package']} {d['package_version']}"),
        row("torch", lambda d: d["torch_version"]),
        row("embedding dim", lambda d: d["embedding_dim"]),
        row("layer (final, post-LayerNorm)", lambda d: d["layer"]),
        row("precision / device", lambda d: f"{d['precision']} / {d['device']}"),
        row("GPU", lambda d: d["gpu"] or "–"),
        row("EMBED tasks (shards)", lambda d: d["n_shards"]),
        row("model load, per task (s)", lambda d: d["load_seconds"]),
        row("embedding time, total (s)", lambda d: f"{d['embed_seconds']:,.1f}"),
        row("residues per second", lambda d: f"{d['residues_per_second']:,.0f}"),
        row("peak GPU memory (GB)", lambda d: d["peak_gpu_memory_gb"] if d["peak_gpu_memory_gb"] is not None else "–"),
        row("padding check, min cosine",
            lambda d: "skipped" if d["padding_min_cosine"] is None else f"{d['padding_min_cosine']:.6f}"),
        "",
        "## Agreement between the two embedding spaces",
        "",
        "| metric | value | chance baseline | p |",
        "|---|---|---|---|",
        *([null_row(f"linear CKA, mean-pooled, all {n:,} sequences", agree["linear_cka_all"])]
          if "linear_cka_all" in agree else []),
        null_row(f"linear CKA, mean-pooled{on_sample}", agree["linear_cka"]),
        null_row(f"debiased CKA, mean-pooled{on_sample}", agree["debiased_cka"]),
        null_row(f"RSA: Spearman of pairwise cosine distances (Mantel){on_sample}", agree["rsa"]),
        f"| partial RSA, length and composition removed{on_sample} | {agree['partial_rsa']:.3f} | 0 | – |",
        *knn_rows,
        null_row(f"linear CKA, per residue ({residues:,} residues)", agree["residue_linear_cka"]),
        "",
        "## What each model's embeddings track",
        "",
        "| | ESM-2 650M | ESM-C 600M |",
        "|---|---|---|",
        *[f"| {label} | {enc[e2][key]:.3f} | {enc[ec][key]:.3f} |" for label, key in [
            (rf"Spearman(distance, \|Δ length\|){on_sample}", "distance_vs_length_diff"),
            (f"Spearman(distance, composition L1){on_sample}", "distance_vs_composition_l1"),
            ("Spearman(‖mean embedding‖, length)", "norm_vs_length"),
            ("per-residue linear CKA with one-hot amino acid", "residue_cka_vs_amino_acid"),
        ]],
        "",
        "## How to read this",
        "",
        "- CKA and RSA compare geometry, so they work across the different widths (1280 vs 1152). "
        "1 means the same geometry up to rotation and scale.",
        f"- Linear CKA on {min(pairs, n):,} mean-pooled vectors is biased upward unless n is much larger than the "
        "embedding width (1152–1280). Read it against its permutation null, or use the debiased value."
        + (f" The all-sequence value (n = {n:,}) is far less affected." if pairs < n else ""),
        f"- Nulls of feature-space CKA (per residue{', all sequences' if pairs < n else ''}) use {LARGE_NULL_PERMS} "
        f"permutations, as each one is a full pass over the data, so their p cannot go below "
        f"{1 / (1 + LARGE_NULL_PERMS):.2g}.",
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
    ap.add_argument("--esm2", required=True, help="MERGE output dir for esm2_650m")
    ap.add_argument("--esmc", required=True, help="MERGE output dir for esmc_600m")
    ap.add_argument("--manifest", required=True, help="manifest.tsv from the reference build")
    ap.add_argument("--fasta", required=True, help="sequences.fasta from the reference build")
    ap.add_argument("--perms", type=int, default=9999, help="permutations for the sequence-sample nulls")
    ap.add_argument("--pair_sample", type=int, default=2000, help="sequences used for all-vs-all metrics")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    if args.pair_sample < MIN_PAIR_SAMPLE:
        ap.error(f"--pair_sample must be at least {MIN_PAIR_SAMPLE}")

    self_test()
    # IDs stay strings: "0001" must not become 1, nor "NA" a missing value.
    manifest = pd.read_csv(args.manifest, sep="\t", dtype={"sequence_id": str}, keep_default_na=False)
    ids = manifest["sequence_id"].tolist()
    lengths = manifest["length"].to_numpy()
    composition = manifest[[f"frac_{aa}" for aa in AMINO_ACIDS]].to_numpy()
    seqs = read_fasta(args.fasta)
    n = len(ids)

    means, residues, metas, residue_keys = {}, {}, {}, {}
    for model, directory in (("esm2_650m", args.esm2), ("esmc_600m", args.esmc)):
        data, meta = load_embeddings(directory, ids)
        if not np.array_equal(data["lengths"], lengths):
            sys.exit(f"compare: {model} sequence lengths do not match the manifest")
        means[model], residues[model], metas[model] = data["mean"], data["per_residue"], meta
        residue_keys[model] = (data["residue_ids"], data["residue_positions"])
    e2, ec = "esm2_650m", "esmc_600m"
    if not all(np.array_equal(a, b) for a, b in zip(residue_keys[e2], residue_keys[ec])):
        sys.exit("compare: the two models stored different residue samples")
    residue_ids, residue_positions = residue_keys[e2]

    rng = np.random.default_rng(0)
    sample = np.arange(n) if n <= args.pair_sample else np.sort(rng.choice(n, args.pair_sample, replace=False))
    m = len(sample)
    print(f"compare: {n:,} sequences; all-vs-all metrics on {m:,} with {args.perms} permutations", file=sys.stderr)

    # Sequence sample: Gram-based CKA and distance geometry need m x m matrices.
    sub = {model: means[model][sample].astype(np.float64) for model in MODELS}
    perms = [rng.permutation(m) for _ in range(args.perms)]
    k_x, k_y = sub[e2] @ sub[e2].T, sub[ec] @ sub[ec].T
    kc_x, kc_y = centered_gram(sub[e2]), centered_gram(sub[ec])
    linear_null = np.array([gram_cka(kc_x, kc_y[np.ix_(p, p)]) for p in perms])
    debiased_null = debiased_cka_null(k_x, k_y, perms)
    linear = null_summary(linear_null, gram_cka(kc_x, kc_y))
    debiased = null_summary(debiased_null, debiased_cka(k_x, k_y))
    dist = {model: cosine_distances(x) for model, x in sub.items()}
    sample_lengths = lengths[sample]
    length_diff = np.abs(sample_lengths[:, None] - sample_lengths[None, :]).astype(np.float64)
    composition_l1 = cdist(composition[sample], composition[sample], "cityblock")
    rsa = mantel(dist[e2], dist[ec], perms)

    # All sequences: feature-space statistics stream over the rows; no n x n matrix is formed.
    print("compare: all-sequence statistics", file=sys.stderr)
    norms = {model: row_norms(x) for model, x in means.items()}
    linear_all = None
    if m < n:
        full_perms = [rng.permutation(n) for _ in range(LARGE_NULL_PERMS)]
        linear_all = null_summary(linear_cka_null(means[e2], means[ec], full_perms), linear_cka(means[e2], means[ec]))
    pca = {model: pca_2d(x) for model, x in means.items()}
    # Normalise in place instead of copying: at full scale each array is several GB.
    units = means
    for x in units.values():
        unit_rows_inplace(x)
    del means
    mean_distance = {model: mean_cosine_distance(x) for model, x in units.items()}
    neighbours = {model: nearest_neighbours(x, sample, max(KNN_K)) for model, x in units.items()}
    knn = {k: knn_overlap(neighbours[e2], neighbours[ec], k) for k in KNN_K}

    # Residue sample: CKA between models (small null) and with amino-acid identity.
    print("compare: residue-level CKA", file=sys.stderr)
    residue_aa = "".join(seqs[sid][pos] for sid, pos in zip(residue_ids.tolist(), residue_positions.tolist()))
    one_hot = (np.frombuffer(residue_aa.encode(), dtype=np.uint8)[:, None]
               == np.frombuffer(AMINO_ACIDS.encode(), dtype=np.uint8)[None, :]).astype(np.float64)
    residue_cka = linear_cka(residues[e2], residues[ec])
    residue_null = linear_cka_null(residues[e2], residues[ec],
                                   [rng.permutation(len(residue_aa)) for _ in range(LARGE_NULL_PERMS)])

    per_model = {
        model: {
            "distance_vs_length_diff": spearman(upper(dist[model]), upper(length_diff)),
            "distance_vs_composition_l1": spearman(upper(dist[model]), upper(composition_l1)),
            "norm_vs_length": spearman(norms[model], lengths),
            "residue_cka_vs_amino_acid": linear_cka(residues[model], one_hot),
        }
        for model in MODELS
    }
    metrics = {
        "input": {"n_sequences": n, "n_residues": int(lengths.sum()),
                  "length_min": int(lengths.min()), "length_max": int(lengths.max()),
                  "pair_sample": m, "residue_sample": len(residue_aa)},
        "models": {
            model: {**{key: metas[model].get(key) for key in (
                "package", "package_version", "embedding_dim", "layer", "precision", "device", "gpu",
                "n_shards", "load_seconds", "embed_seconds", "residues_per_second", "peak_gpu_memory_gb",
                "torch_version")},
                "padding_min_cosine": metas[model].get("padding_check", {}).get("min_cosine")}
            for model in MODELS
        },
        "agreement": {
            **({"linear_cka_all": linear_all} if linear_all else {}),
            "linear_cka": linear,
            "debiased_cka": debiased,
            "rsa": rsa,
            "partial_rsa": partial_rsa(dist[e2], dist[ec], [length_diff, composition_l1]),
            **{f"knn{k}": {"mean": float(v.mean()), "chance": k / (n - 1), "queries": m} for k, v in knn.items()},
            "residue_linear_cka": null_summary(residue_null, residue_cka),
        },
        "per_model": per_model,
    }

    outdir = Path(args.outdir)
    figures = outdir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_summary(outdir / "summary.md", metrics)

    in_sample = np.zeros(n, dtype=int)
    in_sample[sample] = 1
    knn_columns = {}
    for k, v in knn.items():
        column = np.full(n, np.nan)  # kNN overlap exists only for the sampled query sequences
        column[sample] = v
        knn_columns[f"knn{k}_overlap"] = column
    pd.DataFrame({
        "sequence_id": ids,
        "length": lengths,
        "in_pair_sample": in_sample,
        "norm_esm2_650m": norms[e2],
        "norm_esmc_600m": norms[ec],
        "mean_distance_esm2_650m": mean_distance[e2],
        "mean_distance_esmc_600m": mean_distance[ec],
        **knn_columns,
    }).to_csv(outdir / "per_sequence.tsv", sep="\t", index=False, float_format="%.6g")
    i, j = np.triu_indices(m, 1)
    sample_ids = np.asarray(ids)[sample]
    pd.DataFrame({
        "sequence_a": sample_ids[i],
        "sequence_b": sample_ids[j],
        "abs_length_diff": length_diff[i, j].astype(int),
        "composition_l1": composition_l1[i, j],
        "cosine_distance_esm2_650m": dist[e2][i, j],
        "cosine_distance_esmc_600m": dist[ec][i, j],
    }).to_csv(outdir / "pairwise_distances.tsv", sep="\t", index=False, float_format="%.6g")

    style()
    plot_rng = np.random.default_rng(1)
    scope = f" ({m:,}-sequence sample)" if m < n else ""
    plot_distance_agreement(dist[e2], dist[ec], length_diff, rsa["observed"], plot_rng,
                            figures / "distance_agreement.png")
    plot_distance_matrices(dist, sample_lengths, figures / "distance_matrices.png")
    plot_pca(pca, lengths, plot_rng, figures / "pca_by_length.png")
    plot_cka_null({f"Linear CKA, mean-pooled{scope}": (linear_null, linear["observed"]),
                   f"Debiased CKA, mean-pooled{scope}": (debiased_null, debiased["observed"])},
                  figures / "cka_null.png")
    plot_norm_vs_length(norms, lengths, plot_rng, figures / "norm_vs_length.png")
    plot_confounds(per_model, figures / "confounds.png")
    print(f"compare: wrote {outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
