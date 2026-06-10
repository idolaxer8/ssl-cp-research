import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import euclidean_distances


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_cmap(n_classes: int):
    if n_classes <= 10:
        return matplotlib.colormaps.get_cmap('tab10').resampled(n_classes)
    elif n_classes <= 20:
        return matplotlib.colormaps.get_cmap('tab20').resampled(n_classes)
    else:
        return matplotlib.colormaps.get_cmap('nipy_spectral').resampled(n_classes)


def _draw_cov_ellipse(ax, mean_2d, cov_2d, n_std=1.5, **kwargs):
    """Draw a 2D confidence ellipse from a 2×2 covariance matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
    eigenvalues = np.maximum(eigenvalues, 1e-12)   # numerical safety
    angle  = np.degrees(np.arctan2(eigenvectors[1, -1], eigenvectors[0, -1]))
    width  = 2 * n_std * np.sqrt(eigenvalues[-1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])
    e = Ellipse(xy=mean_2d, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(e)


# ---------------------------------------------------------------------------
# Main scatter plot (t-SNE / PCA) with optional ellipse overlay
# ---------------------------------------------------------------------------

def plot_scatter(ax, projections, labels, cmap, unique_labels,
                 show_ellipses: bool = True):
    label_to_idx = {int(lb): i for i, lb in enumerate(unique_labels)}
    for lb in unique_labels:
        mask  = labels == lb
        color = cmap(label_to_idx[int(lb)] / max(len(unique_labels) - 1, 1))
        ax.scatter(projections[mask, 0], projections[mask, 1],
                   color=color, alpha=0.65, s=18, linewidths=0)
        if show_ellipses and mask.sum() >= 3:
            pts = projections[mask]
            mu  = pts.mean(axis=0)
            cov = np.cov(pts.T)
            _draw_cov_ellipse(ax, mu, cov, n_std=1.5,
                              facecolor=color, edgecolor=color,
                              alpha=0.12, linewidth=0.8, zorder=0)


# ---------------------------------------------------------------------------
# Analysis panels
# ---------------------------------------------------------------------------

def plot_spread(ax, embeddings, labels, unique_labels):
    """Per-class average Euclidean distance to centroid (spread in 768-D)."""
    spreads = []
    for c in unique_labels:
        Xc   = embeddings[labels == c]
        mu   = Xc.mean(axis=0)
        spreads.append(float(np.linalg.norm(Xc - mu, axis=1).mean()))

    order   = np.argsort(spreads)[::-1]
    x_pos   = np.arange(len(unique_labels))
    colors  = [plt.cm.RdYlGn_r(i / len(unique_labels)) for i in range(len(unique_labels))]
    ax.bar(x_pos, [spreads[i] for i in order], color=colors, width=1.0, edgecolor='none')
    ax.set_title("Per-class spread (avg dist to centroid)", fontsize=11)
    ax.set_xlabel("Class rank (most spread -> least)")
    ax.set_ylabel("Avg L2 dist to centroid")
    ax.axhline(np.mean(spreads), color='k', linestyle='--', linewidth=1,
               label=f"Mean = {np.mean(spreads):.3f}")
    ax.legend(fontsize=9)
    ax.set_xticks([])


def plot_centroid_heatmap(ax, embeddings, labels, unique_labels):
    """Pairwise Euclidean distances between class centroids (heatmap)."""
    K = len(unique_labels)
    centroids = np.stack([embeddings[labels == c].mean(axis=0) for c in unique_labels])

    # Hierarchical reordering for interpretability
    dist_flat = euclidean_distances(centroids)
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        Z     = linkage(squareform(dist_flat, checks=False), method='ward')
        order = dendrogram(Z, no_plot=True)['leaves']
    except ImportError:
        order = list(range(K))

    D_ord = dist_flat[np.ix_(order, order)]
    im = ax.imshow(D_ord, cmap='viridis_r', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Euclidean dist')
    ax.set_title(f"Centroid distance heatmap ({K}×{K})", fontsize=11)
    ax.set_xlabel("Class (reordered)")
    ax.set_ylabel("Class (reordered)")
    if K <= 20:
        ticks = list(range(K))
        ax.set_xticks(ticks); ax.set_xticklabels([str(unique_labels[order[i]]) for i in ticks], fontsize=6)
        ax.set_yticks(ticks); ax.set_yticklabels([str(unique_labels[order[i]]) for i in ticks], fontsize=6)
    else:
        ax.set_xticks([]); ax.set_yticks([])


def plot_silhouette(ax, pca_result, labels, unique_labels):
    """Per-class mean silhouette score (computed on PCA-50 features)."""
    print("  Computing silhouette scores on PCA-50 features...")
    D    = euclidean_distances(pca_result)
    sil  = silhouette_samples(D, labels, metric='precomputed')

    per_class  = np.array([sil[labels == c].mean() for c in unique_labels])
    order      = np.argsort(per_class)
    colors     = ['#d62728' if per_class[i] < 0 else '#2ca02c' for i in order]

    ax.barh(np.arange(len(unique_labels)), per_class[order], color=colors,
            height=0.85, edgecolor='none')
    ax.axvline(0, color='k', linewidth=0.8)
    ax.axvline(np.mean(per_class), color='steelblue', linewidth=1.2, linestyle='--',
               label=f"Mean = {np.mean(per_class):.3f}")
    ax.set_title("Per-class silhouette score (PCA-50)", fontsize=11)
    ax.set_xlabel("Silhouette score  (↑ better, <0 misclassified)")
    ax.set_ylabel("Class rank")
    ax.set_yticks([])
    ax.legend(fontsize=9)
    neg_frac = (per_class < 0).sum() / len(per_class)
    ax.text(0.98, 0.02, f"{neg_frac*100:.0f}% classes have sil<0",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#d62728')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path",  type=str, default="output/embeddings.pt")
    parser.add_argument("--method",      type=str, default="tsne", choices=["tsne", "pca"])
    parser.add_argument("--analysis",    action="store_true",
                        help="Generate 4-panel analysis figure (spread, heatmap, silhouette)")
    parser.add_argument("--output_dir",  type=str, default="output",
                        help="Directory to save output figures")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load
    print(f"Loading {args.input_path}...")
    data       = torch.load(args.input_path)
    embeddings = data["embeddings"].numpy()
    labels     = data["labels"].numpy()
    n_classes  = len(np.unique(labels))
    unique_labels = np.unique(labels)
    print(f"  {len(embeddings)} samples  |  {n_classes} classes  |  {embeddings.shape[1]}D")

    dataset_tag = os.path.splitext(os.path.basename(args.input_path))[0]
    cmap        = _get_cmap(n_classes)

    # 2. PCA-50 (shared base for t-SNE and silhouette)
    n_comps = min(50, len(embeddings) - 1, embeddings.shape[1])
    print(f"Running PCA -> {n_comps} dims...")
    pca_50     = PCA(n_components=n_comps, random_state=42)
    pca_result = pca_50.fit_transform(embeddings)

    # 3. 2D projection
    if args.method == "pca":
        print("Running PCA -> 2D...")
        projections = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        print("Running t-SNE -> 2D...")
        perp = min(30, len(embeddings) - 1)
        projections = TSNE(n_components=2, perplexity=perp, random_state=42,
                           init='pca', learning_rate='auto').fit_transform(pca_result)

    # 4. Main scatter (with ellipses)
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_scatter(ax, projections, labels, cmap, unique_labels, show_ellipses=True)
    ax.set_title(f"SSL Embeddings ({args.method.upper()}) — {n_classes} classes "
                 f"(ellipses = 1.5σ per class)", fontsize=13)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.grid(True, alpha=0.25)

    if n_classes <= 20:
        handles = [matplotlib.patches.Patch(
                       color=cmap(i / max(n_classes - 1, 1)), label=str(lb))
                   for i, lb in enumerate(unique_labels)]
        ax.legend(handles=handles, title="Class", bbox_to_anchor=(1.01, 1),
                  loc='upper left', fontsize=7, ncol=1)
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap,
             norm=plt.Normalize(vmin=0, vmax=n_classes - 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=f"Class (0–{n_classes-1})",
                     fraction=0.03, pad=0.02)

    plt.tight_layout()
    scatter_path = os.path.join(args.output_dir, f"viz_{dataset_tag}_{args.method}.png")
    plt.savefig(scatter_path, dpi=150)
    print(f"Saved scatter -> {scatter_path}")
    plt.close()

    # 5. Analysis figure (4 panels)
    if args.analysis:
        print("Generating analysis figure...")
        fig2, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig2.suptitle(f"Embedding Analysis — {dataset_tag}  ({n_classes} classes)",
                      fontsize=14, fontweight='bold')

        # Panel A: t-SNE with ellipses (reuse projections)
        plot_scatter(axes[0, 0], projections, labels, cmap, unique_labels,
                     show_ellipses=True)
        axes[0, 0].set_title(f"{args.method.upper()} + 1.5σ ellipses", fontsize=11)
        axes[0, 0].set_xlabel("Dim 1"); axes[0, 0].set_ylabel("Dim 2")
        axes[0, 0].grid(True, alpha=0.2)

        # Panel B: per-class spread
        plot_spread(axes[0, 1], embeddings, labels, unique_labels)

        # Panel C: centroid heatmap
        plot_centroid_heatmap(axes[1, 0], embeddings, labels, unique_labels)

        # Panel D: silhouette
        plot_silhouette(axes[1, 1], pca_result, labels, labels)

        plt.tight_layout()
        analysis_path = os.path.join(args.output_dir, f"analysis_{dataset_tag}.png")
        plt.savefig(analysis_path, dpi=150)
        print(f"Saved analysis -> {analysis_path}")
        plt.close()


if __name__ == "__main__":
    main()
