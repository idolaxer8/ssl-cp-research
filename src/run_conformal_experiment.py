"""
Main script for SSL + Full Conformal Prediction experiments.

Usage:
    python src/run_conformal_experiment.py --embeddings_path output/embeddings.pt --alpha 0.1 --k 5
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from conformal_prediction import (
    FullConformalPredictor,
    CrossValidationPlusPredictor,
    SoftmaxSplitCP,
    create_ncm,
    cal_test_split,
    train_cal_test_split,
    stratified_cal_test_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Conformal Prediction on SSL embeddings")
    
    # Data
    parser.add_argument("--embeddings_path", type=str, default="output/embeddings.pt",
                       help="Path to embeddings .pt file")
    
    # Split ratios (for Full CP, no training set needed)
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                       help="Fraction of data for calibration (rest goes to test)")
    
    # Conformal prediction
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Significance level (target miscoverage rate)")
    parser.add_argument("--ncm", type=str, default="geodesic_topk_mean",
                       choices=["mahal_nn_ratio", "whitened_geodesic",
                                "geodesic_topk_mean", "geodesic_topk_asym", "softmax"],
                       help="NCM for Full CP (and CV+ unless --ncm_cv is set)")
    parser.add_argument("--ncm_cv", type=str, default=None,
                       choices=["mahal_nn_ratio", "whitened_geodesic",
                                "geodesic_topk_mean", "geodesic_topk_asym", "softmax"],
                       help="NCM for CV+ (defaults to --ncm if not set)")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of neighbors for k-NN nonconformity")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save results")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save prediction sets to file")
    parser.add_argument("--compare_methods", action="store_true",
                       help="Compare runtime between k-NN and simplified k-NN")
    parser.add_argument("--compare_k", action="store_true",
                       help="Compare performance across different k values")
    parser.add_argument("--compare_with_baseline", action="store_true",
                       help="Compare SSL embeddings vs raw pixel features (baseline)")
    parser.add_argument("--compare_full_vs_split", action="store_true",
                       help="Compare Full CP (k-NN based) vs Split CP (softmax) across data sizes")
    parser.add_argument("--compare_cp_methods", action="store_true",
                       help="Compare Full CP vs Split CP vs CV+ across data sizes")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds for CV+ (default: 5)")
    parser.add_argument("--data_sizes", type=str, default="200,400,600,800,1000",
                       help="Comma-separated data sizes for comparison experiments")
    parser.add_argument("--n_trials", type=int, default=3,
                       help="Number of trials per data size for comparison experiments")
    parser.add_argument("--test_size", type=int, default=1000,
                       help="Fixed test set size for --compare_cp_methods")
    parser.add_argument("--cal_sizes", type=str, default="100,200,400,600,800,1000",
                       help="Comma-separated calibration sizes to sweep (x-axis)")
    parser.add_argument("--split_train_ratio", type=float, default=0.5,
                       help="Split CP train size as fraction of cal_size (train = ratio * cal)")
    # Data paths
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing raw images (for baseline comparison)")
    parser.add_argument("--num_per_class", type=int, default=None,
                       help="Limit number of images per class for baseline comparison (None = match embeddings)")
    
    # Split strategy
    parser.add_argument("--balanced", action="store_true",
                       help="Use stratified balanced cal/test split (equal samples per class)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def plot_set_size_distribution(set_sizes: np.ndarray, alpha: float, output_dir: str):
    """Plot histogram of prediction set sizes."""
    plt.figure(figsize=(10, 6))
    
    max_size = int(set_sizes.max())
    bins = np.arange(0, max_size + 2) - 0.5
    
    plt.hist(set_sizes, bins=bins, alpha=0.7, edgecolor='black')
    plt.xlabel("Prediction Set Size", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Distribution of Prediction Set Sizes (α={alpha})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, max_size + 1))
    
    # Add statistics
    stats_text = (
        f"Mean: {set_sizes.mean():.2f}\n"
        f"Median: {np.median(set_sizes):.0f}\n"
        f"Singleton rate: {(set_sizes == 1).mean():.2%}"
    )
    plt.text(0.95, 0.95, stats_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = Path(output_dir) / "set_size_distribution.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close()


def plot_coverage_by_alpha(X_cal, y_cal, X_test, y_test, ncm, output_dir: str):
    """Plot coverage and set size vs alpha."""
    alphas = np.linspace(0.01, 0.2, 5)
    coverages = []
    avg_sizes = []
    
    print("\nComputing coverage vs alpha curve...")
    
    for alpha in alphas:
        cp = FullConformalPredictor(ncm, alpha=alpha)
        cp.calibrate(X_cal, y_cal)
        
        # Don't need full evaluation, just predict
        results = cp.predict(X_test, return_p_values=False, verbose=False)
        prediction_sets = results['prediction_sets']
        
        coverage = np.mean([
            y_test[i] in pred_set
            for i, pred_set in enumerate(prediction_sets)
        ])
        avg_size = np.mean(results['set_sizes'])
        
        coverages.append(coverage)
        avg_sizes.append(avg_size)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Coverage vs alpha
    ax1.plot(alphas, coverages, 'o-', linewidth=2, markersize=6, label='Actual coverage')
    ax1.plot(alphas, 1 - alphas, '--', linewidth=2, label='Target coverage (1-α)', color='red')
    ax1.set_xlabel("Significance Level (α)", fontsize=12)
    ax1.set_ylabel("Coverage", fontsize=12)
    ax1.set_title("Coverage vs α", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Average set size vs alpha
    ax2.plot(alphas, avg_sizes, 'o-', linewidth=2, markersize=6, color='green')
    ax2.set_xlabel("Significance Level (α)", fontsize=12)
    ax2.set_ylabel("Average Set Size", fontsize=12)
    ax2.set_title("Efficiency vs α", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / "coverage_vs_alpha.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close()


def analyze_per_class_coverage(y_test, prediction_sets, output_dir: str):
    """Analyze coverage per class."""
    classes = np.unique(y_test)
    
    per_class_coverage = {}
    per_class_avg_size = {}
    
    for cls in classes:
        mask = y_test == cls
        pred_sets_cls = [prediction_sets[i] for i in np.where(mask)[0]]
        
        coverage = np.mean([cls in pred_set for pred_set in pred_sets_cls])
        avg_size = np.mean([len(pred_set) for pred_set in pred_sets_cls])
        
        per_class_coverage[int(cls)] = coverage
        per_class_avg_size[int(cls)] = avg_size
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    classes_list = list(per_class_coverage.keys())
    
    # Coverage per class
    ax1.bar(classes_list, [per_class_coverage[c] for c in classes_list], alpha=0.7)
    ax1.axhline(y=np.mean(list(per_class_coverage.values())), 
                color='r', linestyle='--', label='Average')
    ax1.set_xlabel("Class", fontsize=12)
    ax1.set_ylabel("Coverage", fontsize=12)
    ax1.set_title("Coverage per Class", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Avg set size per class
    ax2.bar(classes_list, [per_class_avg_size[c] for c in classes_list], 
            alpha=0.7, color='green')
    ax2.axhline(y=np.mean(list(per_class_avg_size.values())), 
                color='r', linestyle='--', label='Average')
    ax2.set_xlabel("Class", fontsize=12)
    ax2.set_ylabel("Avg Set Size", fontsize=12)
    ax2.set_title("Average Set Size per Class", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = Path(output_dir) / "per_class_analysis.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close()
    
    return per_class_coverage, per_class_avg_size


def compare_k_values(X_cal, y_cal, X_test, y_test, args, output_dir: str):
    """Compare conformal prediction performance across different k values."""
    k_values = [1, 3, 5, 7, 10, 15, 20]
    
    print("\n" + "="*70)
    print("PERFORMANCE vs k (Number of Neighbors)")
    print("="*70)
    
    results_by_k = {}
    
    for k in k_values:
        print(f"\nEvaluating k={k}...")
        ncm = create_ncm(args.ncm, k=k)
        cp = FullConformalPredictor(ncm, alpha=args.alpha)
        cp.calibrate(X_cal, y_cal)
        metrics = cp.evaluate(X_test, y_test, verbose=False)
        
        results_by_k[k] = metrics
        
        print(f"  Coverage: {metrics['coverage']:.3f}")
        print(f"  Avg set size: {metrics['avg_set_size']:.3f}")
        print(f"  Singleton rate: {metrics['singleton_rate']:.3f}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    coverages = [results_by_k[k]['coverage'] for k in k_values]
    set_sizes = [results_by_k[k]['avg_set_size'] for k in k_values]
    singleton_rates = [results_by_k[k]['singleton_rate'] for k in k_values]
    empty_rates = [results_by_k[k]['empty_set_rate'] for k in k_values]
    
    # Coverage vs k
    ax1.plot(k_values, coverages, 'o-', linewidth=2, markersize=8)
    ax1.axhline(y=1-args.alpha, color='r', linestyle='--', label=f'Target (1-α={1-args.alpha:.3f})')
    ax1.set_xlabel('k (neighbors)', fontsize=12)
    ax1.set_ylabel('Coverage', fontsize=12)
    ax1.set_title('Coverage vs k', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Avg set size vs k
    ax2.plot(k_values, set_sizes, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('k (neighbors)', fontsize=12)
    ax2.set_ylabel('Avg Set Size', fontsize=12)
    ax2.set_title('Efficiency vs k', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Singleton rate vs k
    ax3.plot(k_values, singleton_rates, 'o-', linewidth=2, markersize=8, color='orange')
    ax3.set_xlabel('k (neighbors)', fontsize=12)
    ax3.set_ylabel('Singleton Rate', fontsize=12)
    ax3.set_title('Singleton Rate vs k', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Empty set rate vs k
    ax4.plot(k_values, empty_rates, 'o-', linewidth=2, markersize=8, color='red')
    ax4.set_xlabel('k (neighbors)', fontsize=12)
    ax4.set_ylabel('Empty Set Rate', fontsize=12)
    ax4.set_title('Empty Set Rate vs k', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / "performance_vs_k.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved plot to {save_path}")
    plt.close()
    
    # Save results
    k_comparison_path = Path(output_dir) / "k_comparison.pt"
    torch.save(results_by_k, k_comparison_path)
    print(f"Saved k comparison to {k_comparison_path}")
    print("="*70)
    
    return results_by_k


def load_raw_pixel_features(data_dir: str, num_per_class: int):
    """
    Load raw pixel features from ImageFolder, selecting first `num_per_class` images per class.
    Returns (features, labels) where features are L2-normalized flattened pixels (n_images, 3072).
    """
    from torchvision.datasets import ImageFolder
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    n_classes = len(dataset.classes)

    # Select first num_per_class images per class
    class_count = {c: 0 for c in range(n_classes)}
    indices = []
    labels = []
    for idx, (_, label) in enumerate(dataset.samples):
        if class_count[label] < num_per_class:
            indices.append(idx)
            labels.append(label)
            class_count[label] += 1
        if all(c >= num_per_class for c in class_count.values()):
            break

    if len(indices) != num_per_class * n_classes:
        raise ValueError(f"Not enough images: need {num_per_class} per class, got {class_count}")

    features = np.array([dataset[i][0].numpy().flatten() for i in indices], dtype=np.float32)
    features /= (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    labels = np.array(labels)
    return features, labels


def compare_ssl_vs_baseline(X_ssl, y_ssl, X_raw, y_raw, args, output_dir: str):
    """
    Compare CP performance on SSL embeddings vs raw pixel features.
    Applies PCA to raw features to match SSL embedding dimension for fair comparison.
    """
    print("\n" + "="*70)
    print("SSL EMBEDDINGS vs RAW PIXEL BASELINE COMPARISON")
    print("="*70)
    
    # Report dimensions
    print(f"\nOriginal dimensions:")
    print(f"  SSL embeddings: {X_ssl.shape[0]} samples, {X_ssl.shape[1]}D")
    print(f"  Raw pixels: {X_raw.shape[0]} samples, {X_raw.shape[1]}D")
    
    # Use consistent split for both (same seed ensures same split pattern)
    X_cal_ssl, y_cal_ssl, X_test_ssl, y_test_ssl = cal_test_split(
        X_ssl, y_ssl, cal_ratio=args.cal_ratio, random_state=args.seed
    )
    X_cal_raw, y_cal_raw, X_test_raw, y_test_raw = cal_test_split(
        X_raw, y_raw, cal_ratio=args.cal_ratio, random_state=args.seed
    )
    
    # Apply PCA to raw features to match SSL dimension (fair comparison)
    # k-NN suffers from curse of dimensionality, so we need equal dimensions
    from sklearn.decomposition import PCA
    
    target_dim = X_ssl.shape[1]
    if X_raw.shape[1] > target_dim:
        # PCA n_components must be <= min(n_samples, n_features)
        max_components = min(X_cal_raw.shape[0], X_cal_raw.shape[1], target_dim)
        print(f"\nApplying PCA to raw features: {X_raw.shape[1]}D → {max_components}D")
        print("  (For fair comparison: k-NN is sensitive to dimensionality)")
        
        pca = PCA(n_components=max_components, random_state=args.seed)
        X_cal_raw_pca = pca.fit_transform(X_cal_raw)
        X_test_raw_pca = pca.transform(X_test_raw)
        
        # Normalize after PCA
        X_cal_raw_pca = X_cal_raw_pca / (np.linalg.norm(X_cal_raw_pca, axis=1, keepdims=True) + 1e-8)
        X_test_raw_pca = X_test_raw_pca / (np.linalg.norm(X_test_raw_pca, axis=1, keepdims=True) + 1e-8)
        
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        X_cal_raw_pca = X_cal_raw
        X_test_raw_pca = X_test_raw
    
    comparison_results = {}
    
    for name, X_cal, X_test, y_cal, y_test in [
        ("Raw Pixels (Baseline + PCA)", X_cal_raw_pca, X_test_raw_pca, y_cal_raw, y_test_raw),
        ("SSL Embeddings", X_cal_ssl, X_test_ssl, y_cal_ssl, y_test_ssl),
    ]:
        print(f"\nEvaluating {name}...")
        print(f"  Calibration set: {len(X_cal)}, Test set: {len(X_test)}")
        print(f"  Feature dimension: {X_cal.shape[1]}")
        
        # Create NCM and predictor
        ncm = create_ncm(args.ncm, k=args.k)
        cp = FullConformalPredictor(ncm, alpha=args.alpha)
        cp.calibrate(X_cal, y_cal)
        
        # Predict and evaluate
        metrics = cp.evaluate(X_test, y_test, verbose=False)
        
        comparison_results[name] = metrics
        
        # Print metrics
        print(f"  Coverage: {metrics['coverage']:.3f} (target: {metrics['target_coverage']:.3f})")
        print(f"  Avg set size: {metrics['avg_set_size']:.3f}")
        print(f"  Singleton rate: {metrics['singleton_rate']:.3f}")
        print(f"  Empty set rate: {metrics['empty_set_rate']:.3f}")
    
    # Compute improvements
    print("\n" + "="*70)
    print("SSL IMPROVEMENT over BASELINE")
    print("="*70)
    
    baseline = comparison_results["Raw Pixels (Baseline + PCA)"]
    ssl = comparison_results["SSL Embeddings"]
    
    # Coverage improvement
    cov_improvement = (ssl['coverage'] - baseline['coverage']) / (baseline['coverage'] + 1e-8) * 100
    print(f"Coverage:       {baseline['coverage']:.3f} → {ssl['coverage']:.3f} (Δ: {cov_improvement:+.1f}%)")
    
    # Set size reduction (lower is better)
    setsize_improvement = (baseline['avg_set_size'] - ssl['avg_set_size']) / (baseline['avg_set_size'] + 1e-8) * 100
    print(f"Avg set size:   {baseline['avg_set_size']:.3f} → {ssl['avg_set_size']:.3f} (Δ: {setsize_improvement:+.1f}% reduction)")
    
    # Singleton improvement
    singleton_improvement = (ssl['singleton_rate'] - baseline['singleton_rate']) / (baseline['singleton_rate'] + 1e-8 + 1e-8) * 100
    print(f"Singleton rate: {baseline['singleton_rate']:.3f} → {ssl['singleton_rate']:.3f} (Δ: {singleton_improvement:+.1f}%)")
    
    # Empty set reduction (lower is better)
    empty_improvement = (baseline['empty_set_rate'] - ssl['empty_set_rate']) / (baseline['empty_set_rate'] + 1e-8) * 100
    if baseline['empty_set_rate'] > 0:
        print(f"Empty set rate: {baseline['empty_set_rate']:.3f} → {ssl['empty_set_rate']:.3f} (Δ: {empty_improvement:+.1f}% reduction)")
    else:
        print(f"Empty set rate: {baseline['empty_set_rate']:.3f} → {ssl['empty_set_rate']:.3f}")
    
    print("="*70)
    
    # Save comparison results
    comparison_path = Path(output_dir) / "ssl_vs_baseline_comparison.txt"
    with open(comparison_path, "w") as f:
        f.write("SSL vs Raw Baseline Comparison\n")
        f.write("="*60 + "\n\n")
        for name, metrics in comparison_results.items():
            f.write(f"{name}\n")
            f.write(f"  Coverage: {metrics['coverage']:.4f} (target {metrics['target_coverage']:.4f})\n")
            f.write(f"  Avg set size: {metrics['avg_set_size']:.4f}\n")
            f.write(f"  Median set size: {metrics['median_set_size']:.4f}\n")
            f.write(f"  Singleton rate: {metrics['singleton_rate']:.4f}\n")
            f.write(f"  Singleton accuracy: {metrics['singleton_accuracy']:.4f}\n")
            f.write(f"  Empty set rate: {metrics['empty_set_rate']:.4f}\n")
            f.write("\n")
        f.write("Improvements (SSL over Baseline)\n")
        f.write("="*60 + "\n")
        f.write(f"Coverage delta (%): {cov_improvement:+.2f}\n")
        f.write(f"Avg set size delta (% reduction): {setsize_improvement:+.2f}\n")
        f.write(f"Singleton rate delta (%): {singleton_improvement:+.2f}\n")
        if baseline['empty_set_rate'] > 0:
            f.write(f"Empty set rate delta (% reduction): {empty_improvement:+.2f}\n")
        else:
            f.write("Empty set rate: baseline already 0.0000\n")
    print(f"\nSaved comparison to {comparison_path}")
    
    return comparison_results


def compare_full_cp_vs_split_cp(
    embeddings: np.ndarray,
    labels: np.ndarray,
    data_sizes: list,
    args,
    output_dir: str
):
    """
    Compare Full CP (e.g., simplified k-NN) vs Split CP (softmax) across different data sizes.
    
    Full CP uses cal_test_split (no training set needed).
    Split CP uses train_cal_test_split (requires training set for classifier).
    
    Args:
        embeddings: Full embedding matrix
        labels: Full label array
        data_sizes: List of data sizes to test
        args: Command line arguments
        output_dir: Directory to save results
    """
    import json
    import time
    from tqdm import tqdm
    
    print("\n" + "="*70)
    print("FULL CP vs SPLIT CP COMPARISON ACROSS DATA SIZES")
    print("="*70)
    print(f"Data sizes: {data_sizes}")
    print(f"Trials per size: {args.n_trials}")
    # Only show k for NCMs that use it
    if args.ncm in ["knn", "simplified_knn"]:
        print(f"Full CP NCM: {args.ncm} (k={args.k})")
    else:
        print(f"Full CP NCM: {args.ncm}")
    print(f"Split CP: SoftmaxSplitCP (logistic regression)")
    print(f"Alpha: {args.alpha}")
    print("="*70)
    
    results = {
        'data_sizes': data_sizes,
        'alpha': args.alpha,
        'ncm': args.ncm,
        'k': args.k,
        'n_trials': args.n_trials,
        'full_cp': {},
        'split_cp': {}
    }
    
    # Track aggregated metrics per data size
    for data_size in data_sizes:
        results['full_cp'][data_size] = {
            'coverages': [], 'set_sizes': [], 'singleton_rates': [], 'empty_rates': [], 'times': []
        }
        results['split_cp'][data_size] = {
            'coverages': [], 'set_sizes': [], 'singleton_rates': [], 'empty_rates': [], 'times': []
        }
    
    for data_size in data_sizes:
        print(f"\n{'='*70}")
        print(f"Data Size: {data_size} samples")
        print("="*70)
        
        if data_size > len(embeddings):
            print(f"  Warning: Requested {data_size} samples but only {len(embeddings)} available. Skipping.")
            continue
        
        for trial in range(args.n_trials):
            # Use different seed for each trial
            trial_seed = args.seed + trial * 1000
            np.random.seed(trial_seed)
            
            # Subsample data
            indices = np.random.permutation(len(embeddings))[:data_size]
            X_subset = embeddings[indices]
            y_subset = labels[indices]
            
            print(f"\nTrial {trial+1}/{args.n_trials} (seed={trial_seed})")
            
            # ========== FULL CP ==========
            # Full CP only needs cal + test split
            X_cal, y_cal, X_test, y_test = cal_test_split(
                X_subset, y_subset,
                cal_ratio=0.5,  # 50% cal, 50% test
                random_state=trial_seed
            )
            
            # Create NCM
            ncm = create_ncm(args.ncm, k=args.k)

            print("  Running Full CP...")
            t_start_full = time.time()
            cp_full = FullConformalPredictor(ncm, alpha=args.alpha)
            cp_full.calibrate(X_cal, y_cal)
            metrics_full = cp_full.evaluate(X_test, y_test, verbose=False)
            t_full = time.time() - t_start_full
            
            results['full_cp'][data_size]['coverages'].append(metrics_full['coverage'])
            results['full_cp'][data_size]['set_sizes'].append(metrics_full['avg_set_size'])
            results['full_cp'][data_size]['singleton_rates'].append(metrics_full['singleton_rate'])
            results['full_cp'][data_size]['empty_rates'].append(metrics_full['empty_set_rate'])
            results['full_cp'][data_size]['times'].append(t_full)
            
            print(f"    Coverage: {metrics_full['coverage']:.3f}, Avg size: {metrics_full['avg_set_size']:.2f}, Time: {t_full:.2f}s")
            
            # ========== SPLIT CP (Softmax) ==========
            # Split CP needs train + cal + test split
            X_train, y_train, X_cal_s, y_cal_s, X_test_s, y_test_s = train_cal_test_split(
                X_subset, y_subset,
                train_ratio=0.25,  # 25% train, 25% cal, 50% test
                cal_ratio=0.25,
                random_state=trial_seed
            )
            
            print("  Running Split CP (Softmax)...")
            t_start_split = time.time()
            cp_split = SoftmaxSplitCP(alpha=args.alpha)
            cp_split.fit(X_train, y_train)
            cp_split.calibrate(X_cal_s, y_cal_s)
            metrics_split = cp_split.evaluate(X_test_s, y_test_s, verbose=False)
            t_split = time.time() - t_start_split
            
            results['split_cp'][data_size]['coverages'].append(metrics_split['coverage'])
            results['split_cp'][data_size]['set_sizes'].append(metrics_split['avg_set_size'])
            results['split_cp'][data_size]['singleton_rates'].append(metrics_split['singleton_rate'])
            results['split_cp'][data_size]['empty_rates'].append(metrics_split['empty_set_rate'])
            results['split_cp'][data_size]['times'].append(t_split)
            
            print(f"    Coverage: {metrics_split['coverage']:.3f}, Avg size: {metrics_split['avg_set_size']:.2f}, Time: {t_split:.2f}s")
        
        # Print aggregated results for this data size
        fc = results['full_cp'][data_size]
        sc = results['split_cp'][data_size]
        print(f"\nAggregate (n={args.n_trials} trials):")
        print(f"  Full CP:  Coverage={np.mean(fc['coverages'])*100:.1f}%±{np.std(fc['coverages'])*100:.1f}%, "
              f"Size={np.mean(fc['set_sizes']):.2f}±{np.std(fc['set_sizes']):.2f}, "
              f"Time={np.mean(fc['times']):.2f}s±{np.std(fc['times']):.2f}s")
        print(f"  Split CP: Coverage={np.mean(sc['coverages'])*100:.1f}%±{np.std(sc['coverages'])*100:.1f}%, "
              f"Size={np.mean(sc['set_sizes']):.2f}±{np.std(sc['set_sizes']):.2f}, "
              f"Time={np.mean(sc['times']):.2f}s±{np.std(sc['times']):.2f}s")
    
    # ========== PLOT RESULTS ==========
    plot_full_vs_split_comparison(results, args, output_dir)
    
    # ========== SAVE RESULTS ==========
    # Save JSON results
    results_path = Path(output_dir) / f"full_vs_split_comparison_{args.ncm}.json"
    
    # Convert numpy values to Python types for JSON serialization
    results_serializable = {
        'data_sizes': data_sizes,
        'alpha': args.alpha,
        'ncm': args.ncm,
        'k': args.k,
        'n_trials': args.n_trials,
        'full_cp': {str(k): {kk: [float(v) for v in vv] for kk, vv in v.items()} 
                    for k, v in results['full_cp'].items()},
        'split_cp': {str(k): {kk: [float(v) for v in vv] for kk, vv in v.items()} 
                    for k, v in results['split_cp'].items()}
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Save summary text
    summary_path = Path(output_dir) / f"full_vs_split_comparison_{args.ncm}.txt"
    with open(summary_path, 'w') as f:
        f.write("Full CP vs Split CP Comparison\n")
        f.write("="*60 + "\n\n")
        # Only show k for NCMs that use it
        if args.ncm in ["knn", "simplified_knn"]:
            f.write(f"Full CP NCM: {args.ncm} (k={args.k})\n")
        else:
            f.write(f"Full CP NCM: {args.ncm}\n")
        f.write(f"Split CP: SoftmaxSplitCP (logistic regression)\n")
        f.write(f"Alpha: {args.alpha}, Target coverage: {1-args.alpha:.2f}\n")
        f.write(f"Trials per data size: {args.n_trials}\n\n")
        
        f.write(f"{'Data Size':<12} {'Method':<12} {'Coverage':<18} {'Avg Set Size':<18} {'Time (s)':<18}\n")
        f.write("-"*78 + "\n")
        
        for data_size in data_sizes:
            if data_size not in results['full_cp'] or not results['full_cp'][data_size]['coverages']:
                continue
            fc = results['full_cp'][data_size]
            sc = results['split_cp'][data_size]
            
            f.write(f"{data_size:<12} {'Full CP':<12} "
                   f"{np.mean(fc['coverages'])*100:.1f}%±{np.std(fc['coverages'])*100:.1f}%{'':>4} "
                   f"{np.mean(fc['set_sizes']):.2f}±{np.std(fc['set_sizes']):.2f}{'':>6} "
                   f"{np.mean(fc['times']):.2f}±{np.std(fc['times']):.2f}\n")
            f.write(f"{'':<12} {'Split CP':<12} "
                   f"{np.mean(sc['coverages'])*100:.1f}%±{np.std(sc['coverages'])*100:.1f}%{'':>4} "
                   f"{np.mean(sc['set_sizes']):.2f}±{np.std(sc['set_sizes']):.2f}{'':>6} "
                   f"{np.mean(sc['times']):.2f}±{np.std(sc['times']):.2f}\n")
            f.write("-"*78 + "\n")
    
    print(f"Saved summary to {summary_path}")
    
    return results


def plot_full_vs_split_comparison(results: dict, args, output_dir: str):
    """
    Plot coverage and set size comparison between Full CP and Split CP across data sizes.
    """
    data_sizes = results['data_sizes']
    
    # Filter out data sizes that weren't actually computed
    valid_sizes = [ds for ds in data_sizes 
                   if ds in results['full_cp'] and results['full_cp'][ds]['coverages']]
    
    if not valid_sizes:
        print("No valid data sizes to plot.")
        return
    
    # Compute means and stds
    full_cov_means = [np.mean(results['full_cp'][ds]['coverages']) for ds in valid_sizes]
    full_cov_stds = [np.std(results['full_cp'][ds]['coverages']) for ds in valid_sizes]
    split_cov_means = [np.mean(results['split_cp'][ds]['coverages']) for ds in valid_sizes]
    split_cov_stds = [np.std(results['split_cp'][ds]['coverages']) for ds in valid_sizes]
    
    full_size_means = [np.mean(results['full_cp'][ds]['set_sizes']) for ds in valid_sizes]
    full_size_stds = [np.std(results['full_cp'][ds]['set_sizes']) for ds in valid_sizes]
    split_size_means = [np.mean(results['split_cp'][ds]['set_sizes']) for ds in valid_sizes]
    split_size_stds = [np.std(results['split_cp'][ds]['set_sizes']) for ds in valid_sizes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== Coverage Plot =====
    ax1.errorbar(valid_sizes, full_cov_means, yerr=full_cov_stds, 
                 fmt='o-', linewidth=2, markersize=8, capsize=5,
                 label=f'Full CP ({args.ncm})', color='blue')
    ax1.errorbar(valid_sizes, split_cov_means, yerr=split_cov_stds,
                 fmt='s--', linewidth=2, markersize=8, capsize=5,
                 label='Split CP (Softmax)', color='orange')
    ax1.axhline(y=1-args.alpha, color='red', linestyle=':', linewidth=2,
                label=f'Target (1-α={1-args.alpha:.2f})')
    
    ax1.set_xlabel('Data Size (samples)', fontsize=12)
    ax1.set_ylabel('Coverage', fontsize=12)
    ax1.set_title(f'Coverage vs Data Size (α={args.alpha})', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # ===== Set Size Plot =====
    ax2.errorbar(valid_sizes, full_size_means, yerr=full_size_stds,
                 fmt='o-', linewidth=2, markersize=8, capsize=5,
                 label=f'Full CP ({args.ncm})', color='blue')
    ax2.errorbar(valid_sizes, split_size_means, yerr=split_size_stds,
                 fmt='s--', linewidth=2, markersize=8, capsize=5,
                 label='Split CP (Softmax)', color='orange')
    
    ax2.set_xlabel('Data Size (samples)', fontsize=12)
    ax2.set_ylabel('Average Set Size', fontsize=12)
    ax2.set_title('Efficiency vs Data Size', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / f"full_vs_split_comparison_{args.ncm}.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot to {save_path}")
    plt.close()
    
    # ===== Additional: Coverage gap from target =====
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
    
    target = 1 - args.alpha
    full_gap = [m - target for m in full_cov_means]
    split_gap = [m - target for m in split_cov_means]
    
    ax3.bar(np.array(valid_sizes) - 15, full_gap, width=30, 
            label=f'Full CP ({args.ncm})', color='blue', alpha=0.7)
    ax3.bar(np.array(valid_sizes) + 15, split_gap, width=30,
            label='Split CP (Softmax)', color='orange', alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Target')
    
    ax3.set_xlabel('Data Size (samples)', fontsize=12)
    ax3.set_ylabel('Coverage Gap (Actual - Target)', fontsize=12)
    ax3.set_title('Coverage Gap from Target', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Singleton rates
    full_sing_means = [np.mean(results['full_cp'][ds]['singleton_rates']) for ds in valid_sizes]
    split_sing_means = [np.mean(results['split_cp'][ds]['singleton_rates']) for ds in valid_sizes]
    
    ax4.plot(valid_sizes, full_sing_means, 'o-', linewidth=2, markersize=8,
             label=f'Full CP ({args.ncm})', color='blue')
    ax4.plot(valid_sizes, split_sing_means, 's--', linewidth=2, markersize=8,
             label='Split CP (Softmax)', color='orange')
    
    ax4.set_xlabel('Data Size (samples)', fontsize=12)
    ax4.set_ylabel('Singleton Rate', fontsize=12)
    ax4.set_title('Singleton Rate vs Data Size', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path2 = Path(output_dir) / f"full_vs_split_detailed_{args.ncm}.png"
    plt.savefig(save_path2, dpi=150)
    print(f"Saved detailed comparison plot to {save_path2}")
    plt.close()
    
    # ===== Timing Comparison Plot =====
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute timing means and stds
    full_time_means = [np.mean(results['full_cp'][ds]['times']) for ds in valid_sizes]
    full_time_stds = [np.std(results['full_cp'][ds]['times']) for ds in valid_sizes]
    split_time_means = [np.mean(results['split_cp'][ds]['times']) for ds in valid_sizes]
    split_time_stds = [np.std(results['split_cp'][ds]['times']) for ds in valid_sizes]
    
    # Time vs Data Size
    ax5.errorbar(valid_sizes, full_time_means, yerr=full_time_stds,
                 fmt='o-', linewidth=2, markersize=8, capsize=5,
                 label=f'Full CP ({args.ncm})', color='blue')
    ax5.errorbar(valid_sizes, split_time_means, yerr=split_time_stds,
                 fmt='s--', linewidth=2, markersize=8, capsize=5,
                 label='Split CP (Softmax)', color='orange')
    
    ax5.set_xlabel('Data Size (samples)', fontsize=12)
    ax5.set_ylabel('Time (seconds)', fontsize=12)
    ax5.set_title('Computation Time vs Data Size', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Speedup ratio (Split CP time / Full CP time)
    speedup = [split_time_means[i] / full_time_means[i] if full_time_means[i] > 0 else 1 
               for i in range(len(valid_sizes))]
    
    ax6.bar(valid_sizes, speedup, width=50, color='green', alpha=0.7)
    ax6.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Equal time')
    ax6.set_xlabel('Data Size (samples)', fontsize=12)
    ax6.set_ylabel('Time Ratio (Split CP / Full CP)', fontsize=12)
    ax6.set_title('Relative Speed: >1 means Full CP is faster', fontsize=14)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path3 = Path(output_dir) / f"full_vs_split_timing_{args.ncm}.png"
    plt.savefig(save_path3, dpi=150)
    print(f"Saved timing comparison plot to {save_path3}")
    plt.close()


def compare_cp_methods(
    embeddings: np.ndarray,
    labels: np.ndarray,
    args,
    output_dir: str
):
    """
    Compare Full CP vs Split CP vs CV+ with a **fixed test set** and
    **varying calibration sizes** (the x-axis).

    For each ``cal_size`` in ``args.cal_sizes``:

    * Full CP / CV+: use all ``cal_size`` samples for calibration.
    * Split CP: split the same ``cal_size`` budget into
      ``train = int(split_train_ratio * cal_size)`` and
      ``cal = cal_size - train``, so train + cal == cal_size.

    Total pool per trial = ``test_size + max(cal_sizes)``, sampled
    with ``num_per_class = ceil(pool / n_classes)`` per class.
    The test set is carved first and shared across all methods/sizes.
    """
    import json
    import time
    import math

    test_size = args.test_size
    cal_sizes = [int(s.strip()) for s in args.cal_sizes.split(',')]
    split_train_ratio = args.split_train_ratio

    # Largest pool we'll ever need
    # Split CP splits cal_size into train + cal, so total non-test = cal_size
    max_cal = max(cal_sizes)
    max_needed = test_size + max_cal

    all_classes = np.unique(labels)
    n_classes = len(all_classes)
    class_counts = {int(c): int(np.sum(labels == c)) for c in all_classes}
    min_class_count = min(class_counts.values())

    # Deduce per-class budget
    num_per_class = math.ceil(max_needed / n_classes)
    if num_per_class > min_class_count:
        print(f"  Warning: need {num_per_class} per class but smallest class has "
              f"{min_class_count}. Clamping.")
        num_per_class = min_class_count
        # Reduce max_needed accordingly
        max_needed = num_per_class * n_classes

    print("\n" + "=" * 70)
    print("FULL CP vs CV+ vs SPLIT CP  --  VARYING CALIBRATION SIZE")
    print("=" * 70)
    print(f"Test size (fixed): {test_size}")
    print(f"Calibration sizes: {cal_sizes}")
    print(f"Split CP train ratio (of cal): {split_train_ratio}")
    print(f"Max pool needed: {max_needed}  ({num_per_class}/class x {n_classes} classes)")
    print(f"Trials per size: {args.n_trials}")
    ncm_cv_type = args.ncm_cv if args.ncm_cv is not None else args.ncm
    if args.ncm != "softmax":
        print(f"NCM (Full CP): {args.ncm}" + (f" (k={args.k})" if args.ncm in ["knn", "simplified_knn"] else ""))
    else:
        print(f"Full CP: skipped (softmax has no hypothetical update)")
    print(f"NCM (CV+): {ncm_cv_type}" + (f" (k={args.k})" if ncm_cv_type in ["knn", "simplified_knn"] else ""))
    print(f"Split CP: SoftmaxSplitCP (logistic regression)")
    print(f"CV+ folds: {args.n_folds}")
    print(f"Alpha: {args.alpha}")
    print("=" * 70)

    method_names = ['full_cp', 'cv_plus', 'split_cp']
    results = {
        'cal_sizes': cal_sizes,
        'test_size': test_size,
        'split_train_ratio': split_train_ratio,
        'num_per_class': num_per_class,
        'alpha': args.alpha,
        'ncm': args.ncm,
        'k': args.k,
        'n_trials': args.n_trials,
        'n_folds': args.n_folds,
    }
    for m in method_names:
        results[m] = {}
        for cs in cal_sizes:
            results[m][cs] = {
                'coverages': [], 'set_sizes': [], 'singleton_rates': [],
                'empty_rates': [], 'times': [],
                'n_cal': [], 'n_test': [], 'n_train': []
            }

    for trial in range(args.n_trials):
        trial_seed = args.seed + trial * 1000
        np.random.seed(trial_seed)

        # --- Stratified subsample: num_per_class per class ---
        pool_indices = []
        for c in np.unique(labels):
            c_idx = np.where(labels == c)[0]
            chosen = np.random.choice(c_idx, size=num_per_class, replace=False)
            pool_indices.append(chosen)
        pool_indices = np.concatenate(pool_indices)
        np.random.shuffle(pool_indices)

        X_pool = embeddings[pool_indices]
        y_pool = labels[pool_indices]

        # --- Carve out the FIXED test set (last test_size samples) ---
        actual_test_size = min(test_size, len(X_pool))
        X_test = X_pool[-actual_test_size:]
        y_test = y_pool[-actual_test_size:]
        X_remain = X_pool[:-actual_test_size]
        y_remain = y_pool[:-actual_test_size]

        print(f"\n{'=' * 70}")
        print(f"Trial {trial + 1}/{args.n_trials}  (seed={trial_seed}, "
              f"pool={len(X_pool)}, test={len(X_test)}, avail={len(X_remain)})")
        print("=" * 70)

        for cal_size in cal_sizes:
            if cal_size > len(X_remain):
                print(f"\n  cal_size={cal_size}: need {cal_size} non-test "
                      f"but only {len(X_remain)} available. Skipping.")
                continue

            # Full CP / CV+: first cal_size from remaining pool
            X_cal = X_remain[:cal_size]
            y_cal = y_remain[:cal_size]

            # Split CP: split the SAME cal_size budget into train + cal
            train_size = int(split_train_ratio * cal_size)
            cal_size_split = cal_size - train_size
            X_train_s = X_remain[:train_size]
            y_train_s = y_remain[:train_size]
            X_cal_s = X_remain[train_size:train_size + cal_size_split]
            y_cal_s = y_remain[train_size:train_size + cal_size_split]

            print(f"\n  cal_size={cal_size}  (Full/CV+: cal={len(X_cal)}, "
                  f"Split: train={len(X_train_s)}+cal={len(X_cal_s)}={len(X_train_s)+len(X_cal_s)}, "
                  f"test={len(X_test)})")

            # ---------- FULL CP ----------
            # softmax NCM doesn't support Full CP (no hypothetical update)
            if args.ncm != "softmax":
                ncm = create_ncm(args.ncm, k=args.k)
                print(f"    Full CP...")
                t0 = time.time()
                cp_full = FullConformalPredictor(ncm, alpha=args.alpha)
                cp_full.calibrate(X_cal, y_cal, all_classes=all_classes)
                m_full = cp_full.evaluate(X_test, y_test, verbose=False)
                t_full = time.time() - t0

                results['full_cp'][cal_size]['coverages'].append(m_full['coverage'])
                results['full_cp'][cal_size]['set_sizes'].append(m_full['avg_set_size'])
                results['full_cp'][cal_size]['singleton_rates'].append(m_full['singleton_rate'])
                results['full_cp'][cal_size]['empty_rates'].append(m_full['empty_set_rate'])
                results['full_cp'][cal_size]['times'].append(t_full)
                results['full_cp'][cal_size]['n_cal'].append(len(X_cal))
                results['full_cp'][cal_size]['n_test'].append(len(X_test))

                print(f"      Cov={m_full['coverage']:.3f}, Size={m_full['avg_set_size']:.2f}, "
                      f"T={t_full:.2f}s")

            # ---------- CV+ ----------
            ncm_factory = lambda: create_ncm(ncm_cv_type, k=args.k)
            print(f"    CV+...")
            t0 = time.time()
            cp_cv = CrossValidationPlusPredictor(
                ncm_factory, alpha=args.alpha, n_folds=args.n_folds
            )
            cp_cv.calibrate(X_cal, y_cal, all_classes=all_classes)
            m_cv = cp_cv.evaluate(X_test, y_test, verbose=False)
            t_cv = time.time() - t0

            results['cv_plus'][cal_size]['coverages'].append(m_cv['coverage'])
            results['cv_plus'][cal_size]['set_sizes'].append(m_cv['avg_set_size'])
            results['cv_plus'][cal_size]['singleton_rates'].append(m_cv['singleton_rate'])
            results['cv_plus'][cal_size]['empty_rates'].append(m_cv['empty_set_rate'])
            results['cv_plus'][cal_size]['times'].append(t_cv)
            results['cv_plus'][cal_size]['n_cal'].append(len(X_cal))
            results['cv_plus'][cal_size]['n_test'].append(len(X_test))

            print(f"      Cov={m_cv['coverage']:.3f}, Size={m_cv['avg_set_size']:.2f}, "
                  f"T={t_cv:.2f}s")

            # ---------- SPLIT CP ----------
            print(f"    Split CP...")
            t0 = time.time()
            cp_split = SoftmaxSplitCP(alpha=args.alpha)
            cp_split.fit(X_train_s, y_train_s)
            cp_split.calibrate(X_cal_s, y_cal_s, all_classes=all_classes)
            m_split = cp_split.evaluate(X_test, y_test, verbose=False)
            t_split = time.time() - t0

            results['split_cp'][cal_size]['coverages'].append(m_split['coverage'])
            results['split_cp'][cal_size]['set_sizes'].append(m_split['avg_set_size'])
            results['split_cp'][cal_size]['singleton_rates'].append(m_split['singleton_rate'])
            results['split_cp'][cal_size]['empty_rates'].append(m_split['empty_set_rate'])
            results['split_cp'][cal_size]['times'].append(t_split)
            results['split_cp'][cal_size]['n_cal'].append(len(X_cal_s))
            results['split_cp'][cal_size]['n_test'].append(len(X_test))
            results['split_cp'][cal_size]['n_train'].append(len(X_train_s))

            print(f"      Cov={m_split['coverage']:.3f}, Size={m_split['avg_set_size']:.2f}, "
                  f"T={t_split:.2f}s")

        # Aggregate for this trial
        print(f"\n  --- Trial {trial + 1} summary ---")
        for cs in cal_sizes:
            if not results['full_cp'][cs]['coverages']:
                continue
            # Only show last entry (this trial)
            print(f"  cal={cs:>5d}:  "
                  f"Full={results['full_cp'][cs]['coverages'][-1]*100:.1f}%  "
                  f"CV+={results['cv_plus'][cs]['coverages'][-1]*100:.1f}%  "
                  f"Split={results['split_cp'][cs]['coverages'][-1]*100:.1f}%")

    # ---------- Final aggregate ----------
    print(f"\n{'=' * 70}")
    print(f"FINAL AGGREGATE  ({args.n_trials} trials)")
    print("=" * 70)
    for cs in cal_sizes:
        for mname, label in [('full_cp', 'Full CP'), ('cv_plus', 'CV+'), ('split_cp', 'Split CP')]:
            d = results[mname][cs]
            if not d['coverages']:
                continue
            print(f"  cal={cs:>5d} {label:10s}: "
                  f"Cov={np.mean(d['coverages'])*100:.1f}%+/-{np.std(d['coverages'])*100:.1f}%, "
                  f"Size={np.mean(d['set_sizes']):.2f}+/-{np.std(d['set_sizes']):.2f}, "
                  f"T={np.mean(d['times']):.2f}s")

    # ---------- PLOTS ----------
    plot_cp_methods_comparison(results, args, output_dir)

    # ---------- SAVE ----------
    results_path = Path(output_dir) / f"cp_methods_comparison_{args.ncm}.json"

    def _to_list(obj):
        """Convert numpy types for JSON serialisation."""
        if isinstance(obj, dict):
            return {str(k): _to_list(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_list(i) for i in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        return obj

    with open(results_path, 'w') as f:
        json.dump(_to_list(results), f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Text summary
    summary_path = Path(output_dir) / f"cp_methods_comparison_{args.ncm}.txt"
    with open(summary_path, 'w') as f:
        f.write("Full CP vs CV+ vs Split CP Comparison (Varying Calibration Size)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test size (fixed): {test_size}\n")
        f.write(f"Num per class: {num_per_class}\n")
        f.write(f"NCM (Full CP): {args.ncm}\n")
        f.write(f"NCM (CV+): {ncm_cv_type}\n")
        f.write(f"Split CP: SoftmaxSplitCP (logistic regression)\n")
        f.write(f"Split CP train ratio (of cal): {split_train_ratio}\n")
        f.write(f"CV+ folds: {args.n_folds}\n")
        f.write(f"Alpha: {args.alpha}, Target coverage: {1 - args.alpha:.2f}\n")
        f.write(f"Trials: {args.n_trials}\n\n")

        header = (f"{'Cal Size':<10} {'Method':<12} {'Coverage':<18} "
                  f"{'Avg Set Size':<18} {'Time (s)':<18}\n")
        f.write(header)
        f.write("-" * 76 + "\n")

        for cs in cal_sizes:
            for mname, label in [('full_cp', 'Full CP'), ('cv_plus', 'CV+'), ('split_cp', 'Split CP')]:
                d = results[mname].get(cs, {})
                if not d.get('coverages'):
                    continue
                f.write(
                    f"{cs if mname == 'full_cp' else '':<10} {label:<12} "
                    f"{np.mean(d['coverages'])*100:.1f}%+/-{np.std(d['coverages'])*100:.1f}%{'':>4} "
                    f"{np.mean(d['set_sizes']):.2f}+/-{np.std(d['set_sizes']):.2f}{'':>6} "
                    f"{np.mean(d['times']):.2f}+/-{np.std(d['times']):.2f}\n"
                )
            f.write("-" * 76 + "\n")

    print(f"Saved summary to {summary_path}")
    return results


def plot_cp_methods_comparison(results: dict, args, output_dir: str):
    """Plot coverage, efficiency, singleton rate, and timing for Full CP vs CV+ vs Split CP.

    X-axis is calibration size (int); test size is fixed.
    """
    cal_sizes = results['cal_sizes']
    test_size = results['test_size']

    # Filter to sizes that actually have data (check any method)
    valid_sizes = [cs for cs in cal_sizes
                   if any(cs in results[m] and results[m][cs].get('coverages')
                          for m in ['full_cp', 'cv_plus', 'split_cp'])]
    if not valid_sizes:
        print("No valid calibration sizes to plot.")
        return

    # Only include methods that actually produced results
    all_method_styles = {
        'full_cp':  {'label': 'Full CP',  'color': 'blue',   'fmt': 'o-'},
        'cv_plus':  {'label': 'CV+',      'color': 'green',  'fmt': 'D-'},
        'split_cp': {'label': 'Split CP', 'color': 'orange', 'fmt': 's--'},
    }
    method_styles = {mk: sty for mk, sty in all_method_styles.items()
                     if any(results[mk][cs].get('coverages') for cs in valid_sizes
                            if cs in results[mk])}

    def _means_stds(method_key, metric_key):
        means = [np.mean(results[method_key][cs][metric_key]) for cs in valid_sizes]
        stds  = [np.std(results[method_key][cs][metric_key])  for cs in valid_sizes]
        return means, stds

    # ---- Figure 1: Coverage + Set Size ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for mk, sty in method_styles.items():
        m, s = _means_stds(mk, 'coverages')
        ax1.errorbar(valid_sizes, m, yerr=s, fmt=sty['fmt'], color=sty['color'],
                     label=sty['label'], linewidth=2, markersize=8, capsize=5)
    ax1.axhline(y=1 - args.alpha, color='red', linestyle=':', linewidth=2,
                label=f'Target (1-alpha={1 - args.alpha:.2f})')
    ax1.set_xlabel('Calibration Size', fontsize=12)
    ax1.set_ylabel('Coverage', fontsize=12)
    ax1.set_title(f'Coverage vs Cal Size (test={test_size}, alpha={args.alpha})', fontsize=14)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    for mk, sty in method_styles.items():
        m, s = _means_stds(mk, 'set_sizes')
        ax2.errorbar(valid_sizes, m, yerr=s, fmt=sty['fmt'], color=sty['color'],
                     label=sty['label'], linewidth=2, markersize=8, capsize=5)
    ax2.set_xlabel('Calibration Size', fontsize=12)
    ax2.set_ylabel('Average Set Size', fontsize=12)
    ax2.set_title(f'Efficiency vs Cal Size (test={test_size})', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(output_dir) / f"cp_methods_comparison_{args.ncm}.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison plot to {save_path}")
    plt.close()

    # ---- Figure 2: Coverage gap + Singleton rates ----
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))

    target = 1 - args.alpha
    n_sizes = len(valid_sizes)
    if n_sizes > 1:
        bar_width = max(1, (valid_sizes[-1] - valid_sizes[0]) / n_sizes / 4)
    else:
        bar_width = 20
    offsets = {'full_cp': -bar_width, 'cv_plus': 0, 'split_cp': bar_width}

    for mk, sty in method_styles.items():
        m, _ = _means_stds(mk, 'coverages')
        gap = [v - target for v in m]
        ax3.bar(np.array(valid_sizes, dtype=float) + offsets[mk], gap, width=bar_width,
                label=sty['label'], color=sty['color'], alpha=0.7)
    ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Target')
    ax3.set_xlabel('Calibration Size', fontsize=12)
    ax3.set_ylabel('Coverage Gap (Actual - Target)', fontsize=12)
    ax3.set_title(f'Coverage Gap from Target (test={test_size})', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    for mk, sty in method_styles.items():
        m, _ = _means_stds(mk, 'singleton_rates')
        ax4.plot(valid_sizes, m, sty['fmt'], color=sty['color'],
                 label=sty['label'], linewidth=2, markersize=8)
    ax4.set_xlabel('Calibration Size', fontsize=12)
    ax4.set_ylabel('Singleton Rate', fontsize=12)
    ax4.set_title(f'Singleton Rate vs Cal Size (test={test_size})', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path2 = Path(output_dir) / f"cp_methods_detailed_{args.ncm}.png"
    plt.savefig(save_path2, dpi=150)
    print(f"Saved detailed comparison plot to {save_path2}")
    plt.close()

    # ---- Figure 3: Timing ----
    fig, ax5 = plt.subplots(1, 1, figsize=(8, 5))

    for mk, sty in method_styles.items():
        m, s = _means_stds(mk, 'times')
        ax5.errorbar(valid_sizes, m, yerr=s, fmt=sty['fmt'], color=sty['color'],
                     label=sty['label'], linewidth=2, markersize=8, capsize=5)
    ax5.set_xlabel('Calibration Size', fontsize=12)
    ax5.set_ylabel('Time (seconds)', fontsize=12)
    ax5.set_title(f'Computation Time vs Cal Size (test={test_size})', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path3 = Path(output_dir) / f"cp_methods_timing_{args.ncm}.png"
    plt.savefig(save_path3, dpi=150)
    print(f"Saved timing comparison plot to {save_path3}")
    plt.close()


def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*70)
    print("SSL + FULL CONFORMAL PREDICTION EXPERIMENT")
    print("="*70)
    
    # 1. Load embeddings
    print(f"\n[1/5] Loading embeddings from {args.embeddings_path}...")
    data = torch.load(args.embeddings_path)
    embeddings = data["embeddings"].numpy()
    labels = data["labels"].numpy()
    
    print(f"  Shape: {embeddings.shape}, Labels: {len(np.unique(labels))} classes")
    
    # 2. Split data (Full CP uses only calibration + test)
    if args.balanced:
        n_total = len(embeddings)
        cal_size = int(n_total * args.cal_ratio)
        test_size = n_total - cal_size
        print(f"\n[2/5] Stratified balanced split (cal={cal_size}, test={test_size})...")
        X_cal, y_cal, X_test, y_test = stratified_cal_test_split(
            embeddings, labels,
            cal_size=cal_size,
            test_size=test_size,
            balanced=True,
            random_state=args.seed
        )
    else:
        print(f"\n[2/5] Splitting data (cal={args.cal_ratio}, test={1-args.cal_ratio})...")
        X_cal, y_cal, X_test, y_test = cal_test_split(
            embeddings, labels,
            cal_ratio=args.cal_ratio,
            random_state=args.seed
        )
    
    # 3. Create nonconformity measure
    if args.ncm in ["knn", "simplified_knn"]:
        print(f"\n[3/5] Creating nonconformity measure: {args.ncm} (k={args.k})...")
    else:
        print(f"\n[3/5] Creating nonconformity measure: {args.ncm}...")
    ncm = create_ncm(args.ncm, k=args.k)
    
    # 3b. Optional: Compare both methods
    if args.compare_methods:
        print(f"\n[3b] Runtime Comparison: k-NN vs Simplified k-NN...")
        print("="*70)
        
        methods = {
            'mahal_nn_ratio': create_ncm('mahal_nn_ratio', k=args.k),
            'geodesic_topk_mean': create_ncm('geodesic_topk_mean', k=args.k)
        }
        
        comparison_results = {}
        
        for method_name, method_ncm in methods.items():
            print(f"\nTesting {method_name}...")
            cp_temp = FullConformalPredictor(method_ncm, alpha=args.alpha)
            cp_temp.calibrate(X_cal, y_cal)
            
            # Time prediction on test set
            result_temp = cp_temp.predict(X_test, return_p_values=False, verbose=True)
            
            comparison_results[method_name] = {
                'prediction_time': result_temp['prediction_time'],
                'avg_set_size': np.mean(result_temp['set_sizes']),
                'coverage': np.mean([y_test[i] in pred for i, pred in enumerate(result_temp['prediction_sets'])])
            }
        
        # Print comparison
        print("\n" + "="*70)
        print("RUNTIME COMPARISON")
        print("="*70)
        for method_name, results in comparison_results.items():
            print(f"\n{method_name}:")
            print(f"  Prediction time: {results['prediction_time']:.2f}s")
            print(f"  Time per sample: {results['prediction_time']/len(X_test)*1000:.1f}ms")
            print(f"  Avg set size: {results['avg_set_size']:.2f}")
            print(f"  Coverage: {results['coverage']:.3f}")
        
        speedup = comparison_results['k-NN']['prediction_time'] / comparison_results['Simplified k-NN']['prediction_time']
        print(f"\nSpeedup: {speedup:.2f}x ({comparison_results['Simplified k-NN']['prediction_time']/comparison_results['k-NN']['prediction_time']*100:.1f}% of k-NN time)")
        print("="*70)
    
    # 3b2. Optional: Compare different k values
    if args.compare_k:
        print(f"\n[3b2] Comparing performance across different k values...")
        compare_k_values(X_cal, y_cal, X_test, y_test, args, args.output_dir)
    
    # 3c. Optional: Compare SSL vs Baseline (raw pixels)
    if args.compare_with_baseline:
        if args.num_per_class is None:
            raise ValueError("--num_per_class is required for baseline comparison")
        print(f"\n[3c] Loading raw pixel features for baseline comparison...")
        raw_features, raw_labels = load_raw_pixel_features(args.data_dir, args.num_per_class)
        print(f"  Raw features shape: {raw_features.shape}")
        # Use first num_per_class*n_classes from SSL embeddings to match raw
        n_raw = len(raw_features)
        compare_ssl_vs_baseline(embeddings[:n_raw], labels[:n_raw], raw_features, raw_labels, args, args.output_dir)
    
    # 3d. Optional: Compare Full CP vs Split CP across data sizes
    if args.compare_full_vs_split:
        print(f"\n[3d] Comparing Full CP vs Split CP across data sizes...")
        data_sizes = [int(s.strip()) for s in args.data_sizes.split(',')]
        Path(args.output_dir).mkdir(exist_ok=True)
        compare_full_cp_vs_split_cp(embeddings, labels, data_sizes, args, args.output_dir)
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
        return  # Skip default experiments when running comparison

    # 3e. Optional: Compare Full CP vs CV+ vs Split CP
    if args.compare_cp_methods:
        print(f"\n[3e] Comparing Full CP vs CV+ vs Split CP across calibration ratios...")
        Path(args.output_dir).mkdir(exist_ok=True)
        compare_cp_methods(embeddings, labels, args, args.output_dir)
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE!")
        print("="*70)
        return
    
    # 4. Run Full Conformal Prediction
    print(f"\n[4/5] Running Full Conformal Prediction (α={args.alpha})...")
    all_classes = np.unique(labels)

    cp = FullConformalPredictor(ncm, alpha=args.alpha)
    cp.calibrate(X_cal, y_cal, all_classes=all_classes)
    
    # Evaluate
    metrics = cp.evaluate(X_test, y_test, verbose=True)
    
    # Get predictions for visualization
    results = cp.predict(X_test, return_p_values=True, verbose=False)
    prediction_sets = results['prediction_sets']
    set_sizes = results['set_sizes']
    p_values = results['p_values']
    
    # 5. Visualizations and analysis
    print(f"\n[5/5] Generating visualizations...")
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Plot set size distribution
    plot_set_size_distribution(set_sizes, args.alpha, args.output_dir)
    
    # Plot coverage vs alpha
    plot_coverage_by_alpha(X_cal, y_cal, X_test, y_test, ncm, args.output_dir)
    
    # Per-class analysis
    per_class_cov, per_class_size = analyze_per_class_coverage(
        y_test, prediction_sets, args.output_dir
    )
    
    # Save results
    results_path = Path(args.output_dir) / "cp_results.txt"
    with open(results_path, "w") as f:
        f.write("Conformal Prediction Results\n")
        f.write("="*60 + "\n\n")
        f.write("Overall Metrics\n")
        f.write(f"  Alpha: {metrics['alpha']:.4f}\n")
        f.write(f"  Target coverage: {metrics['target_coverage']:.4f}\n")
        f.write(f"  Coverage: {metrics['coverage']:.4f}\n")
        f.write(f"  Avg set size: {metrics['avg_set_size']:.4f}\n")
        f.write(f"  Median set size: {metrics['median_set_size']:.4f}\n")
        f.write(f"  Singleton rate: {metrics['singleton_rate']:.4f}\n")
        f.write(f"  Singleton accuracy: {metrics['singleton_accuracy']:.4f}\n")
        f.write(f"  Empty set rate: {metrics['empty_set_rate']:.4f}\n\n")
        f.write("Per-class Coverage\n")
        for cls, cov in sorted(per_class_cov.items()):
            f.write(f"  Class {cls}: coverage={cov:.4f}, avg_set_size={per_class_size[cls]:.4f}\n")
        f.write("\nArgs\n")
        for k, v in vars(args).items():
            f.write(f"  {k}: {v}\n")
    print(f"\nSaved results to {results_path}")
    
    # Optionally save predictions
    if args.save_predictions:
        predictions_dict = {
            'prediction_sets': prediction_sets,
            'p_values': p_values,
            'y_test': y_test,
            'y_true': y_test
        }
        pred_path = Path(args.output_dir) / "predictions.pt"
        torch.save(predictions_dict, pred_path)
        print(f"Saved predictions to {pred_path}")
    
    # Print some example predictions
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS (first 10 test examples)")
    print("="*70)
    for i in range(min(10, len(y_test))):
        true_label = y_test[i]
        pred_set = prediction_sets[i]
        correct = "✓" if true_label in pred_set else "✗"
        print(f"Test {i}: True={true_label}, Prediction Set={pred_set}, Size={len(pred_set)} {correct}")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
