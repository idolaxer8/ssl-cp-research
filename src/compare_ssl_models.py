"""
Compare SSL Models on Conformal Prediction

Runs conformal prediction experiments across multiple SSL models and alpha values,
then generates a comparison plot of efficiency (set size) vs alpha.

Usage:
    python src/compare_ssl_models.py --data_dir data/cifar100 --num_per_class 100 --output_dir output/ssl_comparison
    
    # Skip feature extraction if embeddings already exist
    python src/compare_ssl_models.py --data_dir data/cifar100 --skip_extraction --output_dir output/ssl_comparison
"""

import argparse
import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time

# Import from local modules
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformal_prediction import (
    SimplifiedKNNNonconformity, 
    FullConformalPredictor, 
    cal_test_split
)

# Base SSL models to compare (only base/standard sizes)
SSL_MODELS = {
    "dinov2-base": {"family": "Self-Distillation", "color": "#1f77b4", "marker": "o"},
    "clip-base": {"family": "Vision-Language", "color": "#ff7f0e", "marker": "s"},
    "beit-base": {"family": "Masked Image Modeling", "color": "#2ca02c", "marker": "^"},
    "mae-base": {"family": "Masked Autoencoder", "color": "#d62728", "marker": "D"},
    "ssl-resnet50": {"family": "Contrastive (SSL)", "color": "#9467bd", "marker": "v"},
    "swsl-resnet50": {"family": "Semi-Weakly Supervised", "color": "#8c564b", "marker": "p"},
}

# Alpha values to test
ALPHA_VALUES = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SSL Models on Conformal Prediction")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/cifar100",
                       help="Path to dataset directory")
    parser.add_argument("--num_per_class", type=int, default=100,
                       help="Number of images per class")
    
    # Feature extraction
    parser.add_argument("--skip_extraction", action="store_true",
                       help="Skip feature extraction (use existing embeddings)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for feature extraction")
    
    # Conformal prediction
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                       help="Calibration set ratio")
    parser.add_argument("--ncm", type=str, default="simplified_knn",
                       choices=["simplified_knn", "knn", "centroid", "relative_centroid", "ridge"],
                       help="Nonconformity measure")
    parser.add_argument("--k", type=int, default=5,
                       help="k for k-NN based NCM")
    
    # Alpha values
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHA_VALUES,
                       help="Alpha values to test")
    
    # Subsampling
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum total samples to use from embeddings (e.g., 2000 for 1k cal + 1k test)")
    parser.add_argument("--samples_per_class", type=int, default=None,
                       help="Maximum samples per class to use (stratified subsampling)")
    
    # Models
    parser.add_argument("--models", type=str, nargs="+", default=list(SSL_MODELS.keys()),
                       help="SSL models to compare")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output/ssl_comparison",
                       help="Output directory")
    
    # Calibration plot
    parser.add_argument("--calibration_plot", action="store_true",
                       help="Generate calibration plot (confidence vs accuracy)")
    parser.add_argument("--n_bins", type=int, default=10,
                       help="Number of bins for calibration plot")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def extract_features(model_name, data_dir, output_dir, num_per_class, batch_size):
    """Extract features using a specific SSL model."""
    embeddings_path = Path(output_dir) / f"embeddings_{model_name.replace('-', '_')}.pt"
    
    if embeddings_path.exists():
        print(f"  Embeddings already exist: {embeddings_path}")
        return embeddings_path
    
    print(f"  Extracting features with {model_name}...")
    cmd = [
        "python", "src/extract_features.py",
        "--data_dir", data_dir,
        "--model", model_name,
        "--output_name", f"embeddings_{model_name.replace('-', '_')}.pt",
        "--batch_size", str(batch_size),
    ]
    if num_per_class:
        cmd.extend(["--num_per_class", str(num_per_class)])
    
    # Run extraction
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR extracting {model_name}:")
        print(result.stderr)
        return None
    
    # Move to output dir
    src_path = Path("output") / f"embeddings_{model_name.replace('-', '_')}.pt"
    if src_path.exists():
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.rename(embeddings_path)
    
    return embeddings_path


def subsample_data(embeddings, labels, max_samples=None, samples_per_class=None, seed=42):
    """Subsample embeddings to reduce dataset size.
    
    Args:
        embeddings: Full embeddings array
        labels: Full labels array
        max_samples: Maximum total samples (random subsample)
        samples_per_class: Maximum samples per class (stratified subsample)
        seed: Random seed
    
    Returns:
        Subsampled embeddings and labels
    """
    np.random.seed(seed)
    n_total = len(labels)
    
    if samples_per_class is not None:
        # Stratified subsampling: take at most samples_per_class from each class
        unique_classes = np.unique(labels)
        selected_indices = []
        
        for cls in unique_classes:
            cls_indices = np.where(labels == cls)[0]
            n_select = min(len(cls_indices), samples_per_class)
            selected = np.random.choice(cls_indices, n_select, replace=False)
            selected_indices.extend(selected)
        
        selected_indices = np.array(selected_indices)
        np.random.shuffle(selected_indices)
        
        return embeddings[selected_indices], labels[selected_indices]
    
    elif max_samples is not None and max_samples < n_total:
        # Random subsampling
        indices = np.random.choice(n_total, max_samples, replace=False)
        return embeddings[indices], labels[indices]
    
    return embeddings, labels


def compute_calibration_metrics(X_train, y_train, X_test, y_test, n_bins=10):
    """Compute calibration metrics: confidence vs accuracy per bin, and ECE.
    
    Uses logistic regression with softmax to get confidence scores.
    Returns dict with bin_centers, bin_accuracies, bin_confidences, bin_counts, ece.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features for better logistic regression convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit logistic regression (softmax for multi-class)
    clf = LogisticRegression(
        max_iter=1000, 
        solver='lbfgs'
    )
    clf.fit(X_train_scaled, y_train)
    
    # Get predicted probabilities
    probs = clf.predict_proba(X_test_scaled)
    
    # Top-1 confidence and predictions
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == y_test).astype(float)
    
    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:  # Include right edge for last bin
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        
        bin_count = mask.sum()
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            bin_accuracies.append(correct[mask].mean())
            bin_confidences.append(confidences[mask].mean())
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
    
    # Compute ECE (Expected Calibration Error)
    ece = 0.0
    total_samples = len(y_test)
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_confidences[i])
    
    return {
        "bin_centers": bin_centers,
        "bin_boundaries": bin_boundaries,
        "bin_accuracies": np.array(bin_accuracies),
        "bin_confidences": np.array(bin_confidences),
        "bin_counts": np.array(bin_counts),
        "ece": ece,
        "accuracy": correct.mean(),
    }


def run_conformal_experiment(embeddings_path, alphas, cal_ratio, ncm_type, k, seed,
                             max_samples=None, samples_per_class=None, compute_calibration=False, n_bins=10):
    """Run conformal prediction for multiple alpha values."""
    # Load embeddings
    data = torch.load(embeddings_path, weights_only=False)
    embeddings = data["embeddings"].numpy()
    labels = data["labels"].numpy()
    
    # Subsample if requested
    original_size = len(labels)
    embeddings, labels = subsample_data(
        embeddings, labels, max_samples, samples_per_class, seed
    )
    if len(labels) < original_size:
        print(f"  Subsampled: {original_size} → {len(labels)} samples")
    
    # Split data
    X_cal, y_cal, X_test, y_test = cal_test_split(
        embeddings, labels, cal_ratio=cal_ratio, random_state=seed
    )
    
    # Create NCM
    if ncm_type == "simplified_knn":
        ncm = SimplifiedKNNNonconformity(k=k)
    else:
        from conformal_prediction import (
            KNNNonconformity, CentroidNonconformity, 
            RelativeCentroidNonconformity, RidgeRegressionNonconformity
        )
        if ncm_type == "knn":
            ncm = KNNNonconformity(k=k)
        elif ncm_type == "centroid":
            ncm = CentroidNonconformity()
        elif ncm_type == "relative_centroid":
            ncm = RelativeCentroidNonconformity()
        elif ncm_type == "ridge":
            ncm = RidgeRegressionNonconformity()
    
    results = {"alphas": [], "coverages": [], "avg_set_sizes": [], "median_set_sizes": []}
    
    # Compute calibration metrics if requested
    if compute_calibration:
        calibration = compute_calibration_metrics(X_cal, y_cal, X_test, y_test, n_bins=n_bins)
        results["calibration"] = calibration
    
    for alpha in alphas:
        # Create predictor
        predictor = FullConformalPredictor(ncm, alpha=alpha)
        predictor.calibrate(X_cal, y_cal)
        
        # Evaluate
        metrics = predictor.evaluate(X_test, y_test, verbose=False)
        
        results["alphas"].append(alpha)
        results["coverages"].append(metrics["coverage"])
        results["avg_set_sizes"].append(metrics["avg_set_size"])
        results["median_set_sizes"].append(metrics["median_set_size"])
    
    return results


def plot_comparison(all_results, output_dir, ncm_type):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Efficiency (Avg Set Size) vs Alpha
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Average set size vs alpha
    ax1 = axes[0]
    for model_name, results in all_results.items():
        if results is None:
            continue
        info = SSL_MODELS.get(model_name, {"color": "gray", "marker": "x"})
        ax1.plot(results["alphas"], results["avg_set_sizes"], 
                marker=info["marker"], color=info["color"], 
                label=f"{model_name}", linewidth=2, markersize=8)
    
    ax1.set_xlabel("α (Significance Level)", fontsize=12)
    ax1.set_ylabel("Average Set Size (↓ better)", fontsize=12)
    ax1.set_title("Efficiency: Average Prediction Set Size vs α", fontsize=14)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, max(ALPHA_VALUES) + 0.02])
    
    # Right plot: Coverage vs Alpha (should be close to 1-alpha)
    ax2 = axes[1]
    for model_name, results in all_results.items():
        if results is None:
            continue
        info = SSL_MODELS.get(model_name, {"color": "gray", "marker": "x"})
        ax2.plot(results["alphas"], results["coverages"], 
                marker=info["marker"], color=info["color"], 
                label=f"{model_name}", linewidth=2, markersize=8)
    
    # Add target coverage line (1 - alpha)
    target_alphas = np.array(ALPHA_VALUES)
    ax2.plot(target_alphas, 1 - target_alphas, 'k--', linewidth=2, label="Target (1-α)")
    
    ax2.set_xlabel("α (Significance Level)", fontsize=12)
    ax2.set_ylabel("Coverage", fontsize=12)
    ax2.set_title("Validity: Coverage vs α (should match 1-α)", fontsize=14)
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, max(ALPHA_VALUES) + 0.02])
    ax2.set_ylim([0.6, 1.02])
    
    plt.tight_layout()
    save_path = output_dir / f"ssl_comparison_{ncm_type}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {save_path}")
    plt.close()
    
    # Plot 2: Efficiency at fixed coverage (e.g., 90%)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Bar chart of set sizes at alpha=0.1 (90% coverage target)
    # Get alphas from first valid result
    result_alphas = None
    for _model_name, _results in all_results.items():
        if _results is not None and "alphas" in _results:
            result_alphas = _results["alphas"]
            break
    
    if result_alphas is None:
        print("No results for efficiency ranking plot.")
        return
    
    # Find closest alpha to 0.1
    alpha_idx = min(range(len(result_alphas)), key=lambda i: abs(result_alphas[i] - 0.1))
    target_alpha = result_alphas[alpha_idx]
    
    model_names = []
    set_sizes = []
    colors = []
    
    for model_name, results in all_results.items():
        if results is None:
            continue
        model_names.append(model_name)
        set_sizes.append(results["avg_set_sizes"][alpha_idx])
        colors.append(SSL_MODELS.get(model_name, {"color": "gray"})["color"])
    
    # Sort by set size
    sorted_idx = np.argsort(set_sizes)
    model_names = [model_names[i] for i in sorted_idx]
    set_sizes = [set_sizes[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    
    bars = ax.barh(model_names, set_sizes, color=colors)
    ax.set_xlabel("Average Set Size", fontsize=12)
    ax.set_title(f"Efficiency at α={target_alpha} ({100*(1-target_alpha):.0f}% Target Coverage)\nLower is Better", fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, size in zip(bars, set_sizes):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
               f'{size:.2f}', va='center', fontsize=10)
    
    plt.tight_layout()
    save_path = output_dir / f"ssl_efficiency_ranking_{ncm_type}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved efficiency ranking to {save_path}")
    plt.close()


def plot_calibration(all_results, output_dir):
    """Generate calibration plot (reliability diagram) comparing all models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if any model has calibration data
    has_calibration = any(
        results is not None and "calibration" in results 
        for results in all_results.values()
    )
    if not has_calibration:
        print("No calibration data available for plotting.")
        return
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Reliability diagram (confidence vs accuracy)
    ax1 = axes[0]
    
    # Plot perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect calibration')
    
    for model_name, results in all_results.items():
        if results is None or "calibration" not in results:
            continue
        
        cal = results["calibration"]
        info = SSL_MODELS.get(model_name, {"color": "gray", "marker": "x"})
        
        # Filter out empty bins
        mask = ~np.isnan(cal["bin_accuracies"])
        bin_conf = cal["bin_confidences"][mask]
        bin_acc = cal["bin_accuracies"][mask]
        
        ax1.plot(bin_conf, bin_acc,
                marker=info["marker"], color=info["color"],
                label=f"{model_name} (ECE={cal['ece']:.3f})",
                linewidth=2, markersize=8)
    
    ax1.set_xlabel("Confidence (Softmax Probability)", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Calibration: Confidence vs Accuracy\n(Reliability Diagram)", fontsize=14)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    
    # Right: ECE bar chart
    ax2 = axes[1]
    
    model_names = []
    eces = []
    accuracies = []
    colors = []
    
    for model_name, results in all_results.items():
        if results is None or "calibration" not in results:
            continue
        model_names.append(model_name)
        eces.append(results["calibration"]["ece"])
        accuracies.append(results["calibration"]["accuracy"])
        colors.append(SSL_MODELS.get(model_name, {"color": "gray"})["color"])
    
    # Sort by ECE (lower is better)
    sorted_idx = np.argsort(eces)
    model_names = [model_names[i] for i in sorted_idx]
    eces = [eces[i] for i in sorted_idx]
    accuracies = [accuracies[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, eces, width, label='ECE (↓ better)', color=colors, alpha=0.7)
    bars2 = ax2.bar(x + width/2, accuracies, width, label='Accuracy (↑ better)', color=colors, alpha=1.0, hatch='//')
    
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Score", fontsize=12)
    ax2.set_title("Calibration Error & Accuracy Comparison", fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, val in zip(bars1, eces):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    for bar, val in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = output_dir / "ssl_calibration_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved calibration plot to {save_path}")
    plt.close()
    
    # Also create a histogram version showing confidence distribution
    n_models = sum(1 for r in all_results.values() if r is not None and "calibration" in r)
    if n_models > 0:
        fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4), squeeze=False)
        axes = axes.flatten()
        
        idx = 0
        for model_name, results in all_results.items():
            if results is None or "calibration" not in results:
                continue
            
            cal = results["calibration"]
            info = SSL_MODELS.get(model_name, {"color": "gray"})
            ax = axes[idx]
            
            # Bar chart of bin counts
            bin_centers = cal["bin_centers"]
            bin_counts = cal["bin_counts"]
            bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.1
            
            # Color bars by gap between confidence and accuracy
            gap_colors = []
            for i in range(len(bin_counts)):
                if bin_counts[i] > 0 and not np.isnan(cal["bin_accuracies"][i]):
                    gap = cal["bin_confidences"][i] - cal["bin_accuracies"][i]
                    # Red if overconfident, blue if underconfident
                    if gap > 0:
                        gap_colors.append('salmon')
                    else:
                        gap_colors.append('lightblue')
                else:
                    gap_colors.append('lightgray')
            
            ax.bar(bin_centers, bin_counts, width=bin_width*0.9, color=gap_colors, edgecolor='black')
            ax.set_xlabel("Confidence", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"{model_name}\nECE={cal['ece']:.3f}, Acc={cal['accuracy']:.3f}", fontsize=11)
            ax.set_xlim([0, 1])
            ax.grid(True, alpha=0.3, axis='y')
            
            idx += 1
        
        plt.suptitle("Confidence Distribution (Red=Overconfident, Blue=Underconfident)", fontsize=12, y=1.02)
        plt.tight_layout()
        save_path = output_dir / "ssl_confidence_distribution.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confidence distribution to {save_path}")
        plt.close()


def main():
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SSL MODEL COMPARISON ON CONFORMAL PREDICTION")
    print("="*70)
    print(f"Dataset: {args.data_dir}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Alpha values: {args.alphas}")
    print(f"NCM: {args.ncm} (k={args.k})")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Step 1: Extract features for all models
    if not args.skip_extraction:
        print("\n" + "="*70)
        print("STEP 1: FEATURE EXTRACTION")
        print("="*70)
        
        for model_name in args.models:
            print(f"\n[{model_name}]")
            embeddings_path = extract_features(
                model_name, args.data_dir, output_dir, 
                args.num_per_class, args.batch_size
            )
            if embeddings_path is None:
                print(f"  WARNING: Failed to extract features for {model_name}")
    
    # Step 2: Run conformal prediction experiments
    print("\n" + "="*70)
    print("STEP 2: CONFORMAL PREDICTION EXPERIMENTS")
    print("="*70)
    
    all_results = {}
    
    for model_name in tqdm(args.models, desc="Models"):
        embeddings_path = output_dir / f"embeddings_{model_name.replace('-', '_')}.pt"
        
        if not embeddings_path.exists():
            print(f"\n  WARNING: Embeddings not found for {model_name}: {embeddings_path}")
            all_results[model_name] = None
            continue
        
        print(f"\n[{model_name}]")
        print(f"  Loading: {embeddings_path}")
        
        try:
            results = run_conformal_experiment(
                embeddings_path, args.alphas, args.cal_ratio, 
                args.ncm, args.k, args.seed,
                args.max_samples, args.samples_per_class,
                compute_calibration=args.calibration_plot, n_bins=args.n_bins
            )
            all_results[model_name] = results
            
            # Print summary
            print(f"  Results at α=0.1: Coverage={results['coverages'][args.alphas.index(0.1)]:.3f}, "
                  f"Avg Set Size={results['avg_set_sizes'][args.alphas.index(0.1)]:.2f}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results[model_name] = None
    
    # Step 3: Generate comparison plots
    print("\n" + "="*70)
    print("STEP 3: GENERATING COMPARISON PLOTS")
    print("="*70)
    
    plot_comparison(all_results, output_dir, args.ncm)
    
    # Generate calibration plot if requested
    if args.calibration_plot:
        plot_calibration(all_results, output_dir)
    
    # Save results to file
    results_path = output_dir / "comparison_results.txt"
    with open(results_path, 'w') as f:
        f.write("SSL Model Comparison Results\n")
        f.write("="*70 + "\n")
        f.write(f"Dataset: {args.data_dir}\n")
        f.write(f"NCM: {args.ncm} (k={args.k})\n")
        f.write(f"Cal ratio: {args.cal_ratio}\n\n")
        
        f.write("Results at α=0.1 (90% target coverage):\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Model':<20} {'Coverage':<12} {'Avg Set Size':<15}\n")
        f.write("-"*50 + "\n")
        
        alpha_idx = args.alphas.index(0.1) if 0.1 in args.alphas else 4
        for model_name, results in all_results.items():
            if results is None:
                f.write(f"{model_name:<20} {'N/A':<12} {'N/A':<15}\n")
            else:
                f.write(f"{model_name:<20} {results['coverages'][alpha_idx]:<12.3f} "
                       f"{results['avg_set_sizes'][alpha_idx]:<15.2f}\n")
    
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
