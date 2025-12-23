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
    KNNNonconformity,
    SimplifiedKNNNonconformity,
    cal_test_split,
    train_val_test_split  # Kept for backward compatibility
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Conformal Prediction on SSL embeddings")
    
    # Data
    parser.add_argument("--embeddings_path", type=str, default="output/embeddings.pt",
                       help="Path to embeddings .pt file")
    
    # Split ratios (for Full CP, no training set needed)
    parser.add_argument("--cal_ratio", type=float, default=0.85,
                       help="Fraction of data for calibration (rest goes to test)")
    
    # Conformal prediction
    parser.add_argument("--alpha", type=float, default=0.1,
                       help="Significance level (target miscoverage rate)")
    parser.add_argument("--ncm", type=str, default="knn",
                       choices=["knn", "simplified_knn"],
                       help="Nonconformity measure (simplified_knn is faster)")
    parser.add_argument("--k", type=int, default=5,
                       help="Number of neighbors for k-NN nonconformity")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save results")
    parser.add_argument("--save_predictions", action="store_true",
                       help="Save prediction sets to file")
    parser.add_argument("--compare_methods", action="store_true",
                       help="Compare runtime between k-NN and simplified k-NN")
    parser.add_argument("--compare_with_baseline", action="store_true",
                       help="Compare SSL embeddings vs raw pixel features (baseline)")
    
    # Data paths
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Directory containing raw images (for baseline comparison)")
    
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
    alphas = np.linspace(0.05, 0.2, 3)
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


def load_raw_pixel_features(data_dir: str, labels: np.ndarray) -> np.ndarray:
    """
    Load raw pixel features from image folder.
    Assumes images organized as: data/class_name/*.png
    Returns flattened, normalized pixel features matching the label order.
    """
    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR-10 size
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root=data_dir, transform=transform)
    
    # Collect features in same order as embeddings
    features_list = []
    for idx in range(len(dataset)):
        img, _ = dataset[idx]
        # Flatten: (3, 32, 32) -> (3072,)
        features_list.append(img.numpy().flatten())
    
    features = np.array(features_list, dtype=np.float32)
    
    # Normalize to unit norm for fair comparison
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    
    return features


def compare_ssl_vs_baseline(X_ssl, X_raw, y, args, output_dir: str):
    """
    Compare CP performance on SSL embeddings vs raw pixel features.
    """
    print("\n" + "="*70)
    print("SSL EMBEDDINGS vs RAW PIXEL BASELINE COMPARISON")
    print("="*70)
    
    # Use consistent split for both
    X_cal_ssl, y_cal, X_test_ssl, y_test = cal_test_split(
        X_ssl, y, cal_ratio=args.cal_ratio, random_state=args.seed
    )
    X_cal_raw, _, X_test_raw, _ = cal_test_split(
        X_raw, y, cal_ratio=args.cal_ratio, random_state=args.seed
    )
    
    comparison_results = {}
    
    for name, X_cal, X_test in [
        ("Raw Pixels (Baseline)", X_cal_raw, X_test_raw),
        ("SSL Embeddings (DINOv2)", X_cal_ssl, X_test_ssl),
    ]:
        print(f"\nEvaluating {name}...")
        print(f"  Calibration set: {len(X_cal)}, Test set: {len(X_test)}")
        print(f"  Feature dimension: {X_cal.shape[1]}")
        
        # Create NCM and predictor
        if args.ncm == "simplified_knn":
            ncm = SimplifiedKNNNonconformity(k=args.k)
        else:
            ncm = KNNNonconformity(k=args.k, metric='euclidean')
        
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
    
    baseline = comparison_results["Raw Pixels (Baseline)"]
    ssl = comparison_results["SSL Embeddings (DINOv2)"]
    
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
    comparison_path = Path(output_dir) / "ssl_vs_baseline_comparison.pt"
    torch.save(comparison_results, comparison_path)
    print(f"\nSaved comparison to {comparison_path}")
    
    return comparison_results


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
    print(f"\n[2/5] Splitting data (cal={args.cal_ratio}, test={1-args.cal_ratio})...")
    X_cal, y_cal, X_test, y_test = cal_test_split(
        embeddings, labels,
        cal_ratio=args.cal_ratio,
        random_state=args.seed
    )
    
    # 3. Create nonconformity measure
    print(f"\n[3/5] Creating nonconformity measure: {args.ncm} (k={args.k})...")
    if args.ncm == "knn":
        ncm = KNNNonconformity(k=args.k, metric='euclidean')
    elif args.ncm == "simplified_knn":
        ncm = SimplifiedKNNNonconformity(k=args.k)
    else:
        raise ValueError(f"Unknown NCM: {args.ncm}")
    
    # 3b. Optional: Compare both methods
    if args.compare_methods:
        print(f"\n[3b] Runtime Comparison: k-NN vs Simplified k-NN...")
        print("="*70)
        
        methods = {
            'k-NN': KNNNonconformity(k=args.k, metric='euclidean'),
            'Simplified k-NN': SimplifiedKNNNonconformity(k=args.k)
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
    
    # 3c. Optional: Compare SSL vs Baseline (raw pixels)
    if args.compare_with_baseline:
        print(f"\n[3c] Loading raw pixel features for baseline comparison...")
        try:
            raw_features = load_raw_pixel_features(args.data_dir, labels)
            print(f"  Raw features shape: {raw_features.shape}")
            
            # Run comparison
            compare_ssl_vs_baseline(embeddings, raw_features, labels, args, args.output_dir)
        except Exception as e:
            print(f"  ⚠️  Could not load raw features for baseline: {e}")
            print(f"     Skipping baseline comparison. Make sure --data_dir points to images folder.")
    
    # 4. Run Full Conformal Prediction
    print(f"\n[4/5] Running Full Conformal Prediction (α={args.alpha})...")
    cp = FullConformalPredictor(ncm, alpha=args.alpha)
    cp.calibrate(X_cal, y_cal)
    
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
    results_dict = {
        'metrics': metrics,
        'per_class_coverage': per_class_cov,
        'per_class_avg_size': per_class_size,
        'args': vars(args)
    }
    
    results_path = Path(args.output_dir) / "cp_results.pt"
    torch.save(results_dict, results_path)
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
