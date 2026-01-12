"""
k-Nearest Neighbors Baseline Classifier (No Conformal Prediction)

Standard k-NN classification on SSL embeddings for comparison with conformal prediction.

Usage:
    python src/knn_baseline.py --embeddings_path output/embeddings.pt --k 5 --cal_ratio 0.5
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time


def parse_args():
    parser = argparse.ArgumentParser(description="k-NN Baseline Classifier on SSL Embeddings")
    
    # Data
    parser.add_argument("--embeddings_path", type=str, default="output/embeddings.pt",
                       help="Path to embeddings .pt file")
    
    # Split
    parser.add_argument("--cal_ratio", type=float, default=0.5,
                       help="Fraction of data for training/calibration (rest goes to test)")
    
    # k-NN parameters
    parser.add_argument("--k", type=int, default=5,
                       help="Number of neighbors for k-NN classification")
    parser.add_argument("--metric", type=str, default="euclidean",
                       choices=["euclidean", "cosine", "manhattan"],
                       help="Distance metric for k-NN")
    parser.add_argument("--weights", type=str, default="uniform",
                       choices=["uniform", "distance"],
                       help="Weight function: uniform or distance-weighted")
    
    # Evaluation
    parser.add_argument("--compare_k", action="store_true",
                       help="Compare accuracy across different k values")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Directory to save results")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def cal_test_split(embeddings, labels, cal_ratio=0.5, random_state=42):
    """Split data into calibration/training and test sets."""
    np.random.seed(random_state)
    n = len(embeddings)
    
    indices = np.random.permutation(n)
    n_cal = int(n * cal_ratio)
    
    cal_idx = indices[:n_cal]
    test_idx = indices[n_cal:]
    
    X_train = embeddings[cal_idx]
    y_train = labels[cal_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]
    
    print(f"Split: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, y_train, X_test, y_test


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True label',
           xlabel='Predicted label',
           title='k-NN Confusion Matrix')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    save_path = Path(output_dir) / "knn_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def compare_k_values(X_train, y_train, X_test, y_test, k_values, metric, weights, output_dir):
    """Compare accuracy across different k values."""
    print("\n" + "="*60)
    print("COMPARING DIFFERENT k VALUES")
    print("="*60)
    
    accuracies = []
    times = []
    
    for k in k_values:
        print(f"\nTesting k={k}...")
        
        # Train k-NN
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
        knn.fit(X_train, y_train)
        
        # Predict
        y_pred = knn.predict(X_test)
        pred_time = time.time() - start_time
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        times.append(pred_time)
        
        print(f"  Accuracy: {acc:.4f}, Time: {pred_time:.2f}s")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs k
    ax1.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('k (number of neighbors)', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('k-NN Accuracy vs k', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(accuracies) - 0.02, max(accuracies) + 0.02])
    
    # Time vs k
    ax2.plot(k_values, times, 's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('k (number of neighbors)', fontsize=12)
    ax2.set_ylabel('Prediction Time (s)', fontsize=12)
    ax2.set_title('k-NN Prediction Time vs k', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(output_dir) / "knn_k_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved k comparison plot to {save_path}")
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    best_k_idx = np.argmax(accuracies)
    print(f"Best k: {k_values[best_k_idx]} (accuracy: {accuracies[best_k_idx]:.4f})")
    print("="*60)


def main():
    args = parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("k-NEAREST NEIGHBORS BASELINE")
    print("="*60)
    
    # Load embeddings
    print(f"\nLoading embeddings from {args.embeddings_path}...")
    data = torch.load(args.embeddings_path, weights_only=False)
    embeddings = data["embeddings"].numpy()
    labels = data["labels"].numpy()
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # Split data
    X_train, y_train, X_test, y_test = cal_test_split(
        embeddings, labels, 
        cal_ratio=args.cal_ratio, 
        random_state=args.seed
    )
    
    # If compare_k is requested, run comparison
    if args.compare_k:
        k_values = [1, 3, 5, 7, 10, 15, 20, 30, 50]
        compare_k_values(X_train, y_train, X_test, y_test, k_values, 
                        args.metric, args.weights, args.output_dir)
        return
    
    # Train k-NN classifier
    print(f"\n{'='*60}")
    print(f"Training k-NN classifier (k={args.k}, metric={args.metric}, weights={args.weights})...")
    print(f"{'='*60}")
    
    start_time = time.time()
    knn = KNeighborsClassifier(
        n_neighbors=args.k,
        metric=args.metric,
        weights=args.weights,
        n_jobs=-1  # Use all CPU cores
    )
    knn.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    print(f"Training time: {train_time:.2f}s")
    
    # Predict on test set
    print("\nMaking predictions...")
    start_time = time.time()
    y_pred = knn.predict(X_test)
    pred_time = time.time() - start_time
    
    print(f"Prediction time: {pred_time:.2f}s ({pred_time/len(X_test)*1000:.2f}ms per sample)")
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct: {int(accuracy * len(y_test))}/{len(y_test)}")
    print("="*60)
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Plot confusion matrix
    class_names = [str(i) for i in range(len(np.unique(labels)))]
    plot_confusion_matrix(y_test, y_pred, class_names, args.output_dir)
    
    # Save results
    results = {
        'k': args.k,
        'metric': args.metric,
        'weights': args.weights,
        'accuracy': float(accuracy),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_time': train_time,
        'pred_time': pred_time,
        'seed': args.seed
    }
    
    results_path = Path(args.output_dir) / "knn_results.txt"
    with open(results_path, 'w') as f:
        f.write("k-NN Baseline Results\n")
        f.write("="*60 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
