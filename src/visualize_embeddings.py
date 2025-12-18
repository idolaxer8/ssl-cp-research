import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="output/embeddings.pt", help="Path to .pt file")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "pca"], help="Visualization method")
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading {args.input_path}...")
    data = torch.load(args.input_path)
    embeddings = data["embeddings"].numpy()
    labels = data["labels"].numpy()
    
    # 2. Reduce Dimensions (768 -> 2)
    print(f"Running {args.method.upper()} on {len(embeddings)} vectors...")
    
    if args.method == "pca":
        reducer = PCA(n_components=2)
        projections = reducer.fit_transform(embeddings)
    else:
        # t-SNE is better for clusters, but slower. 
        # We reduce to 50 dims with PCA first to make t-SNE faster/cleaner
        n_samples = embeddings.shape[0]
        # Use 50 components OR the number of samples if less than 50
        n_comps = min(50, n_samples, embeddings.shape[1])
        
        pca_50 = PCA(n_components=n_comps)
        pca_result = pca_50.fit_transform(embeddings)
        
        # Perplexity must be less than number of samples
        n_samples = embeddings.shape[0]
        perp = min(30, n_samples - 1) if n_samples > 1 else 1
        
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init='pca', learning_rate='auto')
        projections = reducer.fit_transform(pca_result)

    # 3. Plot
    print("Plotting...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(projections[:, 0], projections[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Class Label")
    plt.title(f"SSL Embeddings Visualization ({args.method.upper()})")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    
    # Save
    out_file = f"output/viz_{args.method}.png"
    plt.savefig(out_file)
    print(f"Saved plot to {out_file}")
    plt.show()

if __name__ == "__main__":
    main()