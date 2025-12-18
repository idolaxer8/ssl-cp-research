"""
Download CIFAR-10 dataset and save as image files in ImageFolder structure.

Usage:
    python src/download_cifar10.py --output_dir data --split train --num_per_class 1000
"""

import argparse
import os
from pathlib import Path
from PIL import Image
import torchvision
from torchvision import datasets
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Download CIFAR-10 and save as images")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="Output directory for images")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test", "both"],
                       help="Which split to download")
    parser.add_argument("--num_per_class", type=int, default=None,
                       help="Limit number of images per class (None = all)")
    parser.add_argument("--download_dir", type=str, default="./cifar10_download",
                       help="Temporary download directory")
    return parser.parse_args()


def save_cifar_as_images(dataset, output_dir, num_per_class=None):
    """
    Save CIFAR-10 dataset as individual image files in ImageFolder structure.
    
    Args:
        dataset: CIFAR10 dataset object
        output_dir: Root directory to save images
        num_per_class: Maximum images per class (None = all)
    """
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Create directories for each class
    output_path = Path(output_dir)
    for class_name in class_names:
        (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    # Track count per class
    class_counts = {i: 0 for i in range(10)}
    
    print(f"Saving images to {output_dir}...")
    
    # Save images
    for idx, (image, label) in enumerate(tqdm(dataset, desc="Processing images")):
        # Check if we've reached the limit for this class
        if num_per_class is not None and class_counts[label] >= num_per_class:
            # Check if all classes have reached the limit
            if all(count >= num_per_class for count in class_counts.values()):
                break
            continue
        
        class_name = class_names[label]
        filename = f"{class_name}_{class_counts[label]:05d}.png"
        filepath = output_path / class_name / filename
        
        # Save image
        image.save(filepath)
        class_counts[label] += 1
    
    # Print summary
    print("\nDataset saved successfully!")
    print(f"Total images: {sum(class_counts.values())}")
    print("\nImages per class:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:12s}: {class_counts[i]:5d}")
    
    return class_counts


def main():
    args = parse_args()
    
    print("="*70)
    print("CIFAR-10 DATASET DOWNLOADER")
    print("="*70)
    
    # Create download directory
    Path(args.download_dir).mkdir(exist_ok=True)
    
    if args.split in ["train", "both"]:
        print("\n[1] Downloading CIFAR-10 training set...")
        train_dataset = datasets.CIFAR10(
            root=args.download_dir,
            train=True,
            download=True,
            transform=None  # No transform, we want raw PIL images
        )
        
        output_dir = args.output_dir if args.split == "train" else f"{args.output_dir}/train"
        save_cifar_as_images(train_dataset, output_dir, args.num_per_class)
    
    if args.split in ["test", "both"]:
        print("\n[2] Downloading CIFAR-10 test set...")
        test_dataset = datasets.CIFAR10(
            root=args.download_dir,
            train=False,
            download=True,
            transform=None
        )
        
        output_dir = args.output_dir if args.split == "test" else f"{args.output_dir}/test"
        save_cifar_as_images(test_dataset, output_dir, args.num_per_class)
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\nImages saved to: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Extract features:")
    print(f"     python src/extract_features.py --data_dir {args.output_dir} --batch_size 64")
    print(f"  2. Run conformal prediction:")
    print(f"     python src/run_conformal_experiment.py --embeddings_path output/embeddings.pt")


if __name__ == "__main__":
    main()
