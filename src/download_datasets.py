"""
Download datasets (CIFAR-10, CIFAR-100, STL-10, EuroSAT) and save as image files in ImageFolder structure.

Usage:
    python src/download_datasets.py --dataset cifar10 --output_dir data/cifar10 --num_per_class 200
    python src/download_datasets.py --dataset cifar100 --output_dir data/cifar100 --num_per_class 100
    python src/download_datasets.py --dataset stl10 --output_dir data/stl10 --num_per_class 200
    python src/download_datasets.py --dataset eurosat --output_dir data/eurosat --num_per_class 200
"""

import argparse
import os
from pathlib import Path
from PIL import Image
from torchvision import datasets
from tqdm import tqdm


DATASET_INFO = {
    "cifar10": {
        "classes": ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        "size": "32x32",
    },
    "cifar100": {
        "classes": ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
                    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
                    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
                    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
                    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
                    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
                    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'],
        "size": "32x32",
    },
    "stl10": {
        "classes": ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck'],
        "size": "96x96",
    },
    "eurosat": {
        "classes": ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
                    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'],
        "size": "64x64",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset and save as images")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "cifar100", "stl10", "eurosat"],
                       help="Dataset to download")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: data/<dataset>)")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test", "both"],
                       help="Which split to download (cifar10/cifar100/stl10 only)")
    parser.add_argument("--num_per_class", type=int, default=None,
                       help="Limit number of images per class (None = all)")
    parser.add_argument("--download_dir", type=str, default="./dataset_download",
                       help="Temporary download directory")
    return parser.parse_args()


def save_dataset_as_images(dataset, output_dir, class_names, num_per_class=None):
    """
    Save dataset as individual image files in ImageFolder structure.
    
    Args:
        dataset: Dataset object (iterable of (image, label))
        output_dir: Root directory to save images
        class_names: List of class names
        num_per_class: Maximum images per class (None = all)
    """
    n_classes = len(class_names)
    output_path = Path(output_dir)
    
    # Create directories for each class
    for class_name in class_names:
        (output_path / class_name).mkdir(parents=True, exist_ok=True)
    
    class_counts = {i: 0 for i in range(n_classes)}
    
    print(f"Saving images to {output_dir}...")
    
    for idx, (image, label) in enumerate(tqdm(dataset, desc="Processing images")):
        if num_per_class is not None and class_counts[label] >= num_per_class:
            if all(count >= num_per_class for count in class_counts.values()):
                break
            continue
        
        # Convert to PIL if needed (STL-10 returns numpy arrays)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        class_name = class_names[label]
        filename = f"{class_name}_{class_counts[label]:05d}.png"
        filepath = output_path / class_name / filename
        
        image.save(filepath)
        class_counts[label] += 1
    
    print("\nDataset saved successfully!")
    print(f"Total images: {sum(class_counts.values())}")
    print("\nImages per class:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:24s}: {class_counts[i]:5d}")
    
    return class_counts


def download_cifar10(args, output_dir):
    """Download and save CIFAR-10."""
    class_names = DATASET_INFO["cifar10"]["classes"]
    
    if args.split in ["train", "both"]:
        print("\nDownloading CIFAR-10 training set...")
        ds = datasets.CIFAR10(root=args.download_dir, train=True, download=True, transform=None)
        out = output_dir if args.split == "train" else f"{output_dir}/train"
        save_dataset_as_images(ds, out, class_names, args.num_per_class)
    
    if args.split in ["test", "both"]:
        print("\nDownloading CIFAR-10 test set...")
        ds = datasets.CIFAR10(root=args.download_dir, train=False, download=True, transform=None)
        out = output_dir if args.split == "test" else f"{output_dir}/test"
        save_dataset_as_images(ds, out, class_names, args.num_per_class)


def download_cifar100(args, output_dir):
    """Download and save CIFAR-100."""
    class_names = DATASET_INFO["cifar100"]["classes"]
    
    if args.split in ["train", "both"]:
        print("\nDownloading CIFAR-100 training set...")
        ds = datasets.CIFAR100(root=args.download_dir, train=True, download=True, transform=None)
        out = output_dir if args.split == "train" else f"{output_dir}/train"
        save_dataset_as_images(ds, out, class_names, args.num_per_class)
    
    if args.split in ["test", "both"]:
        print("\nDownloading CIFAR-100 test set...")
        ds = datasets.CIFAR100(root=args.download_dir, train=False, download=True, transform=None)
        out = output_dir if args.split == "test" else f"{output_dir}/test"
        save_dataset_as_images(ds, out, class_names, args.num_per_class)


def download_stl10(args, output_dir):
    """Download and save STL-10."""
    class_names = DATASET_INFO["stl10"]["classes"]
    
    if args.split in ["train", "both"]:
        print("\nDownloading STL-10 training set...")
        ds = datasets.STL10(root=args.download_dir, split='train', download=True, transform=None)
        out = output_dir if args.split == "train" else f"{output_dir}/train"
        save_dataset_as_images(ds, out, class_names, args.num_per_class)
    
    if args.split in ["test", "both"]:
        print("\nDownloading STL-10 test set...")
        ds = datasets.STL10(root=args.download_dir, split='test', download=True, transform=None)
        out = output_dir if args.split == "test" else f"{output_dir}/test"
        save_dataset_as_images(ds, out, class_names, args.num_per_class)


def download_eurosat(args, output_dir):
    """Download and save EuroSAT (no train/test split, single dataset)."""
    class_names = DATASET_INFO["eurosat"]["classes"]
    
    print("\nDownloading EuroSAT dataset...")
    ds = datasets.EuroSAT(root=args.download_dir, download=True, transform=None)
    save_dataset_as_images(ds, output_dir, class_names, args.num_per_class)


def main():
    args = parse_args()
    
    # Default output directory
    output_dir = args.output_dir or f"data/{args.dataset}"
    
    info = DATASET_INFO[args.dataset]
    print("="*70)
    print(f"DATASET DOWNLOADER: {args.dataset.upper()}")
    print("="*70)
    print(f"  Classes: {len(info['classes'])}")
    print(f"  Image size: {info['size']}")
    print(f"  Output: {output_dir}")
    if args.num_per_class:
        print(f"  Limit: {args.num_per_class} per class ({args.num_per_class * len(info['classes'])} total)")
    
    Path(args.download_dir).mkdir(exist_ok=True)
    
    if args.dataset == "cifar10":
        download_cifar10(args, output_dir)
    elif args.dataset == "cifar100":
        download_cifar100(args, output_dir)
    elif args.dataset == "stl10":
        download_stl10(args, output_dir)
    elif args.dataset == "eurosat":
        download_eurosat(args, output_dir)
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE!")
    print("="*70)
    print(f"\nImages saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Extract features:")
    print(f"     python src/extract_features.py --data_dir {output_dir} --model dinov2-base --num_per_class 200")
    print(f"  2. Run conformal prediction:")
    print(f"     python src/run_conformal_experiment.py --embeddings_path output/embeddings.pt")


if __name__ == "__main__":
    main()
