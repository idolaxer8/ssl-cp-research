"""
Download datasets and save as image files in ImageFolder structure.

Supported datasets:
  cifar10       -- 10 classes, 32x32
  cifar100      -- 100 classes, 32x32
  stl10         -- 10 classes, 96x96
  eurosat       -- 10 classes, 64x64
  flowers102    -- 102 classes, variable size (~80 images/class after merging all splits)
  cub200        -- 200 classes, variable size (~59 images/class, train+test merged)
  aircraft      -- 100 classes (FGVC-Aircraft variants), ~1-2MP. Fine-grained CLIP/CoOp-suite
                   benchmark. 'train'->trainval (~67/class, labeled), 'test'->test (~33/class,
                   disjoint unlabeled pool). Aircraft variants are NOT ImageNet labels -> no
                   DINOv2 pretraining-label overlap (unlike CUB birds / miniImageNet).
  miniimagenet  -- 100 classes, 84x84 (600/class, all splits merged). Requires: pip install learn2learn
  tiny_imagenet -- 200 classes, 64x64 (500/class train only). Living-17 substitute. No extra deps.

NOTE — Living17 (BREEDS benchmark):
  Living17 is a subset of ImageNet and cannot be downloaded without the full ImageNet dataset
  (~150 GB, registration required at image-net.org). Use tiny_imagenet (200 ImageNet classes,
  freely downloadable) or miniimagenet as substitutes for class-similarity experiments.

Usage:
    python src/download_datasets.py --dataset cifar10        --output_dir data/cifar10        --num_per_class 200
    python src/download_datasets.py --dataset cifar100       --output_dir data/cifar100       --num_per_class 100
    python src/download_datasets.py --dataset stl10          --output_dir data/stl10          --num_per_class 200
    python src/download_datasets.py --dataset eurosat        --output_dir data/eurosat        --num_per_class 200
    python src/download_datasets.py --dataset flowers102     --output_dir data/flowers102     --num_per_class 60
    python src/download_datasets.py --dataset cub200         --output_dir data/cub200         --num_per_class 50
    python src/download_datasets.py --dataset aircraft       --output_dir data/aircraft           --split train   # labeled (trainval)
    python src/download_datasets.py --dataset aircraft       --output_dir data/aircraft_unlabeled --split test    # unlabeled pool
    python src/download_datasets.py --dataset miniimagenet   --output_dir data/miniimagenet   --num_per_class 500
    python src/download_datasets.py --dataset tiny_imagenet  --output_dir data/tiny_imagenet  --num_per_class 500
"""

import argparse
import os
import tarfile
from pathlib import Path
from PIL import Image
from torchvision import datasets
from torchvision.datasets.utils import download_url
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
    "flowers102": {
        "num_classes": 102,
        "size": "variable (~500px)",
        "note": "All splits merged (train+val+test): ~80 images/class total",
    },
    "cub200": {
        "num_classes": 200,
        "size": "variable (~400-600px)",
        "note": "Train+test merged: ~59 images/class total",
    },
    "aircraft": {
        "num_classes": 100,
        "size": "variable (~1-2MP)",
        "note": "FGVC-Aircraft variants (annotation_level='variant'). "
                "'train'->trainval (~67/class, labeled pool), "
                "'test'->test (~33/class, disjoint unlabeled pool).",
    },
    "miniimagenet": {
        "num_classes": 100,
        "size": "84x84",
        "note": "All splits merged (train+val+test): 600/class. Requires: pip install learn2learn",
    },
    "tiny_imagenet": {
        "num_classes": 200,
        "size": "64x64",
        "note": "Train split only: 500/class. Living17 substitute. No extra deps.",
    },
}

# URL for CUB-200-2011 (Caltech official release)
CUB200_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB200_FILENAME = "CUB_200_2011.tgz"

# URL for Tiny ImageNet (Stanford CS231N)
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_FILENAME = "tiny-imagenet-200.zip"


def download_miniimagenet(args, output_dir):
    """Download mini-ImageNet via HuggingFace (timm/mini-imagenet).

    Avoids the abandoned learn2learn package (which fails to build on
    Python 3.11+ due to a Cython-generated longintrepr.h dependency).

    Source: https://huggingface.co/datasets/timm/mini-imagenet
        100 classes, 50K train + 10K val + 5K test = 65K original-resolution
        images. Crucially every split uses the SAME 100 classes (unlike
        learn2learn's disjoint-class splits), so train can be the labeled
        pool and val a naturally disjoint unlabeled pool.

    Supports --split train|val|test|both|all (default: train).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for mini-ImageNet.\n"
            "Install with:  pip install datasets"
        )

    # Map our CLI vocabulary onto HF split names.
    # Note: the parser only exposes train|test|both (see parse_args). We treat
    # HF 'validation' as the disjoint pool a caller asks for with --split test.
    split_map = {
        "train": ["train"],
        "test":  ["validation"],
        "both":  ["train", "validation"],
    }
    split = getattr(args, "split", "train") or "train"
    hf_splits = split_map.get(split, split_map["train"])

    print(f"\nDownloading mini-ImageNet via HuggingFace timm/mini-imagenet")
    print(f"  HF splits to materialize: {hf_splits}")

    ds_dict = load_dataset("timm/mini-imagenet")
    # Class names come from the label feature; stable order = label_id
    label_feature = ds_dict[hf_splits[0]].features["label"]
    class_names = list(label_feature.names)
    print(f"  {len(class_names)} classes")

    # Build merged (PIL, int label) list across requested HF splits
    merged = []
    for hs in hf_splits:
        if hs not in ds_dict:
            print(f"  WARNING: split {hs} not present in dataset — skipping")
            continue
        n = len(ds_dict[hs])
        print(f"  Loading {hs}: {n} images")
        for row in ds_dict[hs]:
            merged.append((row["image"], int(row["label"])))

    print(f"  Total images materialized: {len(merged)}")
    save_dataset_as_images(merged, output_dir, class_names, args.num_per_class)


def download_tiny_imagenet(args, output_dir):
    """Download Tiny ImageNet-200 from Stanford and save in ImageFolder format.

    Source: http://cs231n.stanford.edu/tiny-imagenet-200.zip
    200 classes × 500 images (64×64). Train split only.
    Class names are derived from the included words.txt (WordNet descriptions).
    No extra dependencies required.
    """
    import zipfile
    import urllib.request

    download_dir = Path(args.download_dir)
    zip_path = download_dir / TINY_IMAGENET_FILENAME
    extract_dir = download_dir / "tiny-imagenet-200"

    # --- Download ---
    if not zip_path.exists():
        print(f"\nDownloading Tiny ImageNet-200 (~240 MB) from Stanford CS231N...")
        print(f"  URL: {TINY_IMAGENET_URL}")

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100.0, downloaded * 100.0 / total_size)
                print(f"\r  Progress: {pct:5.1f}%  ({downloaded//1_000_000} / {total_size//1_000_000} MB)",
                      end="", flush=True)

        urllib.request.urlretrieve(TINY_IMAGENET_URL, str(zip_path), _progress)
        print()
    else:
        print(f"\nFound existing archive: {zip_path}")

    # --- Extract ---
    if not extract_dir.exists():
        print(f"Extracting {TINY_IMAGENET_FILENAME} ...")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(download_dir))
        print("Extraction complete.")
    else:
        print(f"Found existing extracted data: {extract_dir}")

    # --- Build synset → readable name map from words.txt ---
    words_file = extract_dir / "words.txt"
    synset_to_name = {}
    if words_file.exists():
        with open(words_file) as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    synset_id, description = parts
                    # Use first synonym, sanitize for filesystem
                    first_word = description.split(",")[0].strip()
                    synset_to_name[synset_id] = _sanitize_name(first_word)

    # --- Load train split (already in ImageFolder structure) ---
    train_dir = extract_dir / "train"
    print(f"Loading train images from {train_dir} ...")
    ds = datasets.ImageFolder(root=str(train_dir))

    # Map numeric labels back to synset IDs, then to readable names
    synset_ids = ds.classes  # e.g. ['n01443537', 'n01629819', ...]
    class_names = []
    for sid in synset_ids:
        name = synset_to_name.get(sid, sid)  # fallback to synset ID
        # Append synset suffix to guarantee uniqueness when names collide
        class_names.append(f"{name}_{sid}")

    print(f"  {len(class_names)} classes, {len(ds)} images total")
    save_dataset_as_images(ds, output_dir, class_names, args.num_per_class)


def parse_args():
    parser = argparse.ArgumentParser(description="Download dataset and save as images")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "cifar100", "stl10", "eurosat", "flowers102",
                                "cub200", "aircraft", "miniimagenet", "tiny_imagenet"],
                       help="Dataset to download")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: data/<dataset>)")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test", "both"],
                       help="Which split to download. cifar10/cifar100/stl10: train|test|both. "
                            "miniimagenet/aircraft: 'train'->labeled pool, 'test'->disjoint "
                            "unlabeled pool. Ignored for eurosat/flowers102/cub200.")
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
        class_names: List of class names indexed by integer label
        num_per_class: Maximum images per class (None = all)
    """
    n_classes = len(class_names)
    output_path = Path(output_dir)

    for class_name in class_names:
        (output_path / class_name).mkdir(parents=True, exist_ok=True)

    class_counts = {i: 0 for i in range(n_classes)}

    print(f"Saving images to {output_dir}...")

    for idx, (image, label) in enumerate(tqdm(dataset, desc="Processing images")):
        if num_per_class is not None and class_counts[label] >= num_per_class:
            if all(count >= num_per_class for count in class_counts.values()):
                break
            continue

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            import torch
            if isinstance(image, torch.Tensor):
                # learn2learn MiniImagenet: float Tensor (C,H,W) → uint8 (H,W,C)
                image = Image.fromarray(image.permute(1, 2, 0).to(torch.uint8).numpy())
            else:
                # STL-10 and others: numpy array (H,W,C)
                image = Image.fromarray(image)

        class_name = class_names[label]
        filename = f"{class_name}_{class_counts[label]:05d}.png"
        filepath = output_path / class_name / filename

        # PNG supports RGB / RGBA / grayscale but not CMYK or other modes;
        # mini-ImageNet has a small fraction of CMYK JPEGs which would crash here.
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P", "I", "1"):
            image = image.convert("RGB")
        image.save(filepath)
        class_counts[label] += 1

    print("\nDataset saved successfully!")
    print(f"Total images: {sum(class_counts.values())}")
    print("\nImages per class:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:40s}: {class_counts[i]:5d}")

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


def _sanitize_name(name: str) -> str:
    """Replace characters illegal on Windows filesystems with underscores."""
    import re
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip()


def download_flowers102(args, output_dir):
    """Download Oxford Flowers-102 and merge all splits.

    Split sizes: train=10/class, val=10/class, test=~60/class → ~80/class total.
    All three splits are merged into a single output directory so that
    --num_per_class 60 gives a usable per-class count.
    The --split argument is ignored for this dataset.
    """
    print("\nDownloading Oxford Flowers-102 (all splits: train + val + test)...")

    class_names = None
    merged = []  # list of (PIL Image, int label) across all splits

    for split in ["train", "val", "test"]:
        print(f"  Loading split: {split}")
        ds = datasets.Flowers102(root=args.download_dir, split=split, download=True, transform=None)
        if class_names is None:
            # torchvision >=0.12 exposes ds.classes; fall back to generic names
            if hasattr(ds, 'classes') and ds.classes:
                class_names = [_sanitize_name(c) for c in ds.classes]
            else:
                class_names = [f"class_{i:03d}" for i in range(102)]
        merged.extend(ds)

    print(f"  Total images across all splits: {len(merged)}")
    save_dataset_as_images(merged, output_dir, class_names, args.num_per_class)


def download_cub200(args, output_dir):
    """Download CUB-200-2011 from Caltech and save in ImageFolder format.

    Downloads the official tgz (~1.1 GB), extracts it, and loads the
    images/ subfolder as an ImageFolder dataset (already organised by class).
    Train and test splits are merged so --num_per_class 50 gives ~50/class.
    The --split argument is ignored for this dataset.
    """
    download_dir = Path(args.download_dir)
    tgz_path = download_dir / CUB200_FILENAME
    extract_dir = download_dir / "CUB_200_2011"
    images_dir = download_dir / "CUB_200_2011" / "images"

    # --- Download ---
    if not tgz_path.exists():
        print(f"\nDownloading CUB-200-2011 (~1.1 GB) from Caltech...")
        download_url(CUB200_URL, root=str(download_dir), filename=CUB200_FILENAME)
    else:
        print(f"\nFound existing archive: {tgz_path}")

    # --- Extract ---
    if not images_dir.exists():
        print(f"Extracting {CUB200_FILENAME}...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=str(download_dir))
        print("Extraction complete.")
    else:
        print(f"Found existing extracted data: {images_dir}")

    # --- Load as ImageFolder (already in class-subfolder structure) ---
    print("Loading images via ImageFolder...")
    ds = datasets.ImageFolder(root=str(images_dir))
    class_names = ds.classes  # e.g. ['001.Black_footed_Albatross', ...]
    print(f"  {len(class_names)} classes, {len(ds)} images total")

    save_dataset_as_images(ds, output_dir, class_names, args.num_per_class)


def download_aircraft(args, output_dir):
    """Download FGVC-Aircraft (100 variant classes) and save in ImageFolder format.

    Source: torchvision FGVCAircraft, which auto-downloads the official VGG
    release (~2.6 GB) on first call. We use annotation_level='variant' -> the
    standard fine-grained 100-class benchmark that appears in the CLIP/CoOp
    transfer suite (and Conf-OT, Silva-Rodriguez CVPR'25). All splits share the
    same 100-class label space, so (mirroring mini-ImageNet's labeled/unlabeled
    convention):
        --split train  -> torchvision 'trainval' (~6667, ~67/class)  = labeled pool
        --split test   -> torchvision 'test'     (~3333, ~33/class)  = disjoint unlabeled pool
        --split both   -> both, merged into one dir

    Notes:
      * FGVC-Aircraft images carry a 20px copyright banner along the bottom edge;
        we leave it in (DINOv2 @518 is robust to it and the ablation is relative).
      * Some variant names contain '/' (e.g. 'F/A-18') -> sanitised for the
        filesystem via _sanitize_name.
    """
    split_map = {"train": ["trainval"], "test": ["test"], "both": ["trainval", "test"]}
    split = getattr(args, "split", "train") or "train"
    tv_splits = split_map.get(split, split_map["train"])
    print(f"\nDownloading FGVC-Aircraft (annotation_level=variant) via torchvision")
    print(f"  torchvision splits to materialize: {tv_splits}")

    if len(tv_splits) == 1:
        # Single split -> stream straight to disk (memory-friendly).
        ds = datasets.FGVCAircraft(root=args.download_dir, split=tv_splits[0],
                                   annotation_level="variant", download=True, transform=None)
        class_names = [_sanitize_name(c) for c in ds.classes]
        print(f"  {len(class_names)} classes, {len(ds)} images ({tv_splits[0]})")
        save_dataset_as_images(ds, output_dir, class_names, args.num_per_class)
        return

    # 'both' -> merge the two splits into a single (PIL, label) list.
    class_names, merged = None, []
    for tv in tv_splits:
        ds = datasets.FGVCAircraft(root=args.download_dir, split=tv,
                                   annotation_level="variant", download=True, transform=None)
        if class_names is None:
            class_names = [_sanitize_name(c) for c in ds.classes]
        print(f"  Loading split {tv}: {len(ds)} images")
        merged.extend(ds)
    print(f"  Total images materialized: {len(merged)}")
    save_dataset_as_images(merged, output_dir, class_names, args.num_per_class)


def main():
    args = parse_args()

    output_dir = args.output_dir or f"data/{args.dataset}"

    info = DATASET_INFO[args.dataset]
    n_classes = len(info["classes"]) if "classes" in info else info["num_classes"]

    print("=" * 70)
    print(f"DATASET DOWNLOADER: {args.dataset.upper()}")
    print("=" * 70)
    print(f"  Classes: {n_classes}")
    print(f"  Image size: {info['size']}")
    if "note" in info:
        print(f"  Note: {info['note']}")
    print(f"  Output: {output_dir}")
    if args.num_per_class:
        print(f"  Limit: {args.num_per_class} per class ({args.num_per_class * n_classes} total)")

    Path(args.download_dir).mkdir(exist_ok=True)

    if args.dataset == "cifar10":
        download_cifar10(args, output_dir)
    elif args.dataset == "cifar100":
        download_cifar100(args, output_dir)
    elif args.dataset == "stl10":
        download_stl10(args, output_dir)
    elif args.dataset == "eurosat":
        download_eurosat(args, output_dir)
    elif args.dataset == "flowers102":
        download_flowers102(args, output_dir)
    elif args.dataset == "cub200":
        download_cub200(args, output_dir)
    elif args.dataset == "aircraft":
        download_aircraft(args, output_dir)
    elif args.dataset == "miniimagenet":
        download_miniimagenet(args, output_dir)
    elif args.dataset == "tiny_imagenet":
        download_tiny_imagenet(args, output_dir)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"\nImages saved to: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Extract features:")
    print(f"     python src/extract_features.py --data_dir {output_dir} --model dinov2-base --output_name embeddings_{args.dataset}.pt")
    print(f"  2. Run conformal prediction:")
    print(f"     python src/run_conformal_experiment.py --embeddings_path output/embeddings_{args.dataset}.pt")


if __name__ == "__main__":
    main()
