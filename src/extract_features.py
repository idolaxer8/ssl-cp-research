import argparse
import os
import torch
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Model presets: short name -> (timm_model_name, input_size)
MODEL_PRESETS = {
    # Self-Distillation (DINOv2)
    "dinov2-base": ("vit_base_patch14_dinov2.lvd142m", 518),
    "dinov2-large": ("vit_large_patch14_dinov2.lvd142m", 518),
    "dinov2-giant": ("vit_giant_patch14_dinov2.lvd142m", 518),
    # Vision-Language (CLIP)
    "clip-base": ("vit_base_patch16_clip_224.openai", 224),
    "clip-large": ("vit_large_patch14_clip_224.openai", 224),
    # Masked Image Modeling (BEiT, MAE)
    "beit-base": ("beit_base_patch16_224.in22k_ft_in22k_in1k", 224),
    "beitv2-base": ("beitv2_base_patch16_224.in1k_ft_in22k_in1k", 224),
    "mae-base": ("vit_base_patch16_224.mae", 224),
    "mae-large": ("vit_large_patch16_224.mae", 224),
    # Contrastive Learning (Facebook SSL - similar to MoCo/SimCLR)
    "ssl-resnet50": ("resnet50.fb_ssl_yfcc100m_ft_in1k", 224),
    # Semi-Weakly Supervised (Facebook SWSL - contrastive + weak labels)
    "swsl-resnet50": ("resnet50.fb_swsl_ig1b_ft_in1k", 224),
}

def get_args():
    preset_list = ", ".join(MODEL_PRESETS.keys())
    parser = argparse.ArgumentParser(description="Extract SSL Features for Conformal Prediction")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing images (e.g. data/train)")
    parser.add_argument("--output_name", type=str, default="embeddings.pt", help="Filename for saved tensors")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", type=str, default="dinov2-base",
                        help=f"Model preset: {preset_list}, or full timm model name")
    parser.add_argument("--num_per_class", type=int, default=None,
                        help="Limit number of images per class (None = all)")
    return parser.parse_args()

def get_transform(input_size: int):
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Resolve model preset or use as timm name directly
    if args.model in MODEL_PRESETS:
        model_name, input_size = MODEL_PRESETS[args.model]
        print(f"Using preset '{args.model}' -> {model_name} (input: {input_size}x{input_size})")
    else:
        model_name, input_size = args.model, 224  # default input size for unknown models
        print(f"Using custom timm model: {model_name} (input: {input_size}x{input_size})")

    # 1. Load SSL Model
    print(f"Loading model: {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    # 2. Prepare Data
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    dataset = datasets.ImageFolder(root=args.data_dir, transform=get_transform(input_size))
    # Optionally limit number of images per class by creating a Subset
    if args.num_per_class is not None:
        class_counts = {i: 0 for i in range(len(dataset.classes))}
        selected_indices = []
        for idx, (_path, label) in enumerate(dataset.samples):
            if class_counts[label] < args.num_per_class:
                selected_indices.append(idx)
                class_counts[label] += 1
            # stop early if all classes reached the limit
            if all(count >= args.num_per_class for count in class_counts.values()):
                break

        dataset = Subset(dataset, selected_indices)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Found {len(dataset)} images. Starting extraction...")

    # 3. Extraction Loop
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            
            # Forward pass
            features = model(images)
            
            # CRITICAL: L2 Normalize for Cosine Similarity
            # This makes Euclidean distance equivalent to Cosine distance
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            embeddings_list.append(features.cpu())
            labels_list.append(labels)

    # 4. Save to Disk
    embeddings = torch.cat(embeddings_list)
    labels = torch.cat(labels_list)
    
    # Create an 'output' folder if it doesn't exist
    os.makedirs("output", exist_ok=True)
    save_path = os.path.join("output", args.output_name)
    
    print(f"Saving shape {embeddings.shape} to {save_path}...")
    torch.save({"embeddings": embeddings, "labels": labels}, save_path)
    print("Done.")

if __name__ == "__main__":
    main()