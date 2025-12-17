import argparse
import os
import torch
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Extract SSL Features for Conformal Prediction")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to folder containing images (e.g. data/train)")
    parser.add_argument("--output_name", type=str, default="embeddings.pt", help="Filename for saved tensors")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="vit_base_patch14_dinov2.lvd142m", help="timm model name")
    return parser.parse_args()

def get_transform():
    # Standard ImageNet normalization (works well for DINOv2)
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load SSL Model
    print(f"Loading model: {args.model_name}...")
    # num_classes=0 tells timm to return the raw pooling layer (features), not logits
    model = timm.create_model(args.model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    # 2. Prepare Data
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    dataset = datasets.ImageFolder(root=args.data_dir, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
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