import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path

from piq import brisque
import pyiqa


def load_image(image_path, device='cpu'):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def get_all_image_files(image_dir):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f'*{ext}'))
        image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    
    return sorted([str(f) for f in image_files])


def evaluate_image_metrics(image_dir, device='cpu'):
    image_files = get_all_image_files(image_dir)
    
    print(f"Found {len(image_files)} images in {image_dir}")
    print("Computing metrics: BRISQUE, NIQE")
    
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    
    brisque_scores = []
    niqe_scores = []

    for img_path in tqdm(image_files, desc="Evaluating images"):
        img_tensor = load_image(img_path, device=device)
        brisque_score = brisque(img_tensor, data_range=1.0, reduction='none')
        brisque_scores.append(brisque_score.item())
        
        niqe_score = niqe_metric(img_tensor)
        niqe_scores.append(niqe_score.item())
    
    results = {
        'mean_brisque': np.mean(brisque_scores) if brisque_scores else 0.0,
        'std_brisque': np.std(brisque_scores) if brisque_scores else 0.0,
        'num_images_brisque': len(brisque_scores),
        'mean_niqe': np.mean(niqe_scores) if niqe_scores else 0.0,
        'std_niqe': np.std(niqe_scores) if niqe_scores else 0.0,
        'num_images_niqe': len(niqe_scores),
        'total_images': len(image_files)}
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate image quality metrics BRISQUE and NIQE')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to directory containing images to evaluate')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'], help='Device to use for computation')
    
    args = parser.parse_args()
    
    if args.device == 'gpu':
        if not torch.cuda.is_available():
            print("GPU not available, using CPU instead")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    results = evaluate_image_metrics(args.image_dir, device=device)
    
    print(f"\nBRISQUE (lower is better):")
    print(f"  Mean: {results['mean_brisque']:.4f}")
    print(f"  Std:  {results['std_brisque']:.4f}")
    print(f"  Images evaluated: {results['num_images_brisque']}")
    
    print(f"\nNIQE (lower is better):")
    print(f"  Mean: {results['mean_niqe']:.4f}")
    print(f"  Std:  {results['std_niqe']:.4f}")
    print(f"  Images evaluated: {results['num_images_niqe']}")


if __name__ == '__main__':
    main()
