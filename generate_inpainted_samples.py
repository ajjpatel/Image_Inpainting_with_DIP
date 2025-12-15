import os
import argparse
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
from data import InpaintingDataset
from model import ContextEncoder
from denoise import denoise_tensor_bm3d, denoise_tensor_nlm
from decompress import deartifact_tensor_bilateral, deartifact_tensor_guided
from deblurring import apply_deblur, create_motion_blur_kernel
import time


def denorm(img_tensor): 
    return img_tensor * 0.5 + 0.5

def save_intermediate_image(img_tensor, output_path):
    """Save a single intermediate image tensor to file."""
    img_np = denorm(img_tensor).cpu().numpy()
    if len(img_np.shape) == 4:  # Batch dimension
        img_np = img_np[0]
    img_np = img_np.transpose(1, 2, 0)
    img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(output_path)

def save_triplet(orig, masked, repainted, out_path, mask_type=None):
    orig_img = denorm(orig).cpu().numpy().transpose(1,2,0)
    masked_img = denorm(masked).cpu().numpy().transpose(1,2,0)
    repainted_img = denorm(repainted).cpu().numpy().transpose(1,2,0)
    triplet = np.concatenate([orig_img, masked_img, repainted_img], axis=1)
    triplet = (triplet * 255).astype(np.uint8)
    
    triplet_pil = Image.fromarray(triplet)
    triplet_pil.save(out_path)

def save_comparison_grid(samples_by_type, output_path):
    if not samples_by_type:
        return
    
    from PIL import Image
    
    # Get dimensions from the first sample
    sample_height = list(samples_by_type.values())[0].shape[0]
    sample_width = list(samples_by_type.values())[0].shape[1]
    
    mask_types = list(samples_by_type.keys())
    grid_height = len(mask_types) * sample_height
    grid_width = sample_width
    
    grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    
    for i, (_, triplet_array) in enumerate(samples_by_type.items()):
        triplet_pil = Image.fromarray(triplet_array)
        y_offset = i * sample_height
        grid_img.paste(triplet_pil, (0, y_offset))
    
    grid_img.save(output_path)

def combine_grids(grid_paths, output_path):
    from PIL import Image
    
    # Load all grid images
    grid_images = [Image.open(path) for path in grid_paths]
    
    # Get dimensions from the first grid
    grid_width = grid_images[0].width
    grid_height = grid_images[0].height
    
    # Create a new image that can hold all grids side by side
    combined_width = grid_width * len(grid_images)
    combined_height = grid_height
    
    combined_img = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
    
    # Paste each grid image
    for i, grid_img in enumerate(grid_images):
        x_offset = i * grid_width
        combined_img.paste(grid_img, (x_offset, 0))
    
    combined_img.save(output_path)

def main():
    parser = argparse.ArgumentParser(description="Generate inpainted samples using ContextEncoder with various mask types")
    parser.add_argument('--data_root', type=str, default="data/final_custom_dataset/", help='Path to dataset root')
    parser.add_argument('--model_path', type=str, default="G_best.pth", help='Path to trained ContextEncoder .pth file')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples to combine in the grid')
    parser.add_argument('--image_size', type=int, default=128, help='Image size (should match training)')
    parser.add_argument('--mask_size', type=int, default=64, help='Mask size (should match training)')
    parser.add_argument('--mask_type', type=str, default='mixed', 
                       choices=['square', 'circle', 'triangle', 'ellipse', 'irregular', 'random_patches', 'mixed', 'all'],
                       help='Type of mask to use for generation. Use "all" to generate samples with all mask types')
    parser.add_argument('--output_dir', type=str, default='gen_outputs', help='Directory to save output images')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--comparison_grid', action='store_true', help='Generate comparison grid showing all mask types for same images')
    parser.add_argument('--save_individual', action='store_true', default=True, help='Save individual triplet images')
    parser.add_argument('--process_all', action='store_true', help='Process all images in the dataset instead of random sampling')
    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create folders for original and generated images
    original_dir = os.path.join(args.output_dir, 'original')
    generated_dir = os.path.join(args.output_dir, 'generated')
    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    
    # Create folders for intermediate outputs
    masked_dir = os.path.join(args.output_dir, 'masked_img')
    deartifact_dir = os.path.join(args.output_dir, 'deartifact_img')
    denoised_dir = os.path.join(args.output_dir, 'denoised_img')
    repainted_dir = os.path.join(args.output_dir, 'repainted_img')
    deblurred_dir = os.path.join(args.output_dir, 'deblurred_repainted_img')
    os.makedirs(masked_dir, exist_ok=True)
    os.makedirs(deartifact_dir, exist_ok=True)
    os.makedirs(denoised_dir, exist_ok=True)
    os.makedirs(repainted_dir, exist_ok=True)
    os.makedirs(deblurred_dir, exist_ok=True)
    
    # Lists to store images for saving at the end
    original_images = []
    generated_images = []
    
    # Lists to store intermediate images for batch saving
    masked_images = []
    deartifact_images = []
    denoised_images = []
    repainted_images = []
    deblurred_images = []

    model = ContextEncoder().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    if args.mask_type == 'all':
        mask_types = ['square', 'circle', 'triangle', 'ellipse', 'irregular', 'random_patches']
    else:
        mask_types = [args.mask_type]

    print(f"Generating samples with mask types: {mask_types}")

    num_grids_generated = 0  # Track number of grids for summary
    
    if args.comparison_grid and len(mask_types) > 1:
        print("Generating comparison grids...")
        
        # Initialize dataset to get its length
        temp_dataset = InpaintingDataset(
            root=args.data_root, 
            dataset='celeba', 
            image_size=args.image_size, 
            mask_size=args.mask_size,
            mask_type=mask_types[0]
        )
        dataset_size = len(temp_dataset)
        
        if args.process_all:
            num_samples = dataset_size
            print(f"Processing all {num_samples} images in the dataset...")
            # Use sequential indices for all images
            selected_indices = list(range(dataset_size))
        else:
            num_samples = args.num_samples
            # Select unique random indices upfront to ensure randomness
            num_samples_to_select = min(num_samples, dataset_size)
            selected_indices = np.random.choice(dataset_size, size=num_samples_to_select, replace=False).tolist()
            print(f"Processing {num_samples} random samples (selected indices: {selected_indices})...")
        
        # Store all triplets: [sample_idx][mask_type] = triplet
        all_triplets = {}
        
        # Generate samples
        for sample_idx in tqdm(range(num_samples), desc="Generating comparison grids"):
            # Generate a random seed between 0 and 2^32 - 1
            random_seed = np.random.randint(0, 2**32 - 1)
            np.random.seed(random_seed)
            
            # Use the pre-selected random index
            base_img_idx = selected_indices[sample_idx]
            
            all_triplets[sample_idx] = {}
            
            for mask_type in mask_types:    
                dataset = InpaintingDataset(
                    root=args.data_root, 
                    dataset='celeba', 
                    image_size=args.image_size, 
                    mask_size=args.mask_size,
                    mask_type=mask_type
                )
                
                masked_img, orig_img, mask = dataset[base_img_idx % len(dataset)]
                orig_img = masked_img
                masked_img_tensor = masked_img.unsqueeze(0).to(device)
                mask = mask.to(device)
                
                # Generate image name for saving intermediates
                if args.process_all:
                    img_name = f'image_{base_img_idx:05d}_{mask_type}'
                else:
                    img_name = f'sample_{sample_idx:03d}_{mask_type}'
                
                # Store masked image for batch saving
                masked_images.append((masked_img_tensor, img_name))
                
                # Apply deartifact (guided filtering)
                deartifact_img = deartifact_tensor_guided(masked_img_tensor)
                deartifact_images.append((deartifact_img, img_name))
                
                # Apply denoising (BM3D)
                denoised_img = denoise_tensor_bm3d(deartifact_img)
                denoised_images.append((denoised_img, img_name))
                
                with torch.no_grad():
                    pred = model(denoised_img)
                
                # Merge using denoised masked image (treating it as the new original)
                repainted = denoised_img.squeeze(0) * (1 - mask) + pred.squeeze(0) * mask
                repainted_images.append((repainted.unsqueeze(0), img_name))
                
                # Apply deblurring
                blur_kernel = create_motion_blur_kernel()
                deblurred_repainted = apply_deblur(repainted, kernel=blur_kernel, return_format='same')
                deblurred_images.append((deblurred_repainted.unsqueeze(0), img_name))
                
                # Use deblurred version for triplet
                masked_img = denoised_img
                repainted = deblurred_repainted
                
                orig_img_np = denorm(orig_img).cpu().numpy().transpose(1,2,0)
                masked_img_np = denorm(masked_img.squeeze(0)).cpu().numpy().transpose(1,2,0)
                repainted_img_np = denorm(repainted).cpu().numpy().transpose(1,2,0)
                triplet = np.concatenate([orig_img_np, masked_img_np, repainted_img_np], axis=1)
                triplet = (triplet * 255).astype(np.uint8)
                
                all_triplets[sample_idx][mask_type] = triplet
        
        # Create a single comprehensive comparison grid
        # Rows: mask types, Columns: samples, Each cell: triplet (original | masked | repainted)
        if all_triplets:
            # Get dimensions from first triplet
            first_triplet = list(all_triplets.values())[0][mask_types[0]]
            triplet_height = first_triplet.shape[0]
            triplet_width = first_triplet.shape[1]
            
            # Create grid: rows = mask types, columns = samples
            grid_height = len(mask_types) * triplet_height
            grid_width = num_samples * triplet_width
            
            grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
            
            # Fill grid: for each mask type (row) and each sample (column)
            for mask_idx, mask_type in enumerate(mask_types):
                for sample_idx in range(num_samples):
                    if sample_idx in all_triplets and mask_type in all_triplets[sample_idx]:
                        triplet_array = all_triplets[sample_idx][mask_type]
                        triplet_pil = Image.fromarray(triplet_array)
                        
                        x_offset = sample_idx * triplet_width
                        y_offset = mask_idx * triplet_height
                        grid_img.paste(triplet_pil, (x_offset, y_offset))
            
            # Save the comprehensive comparison grid
            combined_path = os.path.join(args.output_dir, 'combined_comparison_grid.png')
            grid_img.save(combined_path)
            num_grids_generated = 1
            print(f"Saved comprehensive comparison grid to {combined_path}")
    
    else:
        print("Generating individual samples...")
        
        for mask_type in mask_types:
            print(f"Generating samples with {mask_type} masks...")
            
            dataset = InpaintingDataset(
                root=args.data_root, 
                dataset='celeba', 
                image_size=args.image_size, 
                mask_size=args.mask_size,
                mask_type=mask_type
            )
            
            if args.process_all:
                mask_samples = len(dataset)
                print(f"Processing all {mask_samples} images in the dataset...")
            else:
                mask_samples = args.num_samples if len(mask_types) == 1 else max(1, args.num_samples // len(mask_types))
            
            for i in tqdm(range(mask_samples), desc=f"Generating {mask_type} samples"):
                # Generate a random seed between 0 and 2^32 - 1
                random_seed = np.random.randint(0, 2**32 - 1)
                np.random.seed(random_seed)
                
                if args.process_all:
                    idx = i  # Use sequential indices
                else:
                    idx = np.random.randint(0, len(dataset))  # Random selection
                
                masked_img, orig_img, mask = dataset[idx]
                orig_img = masked_img
                masked_img_tensor = masked_img.unsqueeze(0).to(device)
                mask = mask.to(device)

                # Generate image name for saving intermediates
                if args.process_all:
                    if len(mask_types) == 1:
                        img_name = f'image_{idx:05d}'
                    else:
                        img_name = f'image_{idx:05d}_{mask_type}'
                else:
                    if len(mask_types) == 1:
                        img_name = f'sample_{i+1:03d}'
                    else:
                        img_name = f'sample_{mask_type}_{i+1:03d}'
                
                # Store masked image for batch saving
                masked_images.append((masked_img_tensor, img_name))
                
                # Apply deartifact (guided filtering)
                deartifact_img = deartifact_tensor_guided(masked_img_tensor)
                deartifact_images.append((deartifact_img, img_name))
                
                # Apply denoising (BM3D)
                denoised_img = denoise_tensor_bm3d(deartifact_img)
                denoised_images.append((denoised_img, img_name))
                
                with torch.no_grad():
                    pred = model(denoised_img)
                    
                # Merge using denoised masked image
                repainted = denoised_img.squeeze(0) * (1 - mask) + pred.squeeze(0) * mask
                repainted_images.append((repainted.unsqueeze(0), img_name))
                
                # Apply deblurring
                blur_kernel = create_motion_blur_kernel()
                deblurred_repainted = apply_deblur(repainted, kernel=blur_kernel, return_format='same')
                deblurred_images.append((deblurred_repainted.unsqueeze(0), img_name))
                
                # Store original and generated images for saving at the end
                original_images.append((orig_img, img_name))
                generated_images.append((deblurred_repainted, img_name))
                
                # Use denoised version for triplet display
                masked_img = denoised_img
                repainted = deblurred_repainted
                
                if args.save_individual:
                    if args.process_all:
                        if len(mask_types) == 1:
                            out_path = os.path.join(args.output_dir, f'image_{idx:05d}.png')
                        else:
                            out_path = os.path.join(args.output_dir, f'image_{idx:05d}_{mask_type}.png')
                    else:
                        if len(mask_types) == 1:
                            out_path = os.path.join(args.output_dir, f'sample_{i+1:03d}.png')
                        else:
                            out_path = os.path.join(args.output_dir, f'sample_{mask_type}_{i+1:03d}.png')
                    
                    save_triplet(orig_img, masked_img.squeeze(0), repainted, out_path, mask_type)

    # Save all original and generated images at the end (skip if comparison_grid mode)
    if not (args.comparison_grid and len(mask_types) > 1):
        if original_images or generated_images:
            print("\nSaving original and generated images...")
            for img_tensor, img_name in tqdm(original_images, desc="Saving original images"):
                img_np = denorm(img_tensor).cpu().numpy().transpose(1,2,0)
                img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_path = os.path.join(original_dir, f'{img_name}.png')
                img_pil.save(img_path)
            
            for img_tensor, img_name in tqdm(generated_images, desc="Saving generated images"):
                img_np = denorm(img_tensor).cpu().numpy().transpose(1,2,0)
                img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_path = os.path.join(generated_dir, f'{img_name}.png')
                img_pil.save(img_path)
    
    # Save all intermediate images in batch
    if masked_images or deartifact_images or denoised_images or repainted_images or deblurred_images:
        print("\nSaving intermediate images...")
        for img_tensor, img_name in tqdm(masked_images, desc="Saving masked images"):
            save_intermediate_image(img_tensor, os.path.join(masked_dir, f'{img_name}.png'))
        
        for img_tensor, img_name in tqdm(deartifact_images, desc="Saving deartifact images"):
            save_intermediate_image(img_tensor, os.path.join(deartifact_dir, f'{img_name}.png'))
        
        for img_tensor, img_name in tqdm(denoised_images, desc="Saving denoised images"):
            save_intermediate_image(img_tensor, os.path.join(denoised_dir, f'{img_name}.png'))
        
        for img_tensor, img_name in tqdm(repainted_images, desc="Saving repainted images"):
            save_intermediate_image(img_tensor, os.path.join(repainted_dir, f'{img_name}.png'))
        
        for img_tensor, img_name in tqdm(deblurred_images, desc="Saving deblurred images"):
            save_intermediate_image(img_tensor, os.path.join(deblurred_dir, f'{img_name}.png'))
    
    print(f"\nGeneration complete!")
    print(f"Output directory: {args.output_dir}")
    if not (args.comparison_grid and len(mask_types) > 1):
        print(f"Original images saved to: {original_dir}")
        print(f"Generated images saved to: {generated_dir}")
    print(f"Intermediate outputs saved to:")
    print(f"  - Masked images: {masked_dir}")
    print(f"  - Deartifact images: {deartifact_dir}")
    print(f"  - Denoised images: {denoised_dir}")
    print(f"  - Repainted images: {repainted_dir}")
    print(f"  - Deblurred images: {deblurred_dir}")
    
    if args.comparison_grid and len(mask_types) > 1:
        if args.process_all:
            print(f"Generated {num_grids_generated} comparison grids (one per image) and combined them into one image")
        else:
            print(f"Generated {num_grids_generated} comparison grids and combined them into one image")
    else:
        if args.process_all:
            total_samples = len(original_images)
            print(f"Generated {total_samples} individual samples (all images processed) with mask types: {mask_types}")
        else:
            total_samples = args.num_samples if len(mask_types) == 1 else args.num_samples * len(mask_types)
            print(f"Generated {total_samples} individual samples with mask types: {mask_types}")
    
    print(f"Mask types used: {', '.join(mask_types)}")
    print(f"Model: {args.model_path}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Mask size: {args.mask_size}")

if __name__ == '__main__':
    main() 