# Image Inpainting with Context Encoder and Multi-Stage Preprocessing

This project implements an image inpainting system using a Context Encoder architecture combined with multi-stage preprocessing algorithms for denoising, compression artifact removal, and deblurring. The system is designed to restore corrupted or masked regions in images through a comprehensive pipeline that addresses various types of image distortions.

## Part 1: Project Files and Components

### Core Files

#### Context Encoder Components
- **`model.py`** - Contains the Context Encoder model architecture, a generative adversarial network (GAN) designed for image inpainting tasks
- **`train.py`** - Training script for the Context Encoder model with configurable parameters for datasets, epochs, and training settings
- **`G_best.pth`** - Pre-trained Context Encoder model checkpoint (generator component of the GAN)

#### Image Processing Pipeline
- **`denoise.py`** - Implements denoising algorithms including:
  - BM3D (Block-Matching and 3D filtering) for advanced denoising
  - Non-Local Means (NLM) denoising for texture preservation

- **`decompress.py`** - Contains compression artifact removal algorithms:
  - Bilateral filtering for compression artifact reduction
  - Guided filtering for edge-preserving deartifacting

- **`deblurring.py`** - Implements deblurring algorithms:
  - Motion blur kernel estimation and removal
  - Wiener filtering for deblurring operations
  - Blind deconvolution techniques

#### Data and Utilities
- **`generate_inpainted_samples.py`** - Main inference script that generates inpainted images using the trained Context Encoder, with comprehensive preprocessing pipeline
- **`image_metrics.py`** - Evaluation tool that computes image quality metrics:
  - **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator) - measures natural image quality
  - **NIQE** (Natural Image Quality Evaluator) - assesses perceptual image quality

### Data Directory
- **`Data/`** - Contains compressed dataset archives:
  - `train.tar.gz` - Training dataset
  - `test.tar.gz` - Testing dataset

## Part 2: Context Encoder Usage

### Training the Model

To train the Context Encoder model on your dataset:

```bash
python train.py --data_root <path_to_dataset> --dataset <dataset_name> --epochs 20 --mask_type mixed --gpu
```

**Training Parameters:**
- `--data_root`: Path to the root directory containing your training images
- `--dataset`: Dataset type (`celeba`, `cityscapes`, or `places365`)
- `--epochs`: Number of training epochs (default: 20)
- `--mask_type`: Type of masks to use during training (`mixed`, `center`, `random`)
- `--gpu`: Use GPU acceleration if available
- `--batch_size`: Batch size for training (default: 64)
- `--image_size`: Input image resolution (default: 128)
- `--mask_size`: Size of the inpainting mask (default: 64)
- `--lr`: Learning rate (default: 2e-4)

### Generating Inpainted Samples

To generate inpainted images using the trained model:

```bash
python generate_inpainted_samples.py --comparison_grid --mask_type "all" --mask_size 96 --data_root "data/cityscapes/val/img" --gpu
```

**Generation Parameters:**
- `--comparison_grid`: Generate comparison grids showing different mask types
- `--mask_type`: Mask types to use (`"all"` for all types, or specific type like `"mixed"`)
- `--mask_size`: Size of the inpainting mask (default: 64)
- `--data_root`: Path to input images directory
- `--gpu`: Use GPU acceleration
- `--model_path`: Path to trained model (default: `"G_best.pth"`)
- `--output_dir`: Output directory for generated images (default: `"gen_outputs"`)
- `--process_all`: Process all images in dataset instead of random sampling
- `--num_samples`: Number of samples to generate (default: 4)

### Default Processing Pipeline

By default, the system applies all three preprocessing algorithms in following sequence:
- **Decompression** 
- **Denoising** 
- **Context Encoder**
- **Deblurring** 

### Evaluating Results

To evaluate the quality of generated images:

```bash
python image_metrics.py --image_dir <path_to_generated_images> --device gpu
```

This will compute and display:
- BRISQUE scores (lower is better)
- NIQE scores (lower is better)
- Statistical summaries (mean, standard deviation)