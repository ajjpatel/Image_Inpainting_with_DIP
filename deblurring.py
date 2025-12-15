"""
Deblurring code
"""

import numpy as np
from PIL import Image
from scipy import ndimage, signal
import cv2
import torch


def create_motion_blur_kernel(length=5, angle=0):
 
    kernel = np.zeros((length, length))
    center = length // 2
    kernel[center, :] = np.ones(length)
    kernel = kernel / length
    
    # Rotate kernel to desired angle
    if angle != 0:
        kernel = ndimage.rotate(kernel, angle, reshape=False)
        # Normalize after rotation
        kernel = kernel / kernel.sum()
    
    return kernel


def estimate_noise_variance(image, has_mask=False):
    # Handle different input types
    if isinstance(image, torch.Tensor):
        # Tensor: assume normalized [-1, 1] or [0, 1], convert to [0, 1]
        if image.min() < 0:
            img_array = (image.cpu().numpy() + 1) / 2.0
        else:
            img_array = image.cpu().numpy()
        # Handle different tensor shapes: [B, C, H, W], [C, H, W], or [H, W]
        if len(img_array.shape) == 4:
            # Batch dimension: take first image [B, C, H, W] -> [C, H, W]
            img_array = img_array[0]
        if len(img_array.shape) == 3:
            # CHW format: [C, H, W] -> [H, W, C]
            if img_array.shape[0] in [1, 3]:  # Single channel or RGB
                img_array = img_array.transpose(1, 2, 0)
    elif isinstance(image, np.ndarray):
        # Numpy array: assume [0, 255] uint8 or [0, 1] float
        img_array = image.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        # Handle batch dimension if present
        if len(img_array.shape) == 4:
            img_array = img_array[0]
    else:
        # PIL Image
        img_array = np.array(image).astype(np.float32) / 255.0
    
    # Ensure we have a valid 2D or 3D array
    if len(img_array.shape) < 2:
        # Invalid shape, return default
        noise_var = 0.01
        if has_mask:
            noise_var = noise_var * 1.5
        return noise_var
    
    # Convert to grayscale for noise estimation
    if len(img_array.shape) == 3:
        # HWC format: [H, W, C] -> [H, W]
        gray = np.mean(img_array, axis=2)
    elif len(img_array.shape) == 2:
        # Already 2D
        gray = img_array
    else:
        # Unexpected shape, return default
        noise_var = 0.01
        if has_mask:
            noise_var = noise_var * 1.5
        return noise_var
    
    # Ensure gray is 2D
    if len(gray.shape) != 2:
        noise_var = 0.01
        if has_mask:
            noise_var = noise_var * 1.5
        return noise_var
    
    # Check if image is large enough for gradient computation
    # np.gradient requires at least 2 elements in each dimension (edge_order=1 by default)
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        # Image too small for gradient computation, return default noise variance
        noise_var = 0.01
        if has_mask:
            noise_var = noise_var * 1.5
        return noise_var
    
    # Compute gradients with explicit edge_order to avoid issues
    try:
        # Use edge_order=1 explicitly (default, but being explicit)
        # This requires at least 2 elements in each dimension
        if gray.shape[0] >= 2 and gray.shape[1] >= 2:
            grad_x = np.abs(np.gradient(gray, axis=1, edge_order=1))
            grad_y = np.abs(np.gradient(gray, axis=0, edge_order=1))
            
            all_grads = np.concatenate([grad_x.flatten(), grad_y.flatten()])
            noise_est = np.median(all_grads)
            
            # Scale appropriately - convert gradient-based estimate to variance
            noise_var = max(0.001, min(0.1, noise_est * 0.1))
        else:
            # Shouldn't reach here due to check above, but just in case
            noise_var = 0.01
    except (ValueError, IndexError) as e:
        # Fallback if gradient computation fails for any reason
        noise_var = 0.01
    
    # Increase noise estimate for masked images (black regions cause artifacts)
    if has_mask:
        noise_var = noise_var * 1.5
    
    return noise_var


def wiener_deblur(image, kernel, noise_var=0.01, regularization=0.05, return_format='same'):
    # Handle different input types and convert to numpy [0, 1]
    input_is_tensor = isinstance(image, torch.Tensor)
    input_is_numpy = isinstance(image, np.ndarray)
    
    # Initialize variables for tensor handling
    original_device = None
    original_was_normalized = False
    original_shape = None
    
    if input_is_tensor:
        # Tensor: assume normalized [-1, 1] or [0, 1], convert to [0, 1]
        original_device = image.device
        original_was_normalized = image.min() < 0
        if original_was_normalized:
            img_array = (image.cpu().numpy() + 1) / 2.0
        else:
            img_array = image.cpu().numpy()
        original_shape = image.shape
        
        # Handle batch dimension: [B, C, H, W] -> [C, H, W]
        if len(img_array.shape) == 4:
            img_array = img_array[0]  # Take first image from batch
        
        # Convert from CHW to HWC if needed
        if len(img_array.shape) == 3 and img_array.shape[0] in [1, 3]:
            img_array = img_array.transpose(1, 2, 0)
    elif input_is_numpy:
        # Numpy array: assume [0, 255] uint8 or [0, 1] float
        img_array = image.astype(np.float32)
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
    else:
        # PIL Image
        img_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply deconvolution to each channel
    deblurred_channels = []
    for i in range(img_array.shape[2]):
        channel = img_array[:, :, i]
        
        # Compute FFT of image and kernel
        img_fft = np.fft.fft2(channel)
        kernel_fft = np.fft.fft2(kernel, s=channel.shape)
        
        # Wiener filter 
        kernel_conj = np.conj(kernel_fft)
        kernel_mag_sq = np.abs(kernel_fft) ** 2
        wiener_filter = kernel_conj / (kernel_mag_sq + noise_var + regularization)
        
        # Apply filter and inverse FFT
        deblurred_fft = img_fft * wiener_filter
        deblurred_channel = np.real(np.fft.ifft2(deblurred_fft))
        
        deblurred_channels.append(deblurred_channel)
    
    # Stack channels back together
    deblurred_array = np.stack(deblurred_channels, axis=2)
    
    # Clip values to valid range [0, 1]
    deblurred_array = np.clip(deblurred_array, 0, 1)
    
    # Return in the requested format
    if return_format == 'same':
        if input_is_tensor:
            # Convert back to tensor 
            # deblurred_array is [H, W, C], need to convert to [C, H, W] or [1, C, H, W]
            if len(deblurred_array.shape) == 3:
                deblurred_array = deblurred_array.transpose(2, 0, 1)  # [H, W, C] -> [C, H, W]
            
            # Add batch dimension back if it was there originally
            if len(original_shape) == 4:
                deblurred_array = deblurred_array[np.newaxis, ...]  # [C, H, W] -> [1, C, H, W]
            
            deblurred_tensor = torch.from_numpy(deblurred_array).float()
            # Convert [0, 1] back to [-1, 1] if original was normalized
            if original_was_normalized:
                deblurred_tensor = deblurred_tensor * 2.0 - 1.0
            return deblurred_tensor.to(original_device)
        elif input_is_numpy:
            # Return numpy in same format as input
            if image.max() > 1.0:
                return (deblurred_array * 255).astype(np.uint8)
            else:
                return deblurred_array.astype(np.float32)
        else:
            # PIL Image
            deblurred_array = (deblurred_array * 255).astype(np.uint8)
            return Image.fromarray(deblurred_array)
    elif return_format == 'tensor':
        # Convert to tensor: 
        if len(deblurred_array.shape) == 3:
            deblurred_array = deblurred_array.transpose(2, 0, 1)
        deblurred_tensor = torch.from_numpy(deblurred_array).float() * 2.0 - 1.0
        if input_is_tensor:
            return deblurred_tensor.to(original_device)
        return deblurred_tensor
    elif return_format == 'numpy':
        return deblurred_array.astype(np.float32)
    else:  # 'pil' or default
        deblurred_array = (deblurred_array * 255).astype(np.uint8)
        return Image.fromarray(deblurred_array)


def apply_deblur(image, method='wiener', kernel=None, adaptive_noise=True, has_mask=False, return_format='same', **kwargs):

    if method == 'wiener':
        if kernel is None:
            raise ValueError("Kernel required for Wiener deblurring")
        
        if adaptive_noise:
            noise_var = estimate_noise_variance(image, has_mask=has_mask)
        else:
            noise_var = kwargs.get('noise_var', 0.01)
            if has_mask:
                noise_var = noise_var * 2.0
        
        regularization = kwargs.get('regularization', 0.05)
        if has_mask:
            regularization = regularization * 1.5
        
        return wiener_deblur(image, kernel, noise_var=noise_var, regularization=regularization, return_format=return_format)
    else:
        raise ValueError(f"Unknown deblurring method: {method}. Available methods: 'wiener'")

