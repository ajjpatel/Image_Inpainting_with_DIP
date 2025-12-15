import argparse
import os
import numpy as np
import cv2
import bm3d
import torch


def denoise_tensor_nlm(img_tensor, h=20, template_window_size=7, search_window_size=21):
    original_shape = img_tensor.shape
    device = img_tensor.device
    has_batch = len(original_shape) == 4
    
    if has_batch:
        img_np = img_tensor.cpu().numpy()
        batch_size = img_np.shape[0]
    else:
        img_np = img_tensor.unsqueeze(0).cpu().numpy()
        batch_size = 1
    
    denoised_batch = []
    for i in range(batch_size):
        img = img_np[i]
        
        img_normalized = (img + 1.0) / 2.0
        img_normalized = np.clip(img_normalized, 0, 1)

        img_hwc = np.transpose(img_normalized, (1, 2, 0))
        
        img_rgb = (img_hwc * 255.0).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        if img_bgr.shape[2] == 3:  # Color image
            denoised_bgr = cv2.fastNlMeansDenoisingColored(
                img_bgr,
                None,
                h=h,
                hColor=h,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
        else:
            denoised_bgr = cv2.fastNlMeansDenoising(
                img_bgr,
                None,
                h=h,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
        
        denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)
        
        denoised_normalized = denoised_rgb.astype(np.float32) / 255.0
        denoised_normalized = np.clip(denoised_normalized, 0, 1)
        
        denoised_chw = np.transpose(denoised_normalized, (2, 0, 1))
        
        denoised_tensor_range = denoised_chw * 2.0 - 1.0
        
        denoised_batch.append(denoised_tensor_range)
    
    denoised_np = np.stack(denoised_batch, axis=0) if has_batch else denoised_batch[0]
    denoised_tensor = torch.from_numpy(denoised_np).float().to(device)
    
    if not has_batch:
        denoised_tensor = denoised_tensor.squeeze(0)
    
    return denoised_tensor


def denoise_tensor_bm3d(img_tensor, sigma=5):
    original_shape = img_tensor.shape
    device = img_tensor.device
    has_batch = len(original_shape) == 4
    if has_batch:
        img_np = img_tensor.cpu().numpy()
        batch_size = img_np.shape[0]
    else:
        img_np = img_tensor.unsqueeze(0).cpu().numpy()
        batch_size = 1
    
    denoised_batch = []
    for i in range(batch_size):
        img = img_np[i]
        
        img_normalized = (img + 1.0) / 2.0
        img_normalized = np.clip(img_normalized, 0, 1)

        img_hwc = np.transpose(img_normalized, (1, 2, 0))
        
        img_rgb = (img_hwc * 255.0).astype(np.uint8)
        
        img_rgb_float = img_rgb.astype(np.float32) / 255.0
        denoised_normalized = bm3d.bm3d_rgb(img_rgb_float, sigma_psd=sigma/255.0)
        denoised_rgb = np.clip(denoised_normalized, 0, 1)
        
        denoised_hwc = np.transpose(denoised_rgb, (2, 0, 1))
        
        denoised_normalized_tensor = denoised_hwc * 2.0 - 1.0
        
        denoised_batch.append(denoised_normalized_tensor)
    
    denoised_np = np.stack(denoised_batch, axis=0) if has_batch else denoised_batch[0]
    denoised_tensor = torch.from_numpy(denoised_np).float().to(device)
    
    if not has_batch:
        denoised_tensor = denoised_tensor.squeeze(0)
    
    return denoised_tensor
