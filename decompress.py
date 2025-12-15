import numpy as np
import cv2
import torch


def deartifact_tensor_bilateral(img_tensor, d=5, sigmaColor=20, sigmaSpace=100, borderType=cv2.BORDER_CONSTANT):
    original_shape = img_tensor.shape
    device = img_tensor.device
    has_batch = len(original_shape) == 4

    if has_batch:
        img_np = img_tensor.detach().cpu().numpy()
        batch_size = img_np.shape[0]
    else:
        img_np = img_tensor.unsqueeze(0).detach().cpu().numpy()
        batch_size = 1

    out_batch = []
    for i in range(batch_size):
        img = img_np[i]  # (C,H,W) in [-1,1]

        img_normalized = (img + 1.0) / 2.0
        img_normalized = np.clip(img_normalized, 0, 1)

        img_hwc = np.transpose(img_normalized, (1, 2, 0))
        img_rgb = (img_hwc * 255.0).astype(np.uint8)

        # OpenCV works in BGR for color ops
        if img_rgb.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            filt_bgr = cv2.bilateralFilter(
                img_bgr, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace, borderType=borderType
            )
            filt_rgb = cv2.cvtColor(filt_bgr, cv2.COLOR_BGR2RGB)
        else:
            # grayscale fallback (C==1)
            gray = img_rgb[:, :, 0]
            filt_gray = cv2.bilateralFilter(
                gray, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace, borderType=borderType
            )
            filt_rgb = filt_gray[:, :, None]

        filt_normalized = filt_rgb.astype(np.float32) / 255.0
        filt_normalized = np.clip(filt_normalized, 0, 1)

        filt_chw = np.transpose(filt_normalized, (2, 0, 1))
        filt_tensor_range = filt_chw * 2.0 - 1.0

        out_batch.append(filt_tensor_range)

    out_np = np.stack(out_batch, axis=0)
    out_tensor = torch.from_numpy(out_np).float().to(device)

    if not has_batch:
        out_tensor = out_tensor.squeeze(0)

    return out_tensor


def deartifact_tensor_guided(img_tensor, radius=3, eps=6**2):
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        raise RuntimeError(
            "cv2.ximgproc.guidedFilter not found. Install opencv-contrib-python:\n"
            "pip install opencv-contrib-python"
        )

    original_shape = img_tensor.shape
    device = img_tensor.device
    has_batch = len(original_shape) == 4

    if has_batch:
        img_np = img_tensor.detach().cpu().numpy()
        batch_size = img_np.shape[0]
    else:
        img_np = img_tensor.unsqueeze(0).detach().cpu().numpy()
        batch_size = 1

    out_batch = []
    for i in range(batch_size):
        img = img_np[i]  # (C,H,W) in [-1,1]

        img_normalized = (img + 1.0) / 2.0
        img_normalized = np.clip(img_normalized, 0, 1)

        img_hwc = np.transpose(img_normalized, (1, 2, 0))
        img_rgb = (img_hwc * 255.0).astype(np.uint8)

        if img_rgb.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
            guided_bgr = cv2.ximgproc.guidedFilter(
                guide=img_bgr, src=img_bgr, radius=radius, eps=float(eps)
            )
            guided_bgr = np.clip(guided_bgr, 0, 255).astype(np.uint8)
            guided_rgb = cv2.cvtColor(guided_bgr, cv2.COLOR_BGR2RGB)
        else:
            gray = img_rgb[:, :, 0].astype(np.float32)
            guided_gray = cv2.ximgproc.guidedFilter(
                guide=gray, src=gray, radius=radius, eps=float(eps)
            )
            guided_rgb = np.clip(guided_gray, 0, 255).astype(np.uint8)[:, :, None]

        guided_normalized = guided_rgb.astype(np.float32) / 255.0
        guided_normalized = np.clip(guided_normalized, 0, 1)

        guided_chw = np.transpose(guided_normalized, (2, 0, 1))
        guided_tensor_range = guided_chw * 2.0 - 1.0

        out_batch.append(guided_tensor_range)

    out_np = np.stack(out_batch, axis=0)
    out_tensor = torch.from_numpy(out_np).float().to(device)

    if not has_batch:
        out_tensor = out_tensor.squeeze(0)

    return out_tensor
