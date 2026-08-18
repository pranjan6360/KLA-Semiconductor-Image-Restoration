import math
import numpy as np
import torch
import torch.nn.functional as F

try:
    from skimage.metrics import structural_similarity as skimage_ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from scipy.ndimage import gaussian_filter

def calculate_psnr(img1, img2, data_range=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    Supports PyTorch tensors and NumPy arrays.
    
    Args:
        img1: Prediction image (Tensor or ndarray) in range [0, 1]
        img2: Ground-truth image (Tensor or ndarray) in range [0, 1]
        data_range: Dynamic range of input images (default: 1.0)
        
    Returns:
        float: PSNR value in dB
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
        
    img1 = np.clip(img1.astype(np.float64), 0.0, data_range)
    img2 = np.clip(img2.astype(np.float64), 0.0, data_range)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return float(10.0 * math.log10((data_range ** 2) / mse))


def _numpy_ssim(img1, img2, data_range=1.0, k1=0.01, k2=0.03, sigma=1.5):
    """
    Pure NumPy implementation of 2D SSIM using SciPy gaussian_filter.
    """
    im1 = np.clip(img1.astype(np.float64), 0.0, data_range)
    im2 = np.clip(img2.astype(np.float64), 0.0, data_range)
    
    mu1 = gaussian_filter(im1, sigma=sigma)
    mu2 = gaussian_filter(im2, sigma=sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter(im1 ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(im2 ** 2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(im1 * im2, sigma=sigma) - mu1_mu2
    
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))


def calculate_ssim(img1, img2, data_range=1.0):
    """
    Calculate Structural Similarity Index Measure (SSIM).
    Supports PyTorch tensors and NumPy arrays.
    
    Args:
        img1: Prediction image (Tensor or ndarray) in range [0, 1]
        img2: Ground-truth image (Tensor or ndarray) in range [0, 1]
        data_range: Dynamic range of input images (default: 1.0)
        
    Returns:
        float: Average SSIM value in [0, 1]
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    def ssim_2d(im1, im2):
        if HAS_SKIMAGE:
            return float(skimage_ssim(im1, im2, data_range=data_range))
        else:
            return _numpy_ssim(im1, im2, data_range=data_range)

    # Ensure shape is (H, W) or (B, H, W)
    if img1.ndim == 4: # (B, C, H, W)
        ssim_vals = []
        for b in range(img1.shape[0]):
            for c in range(img1.shape[1]):
                im1_2d = np.clip(img1[b, c], 0.0, data_range)
                im2_2d = np.clip(img2[b, c], 0.0, data_range)
                ssim_vals.append(ssim_2d(im1_2d, im2_2d))
        return float(np.mean(ssim_vals))
        
    elif img1.ndim == 3: # (C, H, W)
        ssim_vals = []
        for c in range(img1.shape[0]):
            im1_2d = np.clip(img1[c], 0.0, data_range)
            im2_2d = np.clip(img2[c], 0.0, data_range)
            ssim_vals.append(ssim_2d(im1_2d, im2_2d))
        return float(np.mean(ssim_vals))
        
    else: # (H, W)
        im1_2d = np.clip(img1, 0.0, data_range)
        im2_2d = np.clip(img2, 0.0, data_range)
        return ssim_2d(im1_2d, im2_2d)
