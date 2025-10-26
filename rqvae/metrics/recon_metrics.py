"""
Reconstruction metrics: rFID, SSIM, PSNR, LPIPS
Combines FID computation with perceptual quality metrics
"""
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
from scipy import linalg

from .fid import get_inception_model, mean_covar_numpy, frechet_distance

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False


def ssim(img1, img2, window_size=11):
    """
    Calculate SSIM between two images in [0, 1] range.
    
    Args:
        img1, img2: Images with shape (B, C, H, W) in range [0, 1]
        window_size: Size of Gaussian window
    
    Returns:
        SSIM value (scalar)
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    channel = img1.size(1)
    sigma = 1.5
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) 
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()


def psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images in [0, 1] range.
    
    Args:
        img1, img2: Images with shape (B, C, H, W) in range [0, 1]
        max_val: Maximum possible pixel value
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))


@torch.no_grad()
def compute_reconstruction_metrics(dataset,
                                   batch_size=500,
                                   inception_model=None,
                                   stage1_model=None,
                                   device=torch.device('cuda'),
                                   image_path="recon_image",
                                   save_images=True,
                                   ):
    """
    Compute all reconstruction metrics: rFID, SSIM, PSNR, LPIPS
    
    Args:
        dataset: Dataset that returns (image, label) pairs with images in [-1, 1]
        batch_size: Batch size for processing
        inception_model: Pre-loaded Inception model (optional)
        stage1_model: VAE/VQVAE model for reconstruction
        device: Device to run on
        image_path: Path to save reconstructed images
        save_images: Whether to save reconstructed images
    
    Returns:
        dict: Dictionary containing all metric values
    """
    
    if stage1_model is None:
        raise ValueError("stage1_model is required for reconstruction metrics")
    
    if inception_model is None:
        inception_model = get_inception_model().to(device)
    
    if LPIPS_AVAILABLE:
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
    
    loader = DataLoader(dataset, shuffle=False, pin_memory=True, batch_size=batch_size, num_workers=16)
    
    inception_model.eval()
    stage1_model.eval()
    
    # For rFID computation
    acts_orig = []
    acts_recon = []
    
    # For SSIM, PSNR, LPIPS
    ssim_scores = []
    psnr_scores = []
    lpips_scores = []
    
    # Image statistics
    sample_size_sum = 0.0
    sample_sum = torch.tensor(0.0, device=device)
    sample_sq_sum = torch.tensor(0.0, device=device)
    sample_max = torch.tensor(float('-inf'), device=device)
    sample_min = torch.tensor(float('inf'), device=device)
    
    if save_images:
        os.makedirs(image_path, exist_ok=True)
        logging.info(f"Reconstructed images will be saved to: {image_path}")
    
    for idx, (xs, _) in enumerate(tqdm(loader, desc="Computing reconstruction metrics")):
        xs = xs.to(device, non_blocking=True)
        
        # Assume dataset returns value in -1 ~ 1 -> remap to 0 ~ 1
        xs_01 = torch.clamp(xs * 0.5 + 0.5, 0, 1)
        
        # Compute image statistics
        sample_sum += xs_01.sum()
        sample_sq_sum += xs_01.pow(2.0).sum()
        sample_size_sum += xs_01.numel()
        sample_max = max(xs_01.max(), sample_max)
        sample_min = min(xs_01.min(), sample_min)
        
        # Get Inception activations for original images
        act_orig = inception_model(xs_01).cpu()
        acts_orig.append(act_orig)
        
        # Get reconstructions (assume model input & output values are in -1 ~ 1 range)
        imgs = xs  # Already in [-1, 1]
        xs_recon = torch.cat([
            stage1_model(imgs[i:i+1])[0] for i in range(imgs.shape[0])
        ], dim=0)
        xs_recon_01 = torch.clamp(xs_recon * 0.5 + 0.5, 0, 1)
        
        # Get Inception activations for reconstructed images
        act_recon = inception_model(xs_recon_01).cpu()
        acts_recon.append(act_recon)
        
        # Compute per-image metrics
        for i in range(xs.shape[0]):
            img_orig_01 = xs_01[i:i+1]
            img_recon_01 = xs_recon_01[i:i+1]
            img_orig = xs[i:i+1]  # [-1, 1] for LPIPS
            img_recon = xs_recon[i:i+1]  # [-1, 1] for LPIPS
            
            # SSIM (on [0, 1] range)
            ssim_val = ssim(img_orig_01, img_recon_01).item()
            ssim_scores.append(ssim_val)
            
            # PSNR (on [0, 1] range)
            psnr_val = psnr(img_orig_01, img_recon_01).item()
            psnr_scores.append(psnr_val)
            
            # LPIPS (on [-1, 1] range)
            if LPIPS_AVAILABLE:
                lpips_val = lpips_model(img_orig, img_recon).item()
                lpips_scores.append(lpips_val)
            
            # Save reconstructed images
            if save_images:
                save_image(img_recon_01[0], f"{image_path}/recon_batch{idx:04d}_img{i:03d}.png")
    
    # Log image statistics
    sample_mean = sample_sum.item() / sample_size_sum
    sample_std = ((sample_sq_sum.item() / sample_size_sum) - (sample_mean ** 2.0)) ** 0.5
    logging.info(f'val imgs. stats :: '
                 f'max: {sample_max:.4f}, min: {sample_min:.4f}, '
                 f'mean: {sample_mean:.4f}, std: {sample_std:.4f}')
    
    # Compute rFID
    acts_orig = torch.cat(acts_orig, dim=0)
    acts_recon = torch.cat(acts_recon, dim=0)
    
    mu_orig, sigma_orig = mean_covar_numpy(acts_orig)
    mu_recon, sigma_recon = mean_covar_numpy(acts_recon)
    
    rfid = frechet_distance(mu_orig, sigma_orig, mu_recon, sigma_recon)
    
    # Compile results
    results = {
        'rfid': float(rfid),
        'ssim_mean': float(np.mean(ssim_scores)),
        'ssim_std': float(np.std(ssim_scores)),
        'psnr_mean': float(np.mean(psnr_scores)),
        'psnr_std': float(np.std(psnr_scores)),
    }
    
    if LPIPS_AVAILABLE and len(lpips_scores) > 0:
        results['lpips_mean'] = float(np.mean(lpips_scores))
        results['lpips_std'] = float(np.std(lpips_scores))
    
    return results
