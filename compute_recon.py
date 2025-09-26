from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import os

import torch

from rqvae.img_datasets import create_dataset
from rqvae.models import create_model
from rqvae.metrics.recon_metrics import compute_reconstruction_metrics
from rqvae.utils.config import load_config, augment_arch_defaults


def load_model(path, ema=False):
    model_config = os.path.join(os.path.dirname(path), 'config.yaml')
    config = load_config(model_config)
    config.arch = augment_arch_defaults(config.arch)

    model, _ = create_model(config.arch, ema=False)
    ckpt = torch.load(path, weights_only=False)['state_dict_ema'] if ema else torch.load(path, weights_only=False)[
        'state_dict']
    model.load_state_dict(ckpt)

    return model, config


def setup_logger(result_path):
    # Ensure directory exists before creating log file
    os.makedirs(result_path, exist_ok=True)

    log_fname = os.path.join(result_path, 'metrics.log')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_fname), logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


if __name__ == '__main__':
    """
    Computes reconstruction metrics: rFID, SSIM, PSNR, LPIPS

    - rFID: FID between val images and reconstructed images
    - SSIM: Structural Similarity Index (higher is better, max=1.0)
    - PSNR: Peak Signal-to-Noise Ratio in dB (higher is better)
    - LPIPS: Learned Perceptual Image Patch Similarity (lower is better)

    Log is saved to `metrics.log` in the image_path directory.
    Reconstructed images are saved in the same directory.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Batch size to use')
    parser.add_argument('--split', type=str, default='val',
                        help='Dataset split to evaluate (val or train)')
    parser.add_argument('--vqvae', type=str, default='', required=True,
                        help='vqvae path for reconstruction metrics')
    parser.add_argument('--no-save-images', action='store_true',
                        help='Do not save reconstructed images')

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    vqvae_model, config = load_model(args.vqvae)
    vqvae_model = vqvae_model.to(device)
    vqvae_model = torch.nn.DataParallel(vqvae_model).eval()

    # Setup image path and logger
    model_name = os.path.splitext(os.path.basename(args.vqvae))[0]  # get "epochxxx_model"
    image_path = os.path.join(os.path.dirname(args.vqvae), model_name)

    logger = setup_logger(image_path)
    logger.info(f'vqvae model loaded from {args.vqvae}')

    # Load dataset
    dataset_trn, dataset_val = create_dataset(config, is_eval=True, logger=logger)
    dataset = dataset_val if args.split in ['val', 'valid'] else dataset_trn
    logger.info(f'measuring reconstruction metrics on {config.dataset.type}/{args.split}')
    logger.info(f'dataset size: {len(dataset)} samples')

    # Compute all reconstruction metrics
    logger.info('=' * 60)
    logger.info('Computing reconstruction metrics...')
    logger.info('=' * 60)

    results = compute_reconstruction_metrics(
        dataset,
        batch_size=args.batch_size,
        stage1_model=vqvae_model,
        device=device,
        image_path=image_path,
        save_images=not args.no_save_images
    )

    # Log results
    logger.info('=' * 60)
    logger.info('RECONSTRUCTION METRICS SUMMARY')
    logger.info('=' * 60)
    logger.info(f'rFID:  {results["rfid"]:.4f}')
    logger.info(f'SSIM:  {results["ssim_mean"]:.4f} ± {results["ssim_std"]:.4f}')
    logger.info(f'PSNR:  {results["psnr_mean"]:.2f} ± {results["psnr_std"]:.2f} dB')

    if 'lpips_mean' in results:
        logger.info(f'LPIPS: {results["lpips_mean"]:.4f} ± {results["lpips_std"]:.4f}')
    else:
        logger.info('LPIPS: Not available (install with: pip install lpips)')

    logger.info('=' * 60)
    logger.info(f'Log saved to: {os.path.join(image_path, "metrics.log")}')
    if not args.no_save_images:
        logger.info(f'Images saved to: {image_path}')