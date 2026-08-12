"""
Evaluation metrics: PSNR, SSIM (via skimage), LPIPS (via the `lpips` package,
lazy-loaded once), a dataloader-level averaging helper, and a properly
warmed-up / synchronized inference-time benchmark.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

_LPIPS_MODEL_CACHE = {}


def _to_numpy_hw(t: torch.Tensor) -> np.ndarray:
    """(1, H, W) or (H, W) tensor -> (H, W) numpy array, clamped to [0, 1]."""
    if t.dim() == 3:
        t = t.squeeze(0)
    return t.detach().cpu().clamp(0, 1).numpy()


def compute_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR for a single (1, H, W) image pair, data_range=1.0."""
    pred_np = _to_numpy_hw(pred)
    gt_np = _to_numpy_hw(gt)
    return float(skimage_psnr(gt_np, pred_np, data_range=1.0))


def compute_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM for a single (1, H, W) image pair, data_range=1.0."""
    pred_np = _to_numpy_hw(pred)
    gt_np = _to_numpy_hw(gt)
    return float(skimage_ssim(gt_np, pred_np, data_range=1.0))


def _get_lpips_model(device: torch.device, net: str = "alex"):
    key = (net, str(device))
    if key not in _LPIPS_MODEL_CACHE:
        import lpips as lpips_pkg

        model = lpips_pkg.LPIPS(net=net).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_MODEL_CACHE[key] = model
    return _LPIPS_MODEL_CACHE[key]


def compute_lpips(pred: torch.Tensor, gt: torch.Tensor, lpips_model=None) -> float:
    """
    LPIPS for a single (1, H, W) image pair. If lpips_model is not provided,
    it is lazily loaded once (net='alex') and cached for subsequent calls.
    """
    device = pred.device
    if lpips_model is None:
        lpips_model = _get_lpips_model(device)

    pred_3ch = pred.unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
    gt_3ch = gt.unsqueeze(0).repeat(1, 3, 1, 1) * 2 - 1
    with torch.no_grad():
        val = lpips_model(pred_3ch.to(device), gt_3ch.to(device))
    return float(val.item())


@torch.no_grad()
def evaluate_batch(model: torch.nn.Module, dataloader, device: torch.device,
                    compute_lpips_metric: bool = True) -> dict:
    """
    Runs the model over an entire dataloader and returns averaged metrics:
    {"psnr": ..., "ssim": ..., "lpips": ...}
    """
    model.eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    lpips_model = _get_lpips_model(device) if compute_lpips_metric else None

    for degraded, gt, _meta in dataloader:
        degraded = degraded.to(device)
        gt = gt.to(device)
        pred = model(degraded)

        for i in range(pred.shape[0]):
            p, g = pred[i], gt[i]
            psnr_vals.append(compute_psnr(p, g))
            ssim_vals.append(compute_ssim(p, g))
            if compute_lpips_metric:
                lpips_vals.append(compute_lpips(p, g, lpips_model=lpips_model))

    results = {
        "psnr": float(np.mean(psnr_vals)) if psnr_vals else float("nan"),
        "ssim": float(np.mean(ssim_vals)) if ssim_vals else float("nan"),
    }
    if compute_lpips_metric:
        results["lpips"] = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    return results


@torch.no_grad()
def measure_inference_time(model: torch.nn.Module, dataloader, device: torch.device,
                            num_warmup: int = 5) -> float:
    """
    Returns average inference time in milliseconds per image. Properly warms
    up CUDA (untimed forward passes) and wraps timed passes with
    torch.cuda.synchronize() so async CUDA kernel launches don't make timing
    look artificially fast.
    """
    model.eval()
    is_cuda = device.type == "cuda"

    data_iter = iter(dataloader)
    warmed_up = 0
    while warmed_up < num_warmup:
        try:
            degraded, _gt, _meta = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            degraded, _gt, _meta = next(data_iter)
        degraded = degraded.to(device)
        _ = model(degraded)
        if is_cuda:
            torch.cuda.synchronize()
        warmed_up += degraded.shape[0]

    total_images = 0
    total_time_s = 0.0
    for degraded, _gt, _meta in dataloader:
        degraded = degraded.to(device)
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(degraded)
        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        total_time_s += end - start
        total_images += degraded.shape[0]

    if total_images == 0:
        return float("nan")
    return (total_time_s / total_images) * 1000.0
