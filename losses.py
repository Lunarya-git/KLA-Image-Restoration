"""
Loss functions for the KLA restoration task.

CombinedLoss defaults to the locked baseline: 0.8 * L1 + 0.2 * (1 - SSIM).
It is built so terms and weights are swappable via config (a list of
(name, weight) pairs) for later experiment tracks, e.g.:

    loss:
      terms:
        - name: charbonnier
          weight: 0.7
        - name: ssim
          weight: 0.3
      lpips:
        enabled: true
        weight: 0.1

Supported term names: "l1", "charbonnier", "ssim". LPIPS is implemented as a
separate optional term (kept off by default -- it needs a forward pass
through a pretrained AlexNet and materially slows down training).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1-like loss, differentiable everywhere (unlike L1 at 0)."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))


class SSIMLoss(nn.Module):
    """
    Differentiable SSIM loss (1 - SSIM), computed with a Gaussian window,
    implemented directly in torch so it can be backpropagated through
    (skimage's SSIM, used in metrics.py, is not differentiable and is only
    used for evaluation/reporting).
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.register_buffer("window", self._make_window(window_size, sigma), persistent=False)

    @staticmethod
    def _make_window(window_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        window_2d = g.t() @ g
        return window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        window = self.window.to(dtype=pred.dtype, device=pred.device)
        pad = self.window_size // 2
        channels = pred.shape[1]
        if channels > 1:
            window = window.expand(channels, 1, self.window_size, self.window_size)

        mu_x = F.conv2d(pred, window, padding=pad, groups=channels)
        mu_y = F.conv2d(target, window, padding=pad, groups=channels)

        mu_x_sq, mu_y_sq, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

        sigma_x_sq = F.conv2d(pred * pred, window, padding=pad, groups=channels) - mu_x_sq
        sigma_y_sq = F.conv2d(target * target, window, padding=pad, groups=channels) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, window, padding=pad, groups=channels) - mu_xy

        c1 = (0.01 * self.data_range) ** 2
        c2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        ssim_val = ssim_map.mean()
        return 1.0 - ssim_val


_TERM_REGISTRY = {
    "l1": lambda: nn.L1Loss(),
    "charbonnier": lambda: CharbonnierLoss(),
    "ssim": lambda: SSIMLoss(),
}


class CombinedLoss(nn.Module):
    """
    Weighted sum of configurable loss terms, plus an optional LPIPS term.

    terms: list of (name, weight) tuples, name in {"l1", "charbonnier", "ssim"}
    lpips_enabled / lpips_weight: optional perceptual term, off by default
    """

    def __init__(self, terms: Optional[list[tuple[str, float]]] = None,
                 lpips_enabled: bool = False, lpips_weight: float = 0.0):
        super().__init__()
        if terms is None:
            terms = [("l1", 0.8), ("ssim", 0.2)]  # locked baseline default

        self.term_names = []
        self.term_weights = []
        self.term_modules = nn.ModuleList()
        for name, weight in terms:
            key = name.lower()
            if key not in _TERM_REGISTRY:
                raise ValueError(f"Unknown loss term '{name}'. Supported: {list(_TERM_REGISTRY)}")
            self.term_names.append(key)
            self.term_weights.append(float(weight))
            self.term_modules.append(_TERM_REGISTRY[key]())

        self.lpips_enabled = lpips_enabled
        self.lpips_weight = float(lpips_weight)
        self._lpips_model = None  # lazy-loaded on first use, mirrors metrics.py

    def _get_lpips_model(self, device: torch.device):
        if self._lpips_model is None:
            import lpips as lpips_pkg

            self._lpips_model = lpips_pkg.LPIPS(net="alex").to(device)
            for p in self._lpips_model.parameters():
                p.requires_grad_(False)
        return self._lpips_model

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Returns (total_loss, breakdown_dict) so callers can log individual terms."""
        total = pred.new_zeros(())
        breakdown = {}
        for name, weight, module in zip(self.term_names, self.term_weights, self.term_modules):
            value = module(pred, target)
            breakdown[name] = value.detach().item()
            total = total + weight * value

        if self.lpips_enabled and self.lpips_weight > 0:
            lpips_model = self._get_lpips_model(pred.device)
            # LPIPS (AlexNet) expects 3-channel input in [-1, 1]
            pred_3ch = pred.repeat(1, 3, 1, 1) * 2 - 1
            target_3ch = target.repeat(1, 3, 1, 1) * 2 - 1
            lpips_val = lpips_model(pred_3ch, target_3ch).mean()
            breakdown["lpips"] = lpips_val.detach().item()
            total = total + self.lpips_weight * lpips_val

        breakdown["total"] = total.detach().item()
        return total, breakdown


def build_loss_from_config(config: dict) -> CombinedLoss:
    loss_cfg = config["loss"]
    terms = [(t["name"], t["weight"]) for t in loss_cfg["terms"]]
    lpips_cfg = loss_cfg.get("lpips", {})
    return CombinedLoss(
        terms=terms,
        lpips_enabled=lpips_cfg.get("enabled", False),
        lpips_weight=lpips_cfg.get("weight", 0.0),
    )
