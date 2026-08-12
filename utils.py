"""
Shared helpers used by train.py, evaluate.py, and inspect_data.py:
  - seed setting
  - YAML config loading with CLI override merging
  - checkpoint save/load
  - side-by-side (degraded | restored | GT) comparison grid saver
  - bit-depth detection (shared with dataset.py)
"""
from __future__ import annotations

import copy
import os
import random
import shutil
from typing import Any, Optional

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fix python, numpy, and torch (CPU + all GPUs) seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config loading + CLI override merging
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file '{path}' is empty or invalid YAML.")
    return cfg


def _set_nested(d: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_cli_overrides(cfg: dict, overrides: dict) -> dict:
    """
    Apply a flat dict of {dotted.key: value} overrides on top of a loaded
    config. Only keys with a non-None value are applied, so argparse defaults
    of None never clobber the YAML. Returns a new dict; the input is not
    mutated.
    """
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if value is None:
            continue
        _set_nested(cfg, key, value)
    return cfg


def save_config_copy(cfg_path: str, run_name: str, configs_used_dir: str) -> str:
    """Copy the exact YAML file used for a run into results/configs_used/<run_name>.yaml."""
    os.makedirs(configs_used_dir, exist_ok=True)
    dest = os.path.join(configs_used_dir, f"{run_name}.yaml")
    shutil.copyfile(cfg_path, dest)
    return dest


def dump_effective_config(cfg: dict, run_name: str, configs_used_dir: str) -> str:
    """
    Dump the *effective* config (after CLI overrides) rather than just copying
    the original file. Preferred over save_config_copy when CLI args were used,
    so results/configs_used/<run_name>.yaml always reflects what actually ran.
    """
    os.makedirs(configs_used_dir, exist_ok=True)
    dest = os.path.join(configs_used_dir, f"{run_name}.yaml")
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return dest


# ---------------------------------------------------------------------------
# Bit-depth detection (shared by dataset.py and inspect_data.py)
# ---------------------------------------------------------------------------
def detect_bit_depth(arr: np.ndarray) -> int:
    """
    Detect the source bit depth of a raw (un-normalized) image array so it can
    be scaled to [0, 1] with a fixed global divisor (never per-image min-max).

    Rules:
      - integer dtype uint8            -> 8-bit  (divide by 255)
      - integer dtype uint16           -> 16-bit (divide by 65535)
      - other integer dtypes           -> inferred from max value
      - float dtypes already in [0, ~1.2] (allowing a little headroom for
        speckle overshoot on degraded images) are treated as already-scaled
        and returned as bit_depth=0 (meaning: do not rescale further)
      - float dtypes with larger values are treated as un-normalized and
        bit depth is inferred from the max value, matching the 8-bit/16-bit
        buckets above
    """
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.uint8:
            return 8
        if arr.dtype == np.uint16:
            return 16
        # fall back to max-value inference for other integer dtypes
        max_val = int(arr.max()) if arr.size else 0
        return 16 if max_val > 255 else 8

    # float dtype
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val <= 1.2:
        return 0  # already normalized (0 = "no further scaling")
    if max_val <= 255.0:
        return 8
    return 16


def normalize_array(arr: np.ndarray, bit_depth: Optional[int] = None) -> np.ndarray:
    """
    Apply fixed global-scale normalization. bit_depth is auto-detected if not
    given. Returns float32. Degraded images may legitimately exceed [0, 1]
    after this and must NOT be clipped here -- clipping only happens at model
    output time (sigmoid / clamp), per the locked protocol.
    """
    arr = arr.astype(np.float32)
    if bit_depth is None:
        bit_depth = detect_bit_depth(arr)
    if bit_depth == 8:
        return arr / 255.0
    if bit_depth == 16:
        return arr / 65535.0
    return arr  # bit_depth == 0 -> already normalized, pass through


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(path: str, model: torch.nn.Module, optimizer=None,
                     scheduler=None, epoch: int = 0, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None,
                     scheduler=None, map_location=None) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


# ---------------------------------------------------------------------------
# Comparison grid saver (degraded | restored | GT)
# ---------------------------------------------------------------------------
def save_comparison_grid(degraded: torch.Tensor, restored: torch.Tensor,
                          gt: torch.Tensor, out_path: str, max_samples: int = 4) -> None:
    """
    Save a grid image with rows = samples, columns = [degraded, restored, GT].
    Tensors are expected in shape (N, 1, H, W), float, roughly in [0, 1]
    (values are clipped to [0, 1] purely for *display* purposes here --
    this function is for visualization only, never used inside the loss
    or metric computation).

    Degraded images are resized (nearest, for a fair "what the model saw"
    view) up to the restored/GT resolution so all three columns line up.
    """
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    n = min(max_samples, degraded.shape[0])
    target_hw = restored.shape[-2:]

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    col_titles = ["Degraded (input)", "Restored (model output)", "Ground Truth"]
    for i in range(n):
        deg_i = degraded[i : i + 1]
        if deg_i.shape[-2:] != target_hw:
            deg_i = F.interpolate(deg_i, size=target_hw, mode="nearest")
        panels = [deg_i[0, 0], restored[i, 0], gt[i, 0]]
        for j, panel in enumerate(panels):
            img = panel.detach().cpu().clamp(0, 1).numpy()
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requested device='cuda' but CUDA is not available.")
    if preference == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
