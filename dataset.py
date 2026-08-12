"""
Dataset loading for the KLA image restoration task.

Discovery convention (primary): a data root containing two subfolders,
    <data_root>/degraded/  and  <data_root>/gt/
with matching filenames between the two (e.g. degraded/sample_001.npy and
gt/sample_001.npy). Files may be .npy arrays (the format the KLA dataset
appears to ship in) or standard raster images (.png/.tif/.tiff/.jpg).

If the real data turns out to use a different convention (e.g. a filename
suffix like `_degraded` / `_gt` in a single flat folder, or per-source
subfolders), edit ONLY the `_discover_pairs` method below -- everything else
(normalization, augmentation, splitting, __getitem__) is convention-agnostic
and does not need to change.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from utils import detect_bit_depth, normalize_array

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
ARRAY_EXTENSIONS = (".npy",)
SUPPORTED_EXTENSIONS = ARRAY_EXTENSIONS + IMAGE_EXTENSIONS


def _load_raw(path: str) -> np.ndarray:
    """Load a single file (either .npy array or a standard image) as a raw,
    un-normalized numpy array with its native dtype/scale intact."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ARRAY_EXTENSIONS:
        arr = np.load(path)
    else:
        from PIL import Image

        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")  # grayscale only, per problem statement
        arr = np.array(img)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # collapse an accidental singleton channel dim, e.g. (H, W, 1)
        if arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(
                f"Expected a single-channel (grayscale) array at '{path}' but got shape {arr.shape}. "
                "Color images are not part of this challenge."
            )
    return arr


class PairedRestorationDataset(Dataset):
    """
    Returns (degraded_tensor, gt_tensor, meta_dict) triples.

    degraded_tensor: float32 tensor, shape (1, h, w), fixed-global-scale
        normalized. May exceed [0, 1] due to speckle noise overshoot --
        this is expected and is NOT clipped here.
    gt_tensor: float32 tensor, shape (1, H, W) where H = h * scale_factor,
        normalized the same way, always intended to lie within [0, 1].
    meta_dict: {"filename": str, "scale_factor": int}
    """

    def __init__(self, data_root: str, extensions=SUPPORTED_EXTENSIONS,
                 transform: Optional[Callable] = None):
        self.data_root = data_root
        self.extensions = tuple(e.lower() for e in extensions)
        self.transform = transform
        self.pairs = self._discover_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No degraded/GT pairs found under '{data_root}'. Expected "
                f"'{data_root}/degraded/' and '{data_root}/gt/' with matching "
                f"filenames (extensions: {self.extensions}). If your data uses "
                f"a different layout, edit PairedRestorationDataset._discover_pairs "
                f"in dataset.py."
            )

    # ------------------------------------------------------------------
    # === ADJUST HERE if the real data's naming/folder convention differs.
    # ------------------------------------------------------------------
    def _discover_pairs(self) -> list[tuple[str, str]]:
        degraded_dir = os.path.join(self.data_root, "degraded")
        gt_dir = os.path.join(self.data_root, "gt")

        if os.path.isdir(degraded_dir) and os.path.isdir(gt_dir):
            return self._discover_from_subfolders(degraded_dir, gt_dir)

        raise RuntimeError(
            f"Could not find 'degraded/' and 'gt/' subfolders under '{self.data_root}'. "
            "Edit PairedRestorationDataset._discover_pairs in dataset.py to match "
            "the real data layout (e.g. a filename-suffix convention or per-source "
            "subfolders) once the dataset has been inspected with inspect_data.py."
        )

    def _discover_from_subfolders(self, degraded_dir: str, gt_dir: str) -> list[tuple[str, str]]:
        def index_by_stem(folder: str) -> dict[str, str]:
            out = {}
            for fname in sorted(os.listdir(folder)):
                stem, ext = os.path.splitext(fname)
                if ext.lower() in self.extensions:
                    out[stem] = os.path.join(folder, fname)
            return out

        degraded_by_stem = index_by_stem(degraded_dir)
        gt_by_stem = index_by_stem(gt_dir)

        common = sorted(set(degraded_by_stem) & set(gt_by_stem))
        missing_gt = sorted(set(degraded_by_stem) - set(gt_by_stem))
        missing_degraded = sorted(set(gt_by_stem) - set(degraded_by_stem))
        if missing_gt:
            print(f"[dataset.py] Warning: {len(missing_gt)} degraded file(s) have no matching GT, skipping "
                  f"(e.g. {missing_gt[:3]})")
        if missing_degraded:
            print(f"[dataset.py] Warning: {len(missing_degraded)} GT file(s) have no matching degraded, skipping "
                  f"(e.g. {missing_degraded[:3]})")

        return [(degraded_by_stem[s], gt_by_stem[s]) for s in common]

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        degraded_path, gt_path = self.pairs[idx]

        degraded_raw = _load_raw(degraded_path)
        gt_raw = _load_raw(gt_path)

        # Fixed global-scale normalization (never per-image min-max), applied
        # uniformly to both images per the locked protocol. Bit depth is
        # detected from the GT image (the "clean" reference); the same divisor
        # is then applied to the degraded image even though its raw values may
        # come from a different effective range due to noise overshoot.
        bit_depth = detect_bit_depth(gt_raw)
        degraded = normalize_array(degraded_raw, bit_depth=bit_depth)
        gt = normalize_array(gt_raw, bit_depth=bit_depth)

        if gt.shape[0] % degraded.shape[0] != 0 or gt.shape[1] % degraded.shape[1] != 0:
            raise ValueError(
                f"GT shape {gt.shape} is not an integer multiple of degraded shape "
                f"{degraded.shape} for pair '{degraded_path}' / '{gt_path}'."
            )
        scale_factor = gt.shape[0] // degraded.shape[0]

        degraded_t = torch.from_numpy(degraded).unsqueeze(0).float()
        gt_t = torch.from_numpy(gt).unsqueeze(0).float()

        if self.transform is not None:
            degraded_t, gt_t = self.transform(degraded_t, gt_t)

        meta = {
            "filename": os.path.basename(degraded_path),
            "scale_factor": scale_factor,
        }
        return degraded_t, gt_t, meta


class PairedAugment:
    """
    Applies the SAME random flip/rotation to both the degraded and GT tensor
    so spatial correspondence is preserved. Rotation is restricted to
    multiples of 90 degrees so it is well-defined across the two different
    resolutions (degraded vs. GT) without needing interpolation.
    """

    def __init__(self, horizontal_flip: bool = True, vertical_flip: bool = True,
                 rotate_90: bool = True):
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate_90 = rotate_90

    def __call__(self, degraded: torch.Tensor, gt: torch.Tensor):
        if self.horizontal_flip and torch.rand(1).item() < 0.5:
            degraded = torch.flip(degraded, dims=[-1])
            gt = torch.flip(gt, dims=[-1])
        if self.vertical_flip and torch.rand(1).item() < 0.5:
            degraded = torch.flip(degraded, dims=[-2])
            gt = torch.flip(gt, dims=[-2])
        if self.rotate_90:
            k = int(torch.randint(0, 4, (1,)).item())
            if k > 0:
                degraded = torch.rot90(degraded, k=k, dims=[-2, -1])
                gt = torch.rot90(gt, k=k, dims=[-2, -1])
        return degraded, gt


def get_splits(dataset: Dataset, config: dict) -> tuple[Subset, Subset, Subset]:
    """
    Fixed-seed 80/10/10 train/val/test split (ratios overridable via
    config['data']['split']). Uses a seeded numpy Generator so the split is
    fully deterministic and independent of global RNG state.
    """
    split_cfg = config["data"]["split"]
    train_ratio = split_cfg.get("train", 0.8)
    val_ratio = split_cfg.get("val", 0.1)
    test_ratio = split_cfg.get("test", 0.1)
    seed = split_cfg.get("seed", 42)

    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    n = len(dataset)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # remainder goes to test so rounding never drops/duplicates a sample
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    assert len(train_idx) + len(val_idx) + len(test_idx) == n

    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist()), Subset(dataset, test_idx.tolist())
