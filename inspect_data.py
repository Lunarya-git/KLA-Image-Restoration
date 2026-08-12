"""
Day-1 data exploration tool. NOT part of the training pipeline -- run this
once against a new data root to understand what you're actually working with
before trusting any of the assumptions baked into dataset.py.

Usage:
    python inspect_data.py --data_root /path/to/data [--sample_size 10]

Prints:
    - total degraded/GT pair count found
    - per-sample dtype/shape/min/max for a sample of pairs
    - detected bit depth (8-bit vs 16-bit vs already-normalized float)
    - detected resolution pairs present (e.g. 128->256, 256->512)
    - whether degraded pixel values exceed the GT range, with actual numbers
"""
from __future__ import annotations

import argparse
import os
from collections import Counter

import numpy as np

ARRAY_EXTENSIONS = (".npy",)
IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
SUPPORTED_EXTENSIONS = ARRAY_EXTENSIONS + IMAGE_EXTENSIONS


def load_raw(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in ARRAY_EXTENSIONS:
        arr = np.load(path)
    else:
        from PIL import Image

        img = Image.open(path)
        arr = np.array(img)
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def detect_bit_depth(arr: np.ndarray) -> str:
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.uint8:
            return "8-bit (uint8)"
        if arr.dtype == np.uint16:
            return "16-bit (uint16)"
        max_val = int(arr.max()) if arr.size else 0
        return f"16-bit (inferred from {arr.dtype}, max={max_val})" if max_val > 255 else \
               f"8-bit (inferred from {arr.dtype}, max={max_val})"
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val <= 1.2:
        return f"already normalized float (max={max_val:.4f})"
    if max_val <= 255.0:
        return f"8-bit-range float (max={max_val:.2f})"
    return f"16-bit-range float (max={max_val:.2f})"


def find_pairs(data_root: str) -> list[tuple[str, str]]:
    """Auto-detect the degraded/GT folder pair from several known conventions."""

    def index_by_stem(folder):
        out = {}
        for fname in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() in SUPPORTED_EXTENSIONS:
                out[stem] = os.path.join(folder, fname)
        return out

    def try_pair(noisy_dir, gt_dir):
        if os.path.isdir(noisy_dir) and os.path.isdir(gt_dir):
            noisy_by_stem = index_by_stem(noisy_dir)
            gt_by_stem = index_by_stem(gt_dir)
            common = sorted(set(noisy_by_stem) & set(gt_by_stem))
            if common:
                print(f"[inspect_data.py] Using layout: '{noisy_dir}'  <->  '{gt_dir}'")
                return [(noisy_by_stem[s], gt_by_stem[s]) for s in common]
        return None

    # Convention 1: <data_root>/degraded/  and  <data_root>/gt/
    result = try_pair(os.path.join(data_root, "degraded"), os.path.join(data_root, "gt"))
    if result is not None:
        return result

    # Convention 2 (actual KLA dataset): <data_root>/train/NoisyLR/  and  <data_root>/train/GT/
    result = try_pair(os.path.join(data_root, "train", "NoisyLR"), os.path.join(data_root, "train", "GT"))
    if result is not None:
        return result

    # Convention 3: <data_root>/NoisyLR/  and  <data_root>/GT/
    result = try_pair(os.path.join(data_root, "NoisyLR"), os.path.join(data_root, "GT"))
    if result is not None:
        return result

    print(f"[inspect_data.py] WARNING: Could not find a recognised layout under '{data_root}'.")
    print(f"  Top-level contents:")
    for entry in sorted(os.listdir(data_root))[:50]:
        print(f"    {entry}")
    return []


def main():
    parser = argparse.ArgumentParser(description="Inspect a KLA restoration data root before training.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=10, help="How many pairs to print detailed stats for.")
    args = parser.parse_args()

    pairs = find_pairs(args.data_root)
    print(f"\n[inspect_data.py] Found {len(pairs)} degraded/GT pairs under '{args.data_root}'.\n")
    if not pairs:
        print("[inspect_data.py] No pairs found -- fix the folder layout or edit "
              "dataset.py's _discover_pairs before doing anything else.")
        return

    resolution_pairs = Counter()
    bit_depths_seen = Counter()
    exceed_count = 0
    exceed_examples = []

    sample = pairs[: args.sample_size]

    print(f"--- Per-sample stats (first {len(sample)} pairs) ---")
    for degraded_path, gt_path in sample:
        degraded = load_raw(degraded_path)
        gt = load_raw(gt_path)

        print(f"\n  {os.path.basename(degraded_path)}  <->  {os.path.basename(gt_path)}")
        print(f"    degraded: dtype={degraded.dtype} shape={degraded.shape} "
              f"min={degraded.min()} max={degraded.max()}")
        print(f"    gt:       dtype={gt.dtype} shape={gt.shape} min={gt.min()} max={gt.max()}")
        print(f"    detected bit depth (from gt): {detect_bit_depth(gt)}")

    print("\n--- Full-dataset scan (resolutions, bit depth, range overshoot) ---")
    for degraded_path, gt_path in pairs:
        degraded = load_raw(degraded_path)
        gt = load_raw(gt_path)

        resolution_pairs[(degraded.shape[:2], gt.shape[:2])] += 1
        bit_depths_seen[detect_bit_depth(gt)] += 1

        # Normalize both with the same fixed global scale (per protocol) to
        # check whether degraded exceeds the GT's [0, 1] range post-normalization.
        if np.issubdtype(gt.dtype, np.integer):
            divisor = 255.0 if gt.dtype == np.uint8 else 65535.0
        else:
            divisor = 1.0 if gt.max() <= 1.2 else (255.0 if gt.max() <= 255.0 else 65535.0)

        degraded_norm = degraded.astype(np.float64) / divisor
        gt_norm = gt.astype(np.float64) / divisor

        if degraded_norm.max() > gt_norm.max() + 1e-6 or degraded_norm.min() < gt_norm.min() - 1e-6:
            exceed_count += 1
            if len(exceed_examples) < 5:
                exceed_examples.append(
                    (os.path.basename(degraded_path), float(degraded_norm.min()), float(degraded_norm.max()),
                     float(gt_norm.min()), float(gt_norm.max()))
                )

    print(f"\nResolution pairs found (degraded_shape -> gt_shape : count):")
    for (deg_shape, gt_shape), count in resolution_pairs.most_common():
        print(f"    {deg_shape} -> {gt_shape} : {count}")

    print(f"\nBit depths detected across dataset:")
    for depth, count in bit_depths_seen.most_common():
        print(f"    {depth} : {count}")

    print(f"\nDegraded-exceeds-GT-range check: {exceed_count} / {len(pairs)} pairs "
          f"({100 * exceed_count / len(pairs):.1f}%) have degraded pixel values outside "
          f"the GT's normalized [min, max] range.")
    if exceed_examples:
        print("  Examples (filename, degraded_min, degraded_max, gt_min, gt_max):")
        for ex in exceed_examples:
            print(f"    {ex[0]}: degraded=[{ex[1]:.4f}, {ex[2]:.4f}]  gt=[{ex[3]:.4f}, {ex[4]:.4f}]")


if __name__ == "__main__":
    main()
