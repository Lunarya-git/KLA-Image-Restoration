"""
Standalone evaluation / inference script.

This file is intentionally self-contained: it does NOT import dataset.py,
losses.py, or anything else that carries training-only concerns (splitting,
augmentation, loss terms) so it cannot break due to unrelated changes
elsewhere in the repo, and so it can be handed to a benchmarking team as-is.

CLI contract (exact):
    python evaluate.py --input_dir PATH --output_dir PATH [--weights PATH]

Default --weights: weights/final_model.pt

Behavior:
    - Loads the model (auto-detects CPU vs GPU).
    - Reads every supported image in input_dir (.npy, .png, .tif, .tiff,
      .jpg, .jpeg, .bmp).
    - Runs inference on each image.
    - Writes the restored image to output_dir under the SAME filename
      (same extension as the input file: .npy in -> .npy out, image in ->
      image out).
    - Prints total images processed and average ms/image at the end.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model definition is duplicated (not imported from model.py) on purpose so
# this script has zero dependency on the rest of the training codebase and
# cannot be broken by unrelated refactors there. Keep this in sync with
# model.py if the architecture changes.
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.act1(self.conv1(x))
        out = self.conv2(out)
        return out + residual


class RestorationNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 num_residual_blocks=8, upscale_factor=2, output_activation="sigmoid"):
        super().__init__()
        self.output_activation = output_activation
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_residual_blocks)])
        self.trunk_fuse = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        upsample_channels = base_channels * (upscale_factor ** 2)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, upsample_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        stem_out = self.stem(x)
        trunk_out = self.trunk(stem_out)
        fused = self.trunk_fuse(trunk_out) + stem_out
        up = self.upsample(fused)
        out = self.output_conv(up)
        if self.output_activation == "sigmoid":
            out = torch.sigmoid(out)
        else:
            out = torch.clamp(out, 0.0, 1.0)
        return out


ARRAY_EXTENSIONS = (".npy",)
IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
SUPPORTED_EXTENSIONS = ARRAY_EXTENSIONS + IMAGE_EXTENSIONS


def detect_bit_depth(arr: np.ndarray) -> int:
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.uint8:
            return 8
        if arr.dtype == np.uint16:
            return 16
        max_val = int(arr.max()) if arr.size else 0
        return 16 if max_val > 255 else 8
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val <= 1.2:
        return 0
    if max_val <= 255.0:
        return 8
    return 16


def normalize_array(arr: np.ndarray, bit_depth: int) -> np.ndarray:
    arr = arr.astype(np.float32)
    if bit_depth == 8:
        return arr / 255.0
    if bit_depth == 16:
        return arr / 65535.0
    return arr


def denormalize_array(arr: np.ndarray, bit_depth: int, out_dtype) -> np.ndarray:
    """Scale a [0, 1] float array back to the original bit-depth range/dtype
    for saving. Only meaningful for image-file outputs; .npy outputs are
    saved as float32 in [0, 1] directly (see save_output)."""
    if bit_depth == 8:
        scaled = np.clip(arr * 255.0, 0, 255)
        return scaled.astype(out_dtype)
    if bit_depth == 16:
        scaled = np.clip(arr * 65535.0, 0, 65535)
        return scaled.astype(out_dtype)
    return arr.astype(out_dtype)


def load_input(path: str) -> tuple[np.ndarray, int, np.dtype]:
    """Returns (raw_array, bit_depth, original_dtype)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ARRAY_EXTENSIONS:
        arr = np.load(path)
    else:
        from PIL import Image

        img = Image.open(path)
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img)

    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    original_dtype = arr.dtype
    bit_depth = detect_bit_depth(arr)
    return arr, bit_depth, original_dtype


def save_output(path: str, restored_01: np.ndarray, bit_depth: int, original_dtype) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext in ARRAY_EXTENSIONS:
        np.save(path, restored_01.astype(np.float32))
    else:
        from PIL import Image

        denorm = denormalize_array(restored_01, bit_depth, np.uint8 if bit_depth != 16 else np.uint16)
        # Standard image formats display best as 8-bit; convert 16-bit down
        # to 8-bit for saving as PNG/JPEG/etc, but keep .npy fully lossless.
        if denorm.dtype != np.uint8:
            denorm = (denorm.astype(np.float32) / (65535.0 if bit_depth == 16 else 1.0) * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(denorm).save(path)


def load_model(weights_path: str, device: torch.device) -> nn.Module:
    checkpoint = torch.load(weights_path, map_location=device)

    # Support both a raw state_dict and a full training checkpoint dict
    # (as produced by utils.save_checkpoint), and recover architecture
    # hyperparameters from the embedded config when available.
    model_kwargs = dict(in_channels=1, out_channels=1, base_channels=64,
                         num_residual_blocks=8, upscale_factor=2, output_activation="sigmoid")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        cfg = checkpoint.get("extra", {}).get("config") if isinstance(checkpoint.get("extra"), dict) else None
        if cfg and "model" in cfg:
            m_cfg = cfg["model"]
            model_kwargs.update(
                in_channels=m_cfg.get("in_channels", 1),
                out_channels=m_cfg.get("out_channels", 1),
                base_channels=m_cfg.get("base_channels", 64),
                num_residual_blocks=m_cfg.get("num_residual_blocks", 8),
                upscale_factor=m_cfg.get("upscale_factor", 2),
                output_activation=m_cfg.get("output_activation", "sigmoid"),
            )
    else:
        state_dict = checkpoint  # assume raw state_dict

    model = RestorationNet(**model_kwargs)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Run restoration inference on a directory of images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of degraded input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write restored images to.")
    parser.add_argument("--weights", type=str, default="weights/final_model.pt", help="Path to trained model weights.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"--input_dir '{args.input_dir}' does not exist or is not a directory.")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(
            f"--weights '{args.weights}' not found. Train a model first (see train.py) and either point "
            f"--weights at the resulting checkpoint, or copy/promote it to weights/final_model.pt."
        )
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate.py] Using device: {device}")

    model = load_model(args.weights, device)

    filenames = sorted(
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    )
    if not filenames:
        print(f"[evaluate.py] No supported files found in '{args.input_dir}' "
              f"(supported extensions: {SUPPORTED_EXTENSIONS}).")
        return

    total_time_s = 0.0
    processed = 0

    with torch.no_grad():
        for fname in filenames:
            in_path = os.path.join(args.input_dir, fname)
            out_path = os.path.join(args.output_dir, fname)

            raw, bit_depth, original_dtype = load_input(in_path)
            normalized = normalize_array(raw, bit_depth)
            tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)

            is_cuda = device.type == "cuda"
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()

            pred = model(tensor)

            if is_cuda:
                torch.cuda.synchronize()
            end = time.perf_counter()
            total_time_s += end - start
            processed += 1

            restored = pred.squeeze(0).squeeze(0).clamp(0, 1).cpu().numpy()
            save_output(out_path, restored, bit_depth, original_dtype)

    avg_ms = (total_time_s / processed) * 1000.0 if processed else float("nan")
    print(f"[evaluate.py] Processed {processed} images. Average inference time: {avg_ms:.3f} ms/image.")


if __name__ == "__main__":
    main()
