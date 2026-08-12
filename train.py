"""
Train the restoration model.

Usage:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/baseline.yaml --data_dir /path/to/data --epochs 50

CLI args (all optional) override the matching YAML field when given. Anything
not passed on the CLI keeps whatever is in the config file.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PairedAugment, PairedRestorationDataset, get_splits
from losses import build_loss_from_config
from metrics import evaluate_batch, measure_inference_time
from model import build_model_from_config, count_parameters
from utils import (
    apply_cli_overrides,
    dump_effective_config,
    get_device,
    load_config,
    save_checkpoint,
    save_comparison_grid,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the KLA restoration model.")
    parser.add_argument("--config", type=str, required=True, help="Path to a YAML config file.")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data.data_dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override optim.epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--lr", type=float, default=None, help="Override optim.lr.")
    parser.add_argument("--run_name", type=str, default=None, help="Override run_name.")
    return parser.parse_args()


def build_dataloaders(config: dict):
    dataset = PairedRestorationDataset(
        data_root=config["data"]["data_dir"],
        extensions=config["data"].get("extensions"),
    )
    train_set, val_set, test_set = get_splits(dataset, config)

    aug_cfg = config["augmentation"]
    batch_size = config["train"]["batch_size"]

    # Wrap each split's Subset in a small adapter so the SAME underlying
    # PairedRestorationDataset instance can back train/val/test with
    # different (or no) augmentation, without different splits racing to
    # mutate a shared `.transform` attribute on the base dataset.
    train_set.dataset = _TransformSwitchableDataset(dataset, train_transform=(
        PairedAugment(
            horizontal_flip=aug_cfg.get("horizontal_flip", True),
            vertical_flip=aug_cfg.get("vertical_flip", True),
            rotate_90=aug_cfg.get("rotate_90", True),
        ) if aug_cfg.get("enabled", True) else None
    ), apply_transform=True)
    val_set.dataset = _TransformSwitchableDataset(dataset, train_transform=None, apply_transform=False)
    test_set.dataset = _TransformSwitchableDataset(dataset, train_transform=None, apply_transform=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=config["train"].get("num_workers", 4), drop_last=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=config["train"].get("num_workers", 4), drop_last=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=config["train"].get("num_workers", 4), drop_last=False)

    return train_loader, val_loader, test_loader


class _TransformSwitchableDataset(torch.utils.data.Dataset):
    """
    Thin wrapper so the SAME underlying PairedRestorationDataset instance can
    back multiple Subsets with different (or no) augmentation, without one
    split's DataLoader workers racing to mutate a shared `.transform`
    attribute on the base dataset.
    """

    def __init__(self, base_dataset, train_transform, apply_transform: bool):
        self.base_dataset = base_dataset
        self.train_transform = train_transform
        self.apply_transform = apply_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        original_transform = self.base_dataset.transform
        self.base_dataset.transform = self.train_transform if self.apply_transform else None
        try:
            item = self.base_dataset[idx]
        finally:
            self.base_dataset.transform = original_transform
        return item


def append_metrics_row(csv_path: str, row: dict) -> None:
    fieldnames = [
        "run_name", "config_file", "epochs_trained", "val_psnr", "val_ssim",
        "val_lpips", "inference_ms_per_img", "num_params", "timestamp",
    ]
    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    config = load_config(args.config)

    overrides = {
        "data.data_dir": args.data_dir,
        "optim.epochs": args.epochs,
        "train.batch_size": args.batch_size,
        "optim.lr": args.lr,
        "run_name": args.run_name,
    }
    config = apply_cli_overrides(config, overrides)

    if not config["data"].get("data_dir"):
        raise ValueError("data.data_dir must be set in the config or passed via --data_dir.")

    run_name = config["run_name"]
    seed = config["data"]["split"].get("seed", 42)
    set_seed(seed)

    device = get_device(config["train"].get("device", "auto"))
    print(f"[train.py] Using device: {device}")

    batch_size = config["train"]["batch_size"]
    max_oom_retries = 3
    retries = 0
    train_loader = val_loader = test_loader = None
    while retries <= max_oom_retries:
        try:
            train_loader, val_loader, test_loader = build_dataloaders(config)
            break
        except torch.cuda.OutOfMemoryError:
            new_bs = max(1, batch_size // config["train"].get("oom_batch_size_reduction_factor", 2))
            print(f"[train.py] WARNING: CUDA OOM while preparing data with batch_size={batch_size}. "
                  f"Reducing to {new_bs} and retrying.")
            batch_size = new_bs
            config["train"]["batch_size"] = batch_size
            retries += 1

    print(f"[train.py] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)} batch_size={batch_size}")

    model = build_model_from_config(config).to(device)
    num_params = count_parameters(model)
    print(f"[train.py] Model parameters: {num_params:,}")

    criterion = build_loss_from_config(config).to(device)

    optim_cfg = config["optim"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"])
    epochs = optim_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ckpt_cfg = config["checkpoint"]
    os.makedirs(ckpt_cfg["dir"], exist_ok=True)
    best_ssim = -float("inf")

    comparisons_every = config["train"].get("comparison_grid_every_n_epochs", 5)
    num_comparison_samples = config["train"].get("num_comparison_samples", 4)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for degraded, gt, _meta in pbar:
            degraded = degraded.to(device)
            gt = gt.to(device)

            try:
                optimizer.zero_grad()
                pred = model(degraded)
                loss, breakdown = criterion(pred, gt)
                loss.backward()
                optimizer.step()
            except torch.cuda.OutOfMemoryError:
                new_bs = max(1, config["train"]["batch_size"] // config["train"].get("oom_batch_size_reduction_factor", 2))
                print(f"\n[train.py] WARNING: CUDA OOM during training step. Reducing batch_size to {new_bs} "
                      f"and rebuilding dataloaders. Restarting current epoch.")
                torch.cuda.empty_cache()
                config["train"]["batch_size"] = new_bs
                train_loader, val_loader, test_loader = build_dataloaders(config)
                break

            running_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / max(1, n_batches)
        scheduler.step()

        if epoch % config["train"].get("val_every_n_epochs", 1) == 0:
            val_metrics = evaluate_batch(model, val_loader, device, compute_lpips_metric=True)
            print(f"[train.py] Epoch {epoch}: train_loss={avg_train_loss:.4f} "
                  f"val_psnr={val_metrics['psnr']:.3f} val_ssim={val_metrics['ssim']:.4f} "
                  f"val_lpips={val_metrics['lpips']:.4f}")

            if val_metrics["ssim"] > best_ssim:
                best_ssim = val_metrics["ssim"]
                save_checkpoint(
                    os.path.join(ckpt_cfg["dir"], ckpt_cfg["best_by_ssim_filename"]),
                    model, optimizer, scheduler, epoch,
                    extra={"val_metrics": val_metrics, "config": config},
                )
                print(f"[train.py] New best val SSIM ({best_ssim:.4f}) -> saved best_by_ssim.pt")

        save_checkpoint(
            os.path.join(ckpt_cfg["dir"], ckpt_cfg["last_filename"]),
            model, optimizer, scheduler, epoch,
            extra={"config": config},
        )

        if epoch % comparisons_every == 0 or epoch == epochs:
            try:
                degraded, gt, _meta = next(iter(val_loader))
                degraded = degraded.to(device)
                gt = gt.to(device)
                with torch.no_grad():
                    pred = model(degraded)
                out_path = os.path.join("results", "comparisons", f"{run_name}_epoch{epoch}.png")
                save_comparison_grid(degraded, pred, gt, out_path, max_samples=num_comparison_samples)
            except StopIteration:
                pass

    # ------------------------------------------------------------------
    # Final evaluation + logging
    # ------------------------------------------------------------------
    final_val_metrics = evaluate_batch(model, val_loader, device, compute_lpips_metric=True)
    inference_ms = measure_inference_time(model, test_loader, device, num_warmup=5)

    logging_cfg = config["logging"]
    row = {
        "run_name": run_name,
        "config_file": args.config,
        "epochs_trained": epochs,
        "val_psnr": round(final_val_metrics["psnr"], 4),
        "val_ssim": round(final_val_metrics["ssim"], 4),
        "val_lpips": round(final_val_metrics["lpips"], 4),
        "inference_ms_per_img": round(inference_ms, 4),
        "num_params": num_params,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    append_metrics_row(logging_cfg["metrics_csv"], row)
    dump_effective_config(config, run_name, logging_cfg["configs_used_dir"])

    print(f"[train.py] Done. Final val metrics: {final_val_metrics}, "
          f"inference: {inference_ms:.3f} ms/img. Logged to {logging_cfg['metrics_csv']}.")


if __name__ == "__main__":
    main()
