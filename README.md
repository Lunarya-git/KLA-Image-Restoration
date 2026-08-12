# KLA Image Restoration

## 1. Problem summary

Semiconductor inspection images are frequently degraded by speckle noise, Gaussian noise, and spatial downsampling (e.g. a 512x512 ground-truth image arriving as a 256x256 or 128x128 degraded input), any combination of which can hide a defect that would otherwise fail a chip. This repo trains a single grayscale image-restoration model that removes noise and performs 2x super-resolution simultaneously -- taking a degraded, low-resolution, possibly noise-clipped input and producing a restored image matching the ground truth as closely as possible, while generalizing to out-of-distribution (unseen-source) test images and keeping inference fast.

## 2. Repo structure

```
KLA-Image-Restoration/
├── README.md                  # this file
├── requirements.txt           # pip dependencies
├── .gitignore
├── train.py                   # training entrypoint (CLI)
├── evaluate.py                # standalone inference entrypoint (CLI) -- the benchmarking contract
├── inspect_data.py            # Day-1 data exploration tool, not part of the pipeline
├── model.py                   # RestorationNet architecture
├── losses.py                  # CombinedLoss (configurable weighted loss terms)
├── dataset.py                 # PairedRestorationDataset, augmentation, fixed-seed splitting
├── metrics.py                 # PSNR / SSIM / LPIPS / inference-time benchmarking
├── utils.py                   # seeding, config loading, checkpoint I/O, comparison grids
├── configs/
│   └── baseline.yaml          # every tunable value for the baseline experiment
├── notebooks/
│   └── kaggle_train_runner.ipynb   # thin Kaggle runner, calls train.py -- no duplicated logic
├── weights/                   # checkpoints land here (best_by_ssim.pt, last.pt, final_model.pt)
├── outputs/                   # scratch space for ad hoc script output
└── results/
    ├── metrics.csv            # one row appended per training run
    ├── configs_used/          # exact config used for each run_name, for reproducibility
    └── comparisons/           # periodic degraded | restored | GT visual grids
```

## 3. Setup

Tested with Python 3.10+. From a fresh clone:

```bash
git clone https://github.com/Lunarya-git/KLA-Image-Restoration.git
cd KLA-Image-Restoration

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` lists the packages needed to run the pipeline (torch, torchvision, numpy, pillow, scikit-image, lpips, tqdm, matplotlib, pyyaml, tifffile). **Before final hackathon submission, regenerate this file from your actual training environment** with `pip freeze > requirements.txt`, since KLA's reproducibility requirement asks for the exact frozen environment, not just the top-level package list.

## 4. Inspecting a new data root (Day 1)

Before trusting any assumption baked into `dataset.py`, run:

```bash
python inspect_data.py --data_root /path/to/kla_data --sample_size 10
```

This prints: total pair count, per-sample dtype/shape/min/max for a sample of images, the detected bit depth, every degraded->GT resolution pair present in the data, and whether (and by how much) degraded pixel values exceed the GT's range -- which is expected due to speckle noise, but you should confirm the actual numbers before training.

`dataset.py` assumes a `data_root/degraded/` and `data_root/gt/` folder layout with matching filenames (`.npy` arrays are the primary supported format, since that's what the KLA dataset appears to ship as, with common image formats like `.png`/`.tif`/`.jpg` also supported). If `inspect_data.py` shows the real data uses a different convention, edit only the clearly marked `_discover_pairs` method in `dataset.py` -- normalization, augmentation, splitting, and everything downstream is convention-agnostic.

## 5. Running a training experiment

**Locally:**

```bash
python train.py --config configs/baseline.yaml --data_dir /path/to/kla_data
```

Optional CLI overrides (each overrides the matching YAML field only when passed):

```bash
python train.py --config configs/baseline.yaml --data_dir /path/to/kla_data \
    --epochs 50 --batch_size 8 --lr 1e-4 --run_name my_experiment
```

**On Kaggle** (open `notebooks/kaggle_train_runner.ipynb` in a Kaggle notebook session with GPU enabled): the notebook clones this repo, installs requirements, then runs

```bash
!python train.py --config configs/baseline.yaml --data_dir /kaggle/input/<dataset-slug>
```

The KLA dataset currently lives in Google Drive rather than as a Kaggle Dataset -- upload it to Kaggle first (Kaggle -> "New Dataset", or the notebook's "Add Data" panel) and swap `<dataset-slug>` for the real slug. `train.py` runs identically regardless of whether `--data_dir` points at a local folder or a `/kaggle/input/...` mount; there are no hardcoded absolute paths anywhere in the pipeline.

## 6. How the config system works (onboarding for the rest of the team)

Every tunable value -- split ratios, seed, normalization mode, augmentation flags, model depth/width, loss terms and weights, optimizer/scheduler settings, batch size, checkpoint/logging paths -- lives in a YAML file under `configs/`, not hardcoded in Python. `configs/baseline.yaml` is fully commented; read it top to bottom once.

**To run a new experiment without touching any core code:**

1. Copy `configs/baseline.yaml` to e.g. `configs/my_experiment.yaml`.
2. Change whatever fields you want (loss terms, model width, augmentation, epochs, etc.) and set a unique `run_name`.
3. Run `python train.py --config configs/my_experiment.yaml --data_dir <path>`.

Every run appends exactly one row to `results/metrics.csv` (columns: `run_name, config_file, epochs_trained, val_psnr, val_ssim, val_lpips, inference_ms_per_img, num_params, timestamp`) and copies the exact effective config (including any CLI overrides) to `results/configs_used/<run_name>.yaml`, so any past run is fully reproducible and comparable. There is no external experiment tracker (no MLflow, no W&B) -- the CSV is the single source of truth.

Swappable loss terms (`losses.py`) currently support `l1`, `charbonnier`, and `ssim` by name in the config's `loss.terms` list, plus an optional LPIPS term (`loss.lpips.enabled`) kept off by default since it adds compute cost. Add a new loss by registering it in `losses.py`'s `_TERM_REGISTRY` and referencing it by name in a config -- no other file needs to change.

## 7. Running evaluation (KLA's benchmarking contract)

```bash
python evaluate.py --input_dir /path/to/test_images --output_dir /path/to/save_restored --weights weights/final_model.pt
```

`--weights` defaults to `weights/final_model.pt` if omitted. This script is intentionally self-contained (it does not import `dataset.py`, `losses.py`, or anything training-only) so it cannot be broken by unrelated changes elsewhere in the repo, matches KLA's exact CLI contract (`--input_dir`, `--output_dir`, weights defaulting to `weights/final_model.pt`), auto-detects CPU vs GPU, reads every supported file in `input_dir` (`.npy`, `.png`, `.tif`, `.tiff`, `.jpg`, `.jpeg`, `.bmp`), writes each restored output to `output_dir` under the same filename, and prints total images processed plus average inference time in ms/image at the end.

## 8. Where results land, and how to read `results/metrics.csv`

- **Checkpoints** (`weights/`): every run saves `best_by_ssim.pt` (best validation SSIM seen during that run) and `last.pt` (final epoch, useful for resuming). `weights/final_model.pt` is never written automatically by `train.py` -- a human reviews `results/metrics.csv`, picks the best run, and manually copies/renames its checkpoint to `weights/final_model.pt`. This is the file `evaluate.py` uses by default, and the file KLA's benchmarking team will actually run against.
- **Metrics log** (`results/metrics.csv`): one row per training run. Sort by `val_ssim` (or `val_psnr` / `val_lpips`, lower-is-better for LPIPS) to compare experiments; `inference_ms_per_img` and `num_params` let you weigh restoration quality against the speed requirement KLA benchmarks on.
- **Configs used** (`results/configs_used/<run_name>.yaml`): exact config for a given `run_name`, for reproducing or auditing any past result.
- **Visual comparisons** (`results/comparisons/`): periodic degraded | restored | GT grids saved during training, for a fast sanity check that the model is actually learning to restore rather than just minimizing a metric.

## 9. Current best model

_Placeholder -- fill in after Day 1 training runs on the real data._

| Run name | Epochs | Val PSNR | Val SSIM | Val LPIPS | Inference (ms/img) | Params |
|---|---|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

### Notes on the KLA hackathon submission requirements

This repo is structured to satisfy the mandatory GitHub repository contents from the hackathon brief:

- **README with complete setup instructions** -- this file (sections 3, 4, 5, 7 above cover clone -> install -> inspect -> train -> evaluate end to end).
- **Standalone evaluation script (not a notebook)** -- `evaluate.py`, matching the exact `--input_dir` / `--output_dir` / `--weights` CLI contract, is the file KLA's benchmarking team will run as-is on their H100.
- **Training script** -- `train.py` (CLI) plus `notebooks/kaggle_train_runner.ipynb` (thin Kaggle wrapper around the same script) reproduce training from scratch.
- **Trained model weights** -- `weights/final_model.pt` is the slot for this; it must be populated by manually promoting a checkpoint after a real training run (see section 8), and is intentionally excluded from `.gitignore`'s checkpoint rule so it stays tracked once added. If the file is too large for a normal git push, use Git LFS or link to Google Drive/HuggingFace from this README.
- **Restored test-set outputs** -- generate these with `evaluate.py` against the released test set once available, and commit the resulting output folder (or link it, same large-file caveat as above).
- **requirements.txt** -- see the note in section 3 about replacing the curated list here with a full `pip freeze` from the actual training environment before submitting.
