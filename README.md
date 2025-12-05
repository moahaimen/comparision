# MOH-DETECTOR

MOH-DETECTOR is a lightweight UNet-style detector for astronomical transients in ZTF alert images. It builds YOLO-format datasets from FITS science images, supports optional object-centered patch generation, and provides training and inference utilities.

## Quickstart

1. Place raw ZTF alert folders (each containing FITS files) under `data/alerts`.
2. Build the YOLO dataset:

   ```bash
   python -m moh_detector.datasets build --alerts-root data/alerts --data-root data/ztf_yolo
   ```

   or programmatically:

   ```python
   from pathlib import Path
   from moh_detector.datasets import build_moh_detector_dataset
   from config import CFG

   build_moh_detector_dataset(CFG.alerts_root, CFG.dataset_root)
   ```

3. (Optional) Generate patches to augment the dataset:

   ```python
   from pathlib import Path
   from moh_detector.datasets import build_patch_dataset_from_full
   from config import CFG

   build_patch_dataset_from_full(
       full_img_dir=CFG.dataset_root / "images" / "train",
       full_lbl_dir=CFG.dataset_root / "labels" / "train",
       patch_img_dir=CFG.dataset_root / "patches" / "images" / "train",
       patch_lbl_dir=CFG.dataset_root / "patches" / "labels" / "train",
       patch_size=CFG.patch_size,
       max_patches_per_image=CFG.max_patches_per_image,
   )
   ```

4. Train the detector on full frames or patches:

   ```bash
   python train_detector.py --data-root data/ztf_yolo --batch-size 4 --epochs 10
   # or
   python train_detector.py --data-root data/ztf_yolo/patches --use-patches
   ```

5. Run inference on a FITS or PNG file:

   ```bash
   python infer_detector.py --weights best_model.pt --fits path/to/file.fits --out-png detections.png --out-json detections.json
   # or
   python infer_detector.py --weights best_model.pt --png path/to/file.png --out-png detections.png --out-json detections.json
   ```

## Repository layout

- `config.py`: Central configuration values.
- `moh_detector/datasets.py`: Dataset building utilities and PyTorch datasets.
- `moh_detector/model.py`: UNet-like architecture for heatmap prediction.
- `moh_detector/losses.py`: BCE and Dice-style losses.
- `moh_detector/metrics.py`: Box conversion, IoU, and evaluation metrics.
- `train_detector.py`: Training loop with validation metrics and model checkpointing.
- `infer_detector.py`: Inference CLI that outputs annotated PNGs and JSON detections.
- `data/`: Placeholder for alerts and generated datasets.

## Notes

- All paths are relative to the repository root and can be overridden via CLI flags.
- The code targets Python 3.10+, PyTorch, NumPy, Astropy, Pillow, and scikit-image. Install dependencies with `pip install torch numpy astropy pillow scikit-image`.
