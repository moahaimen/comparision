"""Dataset creation utilities and PyTorch dataset classes for MOH-DETECTOR."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from astropy.io import fits
from skimage import measure
import torch
from torch.utils.data import Dataset

from config import CFG

BBox = Tuple[float, float, float, float]


def _normalize_sci_image(data: np.ndarray) -> np.ndarray:
    """Normalize a science image to the range [0, 1]."""
    median = np.median(data)
    std = np.std(data) + 1e-6
    norm = (data - median) / std
    norm = np.clip(norm, -5, 5)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)
    return norm.astype(np.float32)


def _detect_blobs(norm: np.ndarray, thr_sigma: float, max_area: int) -> List[BBox]:
    """Detect bright blobs using a simple threshold and connected components."""
    threshold = norm.mean() + thr_sigma * norm.std()
    mask = norm > threshold
    labeled = measure.label(mask, connectivity=2)
    props = measure.regionprops(labeled)
    boxes: List[BBox] = []
    h, w = norm.shape
    for prop in props:
        minr, minc, maxr, maxc = prop.bbox
        area = (maxr - minr) * (maxc - minc)
        if area == 0 or area > max_area:
            continue
        minc = max(minc, 0)
        minr = max(minr, 0)
        maxc = min(maxc, w)
        maxr = min(maxr, h)
        boxes.append((float(minc), float(minr), float(maxc), float(maxr)))
    return boxes


def _resize_image_and_boxes(image: np.ndarray, boxes: Sequence[BBox], size: int) -> Tuple[Image.Image, List[BBox]]:
    """Resize image and boxes to a square of given size."""
    h, w = image.shape
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    pil_img = pil_img.resize((size, size), resample=Image.BILINEAR)
    scale_x = size / w
    scale_y = size / h
    resized_boxes: List[BBox] = []
    for x1, y1, x2, y2 in boxes:
        resized_boxes.append((x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y))
    return pil_img, resized_boxes


def _boxes_to_yolo_lines(boxes: Sequence[BBox], img_size: int) -> List[str]:
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2.0 / img_size
        cy = (y1 + y2) / 2.0 / img_size
        w = (x2 - x1) / img_size
        h = (y2 - y1) / img_size
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def _save_example_overlay(img: Image.Image, boxes: Sequence[BBox], out_path: Path) -> None:
    draw = ImageDraw.Draw(img)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
    img.save(out_path)


def build_moh_detector_dataset(alerts_root: Path, data_root: Path, *,
                               thr_sigma: float | None = None, img_size: int | None = None,
                               val_split: float | None = None, max_blob_area: int | None = None,
                               save_overlays: bool = False) -> None:
    """Convert ZTF alert folders into YOLO-style dataset.

    Args:
        alerts_root: Directory containing subfolders with FITS files.
        data_root: Output dataset root containing images/ and labels/.
        thr_sigma: Sigma multiplier for blob thresholding.
        img_size: Target square image size.
        val_split: Fraction of data for training split; remainder used for validation.
        max_blob_area: Ignore blobs with area larger than this.
        save_overlays: Optionally save debug overlays next to images.
    """

    thr_sigma = thr_sigma or CFG.thr_sigma
    img_size = img_size or CFG.img_size
    val_split = val_split or CFG.train_split
    max_blob_area = max_blob_area or CFG.max_blob_area

    image_dir = data_root / "images"
    label_dir = data_root / "labels"
    for split in ["train", "val"]:
        (image_dir / split).mkdir(parents=True, exist_ok=True)
        (label_dir / split).mkdir(parents=True, exist_ok=True)

    alert_folders = sorted([p for p in alerts_root.iterdir() if p.is_dir()])
    if not alert_folders:
        raise FileNotFoundError(f"No alert folders found in {alerts_root}")

    rng = np.random.default_rng(42)
    for alert_path in alert_folders:
        fits_files = list(alert_path.glob("*sci*.fits")) or list(alert_path.glob("*.fits"))
        if not fits_files:
            continue
        fits_path = fits_files[0]
        data = fits.getdata(fits_path)
        if data.ndim > 2:
            data = data[0]
        norm = _normalize_sci_image(np.array(data, dtype=np.float32))
        boxes = _detect_blobs(norm, thr_sigma=thr_sigma, max_area=max_blob_area)
        if not boxes:
            continue
        pil_img, resized_boxes = _resize_image_and_boxes(norm, boxes, img_size)
        split = "train" if rng.random() < val_split else "val"
        img_out = image_dir / split / f"{alert_path.name}.png"
        lbl_out = label_dir / split / f"{alert_path.name}.txt"
        pil_img.save(img_out)
        lines = _boxes_to_yolo_lines(resized_boxes, img_size)
        lbl_out.write_text("\n".join(lines) + ("\n" if lines else ""))
        if save_overlays:
            overlay = pil_img.copy()
            _save_example_overlay(overlay, resized_boxes, img_out.with_suffix("_overlay.png"))


def build_patch_dataset_from_full(full_img_dir: Path, full_lbl_dir: Path,
                                  patch_img_dir: Path, patch_lbl_dir: Path,
                                  patch_size: int, max_patches_per_image: int) -> None:
    """Generate a patch dataset centered on detections from full-frame labels."""

    patch_img_dir.mkdir(parents=True, exist_ok=True)
    patch_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(full_img_dir.glob("*.png"))
    for img_path in img_files:
        lbl_path = full_lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        image = Image.open(img_path).convert("L")
        img_w, img_h = image.size
        lines = [ln.strip() for ln in lbl_path.read_text().splitlines() if ln.strip()]
        boxes: List[BBox] = []
        for line in lines:
            _, cx, cy, w, h = map(float, line.split())
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append((x1, y1, x2, y2))

        for i, box in enumerate(boxes[:max_patches_per_image]):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            half = patch_size / 2
            left = int(max(cx - half, 0))
            top = int(max(cy - half, 0))
            right = int(min(left + patch_size, img_w))
            bottom = int(min(top + patch_size, img_h))
            patch = image.crop((left, top, right, bottom))
            # Adjust label into patch coordinates
            patch_w, patch_h = patch.size
            bx1 = max(x1 - left, 0)
            by1 = max(y1 - top, 0)
            bx2 = min(x2 - left, patch_w)
            by2 = min(y2 - top, patch_h)
            pcx = (bx1 + bx2) / 2 / patch_w
            pcy = (by1 + by2) / 2 / patch_h
            pw = (bx2 - bx1) / patch_w
            ph = (by2 - by1) / patch_h
            patch.save(patch_img_dir / f"{img_path.stem}_patch{i}.png")
            (patch_lbl_dir / f"{img_path.stem}_patch{i}.txt").write_text(
                f"0 {pcx:.6f} {pcy:.6f} {pw:.6f} {ph:.6f}\n"
            )


@dataclass
class YoloLabel:
    boxes: torch.Tensor  # (N, 4) in xyxy absolute pixel coords
    classes: torch.Tensor  # (N,)


class FullFrameYoloDataset(Dataset):
    """YOLO-style dataset of full-frame PNGs and labels."""

    def __init__(self, root: Path, split: str, img_size: int) -> None:
        self.root = root
        self.split = split
        self.img_size = img_size
        self.images = sorted((root / "images" / split).glob("*.png"))
        self.labels = root / "labels" / split
        if not self.images:
            raise FileNotFoundError(f"No images found in {root}/images/{split}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, YoloLabel]:
        img_path = self.images[idx]
        lbl_path = self.labels / f"{img_path.stem}.txt"
        image = Image.open(img_path).convert("L").resize((self.img_size, self.img_size))
        img_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)
        boxes, classes = self._load_labels(lbl_path)
        return img_tensor, YoloLabel(boxes=boxes, classes=classes)

    def _load_labels(self, lbl_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes: List[List[float]] = []
        classes: List[int] = []
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, w, h = map(float, line.split())
                x1 = (cx - w / 2) * self.img_size
                y1 = (cy - h / 2) * self.img_size
                x2 = (cx + w / 2) * self.img_size
                y2 = (cy + h / 2) * self.img_size
                boxes.append([x1, y1, x2, y2])
                classes.append(int(cls))
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)


class PatchYoloDataset(FullFrameYoloDataset):
    """Dataset for patch-based samples using the same label format."""

    def __init__(self, root: Path, split: str, img_size: int) -> None:
        super().__init__(root, split, img_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, YoloLabel]:
        img_path = self.images[idx]
        lbl_path = self.labels / f"{img_path.stem}.txt"
        image = Image.open(img_path).convert("L").resize((self.img_size, self.img_size))
        img_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)
        boxes, classes = self._load_labels(lbl_path)
        return img_tensor, YoloLabel(boxes=boxes, classes=classes)


__all__ = [
    "build_moh_detector_dataset",
    "build_patch_dataset_from_full",
    "FullFrameYoloDataset",
    "PatchYoloDataset",
    "YoloLabel",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MOH-DETECTOR datasets")
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build YOLO dataset from alerts")
    build_parser.add_argument("--alerts-root", type=Path, default=CFG.alerts_root)
    build_parser.add_argument("--data-root", type=Path, default=CFG.dataset_root)
    build_parser.add_argument("--thr-sigma", type=float, default=CFG.thr_sigma)
    build_parser.add_argument("--img-size", type=int, default=CFG.img_size)
    build_parser.add_argument("--val-split", type=float, default=CFG.train_split)
    build_parser.add_argument("--max-blob-area", type=int, default=CFG.max_blob_area)
    build_parser.add_argument("--save-overlays", action="store_true")

    patch_parser = subparsers.add_parser("patches", help="Generate patch dataset from YOLO labels")
    patch_parser.add_argument("--full-img-dir", type=Path, required=True)
    patch_parser.add_argument("--full-lbl-dir", type=Path, required=True)
    patch_parser.add_argument("--patch-img-dir", type=Path, required=True)
    patch_parser.add_argument("--patch-lbl-dir", type=Path, required=True)
    patch_parser.add_argument("--patch-size", type=int, default=CFG.patch_size)
    patch_parser.add_argument("--max-patches", type=int, default=CFG.max_patches_per_image)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    if args.command == "build":
        build_moh_detector_dataset(
            alerts_root=args.alerts_root,
            data_root=args.data_root,
            thr_sigma=args.thr_sigma,
            img_size=args.img_size,
            val_split=args.val_split,
            max_blob_area=args.max_blob_area,
            save_overlays=args.save_overlays,
        )
    elif args.command == "patches":
        build_patch_dataset_from_full(
            full_img_dir=args.full_img_dir,
            full_lbl_dir=args.full_lbl_dir,
            patch_img_dir=args.patch_img_dir,
            patch_lbl_dir=args.patch_lbl_dir,
            patch_size=args.patch_size,
            max_patches_per_image=args.max_patches,
        )
    else:
        raise SystemExit("Specify a command: build or patches")


if __name__ == "__main__":
    _main()
