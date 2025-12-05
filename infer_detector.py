"""Inference script for MOH-DETECTOR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from astropy.io import fits
import torch

from config import CFG
from moh_detector.model import MohUNet
from moh_detector.metrics import heatmap_to_boxes, draw_boxes
from moh_detector.datasets import _normalize_sci_image  # type: ignore

BBox = Tuple[float, float, float, float]


def load_model(weights_path: Path, device: torch.device) -> MohUNet:
    model = MohUNet(in_channels=1)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image: np.ndarray, img_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    h, w = image.shape
    norm = _normalize_sci_image(image)
    pil_img = Image.fromarray((norm * 255).astype(np.uint8)).resize((img_size, img_size), Image.BILINEAR)
    tensor = torch.from_numpy(np.array(pil_img, dtype=np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
    return tensor, (h, w)


def postprocess_boxes(boxes: List[BBox], orig_size: Tuple[int, int], img_size: int) -> List[BBox]:
    orig_h, orig_w = orig_size
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size
    return [(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y) for x1, y1, x2, y2 in boxes]


def infer_on_array(array: np.ndarray, model: MohUNet, device: torch.device, img_size: int) -> Tuple[List[BBox], np.ndarray]:
    tensor, orig_size = preprocess_image(array, img_size)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
    boxes = heatmap_to_boxes(probs, threshold=0.4)
    boxes = postprocess_boxes(boxes, orig_size, img_size)
    return boxes, probs


def infer_on_fits(fits_path: Path, model: MohUNet, device: torch.device, img_size: int,
                  out_png: Path, out_json: Path) -> None:
    data = fits.getdata(fits_path)
    if data.ndim > 2:
        data = data[0]
    boxes, heatmap = infer_on_array(np.array(data, dtype=np.float32), model, device, img_size)
    vis = Image.fromarray((heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)).convert("RGB")
    vis = draw_boxes(vis, boxes)
    vis.save(out_png)
    out_json.write_text(json.dumps([{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": 1.0} for x1, y1, x2, y2 in boxes], indent=2))


def infer_on_png(png_path: Path, model: MohUNet, device: torch.device, img_size: int,
                 out_png: Path, out_json: Path) -> None:
    image = Image.open(png_path).convert("L")
    array = np.array(image, dtype=np.float32)
    boxes, heatmap = infer_on_array(array, model, device, img_size)
    vis = Image.fromarray((heatmap / (heatmap.max() + 1e-6) * 255).astype(np.uint8)).convert("RGB")
    vis = draw_boxes(vis, boxes)
    vis.save(out_png)
    out_json.write_text(json.dumps([{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": 1.0} for x1, y1, x2, y2 in boxes], indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MOH-DETECTOR inference")
    parser.add_argument("--weights", type=Path, default=Path("best_model.pt"))
    parser.add_argument("--fits", type=Path, help="Path to FITS file", default=None)
    parser.add_argument("--png", type=Path, help="Path to PNG file", default=None)
    parser.add_argument("--out-png", type=Path, default=Path("detections.png"))
    parser.add_argument("--out-json", type=Path, default=Path("detections.json"))
    parser.add_argument("--img-size", type=int, default=CFG.img_size)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(args.weights, device)
    if args.fits is not None:
        infer_on_fits(args.fits, model, device, args.img_size, args.out_png, args.out_json)
    elif args.png is not None:
        infer_on_png(args.png, model, device, args.img_size, args.out_png, args.out_json)
    else:
        raise ValueError("Provide either --fits or --png input")
    print(f"Saved results to {args.out_png} and {args.out_json}")


if __name__ == "__main__":
    main()
