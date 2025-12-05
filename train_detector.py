"""Training script for MOH-DETECTOR."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import CFG
from moh_detector.datasets import FullFrameYoloDataset, PatchYoloDataset, YoloLabel
from moh_detector.model import MohUNet
from moh_detector.losses import BCEWithLogits
from moh_detector.metrics import heatmap_to_boxes, precision_recall_ap


def collate_fn(batch: Tuple[torch.Tensor, YoloLabel]):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return images, labels


def make_heatmap_targets(boxes: torch.Tensor, size: int, sigma: int = 3) -> torch.Tensor:
    target = torch.zeros((size, size), dtype=torch.float32)
    for x1, y1, x2, y2 in boxes:
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        xmin = max(cx - sigma, 0)
        xmax = min(cx + sigma, size)
        ymin = max(cy - sigma, 0)
        ymax = min(cy + sigma, size)
        target[ymin:ymax, xmin:xmax] = 1.0
    return target


def train_one_epoch(model: MohUNet, loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        targets = []
        for label in labels:
            heatmap = make_heatmap_targets(label.boxes, images.shape[-1]).to(device)
            targets.append(heatmap)
        target_tensor = torch.stack(targets).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, target_tensor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / max(1, len(loader))


def evaluate(model: MohUNet, loader: DataLoader, device: torch.device) -> Tuple[float, float, float, float]:
    model.eval()
    criterion = BCEWithLogits()
    losses = []
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            targets = [make_heatmap_targets(lbl.boxes, images.shape[-1]).to(device) for lbl in labels]
            target_tensor = torch.stack(targets).unsqueeze(1)
            logits = model(images)
            loss = criterion(logits, target_tensor)
            losses.append(loss.item())
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(len(labels)):
                pred_boxes = heatmap_to_boxes(probs[i, 0], threshold=0.4)
                gt_boxes = labels[i].boxes.numpy().tolist()
                all_preds.append(pred_boxes)
                all_gts.append(gt_boxes)
    precision, recall, ap = precision_recall_ap(all_preds, all_gts)
    return float(np.mean(losses)), precision, recall, ap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MOH-DETECTOR")
    parser.add_argument("--data-root", type=Path, default=CFG.dataset_root, help="YOLO dataset root")
    parser.add_argument("--use-patches", action="store_true", help="Use patch dataset instead of full frames")
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument("--img-size", type=int, default=CFG.img_size)
    parser.add_argument("--lr", type=float, default=CFG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--num-workers", type=int, default=CFG.num_workers)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_cls = PatchYoloDataset if args.use_patches else FullFrameYoloDataset
    train_set = dataset_cls(args.data_root, "train", args.img_size)
    val_set = dataset_cls(args.data_root, "val", args.img_size)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = MohUNet(in_channels=1).to(device)
    criterion = BCEWithLogits()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ap = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, precision, recall, ap = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} P={precision:.3f} R={recall:.3f} AP@0.5={ap:.3f}")
        if ap > best_ap:
            best_ap = ap
            torch.save({
                "model_state_dict": model.state_dict(),
                "ap": ap,
                "epoch": epoch,
            }, "best_model.pt")
            print(f"Saved new best model at epoch {epoch} with AP {ap:.3f}")


if __name__ == "__main__":
    main()
