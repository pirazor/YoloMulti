"""Training loop for the fine-grained traffic-sign classifier."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sign_classifier.data import ImageFolderDataset, build_train_transform, build_val_transform
from sign_classifier.models import DEFAULT_MODEL, build_model

LOGGER = logging.getLogger("sign_classifier.train")


def train(
    data: Path,
    epochs: int = 30,
    batch: int = 64,
    imgsz: int = 96,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    workers: int = 4,
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    save_dir: Path | str = "runs/sign_classifier/exp",
    label_smoothing: float = 0.1,
) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = ImageFolderDataset(data / "train", transform=build_train_transform(imgsz))
    val_ds = ImageFolderDataset(data / "val", transform=build_val_transform(imgsz))
    classes = train_ds.classes
    LOGGER.info("classes (%d): %s", len(classes), classes)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=max(1, workers // 2), pin_memory=True)

    model = build_model(model_name, num_classes=len(classes)).to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=device_t.type == "cuda")

    best_acc = 0.0
    best_path = save_dir / "best.pt"
    last_path = save_dir / "last.pt"

    for ep in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for img, label in train_loader:
            img = img.to(device_t, non_blocking=True)
            label = label.to(device_t, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device_t.type == "cuda"):
                logits = model(img)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            running += loss.item() * img.size(0)
            n += img.size(0)
        scheduler.step()
        train_loss = running / max(n, 1)

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for img, label in val_loader:
                img = img.to(device_t)
                label = label.to(device_t)
                logits = model(img)
                correct += (logits.argmax(dim=1) == label).sum().item()
                total += img.size(0)
        acc = correct / max(total, 1)
        LOGGER.info("epoch %d/%d  train_loss=%.4f  val_acc=%.4f", ep + 1, epochs, train_loss, acc)

        ckpt = {"model_name": model_name, "state_dict": model.state_dict(), "classes": classes,
                "epoch": ep, "val_acc": acc}
        torch.save(ckpt, last_path)
        if acc > best_acc:
            best_acc = acc
            torch.save(ckpt, best_path)
    LOGGER.info("done. best val acc = %.4f. weights -> %s", best_acc, best_path)
    return best_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    p = argparse.ArgumentParser(description="Train the fine-grained traffic-sign classifier")
    p.add_argument("--data", type=Path, required=True, help="Root with train/<class>/ and val/<class>/")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--imgsz", type=int, default=96)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--save_dir", type=Path, default=Path("runs/sign_classifier/exp"))
    p.add_argument("--label_smoothing", type=float, default=0.1)
    args = p.parse_args(list(argv) if argv is not None else None)
    train(
        data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        lr=args.lr, weight_decay=args.weight_decay, workers=args.workers,
        model_name=args.model, device=args.device, save_dir=args.save_dir,
        label_smoothing=args.label_smoothing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
