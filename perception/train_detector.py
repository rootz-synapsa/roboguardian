"""
G7 — Tiny plate-position detector, training script
======================================================

Deliberately tiny (~50K params, 64x64 input) so CPU training takes
minutes and OpenVINO compile/inference stays fast -- this is a
from-scratch model built for this exact task, not a repurposed
general-purpose detector.

Requires: pip install torch --break-system-packages (CPU build is fine,
no GPU needed for a model this small).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


class PlateDataset(Dataset):
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        meta = json.loads((data_dir / "labels.json").read_text())
        self.samples = meta["samples"]
        self.data_dir = data_dir

        xs = np.array([s["x"] for s in self.samples])
        ys = np.array([s["y"] for s in self.samples])
        self.x_mean, self.x_std = float(xs.mean()), float(xs.std() + 1e-6)
        self.y_mean, self.y_std = float(ys.mean()), float(ys.std() + 1e-6)

        meta_img_size = json.loads((data_dir / "labels.json").read_text())["img_size"]
        self.img_size = meta_img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = np.load(self.data_dir / "images" / s["file"]).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        x_norm = (s["x"] - self.x_mean) / self.x_std
        y_norm = (s["y"] - self.y_mean) / self.y_std
        return torch.from_numpy(img), torch.tensor([x_norm, y_norm], dtype=torch.float32)


class TinyDetector(nn.Module):
    """No global average pooling -- deliberately. GAP makes a network
    translation-invariant (great for classification, actively harmful
    for position regression: it discards exactly the spatial signal
    we need). Flatten the conv feature map instead so FC layers can
    learn where the object is, not just whether it's present."""

    def __init__(self, img_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
        )
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, 3, img_size, img_size)).numel()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/g7_dataset")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-onnx", default="results/plate_detector.onnx")
    parser.add_argument("--out-norm", default="results/plate_detector_norm.json")
    args = parser.parse_args()

    dataset = PlateDataset(args.data)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = TinyDetector(dataset.img_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            opt.zero_grad()
            pred = model(imgs)
            loss = loss_fn(pred, labels)
            loss.backward()
            opt.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                val_loss += loss_fn(model(imgs), labels).item() * imgs.size(0)
        val_loss /= n_val

        print(f"epoch {epoch + 1:2d}/{args.epochs}  train_mse={train_loss:.5f}  val_mse={val_loss:.5f}")

    out_onnx = Path(args.out_onnx)
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 3, dataset.img_size, dataset.img_size)
    torch.onnx.export(
        model, dummy, str(out_onnx),
        input_names=["image"], output_names=["xy_normalized"],
        opset_version=18,
    )

    meta = json.loads((Path(args.data) / "labels.json").read_text())
    Path(args.out_norm).write_text(json.dumps({
        "x_mean": dataset.x_mean, "x_std": dataset.x_std,
        "y_mean": dataset.y_mean, "y_std": dataset.y_std,
        "z_fixed": meta["z_fixed"],
        "img_size": meta["img_size"],
        # Preprocessing metadata carried through so VisionStateProvider
        # replicates the exact same camera/crop/resize pipeline used here
        # -- a mismatch here would silently make the model see different
        # input distribution at inference time than at training time.
        "camera": meta["camera"],
        "render_size": meta["render_size"],
        "crop_rows": meta["crop_rows"],
        "crop_cols": meta["crop_cols"],
    }, indent=2))

    print(f"\nExported ONNX model to: {out_onnx.resolve()}")
    print(f"Normalization stats:    {Path(args.out_norm).resolve()}")
    print("\nNOTE: final val_mse is in NORMALIZED units (roughly unit variance). "
          "Convert back with x_std/y_std from the norm file to judge real-world "
          "accuracy in meters before trusting this for G7 integration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
