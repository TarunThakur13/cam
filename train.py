"""
Train script to fine-tune R3D-18 on collected .npy clips organized by class folders.

Directory layout:
  data/
    FALL/
      FALL_...npy
    FIGHT/
      FIGHT_...npy
    NORMAL/
      NORMAL_...npy

Usage example:
  python train.py --data data --out models --epochs 10 --batch-size 8

This script uses the same preprocessing as the inference code (resize to 224, ImageNet normalization).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import r3d_18, R3D_18_Weights
from torchvision import transforms
from tqdm import tqdm
import torch.cuda.amp as amp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class ClipDataset(Dataset):
    def __init__(self, root: str, clip_len: int = 16):
        self.root = Path(root)
        self.clip_len = clip_len
        self.items = []  # list of (path, class_idx)
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls in self.classes:
            for f in (self.root / cls).glob('*.npy'):
                self.items.append((f, self.class_to_idx[cls]))

        # We will apply simple augmentations in __getitem__ for video clips
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        clip = np.load(str(path))  # (T,H,W,C)
        # if clip length mismatches, try simple trimming/padding
        if clip.shape[0] > self.clip_len:
            clip = clip[-self.clip_len:]
        elif clip.shape[0] < self.clip_len:
            pad_count = self.clip_len - clip.shape[0]
            pad = np.repeat(clip[-1:,...], pad_count, axis=0)
            clip = np.concatenate([clip, pad], axis=0)
        # Simple augmentations: random horizontal flip, random brightness
        if random.random() < 0.5:
            clip = clip[:, :, ::-1, :]
        # brightness jitter
        if random.random() < 0.3:
            factor = 0.8 + 0.4 * random.random()
            clip = np.clip(clip * factor, 0, 255).astype(np.uint8)

        # normalize and to tensor: (T,H,W,C) -> (C,T,H,W)
        clip = clip.astype('float32') / 255.0
        clip = (clip - self.mean[None, None, None, :]) / self.std[None, None, None, :]
        tensor = torch.tensor(clip).permute(3, 0, 1, 2)  # (C,T,H,W)
        return tensor, label


def build_model(num_classes: int, device: torch.device):
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    # replace head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.to(device)
    return model


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = ClipDataset(args.data, clip_len=args.clip_len)
    classes = ds.classes
    print(f'Classes: {classes}')

    # simple train/val split
    n = len(ds)
    idxs = list(range(n))
    random.shuffle(idxs)
    split = int(n * 0.85)
    train_idx, val_idx = idxs[:split], idxs[split:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds   = torch.utils.data.Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(len(classes), device)
    # compute class weights
    counts = [0] * len(classes)
    for _, lbl in ds.items:
        counts[lbl] += 1
    counts = np.array(counts, dtype=np.float32)
    class_weights = torch.tensor((counts.max() / (counts + 1e-6)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    scaler = amp.GradScaler()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [train]')
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            with amp.autocast():
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * xb.size(0)
            pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * xb.size(0)))

        # validation
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc='Validating'):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(yb.cpu().numpy().tolist())
        acc = correct / max(1, total)
        print(f'Epoch {epoch} val acc: {acc:.3f}')

        # scheduler step
        scheduler.step(1.0 - acc)

        # compute confusion matrix and print report
        if len(all_labels) > 0:
            cm = confusion_matrix(all_labels, all_preds)
            print('Classification report:')
            print(classification_report(all_labels, all_preds, target_names=classes))
            # save heatmap
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, ax=ax)
            fig.tight_layout()
            fig.savefig(Path(args.out) / f'confusion_epoch{epoch}.png')
            plt.close(fig)

        # save best
        if acc > best_acc:
            best_acc = acc
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            ckpt = out / f'best_epoch{epoch}_acc{acc:.3f}.pth'
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'classes': classes}, ckpt)
            print('Saved', ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data dir')
    parser.add_argument('--out', default='models', help='Output folder for checkpoints')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--clip-len', type=int, default=16)
    args = parser.parse_args()
    train(args)
