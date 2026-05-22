"""
Evaluation script to run trained checkpoint on a dataset folder and print metrics.
Usage:
  python eval.py --ckpt models/best.pth --data data --clip-len 16
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from train import ClipDataset, build_model
from torchvision.models.video import r3d_18
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    classes = ckpt.get('classes')
    model = r3d_18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    return model, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--clip-len', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes = load_checkpoint(args.ckpt, device)

    ds = ClipDataset(args.data, clip_len=args.clip_len)
    idxs = list(range(len(ds)))

    y_true = []
    y_pred = []
    for i in idxs:
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            p = logits.argmax(dim=1).cpu().item()
        y_true.append(y)
        y_pred.append(p)

    print(classification_report(y_true, y_pred, target_names=ds.classes))
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=ds.classes, yticklabels=ds.classes, ax=ax)
    fig.tight_layout()
    fig.savefig('confusion_eval.png')
    print('Saved confusion_eval.png')

if __name__ == '__main__':
    main()
