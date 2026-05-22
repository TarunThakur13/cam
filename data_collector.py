"""
Simple webcam data collector for creating labelled clips for fine-tuning.
Saves fixed-length clips as .npy arrays under `data/{label}/`.

Usage:
  python data_collector.py --out data --clip-len 16 --interval 0.08

Controls (while running):
  - Type label name and press Enter to set current label
  - Press `r` to record one clip (will capture `clip_len` frames at the specified interval)
  - Press `q` or ESC to quit

"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import time


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def record_clip(cap, clip_len: int, sample_interval: float):
    frames = []
    t0 = time.time()
    while len(frames) < clip_len:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.resize(frame, (224, 224)))
        # wait for sample interval
        elapsed = time.time() - t0
        to_wait = sample_interval - (elapsed - (len(frames)-1) * sample_interval)
        if to_wait > 0:
            time.sleep(to_wait)
    return np.array(frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data', help='Output directory for collected clips')
    parser.add_argument('--clip-len', type=int, default=16)
    parser.add_argument('--interval', type=float, default=0.08, help='Seconds between sampled frames')
    parser.add_argument('--device', type=int, default=0, help='OpenCV camera index')
    args = parser.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print('Cannot open camera')
        return

    print('Camera opened. Press q to quit.')
    current_label = None
    counter = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame not available')
            break

        display = frame.copy()
        h, w = display.shape[:2]
        info = f'label={current_label or "(none)"} | press Enter to set label, r=record, q=quit'
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        cv2.imshow('Collector', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('\r') or key == 13:
            # set label from input() - blocking but acceptable for manual labeling
            lbl = input('Enter label name (e.g. FALL, FIGHT, NORMAL): ').strip()
            if lbl:
                current_label = lbl
                print(f'Label set to: {current_label}')
                ensure_dir(out_dir / current_label)
                counter.setdefault(current_label, 0)
        elif key == ord('r'):
            if not current_label:
                print('Set a label first (press Enter then type label)')
                continue
            print(f'Recording clip for label {current_label}...')
            clip = record_clip(cap, args.clip_len, args.interval)  # shape (T,H,W,C)
            if clip.shape[0] != args.clip_len:
                print('Clip too short, skipping')
                continue
            idx = counter.get(current_label, 0) + 1
            counter[current_label] = idx
            fname = out_dir / current_label / f'{current_label}_{int(time.time())}_{idx}.npy'
            np.save(str(fname), clip)
            print(f'Saved clip: {fname}')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
