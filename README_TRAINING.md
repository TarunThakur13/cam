Fine-tuning pipeline

1) Collect clips using `data_collector.py`:
   - Run:

```powershell
.\.venv\Scripts\Activate.ps1
python data_collector.py --out data --clip-len 16 --interval 0.08
```

   - In the running window: press Enter to type a label, then press `r` to record a clip.
   - Save at least a few dozen clips per class for reasonable fine-tuning.

2) Train the model on the collected data:

```powershell
.\.venv\Scripts\Activate.ps1
python train.py --data data --out models --epochs 10 --batch-size 4 --lr 1e-4
```

3) After training, copy the best checkpoint into your app and adjust inference to load it:

```py
ckpt = torch.load('models/best_epochX_accY.pth', map_location='cpu')
model = r3d_18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, len(ckpt['classes']))
model.load_state_dict(ckpt['model_state'])
model.eval()
```

Notes:
- This is a small, opinionated pipeline for quick transfer learning with limited data.
- For better results: augment data, collect more samples, and consider fine-tuning more layers.
- If you have a GPU, training will be much faster. Ensure CUDA is available and Torch installed with CUDA support.
