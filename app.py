from __future__ import annotations
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import torch
import numpy as np
import threading
import time
import os
import json
from flask import Flask, render_template, Response, request, jsonify
import re
from torchvision.models.video import r3d_18, R3D_18_Weights
from werkzeug.utils import secure_filename
from collections import deque
from pathlib import Path

import os
os.environ['TORCH_HOME'] = './torch_cache'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/screenshots', exist_ok=True)

# Optimize CPU threads for inference
torch.set_num_threads(min(os.cpu_count() or 4, 8))

# ─── Inference / Accuracy Tuning ──────────────────────────────────────────
# Clip length used by the model (Kinetics R3D-18 trained at 16-frame clips)
CLIP_LEN = 16
# Minimum seconds between model inferences
INFERENCE_INTERVAL = 1.0
# Number of recent clip softmax vectors to average for smoothing
SMOOTHING_WINDOW = 3
# Confidence threshold required to trigger an alert for danger categories
ALERT_CONF_THRESHOLD = 0.60
# Minimum seconds to wait before allowing another alert (prevents sticky repeats)
ALERT_COOLDOWN = 3.0
# Whether to use simple test-time augmentation (horizontal flip)
ENABLE_TTA = True
# Operating sensitivity: 'balanced' or 'sensitive' (lower threshold)
DETECTION_MODE = 'balanced'  # options: 'balanced', 'sensitive'
# Danger score threshold across all fight/fall-related classes
DANGER_SCORE_THRESHOLD = 0.18

from statistics import mean
from collections import deque as _deque

# ─── Load Model ───────────────────────────────────────────────────────────────
print("Loading R3D-18 model...")
weights = R3D_18_Weights.DEFAULT
model = r3d_18(weights=weights)
model.eval()
labels = weights.meta["categories"]
print(f"Model loaded. {len(labels)} Kinetics-400 classes available.")

# Flag to indicate a custom fine-tuned model is loaded
custom_classes = None

def load_checkpoint(path: str):
    global model, labels, custom_classes
    ckpt = torch.load(path, map_location='cpu')
    classes = ckpt.get('classes')
    if classes is None:
        raise ValueError('Checkpoint missing `classes` list')
    m = r3d_18(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ckpt['model_state'])
    m.eval()
    model = m
    labels = classes
    custom_classes = classes
    print(f'Loaded custom checkpoint with {len(classes)} classes')


# ─── Action Mapping ───────────────────────────────────────────────────────────
FALL_KEYWORDS   = ['fall', 'stumbl', 'trip', 'collaps', 'toppl', 'tumbl', 'drop']
FIGHT_KEYWORDS  = ['punch', 'slap', 'wrestling', 'fight', 'boxing', 'martial',
                   'karate', 'kick', 'headbutt', 'brawl', 'taekwondo', 'judo',
                   'jiu jitsu', 'sword fight', 'shooting', 'attack', 'hit']
SIT_KEYWORDS    = ['sitting', 'crouch', 'squat', 'kneel', 'yoga', 'meditation']
STAND_KEYWORDS  = ['standing', 'walk', 'run', 'jump', 'danc', 'stretch', 'lift',
                   'wave', 'clap', 'gesture']

DANGER_ACTIONS  = {'FALL', 'FIGHT'}
THREAT_CATEGORIES = {
    'FALL': {'severity': 'HIGH', 'icon': '⬇️', 'color': '#ff3b3b'},
    'FIGHT': {'severity': 'HIGH', 'icon': '⚔️', 'color': '#ff3b3b'},
}

def match_keyword(ll: str, keywords: list[str]) -> bool:
    """Check if any of the keywords match ll using word-boundary matching."""
    for k in keywords:
        pattern = rf'\b{k}\w*'
        if re.search(pattern, ll):
            return True
    return False

def map_to_action(label: str) -> tuple[str, bool, str]:
    """Map a Kinetics-400 label to a human action + alert flag + category.
    Returns: (action_name, is_alert, category)
    """
    ll = label.lower()
    if 'stretching arm' in ll:
        return 'FIGHT / VIOLENCE - Punching/Stretching Arm', True, 'FIGHT'
    if match_keyword(ll, FALL_KEYWORDS):
        return f'FALL DETECTED - {ll}', True, 'FALL'
    if match_keyword(ll, FIGHT_KEYWORDS):
        return f'FIGHT/VIOLENCE DETECTED - {ll}', True, 'FIGHT'
    if match_keyword(ll, SIT_KEYWORDS):
        return f'Sitting Activity - {ll}', False, 'NORMAL'
    if match_keyword(ll, STAND_KEYWORDS):
        return f'Normal Movement - {ll}', False, 'NORMAL'
    # Fallback: show the actual Kinetics class name nicely formatted
    return label.replace('_', ' ').upper(), False, 'NORMAL'

# ─── Live Camera State ────────────────────────────────────────────────────────
cam_state: dict = {
    'active':           False,
    'prediction':       '—',
    'raw_label':        '',
    'alert':            False,
    'category':         'NORMAL',
    'confidence':       0.0,
    'screenshot_url':   None,
    'video_url':        None,
    'threat_count':     0,
    'last_threat_time': None,
}
cam_lock    = threading.Lock()
frame_lock  = threading.Lock()
output_frame: np.ndarray | None = None

from queue import Queue

latest_frame = {'frame': None}   # shared between threads
latest_frame_lock = threading.Lock()
video_writer = None
video_frame_count = 0
video_start_time = None
recording_threat = False

def capture_loop(cap):
    """Thread 1: Reads frames and updates the display output immediately."""
    global output_frame
    while cam_state['active']:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Save for inference thread
        with latest_frame_lock:
            latest_frame['frame'] = frame

        # 2. Build premium overlay for the live stream (independent of inference speed)
        with cam_lock:
            pred = cam_state['prediction']
            alert = cam_state['alert']
            conf = cam_state['confidence']

        disp = frame.copy()
        h, w = disp.shape[:2]
        
        # Status Bar at top
        bg_color = (0, 0, 180) if alert else (40, 40, 40)
        cv2.rectangle(disp, (0, 0), (w, 50), bg_color, -1)
        
        text = f"ACTION: {pred}" if not alert else f"⚠️ ALERT: {pred}"
        cv2.putText(disp, text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Confidence indicator
        conf_text = f"Confidence: {conf:.2%}"
        cv2.putText(disp, conf_text, (w - 220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Subtle border if alert
        if alert:
            cv2.rectangle(disp, (0, 0), (w-1, h-1), (0, 0, 255), 4)

        with frame_lock:
            output_frame = disp

def run_inference_loop():
    """Thread 2: Dedicated solely to running the AI model."""
    # Use CAP_DSHOW on Windows for faster initialization
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        with cam_lock:
            cam_state['prediction'] = 'Camera not found'
        return

    # Start capture in its own thread
    t = threading.Thread(target=capture_loop, args=(cap,), daemon=True)
    t.start()

    frames_buf = deque(maxlen=CLIP_LEN)
    last_sample_time = 0.0
    last_inference_time = 0.0
    last_frame_id = None
    # smoothing buffer for averaged softmax vectors
    smooth_buf = _deque(maxlen=SMOOTHING_WINDOW)
    last_alert_time = 0.0
    last_alert_info = None  # (action_name, raw, category, conf, screenshot_url)

    while cam_state['active']:
        # Always grab the LATEST frame
        with latest_frame_lock:
            frame = latest_frame['frame']

        if frame is None:
            time.sleep(0.01)
            continue
        
        current_frame_id = id(frame)
        if current_frame_id == last_frame_id:
            time.sleep(0.005)
            continue
        last_frame_id = current_frame_id

        # ── Sample for inference (Independent of display) ──
        # Sample a frame every 80ms to form a clip of ~1.28 seconds
        now = time.time()
        if now - last_sample_time >= 0.08:
            last_sample_time = now
            resized = cv2.resize(frame, (224, 224))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            frames_buf.append(rgb)

        # ── Run Inference with Time Throttling (configurable interval) ──
        if len(frames_buf) == CLIP_LEN and (now - last_inference_time) >= INFERENCE_INTERVAL:
            last_inference_time = now
            clip_np = np.array(list(frames_buf))  # shape (T, H, W, C)
            # to tensor shape (1, C, T, H, W)
            clip = torch.tensor(clip_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0

            # normalize using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406], device=clip.device).view(1, 3, 1, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=clip.device).view(1, 3, 1, 1, 1)
            clip = (clip - mean) / std

            with torch.no_grad():
                if ENABLE_TTA:
                    # original
                    logits1 = model(clip)
                    # horizontal flip TTA
                    clip_flipped = torch.flip(clip, dims=[-1])
                    logits2 = model(clip_flipped)
                    probs = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2.0
                else:
                    logits = model(clip)
                    probs = torch.softmax(logits, dim=1)

            # Append to smoothing buffer and compute averaged probabilities
            smooth_buf.append(probs[0].cpu().numpy())
            avg_probs = np.mean(list(smooth_buf), axis=0)
            conf_val = float(np.max(avg_probs))
            pred_idx = int(np.argmax(avg_probs))
            raw = labels[pred_idx]
            # initially map to category, then apply confidence threshold for alert
            action_name, _, category = map_to_action(raw)

            # dynamic thresholding based on mode
            threshold = ALERT_CONF_THRESHOLD if DETECTION_MODE == 'balanced' else max(0.35, ALERT_CONF_THRESHOLD * 0.8)
            now = time.time()
            recently_alerted = (now - last_alert_time) < ALERT_COOLDOWN

            # Determine whether this is a new alert (only when not in cooldown)
            new_alert = (category in ('FALL', 'FIGHT') and conf_val >= threshold) and (not recently_alerted)

            screenshot_url = None
            if new_alert:
                try:
                    ts = int(time.time())
                    prefix = category.lower()
                    sc_filename = f"{prefix}_{ts}.jpg"
                    sc_path = os.path.join('static', 'screenshots', sc_filename)
                    cv2.imwrite(sc_path, frame)
                    screenshot_url = f"/static/screenshots/{sc_filename}"
                except Exception as e:
                    print(f"Error saving screenshot: {e}")

                # record the alert time and info
                last_alert_time = time.time()
                last_alert_info = (action_name, raw, category, conf_val, screenshot_url)

                # increment counters
                with cam_lock:
                    cam_state['threat_count'] += 1
                    cam_state['last_threat_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

                # clear temporal buffers so subsequent clips are analyzed fresh
                try:
                    frames_buf.clear()
                except Exception:
                    frames_buf = deque(maxlen=CLIP_LEN)
                try:
                    smooth_buf.clear()
                except Exception:
                    smooth_buf = _deque(maxlen=SMOOTHING_WINDOW)

            # If still in cooldown, present the last alert info rather than new noisy labels
            effective_alert = False
            effective_action = action_name
            effective_raw = raw
            effective_category = category
            effective_conf = conf_val
            effective_screenshot = screenshot_url

            if recently_alerted and last_alert_info is not None:
                effective_action, effective_raw, effective_category, effective_conf, effective_screenshot = last_alert_info
                effective_alert = True
            elif new_alert:
                effective_alert = True

            with cam_lock:
                cam_state['prediction'] = effective_action
                cam_state['raw_label'] = effective_raw
                cam_state['alert'] = bool(effective_alert)
                cam_state['category'] = effective_category
                cam_state['confidence'] = effective_conf
                if effective_screenshot:
                    cam_state['screenshot_url'] = effective_screenshot

        time.sleep(0.005)  # Yield CPU to allow smooth display rendering

    cap.release()

def generate_stream():
    last_frame_id = None
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.04)
                continue
            frame_id = id(output_frame)   # check if frame is actually new
            if frame_id == last_frame_id:
                time.sleep(0.04)
                continue
            last_frame_id = frame_id
            _, buf = cv2.imencode('.jpg', output_frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 70])  # 70 vs 80 helps

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    with cam_lock:
        if not cam_state['active']:
            cam_state['active']     = True
            cam_state['prediction'] = 'Warming up…'
            cam_state['alert']      = False
            cam_state['confidence'] = 0.0
            cam_state['screenshot_url'] = None
            t = threading.Thread(target=run_inference_loop, daemon=True)
            t.start()
    return jsonify(status='started')

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    with cam_lock:
        cam_state['active'] = False
    return jsonify(status='stopped')

@app.route('/video_feed')
def video_feed():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam_status')
def cam_status():
    with cam_lock:
        sc_url = cam_state.get('screenshot_url')
        if sc_url:
            cam_state['screenshot_url'] = None  # Reset after retrieval so it is sent exactly once
        return jsonify(
            prediction = cam_state['prediction'],
            raw_label  = cam_state['raw_label'],
            alert      = cam_state['alert'],
            category   = cam_state['category'],
            confidence = round(cam_state['confidence'], 3),
            active     = cam_state['active'],
            screenshot_url = sc_url,
            threat_count = cam_state['threat_count'],
            last_threat_time = cam_state['last_threat_time'],
        )

# ─── Video Upload & Processing ───────────────────────────────────────────────
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify(error='No file provided'), 400

    f        = request.files['video']
    filename = secure_filename(f.filename)
    if not filename:
        return jsonify(error='Invalid filename'), 400

    fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(fpath)

    try:
        results, meta = process_video_file(fpath)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)

    return jsonify(results=results, meta=meta)


@app.route('/upload_checkpoint', methods=['POST'])
def upload_checkpoint():
    if 'checkpoint' not in request.files:
        return jsonify(error='No file provided'), 400
    f = request.files['checkpoint']
    filename = secure_filename(f.filename)
    if not filename:
        return jsonify(error='Invalid filename'), 400
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    fpath = models_dir / filename
    f.save(str(fpath))
    try:
        load_checkpoint(str(fpath))
    except Exception as e:
        return jsonify(error=str(e)), 500
    return jsonify(status='loaded', filename=filename)

def process_video_file(filepath: str):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        raise ValueError("Cannot open video file")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    results      = []
    frames_buf   = []
    frame_idx    = 0
    clip_idx     = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        fr_small   = cv2.resize(frame, (224, 224))
        fr_rgb     = cv2.cvtColor(fr_small, cv2.COLOR_BGR2RGB)
        frames_buf.append(fr_rgb)

        if len(frames_buf) == CLIP_LEN:
            clip_np = np.array(frames_buf)
            clip = torch.tensor(clip_np).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=clip.device).view(1, 3, 1, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=clip.device).view(1, 3, 1, 1, 1)
            clip = (clip - mean) / std

            with torch.no_grad():
                if ENABLE_TTA:
                    logits1 = model(clip)
                    logits2 = model(torch.flip(clip, dims=[-1]))
                    probs = (torch.softmax(logits1, dim=1) + torch.softmax(logits2, dim=1)) / 2.0
                else:
                    logits = model(clip)
                    probs = torch.softmax(logits, dim=1)

            # smoothing over recent clips
            if 'video_smooth' not in locals():
                video_smooth = _deque(maxlen=SMOOTHING_WINDOW)
            video_smooth.append(probs[0].cpu().numpy())
            avg_probs = np.mean(list(video_smooth), axis=0)
            conf_val = float(np.max(avg_probs))
            pred_idx = int(np.argmax(avg_probs))
            raw = labels[pred_idx]
            action_name, _, category = map_to_action(raw)

            threshold = ALERT_CONF_THRESHOLD if DETECTION_MODE == 'balanced' else max(0.35, ALERT_CONF_THRESHOLD * 0.8)
            is_alert = category in ('FALL', 'FIGHT') and conf_val >= threshold

            ts = round((frame_idx / fps), 2)

            results.append({
                'clip':       clip_idx + 1,
                'timestamp':  ts,
                'time_fmt':   fmt_time(ts),
                'action':     action_name,
                'raw_label':  raw,
                'category':   category,
                'confidence': round(conf_val, 3),
                'alert':      bool(is_alert),
            })

            clip_idx   += 1
            frames_buf  = []

    cap.release()

    alert_count = sum(1 for r in results if r['alert'])
    meta = {
        'duration':    round(duration, 2),
        'total_clips': clip_idx,
        'alerts':      alert_count,
        'fps':         round(fps, 2),
    }
    return results, meta

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"

if __name__ == '__main__':
    app.run(debug=False, threaded=True, host='0.0.0.0', port=5000)
