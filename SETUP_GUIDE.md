# Motion Detection System - Setup & Run Guide

## ✅ Installation Complete!

All required packages have been successfully installed:
- ✅ Flask 3.0+
- ✅ PyTorch 2.0+ (CPU)
- ✅ TorchVision 0.15+
- ✅ OpenCV 4.8+
- ✅ NumPy 1.24+
- ✅ Werkzeug 3.0+

## 🚀 How to Run

### Option 1: Run using the batch file
1. Double-click `run.bat` file
2. Wait for the app to load the model (first time may take 30-60 seconds)
3. Once loaded, you'll see: `Running on http://0.0.0.0:5000`

### Option 2: Run from Terminal
```bash
cd "Motion-Detection-main"
.\.venv\Scripts\python.exe app.py
```

### Option 3: Run from Virtual Environment
```bash
# Activate virtual environment first
.\.venv\Scripts\Activate.ps1

# Then run
python app.py
```

## 🌐 Access the Application

Once the server is running, open your browser and go to:
```
http://localhost:5000
```

## 📋 Features

### Live Detection Tab
- **Start Camera** - Begin real-time motion detection
- **Threat Statistics** - View total threats and last threat time
- **Current Detection** - See detected action with confidence
- **Event Log** - Complete history with screenshots
- **Last Threat Snapshot** - View captured threat images

### Video Upload Tab
- **Drop Zone** - Upload video files for analysis
- **Analysis Results** - Timeline view of all detections
- **Category Labels** - See threat type for each clip
- **Alert Count** - Summary of dangerous events

## 🎯 Detection Categories

### High Severity (RED ALERTS)
- **FALL** - Person falling, tripping, or collapsing
- **FIGHT** - Physical altercation or violence detected

### Normal (GREEN)
- Regular movements: sitting, standing, walking, running

## 📸 Data Storage

- **Screenshots**: `static/screenshots/` folder
  - Automatically captured on threats
  - Named: `fall_TIMESTAMP.jpg` or `fight_TIMESTAMP.jpg`

- **Uploaded Videos**: `uploads/` folder
  - Temporary storage during processing
  - Automatically cleaned after analysis

## ⚠️ First Launch

The first time you run the app, it will:
1. Initialize the Flask server
2. Load the R3D-18 model from Kinetics-400
3. Set up directories for screenshots and uploads
4. Start listening on port 5000

This may take 30-60 seconds. Please wait for the message:
```
Running on http://0.0.0.0:5000
```

## 🔧 Troubleshooting

### If you see "Module not found" errors:
```bash
# Make sure venv is activated and packages are installed
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### If port 5000 is already in use:
Edit `app.py` and change:
```python
app.run(debug=False, threaded=True, host='0.0.0.0', port=5001)
```

### If camera is not working:
1. Check if camera is connected and not in use by another app
2. Give VS Code permission to access camera
3. Try restarting the app

## 📝 Project Structure

```
Motion-Detection-main/
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── run.bat                   # Batch file to run app
├── CHANGES.md               # Enhancement documentation
├── README.md                # This file
├── static/
│   └── screenshots/         # Captured threat images
├── templates/
│   └── index.html           # Web interface
└── uploads/                 # Temporary video uploads
```

## 🎓 Technology Stack

- **Backend**: Flask + Python
- **AI Model**: PyTorch R3D-18 (Kinetics-400)
- **Video Processing**: OpenCV
- **Frontend**: HTML5 + CSS3 + JavaScript
- **Architecture**: Multi-threaded real-time processing

## 📞 Support

For issues or questions:
1. Check that camera is connected and accessible
2. Verify all dependencies are installed
3. Check browser console (F12) for JavaScript errors
4. Review console output for Python errors

## 🔐 Notes

- The application runs on local network only
- Camera feed is not recorded by default
- Screenshots are saved in `static/screenshots/`
- All processing happens locally on your machine

Enjoy real-time motion detection! 🎥
