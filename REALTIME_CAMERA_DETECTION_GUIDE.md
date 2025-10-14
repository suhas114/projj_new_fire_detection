# Real-time Camera Fire Detection System

## Overview
This system captures live video from your camera and detects fire in real-time, sending email alerts when fire is detected.

## Features
- 🎥 **Real-time camera capture** from your webcam
- 🔥 **Advanced fire detection** using YOLOv8 and color analysis
- 📧 **Email alerts** for both fire detected and no fire detected
- 💾 **Automatic saving** of fire detection images
- 📊 **Live statistics** and detection rates
- ⚡ **Cooldown system** to prevent spam alerts

## Requirements
- Webcam or camera connected to your computer
- Python with required packages installed
- Email configuration set up (`email_config.json`)

## Usage

### Option 1: Simple Launcher (Recommended)
```bash
python run_camera_detection.py
```

### Option 2: Direct Script
```bash
python realtime_camera_fire_detection.py
```

### Option 3: With Custom Camera Index
```bash
python realtime_camera_fire_detection.py --camera 1
```

### Option 4: With Custom Email Config
```bash
python realtime_camera_fire_detection.py --config my_email_config.json
```

## Controls
When the camera window is open:
- **'q' or ESC**: Quit the application
- **'s'**: Save current frame manually
- **'c'**: Capture screenshot

## Email Notifications
The system sends emails for:
- ✅ **Fire detected**: Immediate alert with detection image
- 📧 **No fire detected**: Periodic notifications (every 60 seconds)

## Output Files
Results are saved in `realtime_camera_results/`:
- `detections/`: Images where fire was detected
- `captures/`: Regular captures and screenshots

## Configuration

### Camera Settings
- Default camera index: 0
- Resolution: 640x480
- FPS: 30

### Detection Settings
- Confidence threshold: 0.25
- Alert cooldown: 60 seconds
- Fire detection uses trained YOLOv8 model + color analysis

### Email Settings
Make sure `email_config.json` is properly configured:
```json
{
    "sender_email": "your_email@gmail.com",
    "app_password": "your_app_password",
    "recipient_email": "recipient@gmail.com"
}
```

## Troubleshooting

### Camera Issues
- **Camera not found**: Try different camera index (`--camera 1`, `--camera 2`)
- **Permission denied**: Check camera permissions in your system
- **Poor quality**: Ensure good lighting and camera positioning

### Detection Issues
- **Too many false positives**: The system uses advanced filtering to avoid sunlight
- **Missing real fire**: Check lighting conditions and camera angle
- **No detections**: Ensure fire is clearly visible and well-lit

### Email Issues
- **No emails sent**: Check `email_config.json` configuration
- **Authentication failed**: Verify Gmail app password setup
- **Spam folder**: Check spam/junk folder for alerts

## System Requirements
- Python 3.7+
- OpenCV (`cv2`)
- Ultralytics YOLOv8
- NumPy
- Email configuration

## Performance
- **Processing speed**: ~30 FPS on modern hardware
- **Memory usage**: Moderate (depends on camera resolution)
- **CPU usage**: High during detection (uses GPU if available)

## Safety Notes
⚠️ **Important**: This is a detection system, not a replacement for professional fire safety equipment. Always have proper fire alarms and safety measures in place.

## Support
If you encounter issues:
1. Check camera connection and permissions
2. Verify email configuration
3. Ensure good lighting conditions
4. Check system requirements are met

## Examples

### Basic Usage
```bash
# Start with default settings
python run_camera_detection.py
```

### Advanced Usage
```bash
# Use specific camera and custom config
python realtime_camera_fire_detection.py --camera 1 --config production_email.json
```

### Testing
```bash
# Test with different camera
python realtime_camera_fire_detection.py --camera 2
```

The system will automatically:
- Initialize your camera
- Start real-time detection
- Save fire detection images
- Send email alerts
- Display live statistics
