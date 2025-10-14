# Image Fire Detection System Guide

## 🔥 Overview

The Image Fire Detection System uses your trained YOLOv8 deep learning model to detect fire and smoke in images, then sends email alerts if fire is detected.

## 🚀 Quick Start

### Basic Usage
```bash
python image_fire_detection.py path/to/your/image.jpg
```

### With Custom Model
```bash
python image_fire_detection.py path/to/your/image.jpg --model runs/detect/forest_fire_model6/weights/best.pt
```

### With Custom Confidence Threshold
```bash
python image_fire_detection.py path/to/your/image.jpg --threshold 0.3
```

## 📋 Command Line Options

- `image_path` (required): Path to the image file to analyze
- `--config`: Path to email configuration file (default: email_config.json)
- `--model`: Path to custom YOLO model (default: auto-detect trained model)
- `--threshold`: Confidence threshold for detection (default: 0.25)

## 🎯 Features

### Detection Capabilities
- **Fire Detection**: Identifies actual fire with high accuracy
- **Smoke Detection**: Detects smoke as early fire indicator
- **Dual Classification**: Uses your trained model with 2 classes (smoke, fire)

### Output Features
- **Annotated Images**: Creates images with bounding boxes and labels
- **Email Alerts**: Sends email with detection details and image attachment
- **Detailed Reports**: Console output with confidence scores and bounding boxes
- **Timestamped Results**: All outputs include timestamps

### Model Support
- **Auto-Detection**: Automatically finds your trained fire detection model
- **Fallback Support**: Uses default YOLOv8 if trained model not found
- **Custom Models**: Supports any YOLOv8-compatible model

## 📧 Email Alerts

The system automatically sends email alerts when:
- Fire is detected (any confidence level)
- High-confidence smoke is detected (>50% confidence)

Email includes:
- Detection details (confidence, bounding boxes)
- Annotated image attachment
- Timestamp and location information

## 📁 Output Structure

Results are saved to `fire_detection_results/` directory:
```
fire_detection_results/
├── image1_fire_analysis_20241201_143022.jpg
├── image2_fire_analysis_20241201_143045.jpg
└── ...
```

## 🧪 Testing

### Test with Sample Images
```bash
python test_image_detection.py
```

### Test with Specific Image
```bash
python image_fire_detection.py captured_fires/fire_detected_20250829_212845.jpg
```

## 🔧 Configuration

### Email Setup
Make sure `email_config.json` is configured:
```json
{
    "sender_email": "your-email@gmail.com",
    "app_password": "your-app-password",
    "recipient_email": "alerts@example.com"
}
```

### Model Configuration
The system automatically detects your trained models:
1. `runs/detect/forest_fire_model6/weights/best.pt` (preferred)
2. `runs/detect/forest_fire_model2/weights/best.pt`
3. `yolov8n.pt` (fallback)

## 📊 Detection Classes

Your trained model recognizes:
- **Class 0**: Smoke (early fire indicator)
- **Class 1**: Fire (actual fire)

## 🎨 Visual Output

- **Red boxes**: Fire detections
- **Orange boxes**: Smoke detections
- **Green status**: No fire/smoke detected
- **Red status**: Fire detected
- **Orange status**: Smoke detected

## 🔍 Troubleshooting

### Model Not Found
- Check if trained model exists in `runs/detect/` directory
- Verify model file is named `best.pt`

### Email Not Sending
- Check email configuration in `email_config.json`
- Verify Gmail app password is correct
- Test email with: `python simple_email_test.py`

### No Detections
- Try lowering confidence threshold: `--threshold 0.1`
- Check if image contains fire/smoke
- Verify model is trained for fire detection

## 📈 Performance

- **Speed**: ~50-100ms per image (depending on image size)
- **Accuracy**: High accuracy with trained fire detection model
- **Memory**: Efficient GPU/CPU usage with YOLOv8

## 🔄 Integration

The system can be integrated into:
- Security camera systems
- Forest monitoring stations
- Industrial safety systems
- Automated alert systems

## 📝 Example Output

```
🔥 Initializing Image Fire Detection System...
✅ Loaded trained fire detection model: forest_fire_model6
✅ Notification system initialized

🔍 Analyzing image: test_fire.jpg
📸 Image loaded: (480, 640, 3)
🔥 Fire detected! Confidence: 0.87
💾 Results saved: fire_detection_results/test_fire_fire_analysis_20241201_143022.jpg

📊 ANALYSIS RESULTS:
   Original Image: test_fire.jpg
   Annotated Image: fire_detection_results/test_fire_fire_analysis_20241201_143022.jpg
   Fire Detected: YES
   Fire Detections: 1
   Smoke Detections: 0

🔥 FIRE DETECTION DETAILS:
   Fire Detection 1:
     Confidence: 0.87
     Bounding Box: [150, 200, 350, 400]
     Class: fire

📧 Sending fire alert email...
✅ Fire alert email sent successfully!
```
