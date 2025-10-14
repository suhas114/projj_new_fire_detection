# 🔥 Fire Detection System Architecture

## 📋 System Overview

Your fire detection project uses a **multi-layered architecture** combining YOLOv8 neural networks with traditional computer vision techniques for robust fire detection.

## 🧠 Neural Network Architecture

### **Base Model: YOLOv8n (Nano)**
- **Architecture**: Convolutional Neural Network (CNN)
- **Model Size**: Nano variant (smallest, fastest)
- **Input Size**: 640x640 pixels
- **Classes**: 2 classes (fire=1, smoke=0)
- **Framework**: PyTorch-based Ultralytics YOLO

### **YOLOv8n Architecture Components:**
```
Input Layer (640x640x3) 
    ↓
Backbone (CSPDarknet53-based)
    ↓
Neck (PANet + FPN)
    ↓
Head (Detection Head)
    ↓
Output: Bounding boxes + Confidence scores
```

## 🏗️ System Architecture Layers

### **1. Data Layer**
```
Dataset Structure:
├── dataset/
│   ├── images/
│   │   ├── train/ (1,832 images)
│   │   ├── val/
│   │   └── test/
│   └── labels/ (1,834 label files)
└── data.yaml (configuration)
```

### **2. Model Layer**
- **Training**: Custom fire detection model
- **Inference**: Real-time detection engine
- **Fallback**: Default YOLOv8n.pt

### **3. Detection Pipeline**
```
Input Frame → Preprocessing → YOLO Detection → Post-processing → Alert System
     ↓              ↓              ↓               ↓              ↓
  Camera/Image   Resize(640x640)  Neural Net   Confidence Filter  Notifications
```

### **4. Computer Vision Enhancement**
- **HSV Color Space Analysis**
- **Morphological Operations**
- **Contour Detection**
- **Multi-threshold Filtering**

## 🔧 Detection Components

### **Primary Detection Systems:**

1. **YOLOv8 Neural Network**
   - Real-time object detection
   - Fire and smoke classification
   - Bounding box regression

2. **Computer Vision Enhancement**
   - HSV color filtering
   - Contour analysis
   - Area-based validation

3. **Multi-Modal Detection**
   - Combined neural network + CV
   - False positive reduction
   - Confidence scoring

## 📊 Training Architecture

### **Training Configuration:**
```python
Model: YOLOv8n.yaml
Epochs: 30
Batch Size: 32
Image Size: 640x640
Device: CPU/GPU auto-detection
Optimizer: AdamW
Learning Rate: 0.01 (cosine annealing)
```

### **Training Pipeline:**
```
Raw Images → Data Augmentation → YOLO Training → Validation → Model Export
     ↓              ↓               ↓             ↓           ↓
 1,832 images   Mosaic, MixUp   Backpropagation  mAP50: 95.6%  best.pt
```

## 🚨 Alert System Architecture

### **Notification Pipeline:**
```
Fire Detection → Confidence Check → Cooldown Timer → Email/SMS → Logging
       ↓               ↓               ↓             ↓         ↓
   Neural Net      >0.25 threshold   60s delay   SMTP/Twilio   JSON logs
```

### **Alert Components:**
- **Email System**: Gmail SMTP
- **SMS System**: Twilio integration
- **Sound Alerts**: pygame audio
- **Image Capture**: Alert snapshots
- **Logging**: JSON event logs

## 🎯 Detection Flow Architecture

### **Real-time Detection:**
```
Camera Input → Frame Capture → Preprocessing → YOLO Inference → Post-processing
     ↓              ↓              ↓              ↓               ↓
  OpenCV        640x640 resize   Normalize    Neural Network   NMS + Filtering
     ↓              ↓              ↓              ↓               ↓
 Frame Queue → Background Thread → GPU/CPU → Confidence Check → Alert Decision
```

### **Image Detection:**
```
Image Input → Load & Validate → YOLO Processing → Region Analysis → Output
     ↓              ↓               ↓               ↓              ↓
 File Path      Format Check    Neural Network   Fire Regions   Annotated Image
```

## 🔍 Model Hierarchy

### **Model Loading Priority:**
1. `runs/detect/forest_fire_model6/weights/best.pt` (Primary)
2. `runs/detect/forest_fire_model2/weights/best.pt` (Secondary)
3. `yolov8n.pt` (Fallback)

### **Detection Classes:**
- **Class 0**: Smoke (early fire indicator)
- **Class 1**: Fire (actual fire)

## ⚙️ Performance Architecture

### **Optimization Features:**
- **Early Stopping**: Prevents overfitting
- **Model Quantization**: Reduced inference time
- **Batch Processing**: Efficient GPU utilization
- **Multi-threading**: Parallel processing
- **Memory Management**: Efficient resource usage

### **Accuracy Metrics:**
- **mAP50**: 95.6%
- **mAP50-95**: 85.4%
- **Precision**: 88.6%
- **Recall**: 92.7%

## 🔄 System Integration

### **Modular Components:**
```
Fire Detection Core
├── Neural Network Engine (YOLOv8)
├── Computer Vision Engine (OpenCV)
├── Alert System (Email/SMS)
├── Logging System (JSON)
└── Visualization Engine (Matplotlib)
```

### **Interface Layer:**
- **Real-time Camera**: `realtime_detection_with_notifications.py`
- **Image Processing**: `image_fire_detection.py`
- **Interactive GUI**: `interactive_image_fire_detection.py`
- **Batch Processing**: `user_input_image_detection.py`

This architecture provides a robust, scalable fire detection system with multiple detection methods, comprehensive alerting, and high accuracy performance.
