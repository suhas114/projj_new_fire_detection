#!/usr/bin/env python3
"""
Image-based Fire Detection with Email Alerts
Takes an image as input, detects fire, and sends email if fire is found
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import sys
import argparse
from datetime import datetime
from notification_system import NotificationSystem

class ImageFireDetector:
    def __init__(self, notification_config="email_config.json", model_path=None):
        """Initialize the image fire detector"""
        print("🔥 Initializing Image Fire Detection System...")
        
        # Load YOLO model - try trained fire model first
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
            print(f"✅ Loaded custom fire detection model: {model_path}")
        elif os.path.exists("runs/detect/forest_fire_model6/weights/best.pt"):
            self.model = YOLO("runs/detect/forest_fire_model6/weights/best.pt")
            print("✅ Loaded trained fire detection model: forest_fire_model6")
        elif os.path.exists("runs/detect/forest_fire_model2/weights/best.pt"):
            self.model = YOLO("runs/detect/forest_fire_model2/weights/best.pt")
            print("✅ Loaded trained fire detection model: forest_fire_model2")
        else:
            self.model = YOLO('yolov8n.pt')
            print("⚠️ Using default YOLOv8 model (not trained for fire detection)")
        
        # Initialize notification system
        self.notification_system = None
        if os.path.exists(notification_config):
            try:
                self.notification_system = NotificationSystem(notification_config)
                print("✅ Notification system initialized - emails enabled")
            except Exception as e:
                print(f"❌ Notification system failed: {e}")
        else:
            print("⚠️ No notification config found - email alerts disabled")
        
        # Fire detection parameters
        self.confidence_threshold = 0.25
        self.fire_class_id = 1  # Fire is class 1 in your dataset (smoke=0, fire=1)
        self.smoke_class_id = 0  # Smoke is class 0
        
        # Create output directory
        self.output_dir = "fire_detection_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def detect_fire_in_image(self, image_path):
        """Detect fire in a single image"""
        print(f"\n🔍 Analyzing image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return False, None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return False, None
            
            print(f"📸 Image loaded: {image.shape}")
            
            # Run YOLO detection
            results = self.model(image)
            
            # Process results
            fire_detected = False
            fire_detections = []
            smoke_detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        if confidence > self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            
                            detection_info = {
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
                            }
                            
                            # Check if it's fire or smoke detection
                            if class_id == self.fire_class_id:  # Fire detected
                                fire_detected = True
                                fire_detections.append(detection_info)
                                print(f"🔥 Fire detected! Confidence: {confidence:.2f}")
                            elif class_id == self.smoke_class_id:  # Smoke detected
                                smoke_detections.append(detection_info)
                                print(f"💨 Smoke detected! Confidence: {confidence:.2f}")
                                # Consider smoke as potential fire indicator
                                if confidence > 0.5:  # High confidence smoke
                                    fire_detected = True
                                    print("⚠️ High confidence smoke - treating as fire alert!")
            
            # Create annotated image
            annotated_image = self._create_annotated_image(image, fire_detections, smoke_detections)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save annotated image
            output_path = os.path.join(self.output_dir, f"{base_name}_fire_analysis_{timestamp}.jpg")
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Results saved: {output_path}")
            
            return fire_detected, {
                'fire_detections': fire_detections,
                'smoke_detections': smoke_detections,
                'annotated_image': output_path,
                'original_image': image_path,
                'total_fire_detections': len(fire_detections),
                'total_smoke_detections': len(smoke_detections),
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return False, None
    
    def _is_fire_like(self, image, detection):
        """Check if detection area contains fire-like colors"""
        try:
            x1, y1, x2, y2 = detection['bbox']
            
            # Extract the detection region
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                return False
            
            # Convert to HSV for better color analysis
            hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            
            # Define fire-like color ranges (orange/red/yellow)
            # Lower and upper bounds for fire colors in HSV
            fire_ranges = [
                ([0, 50, 50], [10, 255, 255]),    # Red range 1
                ([170, 50, 50], [180, 255, 255]), # Red range 2
                ([10, 50, 50], [25, 255, 255]),   # Orange range
                ([25, 50, 50], [35, 255, 255]),   # Yellow range
            ]
            
            fire_pixel_count = 0
            total_pixels = region.shape[0] * region.shape[1]
            
            for lower, upper in fire_ranges:
                mask = cv2.inRange(hsv_region, np.array(lower), np.array(upper))
                fire_pixel_count += cv2.countNonZero(mask)
            
            # If more than 10% of pixels are fire-like colors, consider it fire
            fire_ratio = fire_pixel_count / total_pixels
            return fire_ratio > 0.1
            
        except Exception as e:
            print(f"Warning: Could not analyze fire colors: {e}")
            return True  # Default to treating high-confidence detections as fire
    
    def _create_annotated_image(self, image, fire_detections, smoke_detections=None):
        """Create annotated image with fire and smoke detection boxes"""
        annotated = image.copy()
        
        if smoke_detections is None:
            smoke_detections = []
        
        # Draw fire detection boxes (red)
        for detection in fire_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box (red for fire)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label
            label = f"FIRE {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw smoke detection boxes (orange)
        for detection in smoke_detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box (orange for smoke)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 3)
            
            # Draw label
            label = f"SMOKE {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 165, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add timestamp and status
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_detections = len(fire_detections) + len(smoke_detections)
        if fire_detections:
            status = f"FIRE DETECTED: {len(fire_detections)} fire, {len(smoke_detections)} smoke"
            color = (0, 0, 255)  # Red
        elif smoke_detections:
            status = f"SMOKE DETECTED: {len(smoke_detections)} smoke"
            color = (0, 165, 255)  # Orange
        else:
            status = "NO FIRE OR SMOKE DETECTED"
            color = (0, 255, 0)  # Green
        
        cv2.putText(annotated, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated, status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return annotated
    
    def send_fire_alert(self, detection_result):
        """Send email alert if fire is detected"""
        if not self.notification_system:
            print("⚠️ No notification system available")
            return False
        
        try:
            print("📧 Sending fire alert email...")
            
            # Prepare detection data
            detection_data = {
                'fire_detections': detection_result['fire_detections'],
                'smoke_detections': detection_result['smoke_detections'],
                'total_fire_detections': detection_result['total_fire_detections'],
                'total_smoke_detections': detection_result['total_smoke_detections'],
                'timestamp': detection_result['timestamp']
            }
            
            # Send notification with image attachment
            success = self.notification_system.send_fire_alert(
                detection_data=detection_data,
                image_path=detection_result['annotated_image'],
                confidence=max([d['confidence'] for d in detection_result['fire_detections'] + detection_result['smoke_detections']]) if (detection_result['fire_detections'] or detection_result['smoke_detections']) else 0,
                location=f"Image Analysis - {os.path.basename(detection_result['original_image'])}"
            )
            
            if success:
                print("✅ Fire alert email sent successfully!")
            else:
                print("❌ Failed to send fire alert email")
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending fire alert: {e}")
            return False
    
    def process_image(self, image_path):
        """Main method to process an image for fire detection"""
        print("=" * 60)
        print("🔥 IMAGE FIRE DETECTION SYSTEM")
        print("=" * 60)
        
        # Detect fire
        fire_detected, result = self.detect_fire_in_image(image_path)
        
        if result is None:
            print("❌ Failed to process image")
            return False
        
        # Print results
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Original Image: {result['original_image']}")
        print(f"   Annotated Image: {result['annotated_image']}")
        print(f"   Fire Detected: {'YES' if fire_detected else 'NO'}")
        print(f"   Fire Detections: {result['total_fire_detections']}")
        print(f"   Smoke Detections: {result['total_smoke_detections']}")
        
        if fire_detected:
            print(f"\n🔥 FIRE DETECTION DETAILS:")
            for i, detection in enumerate(result['fire_detections'], 1):
                print(f"   Fire Detection {i}:")
                print(f"     Confidence: {detection['confidence']:.2f}")
                print(f"     Bounding Box: {detection['bbox']}")
                print(f"     Class: {detection['class_name']}")
            
            if result['smoke_detections']:
                print(f"\n💨 SMOKE DETECTION DETAILS:")
                for i, detection in enumerate(result['smoke_detections'], 1):
                    print(f"   Smoke Detection {i}:")
                    print(f"     Confidence: {detection['confidence']:.2f}")
                    print(f"     Bounding Box: {detection['bbox']}")
                    print(f"     Class: {detection['class_name']}")
            
            # Send email alert
            self.send_fire_alert(result)
        else:
            print("\n✅ No fire or smoke detected in the image")
        
        return fire_detected

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Image-based Fire Detection with Email Alerts")
    parser.add_argument("image_path", help="Path to the image file to analyze")
    parser.add_argument("--config", default="email_config.json", 
                       help="Path to notification configuration file")
    parser.add_argument("--model", type=str, 
                       help="Path to custom YOLO model (default: auto-detect trained model)")
    parser.add_argument("--threshold", type=float, default=0.25,
                       help="Confidence threshold for fire detection (default: 0.25)")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ImageFireDetector(args.config, args.model)
    detector.confidence_threshold = args.threshold
    
    # Process image
    fire_detected = detector.process_image(args.image_path)
    
    # Exit with appropriate code
    sys.exit(1 if fire_detected else 0)

if __name__ == "__main__":
    main()
