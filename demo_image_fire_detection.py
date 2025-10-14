#!/usr/bin/env python3
"""
Demo Image Fire Detection - Works with default YOLOv8 model
Uses color analysis as fallback when trained model doesn't detect fire
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

class DemoImageFireDetector:
    def __init__(self, notification_config="email_config.json"):
        """Initialize the demo image fire detector"""
        print("🔥 Initializing Demo Image Fire Detection System...")
        
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded")
        
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
        
        # Create output directory
        self.output_dir = "demo_fire_detection_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def detect_fire_in_image(self, image_path):
        """Detect fire in a single image using color analysis"""
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
            
            # Run YOLO detection first
            results = self.model(image)
            
            # Check for any high-confidence detections
            yolo_fire_detected = False
            yolo_detections = []
            
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf[0])
                        if confidence > self.confidence_threshold:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            yolo_detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': confidence,
                                'class_name': 'object'
                            })
            
            # Use color analysis for fire detection
            fire_detected, fire_regions = self._analyze_fire_colors(image)
            
            # Create annotated image
            annotated_image = self._create_annotated_image(image, fire_regions, yolo_detections)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save annotated image
            output_path = os.path.join(self.output_dir, f"{base_name}_demo_analysis_{timestamp}.jpg")
            cv2.imwrite(output_path, annotated_image)
            print(f"💾 Results saved: {output_path}")
            
            return fire_detected, {
                'fire_regions': fire_regions,
                'yolo_detections': yolo_detections,
                'annotated_image': output_path,
                'original_image': image_path,
                'total_fire_regions': len(fire_regions),
                'timestamp': timestamp
            }
            
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return False, None
    
    def _analyze_fire_colors(self, image):
        """Analyze image for fire-like colors"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define fire color ranges in HSV
        fire_ranges = [
            # Red ranges
            ([0, 50, 50], [10, 255, 255]),
            ([170, 50, 50], [180, 255, 255]),
            # Orange range
            ([10, 50, 50], [25, 255, 255]),
            # Yellow range
            ([25, 50, 50], [35, 255, 255]),
        ]
        
        # Create combined mask for fire colors
        fire_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in fire_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            fire_mask = cv2.bitwise_or(fire_mask, mask)
        
        # Find contours of fire-colored regions
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_regions = []
        total_fire_pixels = 0
        image_pixels = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                fire_ratio = area / image_pixels
                
                # Check if region has high fire color concentration
                region_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.drawContours(region_mask, [contour], -1, 255, -1)
                fire_pixels_in_region = cv2.countNonZero(cv2.bitwise_and(fire_mask, region_mask))
                region_fire_ratio = fire_pixels_in_region / area if area > 0 else 0
                
                if region_fire_ratio > 0.3:  # 30% of region should be fire-colored
                    fire_regions.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': min(region_fire_ratio, 1.0),
                        'area': area,
                        'fire_ratio': fire_ratio
                    })
                    total_fire_pixels += fire_pixels_in_region
        
        # Determine if fire is detected
        total_fire_ratio = total_fire_pixels / image_pixels
        fire_detected = len(fire_regions) > 0 and total_fire_ratio > 0.02  # 2% of image should be fire-colored
        
        if fire_detected:
            print(f"🔥 Fire-like colors detected in {len(fire_regions)} regions")
            print(f"   Total fire pixel ratio: {total_fire_ratio:.2%}")
        
        return fire_detected, fire_regions
    
    def _create_annotated_image(self, image, fire_regions, yolo_detections):
        """Create annotated image with fire detection boxes"""
        annotated = image.copy()
        
        # Draw fire region boxes (red)
        for i, region in enumerate(fire_regions):
            x1, y1, x2, y2 = region['bbox']
            confidence = region['confidence']
            
            # Draw bounding box (red for fire)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label
            label = f"FIRE {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 0, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw YOLO detection boxes (blue)
        for i, detection in enumerate(yolo_detections):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box (blue for YOLO detections)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Draw label
            label = f"YOLO {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (255, 0, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add timestamp and status
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if fire_regions:
            status = f"FIRE DETECTED: {len(fire_regions)} regions"
            color = (0, 0, 255)  # Red
        else:
            status = "NO FIRE DETECTED"
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
                'fire_regions': detection_result['fire_regions'],
                'total_fire_regions': detection_result['total_fire_regions'],
                'timestamp': detection_result['timestamp']
            }
            
            # Send notification with image attachment
            success = self.notification_system.send_fire_alert(
                detection_data=detection_data,
                image_path=detection_result['annotated_image'],
                confidence=max([r['confidence'] for r in detection_result['fire_regions']]) if detection_result['fire_regions'] else 0,
                location=f"Demo Image Analysis - {os.path.basename(detection_result['original_image'])}"
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
        print("🔥 DEMO IMAGE FIRE DETECTION SYSTEM")
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
        print(f"   Fire Regions: {result['total_fire_regions']}")
        print(f"   YOLO Detections: {len(result['yolo_detections'])}")
        
        if fire_detected:
            print(f"\n🔥 FIRE DETECTION DETAILS:")
            for i, region in enumerate(result['fire_regions'], 1):
                print(f"   Fire Region {i}:")
                print(f"     Confidence: {region['confidence']:.2f}")
                print(f"     Bounding Box: {region['bbox']}")
                print(f"     Area: {region['area']} pixels")
                print(f"     Fire Ratio: {region['fire_ratio']:.2%}")
            
            # Send email alert
            self.send_fire_alert(result)
        else:
            print("\n✅ No fire detected in the image")
        
        return fire_detected

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Demo Image-based Fire Detection with Email Alerts")
    parser.add_argument("image_path", help="Path to the image file to analyze")
    parser.add_argument("--config", default="email_config.json", 
                       help="Path to notification configuration file")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DemoImageFireDetector(args.config)
    
    # Process image
    fire_detected = detector.process_image(args.image_path)
    
    # Exit with appropriate code
    sys.exit(1 if fire_detected else 0)

if __name__ == "__main__":
    main()
