#!/usr/bin/env python3
"""
Real-time Camera Fire Detection System
Captures images from camera and detects fire in real-time
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import sys
import time
from datetime import datetime
from notification_system import NotificationSystem
import threading
import queue

class RealtimeCameraFireDetector:
    def __init__(self, camera_index=0, notification_config="email_config.json"):
        """Initialize the real-time camera fire detector"""
        print("🔥 Initializing Real-time Camera Fire Detection System...")
        
        # Load YOLO model - try trained fire model first
        if os.path.exists("runs/detect/forest_fire_model6/weights/best.pt"):
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
        
        # Camera settings
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.last_alert_time = 0
        self.alert_cooldown = 60  # 60 seconds between alerts
        
        # Create output directories
        self.output_dir = "realtime_camera_results"
        self.captures_dir = os.path.join(self.output_dir, "captures")
        self.detections_dir = os.path.join(self.output_dir, "detections")
        os.makedirs(self.captures_dir, exist_ok=True)
        os.makedirs(self.detections_dir, exist_ok=True)
        
        # Detection statistics
        self.total_frames = 0
        self.fire_detections = 0
        self.no_fire_detections = 0
        self.last_detection_time = None
        
        print("🎥 Camera fire detection system ready!")
    
    def initialize_camera(self):
        """Initialize camera capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"❌ Error: Could not open camera {self.camera_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"✅ Camera {self.camera_index} initialized successfully")
            print(f"   Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"   FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
            return True
            
        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False
    
    def detect_fire_in_frame(self, frame):
        """Detect fire in a single frame"""
        try:
            # Run YOLO detection
            results = self.model(frame)
            
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
            
            # If no detections with trained model, try improved color analysis
            if not fire_detected and not smoke_detections:
                fire_detected, fire_regions = self._analyze_fire_colors_improved(frame)
                if fire_detected:
                    print("🔥 Fire-like patterns detected with improved analysis")
                    # Convert color analysis results to detection format
                    for region in fire_regions:
                        detection_info = {
                            'bbox': region['bbox'],
                            'confidence': region['confidence'],
                            'class_id': 1,  # Treat as fire
                            'class_name': 'fire_color_analysis'
                        }
                        fire_detections.append(detection_info)
            
            return fire_detected, fire_detections, smoke_detections
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return False, [], []
    
    def _analyze_fire_colors_improved(self, image):
        """Improved fire color analysis that distinguishes fire from sunlight"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define fire color ranges in HSV (more restrictive)
        fire_ranges = [
            # Red ranges (more specific)
            ([0, 120, 100], [10, 255, 255]),      # Bright red
            ([170, 120, 100], [180, 255, 255]),   # Bright red
            # Orange range (more specific)
            ([10, 120, 120], [25, 255, 255]),     # Bright orange
        ]
        
        # Create combined mask for fire colors
        fire_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in fire_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            fire_mask = cv2.bitwise_or(fire_mask, mask)
        
        # Additional analysis to distinguish fire from sunlight
        # Convert to LAB color space for better brightness analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # Luminance channel
        
        # Find contours of fire-colored regions
        contours, _ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        fire_regions = []
        total_fire_pixels = 0
        image_pixels = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Higher minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                fire_ratio = area / image_pixels
                
                # Check if region has high fire color concentration
                region_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                cv2.drawContours(region_mask, [contour], -1, 255, -1)
                fire_pixels_in_region = cv2.countNonZero(cv2.bitwise_and(fire_mask, region_mask))
                region_fire_ratio = fire_pixels_in_region / area if area > 0 else 0
                
                # Additional fire characteristics analysis
                if region_fire_ratio > 0.2:  # Lower threshold - 20% of region should be fire-colored
                    # Extract region for analysis
                    region_hsv = hsv[y:y+h, x:x+w]
                    region_lab = l_channel[y:y+h, x:x+w]
                    region_mask_small = region_mask[y:y+h, x:x+w]
                    
                    # Calculate fire characteristics
                    fire_score = self._calculate_fire_characteristics(region_hsv, region_lab, region_mask_small)
                    
                    # Only consider it fire if it meets multiple criteria (lower threshold)
                    if fire_score > 0.3:  # Lower fire score threshold
                        fire_regions.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': min(region_fire_ratio * fire_score, 1.0),
                            'area': area,
                            'fire_ratio': fire_ratio,
                            'fire_score': fire_score
                        })
                        total_fire_pixels += fire_pixels_in_region
        
        # Determine if fire is detected with balanced criteria
        total_fire_ratio = total_fire_pixels / image_pixels
        fire_detected = (len(fire_regions) > 0 and 
                        total_fire_ratio > 0.01 and  # Lower threshold - 1% of image should be fire-colored
                        len(fire_regions) <= 15)  # Allow more regions but still avoid large uniform areas
        
        return fire_detected, fire_regions
    
    def _calculate_fire_characteristics(self, region_hsv, region_lab, region_mask):
        """Calculate fire characteristics score to distinguish from sunlight"""
        fire_score = 0.0
        
        if region_hsv.size == 0 or region_lab.size == 0:
            return 0.0
        
        # 1. Brightness analysis - fire is usually bright (lower threshold)
        bright_pixels = np.sum(region_lab > 120)  # Lower brightness threshold
        brightness_score = bright_pixels / region_lab.size
        fire_score += brightness_score * 0.25
        
        # 2. Saturation analysis - fire has medium to high saturation (lower threshold)
        saturation = region_hsv[:, :, 1]
        high_sat_pixels = np.sum(saturation > 100)  # Lower saturation threshold
        saturation_score = high_sat_pixels / saturation.size
        fire_score += saturation_score * 0.25
        
        # 3. Color distribution analysis - fire has red/orange colors
        hue = region_hsv[:, :, 0]  # Use hue channel (0) not saturation (1)
        red_orange_pixels = np.sum((hue >= 0) & (hue <= 30))  # Red-orange hue range
        color_score = red_orange_pixels / hue.size
        fire_score += color_score * 0.3
        
        # 4. Shape irregularity - fire has irregular shapes, sunlight is more uniform
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Calculate perimeter to area ratio (irregularity)
            perimeter = cv2.arcLength(contours[0], True)
            area = cv2.contourArea(contours[0])
            if area > 0:
                irregularity = perimeter / np.sqrt(area)
                # Fire typically has higher irregularity than sunlight
                irregularity_score = min(irregularity / 20.0, 1.0)  # Normalize
                fire_score += irregularity_score * 0.2
        
        return min(fire_score, 1.0)
    
    def create_annotated_frame(self, frame, fire_detections, smoke_detections=None):
        """Create annotated frame with fire and smoke detection boxes"""
        annotated = frame.copy()
        
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
        
        # Add statistics
        stats_text = f"Frames: {self.total_frames} | Fire: {self.fire_detections} | No Fire: {self.no_fire_detections}"
        cv2.putText(annotated, stats_text, (10, annotated.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated
    
    def save_frame(self, frame, fire_detected, fire_detections, smoke_detections):
        """Save frame with detection results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        
        # Create annotated frame
        annotated_frame = self.create_annotated_frame(frame, fire_detections, smoke_detections)
        
        if fire_detected:
            # Save to detections folder
            filename = f"fire_detected_{timestamp}.jpg"
            filepath = os.path.join(self.detections_dir, filename)
            cv2.imwrite(filepath, annotated_frame)
            
            # Also save original frame
            original_filename = f"fire_original_{timestamp}.jpg"
            original_filepath = os.path.join(self.detections_dir, original_filename)
            cv2.imwrite(original_filepath, frame)
            
            print(f"💾 Fire detection saved: {filename}")
            return filepath
        else:
            # Save to captures folder (less frequent for no-fire)
            if self.total_frames % 30 == 0:  # Save every 30 frames
                filename = f"capture_{timestamp}.jpg"
                filepath = os.path.join(self.captures_dir, filename)
                cv2.imwrite(filepath, annotated_frame)
                return filepath
        
        return None
    
    def send_fire_alert(self, frame, fire_detections, smoke_detections):
        """Send email alert if fire is detected"""
        if not self.notification_system:
            print("⚠️ No notification system available")
            return False
        
        try:
            print("📧 Sending real-time fire alert email...")
            
            # Save the frame first
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_fire_{timestamp}.jpg"
            filepath = os.path.join(self.detections_dir, filename)
            
            annotated_frame = self.create_annotated_frame(frame, fire_detections, smoke_detections)
            cv2.imwrite(filepath, annotated_frame)
            
            # Prepare detection data
            detection_data = {
                'fire_detections': fire_detections,
                'smoke_detections': smoke_detections,
                'total_fire_detections': len(fire_detections),
                'total_smoke_detections': len(smoke_detections),
                'timestamp': timestamp,
                'source': 'realtime_camera'
            }
            
            # Send notification with image attachment
            success = self.notification_system.send_fire_alert(
                detection_data=detection_data,
                image_path=filepath,
                confidence=max([d['confidence'] for d in fire_detections + smoke_detections]) if (fire_detections or smoke_detections) else 0,
                location=f"Real-time Camera Detection - Camera {self.camera_index}"
            )
            
            if success:
                print("✅ Real-time fire alert email sent successfully!")
                self.last_alert_time = time.time()
            else:
                print("❌ Failed to send real-time fire alert email")
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending real-time fire alert: {e}")
            return False
    
    def run_detection(self):
        """Main detection loop"""
        if not self.initialize_camera():
            return
        
        print("\n" + "=" * 60)
        print("🔥 REAL-TIME CAMERA FIRE DETECTION SYSTEM")
        print("=" * 60)
        print("Controls:")
        print("  - Press 'q' or ESC to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'c' to capture screenshot")
        print("=" * 60)
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Error reading frame from camera")
                    break
                
                self.total_frames += 1
                
                # Detect fire in current frame
                fire_detected, fire_detections, smoke_detections = self.detect_fire_in_frame(frame)
                
                # Update statistics
                if fire_detected:
                    self.fire_detections += 1
                    self.last_detection_time = datetime.now()
                    
                    # Save frame and send alert (with cooldown)
                    current_time = time.time()
                    if current_time - self.last_alert_time > self.alert_cooldown:
                        saved_file = self.save_frame(frame, fire_detected, fire_detections, smoke_detections)
                        if saved_file:
                            self.send_fire_alert(frame, fire_detections, smoke_detections)
                    else:
                        print(f"⏳ Alert cooldown active ({int(self.alert_cooldown - (current_time - self.last_alert_time))}s remaining)")
                        self.save_frame(frame, fire_detected, fire_detections, smoke_detections)
                else:
                    self.no_fire_detections += 1
                
                # Create annotated frame for display
                annotated_frame = self.create_annotated_frame(frame, fire_detections, smoke_detections)
                
                # Display the frame
                cv2.imshow('Real-time Fire Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n👋 Stopping detection...")
                    break
                elif key == ord('s'):  # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_save_{timestamp}.jpg"
                    filepath = os.path.join(self.captures_dir, filename)
                    cv2.imwrite(filepath, annotated_frame)
                    print(f"💾 Manual save: {filename}")
                elif key == ord('c'):  # Capture screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_{timestamp}.jpg"
                    filepath = os.path.join(self.captures_dir, filename)
                    cv2.imwrite(filepath, annotated_frame)
                    print(f"📸 Screenshot saved: {filename}")
                
                # Print periodic status
                if self.total_frames % 100 == 0:
                    print(f"📊 Status: {self.total_frames} frames processed | "
                          f"Fire: {self.fire_detections} | No Fire: {self.no_fire_detections}")
        
        except KeyboardInterrupt:
            print("\n👋 Detection stopped by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("📊 FINAL DETECTION STATISTICS")
        print("=" * 60)
        print(f"Total Frames Processed: {self.total_frames}")
        print(f"Fire Detections: {self.fire_detections}")
        print(f"No Fire Detections: {self.no_fire_detections}")
        if self.total_frames > 0:
            fire_rate = (self.fire_detections / self.total_frames) * 100
            print(f"Fire Detection Rate: {fire_rate:.2f}%")
        if self.last_detection_time:
            print(f"Last Fire Detection: {self.last_detection_time}")
        print(f"Results saved in: {self.output_dir}")
        print("=" * 60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Camera Fire Detection System")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index (default: 0)")
    parser.add_argument("--config", default="email_config.json", 
                       help="Path to notification configuration file")
    
    args = parser.parse_args()
    
    try:
        detector = RealtimeCameraFireDetector(args.camera, args.config)
        detector.run_detection()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
