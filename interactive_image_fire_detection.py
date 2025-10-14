#!/usr/bin/env python3
"""
Interactive Image Fire Detection - Asks user to select image from device
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os
import sys
from datetime import datetime
from notification_system import NotificationSystem

class InteractiveFireDetector:
    def __init__(self):
        """Initialize the interactive fire detector"""
        print("🔥 Initializing Interactive Image Fire Detection System...")
        
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
        if os.path.exists("email_config.json"):
            try:
                self.notification_system = NotificationSystem("email_config.json")
                print("✅ Notification system initialized")
            except Exception as e:
                print(f"❌ Notification system failed: {e}")
        else:
            print("⚠️ No notification config found - email alerts disabled")
        
        # Fire detection parameters
        self.confidence_threshold = 0.25
        self.fire_class_id = 1  # Fire is class 1 in your dataset (smoke=0, fire=1)
        self.smoke_class_id = 0  # Smoke is class 0
        
        # Create output directory
        self.output_dir = "interactive_fire_detection_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def select_image(self):
        """Open file dialog to select image from device"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Configure file dialog
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        # Open file dialog
        image_path = filedialog.askopenfilename(
            title="Select Image for Fire Detection",
            filetypes=filetypes
        )
        
        root.destroy()
        return image_path
    
    def detect_fire_in_image(self, image_path):
        """Detect fire in the selected image"""
        print(f"\n🔍 Analyzing image: {image_path}")
        
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
            
            # If no detections with trained model, try improved color analysis
            if not fire_detected and not smoke_detections:
                print("🔍 No detections with trained model, trying improved color analysis...")
                fire_detected, fire_regions = self._analyze_fire_colors_improved(image)
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
                else:
                    print("✅ No fire patterns detected - avoiding false positives from sunlight")
            
            # Create annotated image
            annotated_image = self._create_annotated_image(image, fire_detections, smoke_detections)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save annotated image
            output_path = os.path.join(self.output_dir, f"{base_name}_analysis_{timestamp}.jpg")
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
        
        if fire_detected:
            print(f"🔥 Fire-like patterns detected in {len(fire_regions)} regions")
            print(f"   Total fire pixel ratio: {total_fire_ratio:.2%}")
            print(f"   Average fire score: {sum(r['fire_score'] for r in fire_regions) / len(fire_regions):.2f}")
        else:
            print("✅ No fire patterns detected - avoiding false positives from sunlight")
        
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
                location=f"Interactive Analysis - {os.path.basename(detection_result['original_image'])}"
            )
            
            if success:
                print("✅ Fire alert email sent successfully!")
            else:
                print("❌ Failed to send fire alert email")
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending fire alert: {e}")
            return False
    
    def send_no_fire_alert(self, detection_result):
        """Send email notification when no fire is detected"""
        if not self.notification_system:
            print("⚠️ No notification system available")
            return False
        
        try:
            print("📧 Sending no-fire notification email...")
            
            # Prepare detection data
            detection_data = {
                'fire_detections': [],
                'smoke_detections': [],
                'total_fire_detections': 0,
                'total_smoke_detections': 0,
                'timestamp': detection_result['timestamp'],
                'status': 'NO_FIRE_DETECTED'
            }
            
            # Send notification with image attachment
            success = self.notification_system.send_fire_alert(
                detection_data=detection_data,
                image_path=detection_result['annotated_image'],
                confidence=0.0,
                location=f"Interactive Analysis - {os.path.basename(detection_result['original_image'])} - NO FIRE DETECTED"
            )
            
            if success:
                print("✅ No-fire notification email sent successfully!")
            else:
                print("❌ Failed to send no-fire notification email")
            
            return success
            
        except Exception as e:
            print(f"❌ Error sending no-fire notification: {e}")
            return False
    
    def show_results_popup(self, fire_detected, result):
        """Show results in a popup dialog"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        if fire_detected:
            title = "🔥 FIRE DETECTED!"
            message = f"""Fire detected in your image!

Fire Detections: {result['total_fire_detections']}
Smoke Detections: {result['total_smoke_detections']}

Email alert has been sent!
Results saved to: {result['annotated_image']}

Click OK to view the annotated image."""
            messagebox.showwarning(title, message)
            
            # Ask if user wants to view the image
            view_image = messagebox.askyesno("View Image", "Do you want to view the annotated image?")
            if view_image:
                self._view_image(result['annotated_image'])
        else:
            title = "✅ No Fire Detected"
            message = f"""No fire or smoke detected in your image.

Results saved to: {result['annotated_image']}

Click OK to view the annotated image."""
            messagebox.showinfo(title, message)
            
            # Ask if user wants to view the image
            view_image = messagebox.askyesno("View Image", "Do you want to view the annotated image?")
            if view_image:
                self._view_image(result['annotated_image'])
        
        root.destroy()
    
    def _view_image(self, image_path):
        """View the annotated image"""
        try:
            # Try to open with default image viewer
            if os.name == 'nt':  # Windows
                os.startfile(image_path)
            elif os.name == 'posix':  # macOS and Linux
                os.system(f'open "{image_path}"')
            else:
                print(f"📸 Please open: {image_path}")
        except Exception as e:
            print(f"❌ Could not open image: {e}")
            print(f"📸 Please manually open: {image_path}")
    
    def run_interactive_detection(self):
        """Main interactive detection process"""
        print("=" * 60)
        print("🔥 INTERACTIVE IMAGE FIRE DETECTION SYSTEM")
        print("=" * 60)
        
        while True:
            try:
                # Select image
                print("\n📁 Please select an image from your device...")
                image_path = self.select_image()
                
                if not image_path:
                    print("❌ No image selected. Exiting...")
                    break
                
                print(f"✅ Image selected: {image_path}")
                
                # Detect fire
                fire_detected, result = self.detect_fire_in_image(image_path)
                
                if result is None:
                    print("❌ Failed to process image")
                    continue
                
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
                    
                    # Send fire alert email
                    self.send_fire_alert(result)
                    print("\n🔥 FIRE DETECTED: Email sent!")
                else:
                    print("\n✅ No fire or smoke detected in the image")
                    # Send "no fire" email notification
                    self.send_no_fire_alert(result)
                    print("📧 NO FIRE DETECTED: Email sent!")
                
                # Show results popup
                self.show_results_popup(fire_detected, result)
                
                # Ask if user wants to analyze another image
                root = tk.Tk()
                root.withdraw()
                continue_detection = messagebox.askyesno("Continue", "Do you want to analyze another image?")
                root.destroy()
                
                if not continue_detection:
                    print("👋 Thank you for using the Fire Detection System!")
                    break
                    
            except KeyboardInterrupt:
                print("\n👋 Detection cancelled by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

def main():
    """Main function"""
    try:
        detector = InteractiveFireDetector()
        detector.run_interactive_detection()
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
