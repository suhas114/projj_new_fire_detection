#!/usr/bin/env python3
"""
Simple test script for image fire detection
"""

from image_fire_detection import ImageFireDetector
import os

def test_with_sample_images():
    """Test fire detection with sample images"""
    print("🔥 Testing Image Fire Detection System")
    print("=" * 50)
    
    # Initialize detector
    detector = ImageFireDetector()
    
    # Look for test images in various directories
    test_dirs = [
        "captured_fires",
        "demo_fires", 
        "firee/Forest_Fire_BigData/fire_images",
        "dataset/images/test"
    ]
    
    test_images = []
    for directory in test_dirs:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(directory, file))
                    break  # Just take one image from each directory
    
    if not test_images:
        print("❌ No test images found in expected directories")
        print("Please provide an image path manually:")
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            test_images = [image_path]
        else:
            print("❌ Invalid image path")
            return
    
    print(f"📸 Found {len(test_images)} test images")
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n--- Test {i}/{len(test_images)} ---")
        fire_detected = detector.process_image(image_path)
        
        if fire_detected:
            print("🔥 FIRE ALERT: Email sent!")
        else:
            print("✅ No fire detected")
        
        print("-" * 40)

if __name__ == "__main__":
    test_with_sample_images()
