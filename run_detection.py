#!/usr/bin/env python3
"""
Simple script to run fire detection with notifications always enabled
"""

from realtime_detection_with_notifications import EnhancedDetectionVisualizer

def main():
    """Run detection system with notifications always enabled"""
    print("Starting Fire Detection System with Notifications...")
    
    # Initialize with default email config
    detector = EnhancedDetectionVisualizer()
    
    # Run detection
    detector.run_detection()

if __name__ == "__main__":
    main()
