#!/usr/bin/env python3
"""
Simple launcher for Real-time Camera Fire Detection
"""

import subprocess
import sys
import os

def main():
    """Launch the camera fire detection system"""
    script_path = "realtime_camera_fire_detection.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found!")
        sys.exit(1)
    
    print("Starting Real-time Camera Fire Detection System...")
    print("Make sure email_config.json is configured for notifications")
    print("Make sure your camera is connected and working")
    print("=" * 60)
    
    try:
        # Run the camera detection script
        subprocess.run([sys.executable, script_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Camera detection stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running camera detection: {e}")
    except FileNotFoundError:
        print("❌ Error: Python interpreter not found")
        sys.exit(1)

if __name__ == "__main__":
    main()
