#!/usr/bin/env python3
"""
Test the notification system directly
"""

from notification_system import NotificationSystem

def test_notification():
    """Test notification system"""
    print("Testing Notification System...")
    print("=" * 40)
    
    try:
        # Initialize notification system
        notification_system = NotificationSystem("email_config.json")
        
        print(f"Email enabled: {notification_system.email_enabled}")
        print(f"SMS enabled: {notification_system.sms_enabled}")
        
        if notification_system.email_enabled:
            print("Sending test notification...")
            success = notification_system.send_test_notification()
            if success:
                print("SUCCESS: Test notification sent!")
                return True
            else:
                print("FAILED: Test notification failed")
                return False
        else:
            print("Email not enabled in notification system")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    test_notification()
