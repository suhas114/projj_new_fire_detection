#!/usr/bin/env python3
"""
Simple email test script to verify Gmail configuration
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import os

def test_email_config(config_file="email_config.json"):
    """Test email configuration"""
    print("Testing Email Configuration...")
    print("=" * 50)
    
    # Load config
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"✅ Config loaded from: {config_file}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    # Check required fields
    sender_email = config.get("sender_email")
    app_password = config.get("app_password")
    recipient_email = config.get("recipient_email")
    
    if not sender_email:
        print("❌ Missing sender_email in config")
        return False
    if not app_password:
        print("❌ Missing app_password in config")
        return False
    if not recipient_email:
        print("❌ Missing recipient_email in config")
        return False
    
    print(f"📧 Sender: {sender_email}")
    print(f"📬 Recipient: {recipient_email}")
    print(f"🔑 App Password: {'*' * len(app_password)}")
    
    # Test email sending
    try:
        print("\n🚀 Attempting to send test email...")
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = "🔥 Fire Detection System - Test Email"
        
        body = """
This is a test email from your Fire Detection System.

If you receive this email, your email configuration is working correctly!

System Details:
- Sender: {sender}
- Recipient: {recipient}
- Test Time: {time}

Your fire detection system is ready to send alerts!
        """.format(
            sender=sender_email,
            recipient=recipient_email,
            time=__import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        msg.attach(MIMEText(body, "plain"))
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        
        print("✅ Test email sent successfully!")
        print("📬 Check your inbox (and spam folder)")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"❌ Authentication failed: {e}")
        print("💡 Check your app password - make sure it's correct and no spaces")
        return False
    except smtplib.SMTPRecipientsRefused as e:
        print(f"❌ Recipient refused: {e}")
        return False
    except Exception as e:
        print(f"❌ Email send failed: {e}")
        return False

def test_notification_system():
    """Test the notification system directly"""
    print("\n🔧 Testing Notification System...")
    print("=" * 50)
    
    try:
        from notification_system import NotificationSystem
        
        notification_system = NotificationSystem("email_config.json")
        
        print(f"📧 Email enabled: {notification_system.email_enabled}")
        print(f"📱 SMS enabled: {notification_system.sms_enabled}")
        
        if notification_system.email_enabled:
            print("\n🚀 Sending test notification...")
            success = notification_system.send_test_notification()
            if success:
                print("✅ Notification system test successful!")
                return True
            else:
                print("❌ Notification system test failed")
                return False
        else:
            print("❌ Email not enabled in notification system")
            return False
            
    except Exception as e:
        print(f"❌ Notification system test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Fire Detection Email Test")
    print("=" * 60)
    
    # Test 1: Direct email test
    email_success = test_email_config()
    
    # Test 2: Notification system test
    notification_success = test_notification_system()
    
    print("\n📊 Test Results:")
    print("=" * 30)
    print(f"Direct Email Test: {'✅ PASS' if email_success else '❌ FAIL'}")
    print(f"Notification System: {'✅ PASS' if notification_success else '❌ FAIL'}")
    
    if email_success and notification_success:
        print("\n🎉 All tests passed! Your email system is ready!")
        print("Run: python realtime_detection_with_notifications.py --config email_config.json")
    else:
        print("\n🔧 Troubleshooting needed:")
        if not email_success:
            print("- Check Gmail app password")
            print("- Verify 2FA is enabled")
            print("- Check internet connection")
        if not notification_success:
            print("- Check notification system configuration")

if __name__ == "__main__":
    main()
