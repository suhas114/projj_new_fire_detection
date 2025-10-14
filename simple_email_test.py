#!/usr/bin/env python3
"""
Simple email test without Unicode characters
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json

def test_email():
    """Test email configuration"""
    print("Testing Email Configuration...")
    print("=" * 50)
    
    # Load config
    try:
        with open('email_config.json', 'r') as f:
            config = json.load(f)
        print("Config loaded successfully")
    except Exception as e:
        print(f"Failed to load config: {e}")
        return False
    
    sender_email = config.get("sender_email")
    app_password = config.get("app_password")
    recipient_email = config.get("recipient_email")
    
    print(f"Sender: {sender_email}")
    print(f"Recipient: {recipient_email}")
    
    # Test email sending
    try:
        print("Attempting to send test email...")
        
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = "Fire Detection System - Test Email"
        
        body = f"""
This is a test email from your Fire Detection System.

Sender: {sender_email}
Recipient: {recipient_email}

If you receive this email, your configuration is working!
        """
        
        msg.attach(MIMEText(body, "plain"))
        
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        
        print("SUCCESS: Test email sent!")
        print("Check your inbox and spam folder")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        print(f"AUTHENTICATION FAILED: {e}")
        print("Check your app password")
        return False
    except Exception as e:
        print(f"EMAIL SEND FAILED: {e}")
        return False

if __name__ == "__main__":
    test_email()