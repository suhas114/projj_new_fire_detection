# Email Alert Setup Guide

## 🔧 Quick Setup

### 1. Configure Email Settings

Edit `email_config.json` with your Gmail credentials:

```json
{
    "sender_email": "your-email@gmail.com",
    "app_password": "your-16-character-app-password",
    "recipient_email": "alert-recipient@example.com",
    "twilio_account_sid": "",
    "twilio_auth_token": "",
    "twilio_from": "",
    "twilio_to": ""
}
```

### 2. Get Gmail App Password

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate App Password:**
   - Go to [Google Account Settings](https://myaccount.google.com/)
   - Security → 2-Step Verification → App passwords
   - Select "Mail" and generate password
   - Copy the 16-character password (no spaces)

### 3. Update Configuration

Replace the placeholders in `email_config.json`:
- `your-email@gmail.com` → Your Gmail address
- `your-16-character-app-password` → Your app password
- `alert-recipient@example.com` → Where to send alerts

### 4. Run with Email Alerts

```bash
python realtime_detection_with_notifications.py --config email_config.json
```

## 🧪 Test Email Setup

Test your email configuration:

```bash
python demo_notification.py
```

## 📧 Email Alert Features

- **Automatic Alerts**: Sends email when fire is detected
- **Image Attachments**: Includes captured fire image
- **Alert Details**: Confidence level, location, timestamp
- **Cooldown Period**: Prevents spam (1 minute default)

## 🔒 Security Notes

- Never commit real credentials to git
- Use app passwords, not your main Gmail password
- Keep `email_config.json` in `.gitignore`

## 🛠️ Troubleshooting

### "Authentication failed"
- Verify 2FA is enabled
- Check app password is correct
- Ensure no extra spaces in password

### "No module named 'smtplib'"
- `smtplib` is built-in to Python, should work by default

### "Email send failed"
- Check internet connection
- Verify Gmail credentials
- Try test notification first

## 📱 Alternative: Environment Variables

Instead of config file, you can use environment variables:

```bash
set ALERT_SENDER_EMAIL=your-email@gmail.com
set ALERT_APP_PASSWORD=your-app-password
set ALERT_RECIPIENT_EMAIL=recipient@example.com
python realtime_detection_with_notifications.py
```

## 🎯 Next Steps

1. Set up email credentials
2. Test with demo
3. Run realtime detection
4. Monitor for fire alerts!
