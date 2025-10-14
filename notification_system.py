import os
import json
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Any, Dict, Optional


class NotificationSystem:
    def __init__(self, config: Optional[str | Dict[str, Any]] = None) -> None:
        """Initialize notification system from a JSON file path or a dict.

        Supports email (Gmail app password) and Twilio SMS if configured.
        """
        self.config: Dict[str, Any] = self._load_config(config)

        # Email
        self.email_enabled: bool = bool(
            self.config.get("sender_email") and self.config.get("app_password") and self.config.get("recipient_email")
        )

        # SMS (Twilio)
        self.sms_enabled: bool = bool(
            self.config.get("twilio_account_sid")
            and self.config.get("twilio_auth_token")
            and self.config.get("twilio_from")
            and self.config.get("twilio_to")
        )

    def _load_config(self, config: Optional[str | Dict[str, Any]]) -> Dict[str, Any]:
        # Base from environment
        cfg: Dict[str, Any] = {
            "sender_email": os.environ.get("ALERT_SENDER_EMAIL"),
            "app_password": os.environ.get("ALERT_APP_PASSWORD"),
            "recipient_email": os.environ.get("ALERT_RECIPIENT_EMAIL"),
            "twilio_account_sid": os.environ.get("TWILIO_ACCOUNT_SID"),
            "twilio_auth_token": os.environ.get("TWILIO_AUTH_TOKEN"),
            "twilio_from": os.environ.get("TWILIO_FROM_NUMBER"),
            "twilio_to": os.environ.get("TWILIO_TO_NUMBER"),
        }

        if isinstance(config, str) and os.path.exists(config):
            try:
                with open(config, "r", encoding="utf-8") as f:
                    file_cfg = json.load(f)
                cfg.update({k: v for k, v in file_cfg.items() if v not in (None, "")})
            except Exception:
                # Ignore config file errors; fall back to env-only
                pass
        elif isinstance(config, dict):
            cfg.update({k: v for k, v in config.items() if v not in (None, "")})
        return cfg

    def send_fire_alert(
        self,
        detection_data: Dict[str, Any],
        image_path: Optional[str] = None,
        confidence: Optional[float] = None,
        location: Optional[str] = None,
    ) -> bool:
        """Send fire alert via available channels. Returns True if any succeeded."""
        subject = "FIRE DETECTED - Alert"
        body_lines = [
            "FIRE DETECTION ALERT",
            f"Confidence: {confidence:.2f}" if isinstance(confidence, (int, float)) else None,
            f"Location: {location}" if location else None,
            f"Details: {json.dumps(detection_data, ensure_ascii=False)}" if detection_data else None,
        ]
        body = "\n".join([line for line in body_lines if line])

        any_sent = False
        if self.email_enabled:
            any_sent = self._send_email(subject, body, image_path) or any_sent
        if self.sms_enabled:
            any_sent = self._send_sms(f"FIRE ALERT {('('+location+')') if location else ''} Conf: {confidence}") or any_sent
        return any_sent

    def send_test_notification(self) -> bool:
        """Send a simple test notification."""
        return self.send_fire_alert({"test": True}, None, 1.0, "TEST")

    def _send_email(self, subject: str, body: str, image_path: Optional[str]) -> bool:
        try:
            msg = MIMEMultipart()
            msg["From"] = self.config["sender_email"]
            msg["To"] = self.config["recipient_email"]
            msg["Subject"] = subject
            msg.attach(MIMEText(body, "plain"))

            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img = MIMEImage(f.read())
                img.add_header("Content-Disposition", "attachment", filename=os.path.basename(image_path))
                msg.attach(img)

            context = ssl.create_default_context()
            with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                server.login(self.config["sender_email"], self.config["app_password"])
                server.send_message(msg)
            print("Email alert sent")
            return True
        except Exception as e:
            print(f"Email send failed: {e}")
            return False

    def _send_sms(self, message: str) -> bool:
        try:
            from twilio.rest import Client  # lazy import

            client = Client(self.config["twilio_account_sid"], self.config["twilio_auth_token"])
            client.messages.create(
                body=message,
                from_=self.config["twilio_from"],
                to=self.config["twilio_to"],
            )
            print("SMS alert sent")
            return True
        except Exception as e:
            print(f"SMS send failed: {e}")
            return False



    def _create_alert_message(
        self,
        detection_data: Dict[str, Any],
        confidence: Optional[float] = None,
        location: Optional[str] = None,
    ) -> str:
        """Create a human-readable alert message preview.

        This mirrors the content used by send_fire_alert but formatted for CLI preview.
        """
        lines: list[str] = ["FIRE DETECTION ALERT"]

        if isinstance(confidence, (int, float)):
            lines.append(f"Confidence: {confidence:.2%}")
        if location:
            lines.append(f"Location: {location}")

        # Add common detection details if present
        if isinstance(detection_data, dict) and detection_data:
            bbox = detection_data.get("bbox")
            if bbox is not None:
                lines.append(f"Bounding Box: {bbox}")
            class_id = detection_data.get("class_id")
            if class_id is not None:
                lines.append(f"Class ID: {class_id}")
            coords = detection_data.get("coordinates") or detection_data.get("location")
            if coords:
                lines.append(f"Coordinates: {coords}")

            # Fallback to raw JSON if nothing specific was extracted
            if len(lines) <= 3:  # only header + maybe 1-2 lines added
                try:
                    lines.append(f"Details: {json.dumps(detection_data, ensure_ascii=False)}")
                except Exception:
                    lines.append(f"Details: {detection_data}")

        return "\n".join(lines)
