"""Email notification via QQ SMTP for critical alerts and daily summaries.

Configuration in .env:
  EMAIL_SMTP_USER=your_qq_email@qq.com
  EMAIL_SMTP_PASS=your_qq_smtp_auth_code  (NOT the QQ password — use SMTP auth code)

QQ SMTP: smtp.qq.com:587, TLS
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Non-blocking email notification via QQ SMTP."""

    def __init__(
        self,
        smtp_host: str = "smtp.qq.com",
        smtp_port: int = 587,
        recipient: str = "676236147@qq.com",
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.recipient = recipient
        self._user: str = ""
        self._password: str = ""
        self._load_config()

    def _load_config(self) -> None:
        """Load SMTP credentials from .env or environment."""
        self._user = os.environ.get("EMAIL_SMTP_USER", "")
        self._password = os.environ.get("EMAIL_SMTP_PASS", "")
        if not self._user or not self._password:
            env_path = Path(__file__).resolve().parents[2] / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("EMAIL_SMTP_USER="):
                        self._user = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("EMAIL_SMTP_PASS="):
                        self._password = line.split("=", 1)[1].strip().strip('"').strip("'")

    @property
    def available(self) -> bool:
        return bool(self._user and self._password)

    def send(self, subject: str, body: str, is_html: bool = False) -> bool:
        """Send email in background thread. Returns True if queued."""
        if not self.available:
            logger.debug("Email not configured — skipping notification")
            return False

        def _do_send() -> None:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = subject
                msg["From"] = self._user
                msg["To"] = self.recipient
                msg.attach(MIMEText(body, "html" if is_html else "plain", "utf-8"))

                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as server:
                    server.starttls()
                    server.login(self._user, self._password)
                    server.send_message(msg)
                logger.info("Email sent: %s", subject)
            except Exception as exc:
                logger.warning("Email发送失败 (%s): %s", subject, exc)

        thread = threading.Thread(target=_do_send, daemon=True)
        thread.start()
        return True

    def send_alert(self, alert: dict[str, Any]) -> None:
        """Send a single alert notification."""
        subject = f"[QuantAgent] {alert.get('level', 'info').upper()}: {alert.get('message', 'Alert')[:60]}"
        body = json.dumps(alert, ensure_ascii=False, indent=2, default=str)
        self.send(subject, body)

    def send_daily_summary(self, record: dict[str, Any]) -> None:
        """Send daily pipeline summary report."""
        status = record.get("status", "unknown")
        status_emoji = {"success": "OK", "partial": "~", "failed": "!!!"}.get(status, "?")
        run_date = record.get("run_date", "?")
        evo = record.get("evolution", {})
        oos = record.get("oos_validation", {})
        screening = record.get("screening", {})
        gov = record.get("governance", {})

        body = f"""
        <h2>QuantAgent 日度报告 — {run_date}</h2>
        <p><b>Status:</b> {status_emoji} {status}</p>
        <h3>Pipeline Results</h3>
        <table border='1' cellpadding='4' cellspacing='0'>
        <tr><td>新增因子</td><td>{evo.get('new_approved', '?')}</td></tr>
        <tr><td>OOS 通过/失败</td><td>{oos.get('passed','?')} / {oos.get('failed','?')}</td></tr>
        <tr><td>可交付因子</td><td>{screening.get('deliverable_count','?')}</td></tr>
        <tr><td>市场状态</td><td>{gov.get('regime',{}).get('current','?') if isinstance(gov.get('regime'), dict) else '?'}</td></tr>
        </table>
        <p><i>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i></p>
        """
        self.send(f"[QuantAgent] 日度报告 {run_date} — {status_emoji}", body, is_html=True)
