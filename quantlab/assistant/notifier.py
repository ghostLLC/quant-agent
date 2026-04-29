"""通知与告警系统 —— 因子工厂无人值守运行的感知层。

设计原则：
- 文件日志为基础（始终可用）
- 扩展点接口支持 Telegram/Email/Webhook
- 与 DailyScheduler 集成，关键事件自动触发
- 所有通知经过 AlertBus，下游可插拔
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    source: str  # e.g. "data_refresh", "decay_monitor", "crowding", "drawdown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class AlertBus:
    """告警总线 —— 单例，收集本次运行的所有告警。"""

    def __init__(self) -> None:
        self._alerts: list[Alert] = []

    def emit(self, level: AlertLevel, title: str, message: str, source: str = "", metadata: dict | None = None) -> Alert:
        from uuid import uuid4
        alert = Alert(
            alert_id=uuid4().hex[:8],
            level=level,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {},
        )
        self._alerts.append(alert)
        log_fn = {"info": logger.info, "warning": logger.warning, "critical": logger.error}.get(level.value, logger.info)
        log_fn("[%s] %s: %s", level.value.upper(), title, message)
        return alert

    def info(self, title: str, message: str, source: str = "", **metadata) -> Alert:
        return self.emit(AlertLevel.INFO, title, message, source, metadata)

    def warning(self, title: str, message: str, source: str = "", **metadata) -> Alert:
        return self.emit(AlertLevel.WARNING, title, message, source, metadata)

    def critical(self, title: str, message: str, source: str = "", **metadata) -> Alert:
        return self.emit(AlertLevel.CRITICAL, title, message, source, metadata)

    def all_alerts(self) -> list[Alert]:
        return list(self._alerts)

    def by_level(self, level: AlertLevel) -> list[Alert]:
        return [a for a in self._alerts if a.level == level]

    def has_critical(self) -> bool:
        return any(a.level == AlertLevel.CRITICAL for a in self._alerts)

    def summary(self) -> dict[str, int]:
        counts = {"info": 0, "warning": 0, "critical": 0}
        for a in self._alerts:
            counts[a.level.value] += 1
        return counts


class Notifier:
    """通知器 —— 支持文件持久化和可扩展通道。

    扩展方式:
        notifier = Notifier()
        notifier.add_channel("telegram", telegram_handler)
        notifier.add_channel("email", email_handler)
    """

    def __init__(self, store_dir: str | Path | None = None) -> None:
        self.store_dir = Path(store_dir or (Path(__file__).resolve().parents[2] / "assistant_data" / "alerts"))
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._channels: dict[str, Callable[[Alert], None]] = {}
        self._alert_log_path = self.store_dir / "alert_log.json"

    def add_channel(self, name: str, handler: Callable[[Alert], None]) -> None:
        """注册告警通道。handler 接收 Alert 对象。"""
        self._channels[name] = handler

    def notify(self, alert: Alert) -> None:
        """发送告警到所有通道。"""
        # Always persist to file
        self._persist(alert)
        # Dispatch to channels
        for ch_name, handler in self._channels.items():
            try:
                handler(alert)
            except Exception as exc:
                logger.warning("告警通道 %s 发送失败: %s", ch_name, exc)

    def notify_all(self, alerts: list[Alert]) -> None:
        for a in alerts:
            self.notify(a)

    def _persist(self, alert: Alert) -> None:
        try:
            existing = []
            if self._alert_log_path.exists():
                existing = json.loads(self._alert_log_path.read_text(encoding="utf-8"))
            existing.append(alert.to_dict())
            existing = existing[-500:]  # keep last 500 alerts
            self._alert_log_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("告警持久化失败: %s", exc)

    def recent(self, n: int = 20) -> list[dict]:
        try:
            if self._alert_log_path.exists():
                data = json.loads(self._alert_log_path.read_text(encoding="utf-8"))
                return data[-n:]
        except Exception:
            pass
        return []


def create_telegram_handler(bot_token: str, chat_id: str) -> Callable[[Alert], None] | None:
    """创建 Telegram 通知处理器（如果依赖可用）。"""
    try:
        from urllib import request as urllib_req

        def _send(alert: Alert) -> None:
            text = f"*{alert.level.value.upper()}*: {alert.title}\n{alert.message}\n`{alert.source}`"
            payload = json.dumps({
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode("utf-8")
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            req = urllib_req.Request(url, data=payload, headers={"Content-Type": "application/json"})
            urllib_req.urlopen(req, timeout=10)

        return _send
    except Exception:
        return None
