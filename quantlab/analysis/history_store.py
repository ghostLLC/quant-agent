from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import json
import uuid

import pandas as pd

from quantlab.config import HISTORY_REPORT_DIR


def _stringify_payload(payload: dict) -> dict:
    normalized = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            normalized[key] = str(value)
        else:
            normalized[key] = value
    return normalized


def save_experiment_record(
    experiment_type: str,
    config_payload: dict,
    metrics_payload: dict,
    notes: dict | None = None,
    history_dir: str | Path = HISTORY_REPORT_DIR,
) -> Path:
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    record = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_type": experiment_type,
        "config": _stringify_payload(config_payload),
        "metrics": _stringify_payload(metrics_payload),
        "notes": _stringify_payload(notes or {}),
    }

    output_path = history_path / f"{experiment_id}.json"
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def load_experiment_history(history_dir: str | Path = HISTORY_REPORT_DIR) -> pd.DataFrame:
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for file in sorted(history_path.glob("*.json"), reverse=True):
        record = json.loads(file.read_text(encoding="utf-8"))
        row = {
            "experiment_id": record.get("experiment_id"),
            "timestamp": record.get("timestamp"),
            "experiment_type": record.get("experiment_type"),
            **record.get("metrics", {}),
            "notes": json.dumps(record.get("notes", {}), ensure_ascii=False),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def load_experiment_detail(experiment_id: str, history_dir: str | Path = HISTORY_REPORT_DIR) -> dict | None:
    history_path = Path(history_dir)
    target = history_path / f"{experiment_id}.json"
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
