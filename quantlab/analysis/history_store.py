from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import uuid

import pandas as pd

from quantlab.config import HISTORY_REPORT_DIR


def _normalize_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _stringify_payload(payload: dict) -> dict:
    return {key: _normalize_value(value) for key, value in payload.items()}


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
        metrics = record.get("metrics", {})
        notes = record.get("notes", {})
        row = {
            "experiment_id": record.get("experiment_id"),
            "timestamp": record.get("timestamp"),
            "experiment_type": record.get("experiment_type"),
            "fold_count": notes.get("overview", {}).get("fold_count"),
            "primary_metric": metrics.get("annual_return")
            or metrics.get("test_annual_return")
            or metrics.get("average_test_annual_return")
            or metrics.get("train_annual_return"),
            **metrics,
            "notes": json.dumps(notes, ensure_ascii=False),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    return df


def load_experiment_detail(experiment_id: str, history_dir: str | Path = HISTORY_REPORT_DIR) -> dict | None:
    history_path = Path(history_dir)
    target = history_path / f"{experiment_id}.json"
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))
