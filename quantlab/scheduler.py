"""每日自动调度器 —— 因子工厂无人值守运行。

管线阶段在 quantlab.pipeline_stages 中定义，
此处只负责编排和告警汇总。

使用方式：
    python -m quantlab.scheduler run_daily
    python -m quantlab.scheduler install_cron
    python -m quantlab.scheduler status
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from quantlab.config import DEFAULT_CROSS_SECTION_DATA_PATH, DATA_DIR
from quantlab.factor_discovery.factor_enhancements import ExperienceLoop, OrthogonalityGuide
from quantlab.pipeline_stages import (
    PipelineContext,
    DataRefreshStage,
    DecayMonitorStage,
    EvolutionStage,
    OOSValidationStage,
    CombinationStage,
    GovernanceStage,
    DeliveryScreeningStage,
    PaperTradingStage,
    DeliveryReportStage,
)
from quantlab.pipeline_stages.combination import _benchmark_compare

logger = logging.getLogger(__name__)

SCHEDULER_DIR = DATA_DIR / "scheduler"
SCHEDULER_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG_PATH = SCHEDULER_DIR / "daily_runs.json"


# ---------------------------------------------------------------------------
# 1. 执行记录
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DailyRunRecord:
    """一次日常执行记录。"""
    run_id: str
    run_date: str
    start_time: str
    end_time: str = ""
    status: str = "running"

    data_refresh: dict[str, Any] = field(default_factory=dict)
    decay_monitor: dict[str, Any] = field(default_factory=dict)
    evolution: dict[str, Any] = field(default_factory=dict)
    screening: dict[str, Any] = field(default_factory=dict)
    oos_validation: dict[str, Any] = field(default_factory=dict)
    combination: dict[str, Any] = field(default_factory=dict)
    governance: dict[str, Any] = field(default_factory=dict)
    paper_trading: dict[str, Any] = field(default_factory=dict)
    delivery_reports: list[str] = field(default_factory=list)

    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_run_log() -> list[dict[str, Any]]:
    if not RUN_LOG_PATH.exists():
        return []
    try:
        return json.loads(RUN_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_run_log(records: list[dict[str, Any]]) -> None:
    records = records[-90:]
    RUN_LOG_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 2. 每日执行引擎
# ---------------------------------------------------------------------------

class DailyScheduler:
    """每日因子工厂调度引擎 —— 编排管线阶段并汇总告警。"""

    def __init__(
        self,
        data_path: Path | str | None = None,
        directions: list[str] | None = None,
        evolution_rounds: int = 3,
        max_candidates_per_round: int = 5,
        use_adaptive_directions: bool = True,
        use_multi_agent: bool = True,
    ) -> None:
        self.ctx = PipelineContext(
            data_path=Path(data_path or DEFAULT_CROSS_SECTION_DATA_PATH),
            directions=directions or [
                "momentum_reversal", "quality_earnings",
                "volume_price_divergence", "volatility_regime", "liquidity_premium",
            ],
            evolution_rounds=evolution_rounds,
            max_candidates_per_round=max_candidates_per_round,
            use_adaptive_directions=use_adaptive_directions,
            use_multi_agent=use_multi_agent,
        )
        self.experience_loop = ExperienceLoop()
        self.orth_guide = OrthogonalityGuide()
        self._alert_bus = None

    def run_daily(self) -> DailyRunRecord:
        """执行一次完整的日常任务。"""
        now = datetime.now()
        record = DailyRunRecord(
            run_id=f"daily_{now.strftime('%Y%m%d_%H%M%S')}",
            run_date=now.strftime("%Y-%m-%d"),
            start_time=now.isoformat(),
        )

        logger.info("=== 每日因子工厂启动 === run_id=%s", record.run_id)

        try:
            # 阶段 1: 增量数据刷新
            record.data_refresh = DataRefreshStage().run(self.ctx)
            logger.info("数据刷新完成: %s", record.data_refresh.get("status", "unknown"))

            # 阶段 2: 因子衰减监控
            record.decay_monitor = DecayMonitorStage().run(self.ctx)
            logger.info("衰减监控完成: %d 因子需再发掘", record.decay_monitor.get("decayed_count", 0))

            # 阶段 3: 进化搜索
            evolution_stage = EvolutionStage(
                experience_loop=self.experience_loop,
                orth_guide=self.orth_guide,
            )
            record.evolution = evolution_stage.run(self.ctx)
            logger.info("进化搜索完成: 新增 %d 因子", record.evolution.get("new_approved", 0))

            # 阶段 4: 样本外验证
            try:
                record.oos_validation = OOSValidationStage().run(self.ctx)
                logger.info("OOS验证完成: 通过 %d / 失败 %d",
                           record.oos_validation.get("passed", 0),
                           record.oos_validation.get("failed", 0))
            except Exception as exc:
                logger.warning("OOS验证失败: %s", exc)
                record.oos_validation = {"status": "failed", "error": str(exc)[:200]}

            # 阶段 5: 多因子组合
            try:
                record.combination = CombinationStage().run(self.ctx)
                logger.info("多因子组合完成: 组合IC=%.4f", record.combination.get("combined_ic", 0))
                if record.combination.get("status") == "success":
                    record.combination["benchmark"] = _benchmark_compare(self.ctx, record.combination)
            except Exception as exc:
                logger.warning("多因子组合失败: %s", exc)
                record.combination = {"status": "failed", "error": str(exc)[:200]}

            # 阶段 6: 交付标准筛选
            record.screening = DeliveryScreeningStage().run(self.ctx)
            logger.info("筛选完成: %d 可交付因子", record.screening.get("deliverable_count", 0))

            # 阶段 6.5: 启动纸交易
            deliverable_ids = record.screening.get("deliverable_factor_ids", [])
            self.ctx._meta = {"deliverable_factor_ids": deliverable_ids}
            try:
                record.paper_trading = PaperTradingStage().run(self.ctx)
            except Exception as exc:
                logger.warning("纸交易启动失败: %s", exc)
                record.paper_trading = {"status": "failed", "error": str(exc)[:200]}

            # 阶段 7: 因子库治理
            try:
                record.governance = GovernanceStage().run(self.ctx)
                logger.info("因子库治理完成: 归档 %d 个因子", record.governance.get("archived_count", 0))
            except Exception as exc:
                logger.warning("因子库治理失败: %s", exc)
                record.governance = {"status": "failed", "error": str(exc)[:200]}

            # 阶段 8: 生成交付报告
            record.delivery_reports = DeliveryReportStage().run(self.ctx)
            logger.info("交付报告生成: %d 份", len(record.delivery_reports))

            record.status = "success"

        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)[:500]
            logger.error("日常任务失败: %s", exc)

        record.end_time = datetime.now().isoformat()

        # 告警汇总
        self._emit_alerts(record)

        # 保存记录
        log = _load_run_log()
        log.append(record.to_dict())
        _save_run_log(log)

        logger.info("=== 每日因子工厂结束 === status=%s", record.status)
        return record

    def _emit_alerts(self, record: DailyRunRecord) -> None:
        alert_bus = self._get_alert_bus()
        if record.data_refresh.get("status") == "skipped":
            alert_bus.warning("数据刷新跳过", record.data_refresh.get("reason", ""), source="data_refresh")
        if record.decay_monitor.get("decayed_count", 0) > 0:
            alert_bus.warning("因子衰减告警", f"{record.decay_monitor.get('decayed_count')} 个因子已衰减", source="decay_monitor")
        if record.governance.get("crowding", {}).get("crowded_factor_ids"):
            alert_bus.warning("拥挤度告警", "发现拥挤因子", source="crowding",
                             ids=record.governance["crowding"]["crowded_factor_ids"])
        risk = record.governance.get("risk", {})
        if risk.get("breaches"):
            alert_bus.critical("风控告警", "; ".join(risk["breaches"][:3]), source="risk_control")
        if record.governance.get("regime", {}).get("current") == "bear":
            alert_bus.info("市场状态: 熊市", "因子表现可能出现系统性下降", source="regime")
        if record.status == "failed":
            alert_bus.critical("每日任务失败", record.error_message[:200], source="scheduler")

        summary = alert_bus.summary()
        if summary.get("critical", 0) > 0:
            record.status = "partial" if record.status == "success" else record.status
        if any(summary.values()):
            logger.info("告警汇总: info=%d warning=%d critical=%d",
                       summary.get("info", 0), summary.get("warning", 0), summary.get("critical", 0))

    def _get_alert_bus(self):
        if self._alert_bus is None:
            from quantlab.assistant.notifier import AlertBus
            self._alert_bus = AlertBus()
        return self._alert_bus


# ---------------------------------------------------------------------------
# 3. Windows 计划任务安装
# ---------------------------------------------------------------------------

def install_windows_task(task_name: str = "QuantAgentDaily", hour: int = 18, minute: int = 30) -> dict[str, str]:
    python_exe = sys.executable
    script = f'"{python_exe}" -m quantlab.scheduler run_daily'
    cmd = ["schtasks", "/Create", "/TN", task_name, "/TR", script,
           "/SC", "DAILY", "/ST", f"{hour:02d}:{minute:02d}", "/F"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"status": "installed", "task_name": task_name, "schedule": f"每日 {hour:02d}:{minute:02d}"}
        return {"status": "error", "message": result.stderr.strip()[:500]}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:500]}


def remove_windows_task(task_name: str = "QuantAgentDaily") -> dict[str, str]:
    cmd = ["schtasks", "/Delete", "/TN", task_name, "/F"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"status": "removed", "task_name": task_name}
        return {"status": "error", "message": result.stderr.strip()[:500]}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:500]}


# ---------------------------------------------------------------------------
# 4. CLI 入口
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="quant-agent 每日调度器")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run_daily", help="执行一次日常任务")
    sub.add_parser("status", help="查看最近执行记录")

    cron_parser = sub.add_parser("install_cron", help="安装每日定时任务")
    cron_parser.add_argument("--hour", type=int, default=18)
    cron_parser.add_argument("--minute", type=int, default=30)

    sub.add_parser("remove_cron", help="移除定时任务")

    args = parser.parse_args()

    if args.command == "run_daily":
        scheduler = DailyScheduler()
        record = scheduler.run_daily()
        print(json.dumps(record.to_dict(), ensure_ascii=False, indent=2, default=str))

    elif args.command == "status":
        records = _load_run_log()
        if not records:
            print("暂无执行记录。")
        else:
            for r in records[-10:]:
                status_icon = {
                    "success": "[OK]", "partial": "[~]", "failed": "[!!]", "running": "[..]",
                }.get(r.get("status", ""), "[?]")
                date = r.get("run_date", "?")
                new = r.get("evolution", {}).get("new_approved", 0)
                delivered = r.get("screening", {}).get("deliverable_count", 0)
                print(f"{status_icon} {date} | 新增={new} | 可交付={delivered} | {r.get('status', '')}")

    elif args.command == "install_cron":
        result = install_windows_task(hour=args.hour, minute=args.minute)
        print(json.dumps(result, ensure_ascii=False))

    elif args.command == "remove_cron":
        result = remove_windows_task()
        print(json.dumps(result, ensure_ascii=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
