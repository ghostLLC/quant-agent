"""每日自动调度器 —— 因子工厂无人值守运行。

核心流程：
1. 增量刷新横截面数据（只拉新交易日）
2. 运行因子衰减监控（检查已有因子 IC 是否衰减）
3. 触发进化搜索（对衰减因子或按计划发掘新因子）
4. 交付标准自动筛选（只保留可卖因子）
5. 生成交付报告（可交付因子输出 JSON + Markdown）

使用方式：
    # 手动执行一次日常任务
    python -m quantlab.scheduler run_daily

    # 启动 Windows 计划任务（每日 18:30 执行）
    python -m quantlab.scheduler install_cron

    # 查看最近执行记录
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

import numpy as np
import pandas as pd

from quantlab.config import DEFAULT_CROSS_SECTION_DATA_PATH, DATA_DIR

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
    status: str = "running"  # running / success / partial / failed

    # 各阶段结果摘要
    data_refresh: dict[str, Any] = field(default_factory=dict)
    decay_monitor: dict[str, Any] = field(default_factory=dict)
    evolution: dict[str, Any] = field(default_factory=dict)
    screening: dict[str, Any] = field(default_factory=dict)
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
    # 只保留最近 90 条
    records = records[-90:]
    RUN_LOG_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# 2. 每日执行引擎
# ---------------------------------------------------------------------------

class DailyScheduler:
    """每日因子工厂调度引擎。"""

    def __init__(
        self,
        data_path: Path | str | None = None,
        directions: list[str] | None = None,
        evolution_rounds: int = 3,
        max_candidates_per_round: int = 5,
    ) -> None:
        self.data_path = Path(data_path or DEFAULT_CROSS_SECTION_DATA_PATH)
        self.directions = directions or [
            "momentum_reversal",
            "quality_earnings",
            "volume_price_divergence",
            "volatility_regime",
            "liquidity_premium",
        ]
        self.evolution_rounds = evolution_rounds
        self.max_candidates_per_round = max_candidates_per_round

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
            # ---- 阶段 1: 增量数据刷新 ----
            record.data_refresh = self._refresh_data()
            logger.info("数据刷新完成: %s", record.data_refresh.get("status", "unknown"))

            # ---- 阶段 2: 因子衰减监控 ----
            record.decay_monitor = self._monitor_decay()
            logger.info("衰减监控完成: %d 因子需再发掘", record.decay_monitor.get("decayed_count", 0))

            # ---- 阶段 3: 进化搜索 ----
            record.evolution = self._run_evolution()
            logger.info("进化搜索完成: 新增 %d 因子", record.evolution.get("new_approved", 0))

            # ---- 阶段 4: 交付标准筛选 ----
            record.screening = self._screen_deliverable()
            logger.info("筛选完成: %d 可交付因子", record.screening.get("deliverable_count", 0))

            # ---- 阶段 5: 生成交付报告 ----
            record.delivery_reports = self._generate_delivery_reports(
                record.screening.get("deliverable_factor_ids", [])
            )
            logger.info("交付报告生成: %d 份", len(record.delivery_reports))

            record.status = "success"

        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)[:500]
            logger.error("日常任务失败: %s", exc)

        record.end_time = datetime.now().isoformat()

        # 保存记录
        log = _load_run_log()
        log.append(record.to_dict())
        _save_run_log(log)

        logger.info("=== 每日因子工厂结束 === status=%s", record.status)
        return record

    # -- 阶段实现 --

    def _refresh_data(self) -> dict[str, Any]:
        """增量刷新横截面数据。"""
        try:
            from quantlab.data.tushare_provider import AkShareIncrementalProvider
            provider = AkShareIncrementalProvider()
            result = provider.refresh_cross_section(self.data_path)
            return {"status": "success", "result": result}
        except Exception as exc:
            logger.warning("增量刷新失败，尝试 Tushare Pro: %s", exc)
            try:
                from quantlab.data.tushare_provider import TushareProProvider
                provider = TushareProProvider()
                if provider.available:
                    result = provider.refresh_cross_section(self.data_path)
                    return {"status": "success_fallback", "result": result}
            except Exception as exc2:
                logger.warning("Tushare Pro 也失败: %s", exc2)
            return {"status": "skipped", "reason": str(exc)[:200]}

    def _monitor_decay(self) -> dict[str, Any]:
        """监控因子衰减。"""
        from quantlab.factor_discovery.decay_monitor import FactorDecayMonitor
        monitor = FactorDecayMonitor(data_path=self.data_path)
        return monitor.check_all()

    def _run_evolution(self) -> dict[str, Any]:
        """运行因子进化搜索。"""
        from quantlab.factor_discovery.evolution import EvolutionConfig, FactorEvolutionLoop
        from quantlab.factor_discovery.runtime import PersistentFactorStore

        store = PersistentFactorStore()
        hub = self._load_data()
        if hub.empty:
            return {"status": "skipped", "reason": "数据为空"}

        total_approved = 0
        all_results = []

        for direction in self.directions:
            try:
                loop = FactorEvolutionLoop(
                    store=store,
                    config=EvolutionConfig(
                        max_rounds=self.evolution_rounds,
                        candidates_per_round=self.max_candidates_per_round,
                    ),
                )
                result = loop.run(direction=direction, market_df=hub)
                approved = result.get("approved_count", 0)
                total_approved += approved
                all_results.append({
                    "direction": direction,
                    "approved": approved,
                    "total_candidates": result.get("total_candidates", 0),
                    "best_score": result.get("best_score", 0.0),
                })
            except Exception as exc:
                logger.warning("方向 %s 进化失败: %s", direction, exc)
                all_results.append({"direction": direction, "error": str(exc)[:200]})

        return {
            "status": "success",
            "new_approved": total_approved,
            "directions": all_results,
        }

    def _screen_deliverable(self) -> dict[str, Any]:
        """筛选可交付因子。"""
        from quantlab.factor_discovery.delivery_screener import DeliveryScreener
        screener = DeliveryScreener(data_path=self.data_path)
        return screener.screen()

    def _generate_delivery_reports(self, factor_ids: list[str]) -> list[str]:
        """为可交付因子生成报告。"""
        if not factor_ids:
            return []

        from quantlab.factor_discovery.factor_report import FactorDeliveryReportGenerator
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor

        store = PersistentFactorStore()
        hub = self._load_data()
        if hub.empty:
            return []

        generator = FactorDeliveryReportGenerator()
        executor = SafeFactorExecutor()
        library = store.load_library_entries()
        reports = []

        for entry in library:
            if entry.factor_spec.factor_id not in factor_ids:
                continue
            try:
                computed = executor.execute(entry.factor_spec, hub)
                factor_panel = computed["factor_panel"]
                output_dir = str(DATA_DIR / "delivery_reports" / entry.factor_spec.factor_id)
                report = generator.generate(
                    factor_spec=entry.factor_spec,
                    factor_panel=factor_panel,
                    market_df=hub,
                    evaluation_report=entry.latest_report,
                    output_dir=output_dir,
                )
                reports.append(output_dir)
            except Exception as exc:
                logger.warning("因子 %s 报告生成失败: %s", entry.factor_spec.factor_id, exc)

        return reports

    def _load_data(self) -> pd.DataFrame:
        """加载市场数据。"""
        try:
            from quantlab.factor_discovery.datahub import DataHub
            hub = DataHub()
            return hub.load(str(self.data_path), use_cache=False)
        except Exception:
            if self.data_path.exists():
                return pd.read_csv(self.data_path)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. Windows 计划任务安装
# ---------------------------------------------------------------------------

def install_windows_task(task_name: str = "QuantAgentDaily", hour: int = 18, minute: int = 30) -> dict[str, str]:
    """安装 Windows 计划任务，每日定时执行。

    使用 schtasks 命令创建，无需管理员权限（当前用户作用域）。
    """
    python_exe = sys.executable
    project_root = str(Path(__file__).resolve().parent.parent)
    script = f'"{python_exe}" -m quantlab.scheduler run_daily'

    # 创建计划任务
    cmd = [
        "schtasks", "/Create",
        "/TN", task_name,
        "/TR", script,
        "/SC", "DAILY",
        "/ST", f"{hour:02d}:{minute:02d}",
        "/F",  # 强制覆盖
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return {"status": "installed", "task_name": task_name, "schedule": f"每日 {hour:02d}:{minute:02d}"}
        return {"status": "error", "message": result.stderr.strip()[:500]}
    except Exception as exc:
        return {"status": "error", "message": str(exc)[:500]}


def remove_windows_task(task_name: str = "QuantAgentDaily") -> dict[str, str]:
    """移除 Windows 计划任务。"""
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
    """命令行入口。"""
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
                    "success": "[OK]",
                    "partial": "[~]",
                    "failed": "[!!]",
                    "running": "[..]",
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
