"""因子衰减监控 —— 持续跟踪已有因子的 IC 表现，衰减时自动触发再发掘。

核心能力：
1. 定期重算因子库中所有因子的近期 IC
2. 对比历史 IC，检测衰减幅度
3. 衰减超阈值的因子标记为"需再发掘"
4. 返回衰减摘要 + 建议动作

衰减判断标准：
- 近期 IC（最近 20/60 交易日）vs 全样本 IC 衰减 > 50%
- 近期 IC 绝对值 < 0.015
- ICIR 近期 < 0.15
- Rank IC 正比例近期 < 50%

参考：AlphaAgent 抗衰减评估 + WorldQuant 因子生命周期管理
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.config import DEFAULT_CROSS_SECTION_DATA_PATH, DATA_DIR

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DecayCheckResult:
    """单个因子的衰减检查结果。"""
    factor_id: str
    factor_name: str
    factor_family: str
    status: str  # healthy / warning / decayed / error

    # 全样本 IC
    full_sample_rank_ic: float = 0.0
    full_sample_icir: float = 0.0
    full_sample_ic_positive_ratio: float = 0.0

    # 近期 IC（最近 20 交易日）
    recent_rank_ic: float = 0.0
    recent_icir: float = 0.0
    recent_ic_positive_ratio: float = 0.0

    # 衰减指标
    ic_decay_ratio: float = 0.0  # 1 - recent/full, 越大衰减越严重
    icir_decay_ratio: float = 0.0

    # 建议动作
    recommended_action: str = "monitor"  # monitor / re_evaluate / re_discover / archive
    decay_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DecayMonitorSummary:
    """衰减监控汇总。"""
    check_date: str = ""
    total_factors: int = 0
    healthy_count: int = 0
    warning_count: int = 0
    decayed_count: int = 0
    error_count: int = 0

    # 需要再发掘的因子
    factors_to_rediscover: list[str] = field(default_factory=list)
    # 需要重新评估的因子
    factors_to_reevaluate: list[str] = field(default_factory=list)

    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FactorDecayMonitor:
    """因子衰减监控器。"""

    def __init__(
        self,
        data_path: Path | str | None = None,
        recent_window: int = 20,
        ic_decay_threshold: float = 0.50,
        ic_absolute_floor: float = 0.015,
        icir_floor: float = 0.15,
        ic_positive_ratio_floor: float = 0.50,
    ) -> None:
        self.data_path = Path(data_path or DEFAULT_CROSS_SECTION_DATA_PATH)
        self.recent_window = recent_window
        self.ic_decay_threshold = ic_decay_threshold
        self.ic_absolute_floor = ic_absolute_floor
        self.icir_floor = icir_floor
        self.ic_positive_ratio_floor = ic_positive_ratio_floor

    def check_all(self) -> dict[str, Any]:
        """检查因子库中所有因子的衰减状态。"""
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor

        store = PersistentFactorStore()
        library = store.load_library_entries()

        if not library:
            return DecayMonitorSummary(
                check_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                total_factors=0,
            ).to_dict()

        # 加载市场数据
        market_df = self._load_data()
        if market_df.empty:
            logger.warning("市场数据为空，无法进行衰减检查")
            return DecayMonitorSummary(
                check_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
                total_factors=len(library),
                error_count=len(library),
            ).to_dict()

        # 获取最近交易日期
        market_df["date"] = pd.to_datetime(market_df["date"])
        all_dates = sorted(market_df["date"].unique())
        recent_cutoff = all_dates[-self.recent_window] if len(all_dates) >= self.recent_window else all_dates[0]

        results: list[DecayCheckResult] = []
        executor = SafeFactorExecutor()

        for entry in library:
            try:
                result = self._check_single_factor(
                    entry, market_df, recent_cutoff, executor
                )
                results.append(result)
            except Exception as exc:
                logger.warning("因子 %s 衰减检查失败: %s", entry.factor_spec.factor_id, exc)
                results.append(DecayCheckResult(
                    factor_id=entry.factor_spec.factor_id,
                    factor_name=entry.factor_spec.name,
                    factor_family=entry.factor_spec.family,
                    status="error",
                    decay_reasons=[f"检查失败: {str(exc)[:100]}"],
                ))

        # 汇总
        healthy = sum(1 for r in results if r.status == "healthy")
        warning = sum(1 for r in results if r.status == "warning")
        decayed = sum(1 for r in results if r.status == "decayed")
        errors = sum(1 for r in results if r.status == "error")

        rediscover = [r.factor_id for r in results if r.recommended_action == "re_discover"]
        reevaluate = [r.factor_id for r in results if r.recommended_action == "re_evaluate"]

        summary = DecayMonitorSummary(
            check_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            total_factors=len(results),
            healthy_count=healthy,
            warning_count=warning,
            decayed_count=decayed,
            error_count=errors,
            factors_to_rediscover=rediscover,
            factors_to_reevaluate=reevaluate,
            details=[r.to_dict() for r in results],
        )

        # 保存监控记录
        self._save_monitor_record(summary)

        return summary.to_dict()

    def _check_single_factor(
        self,
        entry: Any,
        market_df: pd.DataFrame,
        recent_cutoff: pd.Timestamp,
        executor: Any,
    ) -> DecayCheckResult:
        """检查单个因子的衰减。"""
        spec = entry.factor_spec
        factor_id = spec.factor_id

        # 计算因子面板
        computed = executor.execute(spec, market_df)
        panel = computed["factor_panel"]

        if panel.empty or "factor_value" not in panel.columns:
            return DecayCheckResult(
                factor_id=factor_id,
                factor_name=spec.name,
                factor_family=spec.family,
                status="error",
                decay_reasons=["因子面板为空"],
            )

        panel["date"] = pd.to_datetime(panel["date"])

        # 计算 forward return
        if "close" in panel.columns:
            panel["forward_return"] = panel.groupby("asset")["close"].shift(-1) / panel["close"] - 1.0
        else:
            return DecayCheckResult(
                factor_id=factor_id,
                factor_name=spec.name,
                factor_family=spec.family,
                status="error",
                decay_reasons=["缺少 close 字段"],
            )

        panel = panel.dropna(subset=["factor_value", "forward_return"])

        if panel.empty:
            return DecayCheckResult(
                factor_id=factor_id,
                factor_name=spec.name,
                factor_family=spec.family,
                status="error",
                decay_reasons=["有效数据不足"],
            )

        # 全样本 IC
        full_ic = self._compute_ic_metrics(panel)
        # 近期 IC
        recent_panel = panel[panel["date"] >= recent_cutoff]
        recent_ic = self._compute_ic_metrics(recent_panel) if len(recent_panel) > 50 else full_ic

        # 衰减比率
        ic_decay = 0.0
        if abs(full_ic["rank_ic"]) > 0.001:
            ic_decay = 1.0 - abs(recent_ic["rank_ic"]) / abs(full_ic["rank_ic"])
        ic_decay = max(0.0, ic_decay)

        icir_decay = 0.0
        if abs(full_ic["icir"]) > 0.01:
            icir_decay = 1.0 - abs(recent_ic["icir"]) / abs(full_ic["icir"])
        icir_decay = max(0.0, icir_decay)

        # 判断状态
        reasons: list[str] = []
        status = "healthy"
        action = "monitor"

        if ic_decay > self.ic_decay_threshold:
            reasons.append(f"IC 衰减 {ic_decay:.0%}（>{self.ic_decay_threshold:.0%}）")
            status = "decayed"
            action = "re_discover"

        if abs(recent_ic["rank_ic"]) < self.ic_absolute_floor:
            reasons.append(f"近期 IC 绝对值 {recent_ic['rank_ic']:.4f}（<{self.ic_absolute_floor}）")
            if status == "healthy":
                status = "warning"
                action = "re_evaluate"

        if abs(recent_ic["icir"]) < self.icir_floor and abs(full_ic["icir"]) >= self.icir_floor:
            reasons.append(f"近期 ICIR {recent_ic['icir']:.3f}（<{self.icir_floor}）")
            if status == "healthy":
                status = "warning"
                action = "re_evaluate"

        if recent_ic["ic_positive_ratio"] < self.ic_positive_ratio_floor:
            reasons.append(f"近期 IC 正比例 {recent_ic['ic_positive_ratio']:.1%}（<{self.ic_positive_ratio_floor:.0%}）")
            if status == "healthy":
                status = "warning"
                action = "re_evaluate"

        return DecayCheckResult(
            factor_id=factor_id,
            factor_name=spec.name,
            factor_family=spec.family,
            status=status,
            full_sample_rank_ic=full_ic["rank_ic"],
            full_sample_icir=full_ic["icir"],
            full_sample_ic_positive_ratio=full_ic["ic_positive_ratio"],
            recent_rank_ic=recent_ic["rank_ic"],
            recent_icir=recent_ic["icir"],
            recent_ic_positive_ratio=recent_ic["ic_positive_ratio"],
            ic_decay_ratio=round(ic_decay, 4),
            icir_decay_ratio=round(icir_decay, 4),
            recommended_action=action,
            decay_reasons=reasons,
        )

    def _compute_ic_metrics(self, panel: pd.DataFrame) -> dict[str, float]:
        """计算 IC 指标。"""
        if panel.empty:
            return {"rank_ic": 0.0, "icir": 0.0, "ic_positive_ratio": 0.0}

        rank_ic_series = panel.groupby("date", sort=False).apply(
            lambda g: g["factor_value"].rank(pct=True).corr(
                g["forward_return"].rank(pct=True)
            ) if len(g) >= 5 and g["factor_value"].nunique() > 1 and g["forward_return"].nunique() > 1
            else np.nan
        ).dropna()

        if rank_ic_series.empty:
            return {"rank_ic": 0.0, "icir": 0.0, "ic_positive_ratio": 0.0}

        ic_mean = float(rank_ic_series.mean())
        ic_std = float(rank_ic_series.std(ddof=0))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0

        return {
            "rank_ic": round(ic_mean, 6),
            "icir": round(icir, 4),
            "ic_positive_ratio": round(float((rank_ic_series > 0).mean()), 4),
        }

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

    def _save_monitor_record(self, summary: DecayMonitorSummary) -> None:
        """保存监控记录。"""
        monitor_dir = DATA_DIR / "scheduler"
        monitor_dir.mkdir(parents=True, exist_ok=True)
        record_path = monitor_dir / "decay_monitor.json"

        records = []
        if record_path.exists():
            try:
                records = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        records.append(summary.to_dict())
        # 只保留最近 30 次
        records = records[-30:]
        record_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
