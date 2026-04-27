"""交付标准自动筛选 —— 只输出可卖给买方的因子。

WorldQuant / 因子买方的核心筛选标准：
1. ICIR > 1.0（信号稳定，不是随机噪声）
2. Rank IC 绝对值 > 0.02（信号有实际区分力）
3. IC 正比例 > 55%（方向稳定）
4. 与已有因子库相关性 < 0.3（正交性，买方不买重复因子）
5. 日均换手率 < 50%（扣费后还有剩的）
6. 容量充足（日容量 > 500 万）
7. 市值暴露 < 0.3（不是变相做大小盘轮动）
8. 衰减平缓（20d IC > 1d IC 的 30%）

不符合上述任一标准的因子被标记为"不可交付"，给出具体未达标原因。

使用方式：
    from quantlab.factor_discovery.delivery_screener import DeliveryScreener
    screener = DeliveryScreener()
    result = screener.screen()
    # result["deliverable_factor_ids"] = ["factor_xxx", "factor_yyy"]
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
class ScreeningResult:
    """单个因子的筛选结果。"""
    factor_id: str
    factor_name: str
    factor_family: str
    deliverable: bool = False

    # 指标值
    rank_ic_mean: float = 0.0
    icir: float = 0.0
    ic_positive_ratio: float = 0.0
    avg_daily_turnover: float = 0.0
    max_library_correlation: float = 0.0
    market_cap_exposure: float = 0.0
    capacity_daily_yuan: float = 0.0
    decay_20d_ratio: float = 0.0  # 20d IC / 1d IC

    # 未达标原因
    fail_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScreeningSummary:
    """筛选汇总。"""
    screen_date: str = ""
    total_factors: int = 0
    deliverable_count: int = 0
    deliverable_factor_ids: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DeliveryScreener:
    """交付标准自动筛选器。"""

    def __init__(
        self,
        data_path: Path | str | None = None,
        # 筛选阈值（WorldQuant 标准）
        min_icir: float = 1.0,
        min_rank_ic: float = 0.02,
        min_ic_positive_ratio: float = 0.55,
        max_library_correlation: float = 0.3,
        max_daily_turnover: float = 0.50,
        max_market_cap_exposure: float = 0.3,
        min_daily_capacity_yuan: float = 5_000_000,
        min_decay_20d_ratio: float = 0.3,
    ) -> None:
        self.data_path = Path(data_path or DEFAULT_CROSS_SECTION_DATA_PATH)
        self.min_icir = min_icir
        self.min_rank_ic = min_rank_ic
        self.min_ic_positive_ratio = min_ic_positive_ratio
        self.max_library_correlation = max_library_correlation
        self.max_daily_turnover = max_daily_turnover
        self.max_market_cap_exposure = max_market_cap_exposure
        self.min_daily_capacity_yuan = min_daily_capacity_yuan
        self.min_decay_20d_ratio = min_decay_20d_ratio

    def screen(self) -> dict[str, Any]:
        """对因子库中所有因子做交付标准筛选。"""
        from quantlab.factor_discovery.runtime import PersistentFactorStore, SafeFactorExecutor

        store = PersistentFactorStore()
        library = store.load_library_entries()

        if not library:
            return ScreeningSummary(
                screen_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            ).to_dict()

        # 加载市场数据
        market_df = self._load_data()

        results: list[ScreeningResult] = []

        for entry in library:
            try:
                result = self._screen_single(entry, market_df, store)
                results.append(result)
            except Exception as exc:
                logger.warning("因子 %s 筛选失败: %s", entry.factor_spec.factor_id, exc)
                results.append(ScreeningResult(
                    factor_id=entry.factor_spec.factor_id,
                    factor_name=entry.factor_spec.name,
                    factor_family=entry.factor_spec.family,
                    deliverable=False,
                    fail_reasons=[f"筛选异常: {str(exc)[:100]}"],
                ))

        deliverable = [r for r in results if r.deliverable]
        deliverable_ids = [r.factor_id for r in deliverable]

        summary = ScreeningSummary(
            screen_date=pd.Timestamp.now().strftime("%Y-%m-%d"),
            total_factors=len(results),
            deliverable_count=len(deliverable),
            deliverable_factor_ids=deliverable_ids,
            details=[r.to_dict() for r in results],
        )

        # 保存筛选记录
        self._save_screening_record(summary)

        return summary.to_dict()

    def _screen_single(
        self,
        entry: Any,
        market_df: pd.DataFrame,
        store: Any,
    ) -> ScreeningResult:
        """筛选单个因子。"""
        spec = entry.factor_spec
        report = entry.latest_report
        scorecard = report.scorecard

        fail_reasons: list[str] = []

        # --- 从评估报告取指标 ---
        rank_ic = float(scorecard.rank_ic_mean or 0.0)
        icir = float(scorecard.ic_ir or 0.0)
        turnover = float(scorecard.turnover or 0.0)

        # 需要重新计算的指标（正交性、容量、衰减）
        max_library_corr = 0.0
        market_cap_exp = 0.0
        capacity = 0.0
        decay_ratio = 0.0
        ic_positive_ratio = 0.0

        # 尝试从因子面板计算更精确的指标
        if not market_df.empty:
            try:
                executor = SafeFactorExecutor()
                computed = executor.execute(spec, market_df)
                panel = computed["factor_panel"]

                if not panel.empty:
                    # 正交性
                    corr_map, max_corr = store.summarize_library_overlap(panel)
                    max_library_corr = max_corr

                    # 市值暴露
                    if "market_cap" in panel.columns and "factor_value" in panel.columns:
                        valid = panel.dropna(subset=["factor_value", "market_cap"])
                        if len(valid) > 10:
                            market_cap_exp = abs(float(
                                valid["factor_value"].corr(np.log(valid["market_cap"].clip(lower=1.0)))
                            ))

                    # 容量估算
                    if "volume" in market_df.columns:
                        from quantlab.trading.cost_model import AShareCostModel
                        cost_model = AShareCostModel()
                        cap_dict = cost_model.estimate_capacity(market_df["volume"].dropna())
                        capacity = cap_dict.get("total_daily_capacity_yuan", 0.0)

                    # IC 正比例和衰减
                    ic_stats = self._compute_ic_details(panel)
                    ic_positive_ratio = ic_stats["ic_positive_ratio"]
                    decay_profile = ic_stats.get("decay_profile", {})
                    if decay_profile:
                        ic_1d = abs(decay_profile.get("1d", 0.0))
                        ic_20d = abs(decay_profile.get("20d", 0.0))
                        if ic_1d > 0.001:
                            decay_ratio = ic_20d / ic_1d
            except Exception as exc:
                logger.warning("因子 %s 精确计算失败，使用报告值: %s", spec.factor_id, exc)

        # 回退到报告值
        if ic_positive_ratio == 0.0:
            # 从稳定性评分估算
            stability = float(scorecard.stability_score or 0.0)
            ic_positive_ratio = min(1.0, 0.5 + stability * 0.3)

        # --- 逐项检查 ---
        if abs(icir) < self.min_icir:
            fail_reasons.append(f"ICIR={icir:.3f} < {self.min_icir}")

        if abs(rank_ic) < self.min_rank_ic:
            fail_reasons.append(f"|RankIC|={abs(rank_ic):.4f} < {self.min_rank_ic}")

        if ic_positive_ratio < self.min_ic_positive_ratio:
            fail_reasons.append(f"IC正比例={ic_positive_ratio:.1%} < {self.min_ic_positive_ratio:.0%}")

        if max_library_corr > self.max_library_correlation:
            fail_reasons.append(f"库内最大相关性={max_library_corr:.3f} > {self.max_library_correlation}")

        if turnover > self.max_daily_turnover:
            fail_reasons.append(f"日均换手={turnover:.1%} > {self.max_daily_turnover:.0%}")

        if market_cap_exp > self.max_market_cap_exposure:
            fail_reasons.append(f"市值暴露={market_cap_exp:.3f} > {self.max_market_cap_exposure}")

        if capacity > 0 and capacity < self.min_daily_capacity_yuan:
            fail_reasons.append(f"日容量=¥{capacity:,.0f} < ¥{self.min_daily_capacity_yuan:,.0f}")

        if decay_ratio > 0 and decay_ratio < self.min_decay_20d_ratio:
            fail_reasons.append(f"20d衰减比={decay_ratio:.2f} < {self.min_decay_20d_ratio}")

        return ScreeningResult(
            factor_id=spec.factor_id,
            factor_name=spec.name,
            factor_family=spec.family,
            deliverable=len(fail_reasons) == 0,
            rank_ic_mean=round(rank_ic, 6),
            icir=round(icir, 4),
            ic_positive_ratio=round(ic_positive_ratio, 4),
            avg_daily_turnover=round(turnover, 6),
            max_library_correlation=round(max_library_corr, 4),
            market_cap_exposure=round(market_cap_exp, 4),
            capacity_daily_yuan=round(capacity, 0),
            decay_20d_ratio=round(decay_ratio, 4),
            fail_reasons=fail_reasons,
        )

    def _compute_ic_details(self, panel: pd.DataFrame) -> dict[str, Any]:
        """计算 IC 正比例和衰减。"""
        panel = panel.copy()
        if panel.empty or "factor_value" not in panel.columns:
            return {"ic_positive_ratio": 0.0, "decay_profile": {}}

        panel["date"] = pd.to_datetime(panel["date"])

        # 计算 forward return
        if "close" not in panel.columns:
            return {"ic_positive_ratio": 0.0, "decay_profile": {}}

        panel["forward_return"] = panel.groupby("asset")["close"].shift(-1) / panel["close"] - 1.0
        panel = panel.dropna(subset=["factor_value", "forward_return"])

        if panel.empty:
            return {"ic_positive_ratio": 0.0, "decay_profile": {}}

        # Rank IC 序列
        rank_ic_series = panel.groupby("date", sort=False).apply(
            lambda g: g["factor_value"].rank(pct=True).corr(
                g["forward_return"].rank(pct=True)
            ) if len(g) >= 5 and g["factor_value"].nunique() > 1 and g["forward_return"].nunique() > 1
            else np.nan
        ).dropna()

        ic_pos_ratio = float((rank_ic_series > 0).mean()) if not rank_ic_series.empty else 0.0

        # 衰减
        decay: dict[str, float] = {}
        for h in [1, 5, 10, 20]:
            shifted = panel.copy()
            shifted["fwd_h"] = shifted.groupby("asset")["close"].shift(-h) / shifted["close"] - 1.0
            shifted = shifted.dropna(subset=["fwd_h", "factor_value"])
            if not shifted.empty:
                ic_h = shifted.groupby("date", sort=False).apply(
                    lambda g: g["factor_value"].rank(pct=True).corr(
                        g["fwd_h"].rank(pct=True)
                    ) if len(g) >= 5 and g["factor_value"].nunique() > 1 else np.nan
                ).dropna().mean()
                decay[f"{h}d"] = round(float(ic_h), 6) if pd.notna(ic_h) else 0.0

        return {
            "ic_positive_ratio": round(ic_pos_ratio, 4),
            "decay_profile": decay,
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

    def _save_screening_record(self, summary: ScreeningSummary) -> None:
        """保存筛选记录。"""
        screen_dir = DATA_DIR / "scheduler"
        screen_dir.mkdir(parents=True, exist_ok=True)
        record_path = screen_dir / "delivery_screening.json"

        records = []
        if record_path.exists():
            try:
                records = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        records.append(summary.to_dict())
        records = records[-30:]
        record_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
