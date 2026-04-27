"""因子模拟交易模块 —— 因子信号 → 组合构建 → 模拟交易（含成本）→ 绩效归因。

设计目标：
- WorldQuant 风格因子卖出：买家关注的是因子扣费后能贡献多少增量收益
- 因子 → 组合 → 模拟盘全链路，让因子研究结论可直接对接买方尽调
- 交易成本建模：佣金 + 印花税 + 滑点 + 冲击成本

模块结构：
- cost_model.py    交易成本模型
- portfolio.py     因子信号 → 组合权重
- simulator.py     模拟交易引擎
"""
from .cost_model import CostModel, AShareCostModel
from .portfolio import FactorPortfolioConstructor, PortfolioWeightScheme
from .simulator import FactorPortfolioSimulator, SimulationResult

__all__ = [
    "CostModel",
    "AShareCostModel",
    "FactorPortfolioConstructor",
    "PortfolioWeightScheme",
    "FactorPortfolioSimulator",
    "SimulationResult",
]
