"""因子工厂管线阶段 —— 每个阶段独立实现，可组合、可单独测试。"""

from .base import PipelineContext, PipelineStage
from .data_refresh import DataRefreshStage
from .decay_monitor import DecayMonitorStage
from .evolution import EvolutionStage
from .oos_validation import OOSValidationStage
from .combination import CombinationStage
from .governance import GovernanceStage
from .delivery import DeliveryScreeningStage, PaperTradingStage, DeliveryReportStage
