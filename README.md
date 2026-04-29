# Quant Agent

<div align="center">

**AI-native Quant Factor Factory — Fully Autonomous Cross-sectional Factor Discovery**

聚焦 **A股横截面 alpha 因子发掘** 的全自动 AI Agent 系统。围绕 **LLM 假设生成 → Block 系统安全执行 → 横截面评估 → 进化搜索 → OOS 验证 → 多因子组合 → 交付筛选 → 报告输出** 构建完整的 WorldQuant 风格因子工厂闭环。

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.2%2B-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26%2B-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="DeepSeek" src="https://img.shields.io/badge/LLM-DeepSeek%20v4-536DFE?style=flat-square">
  <img alt="Tushare" src="https://img.shields.io/badge/Data-Tushare_Pro-red?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

</div>

---

## Overview

Quant Agent 是一个朝 **WorldQuant 卖因子** 目标演进的 AI Agent 系统——**发现因子、验证因子、筛选因子、交付因子**，不是跑回测的脚本集合。

核心设计：**LLM 驱动深度推理 + Block 系统安全执行 + 全链路无人值守自动化**。

```text
        冷启动种子因子引导
              ↓
        每日增量数据刷新（含幸存者偏差修正）
              ↓
        市场状态检测（牛/熊/震荡）
              ↓
        因子衰减监控（4 项检测 → 自动触发再发掘）
              ↓
    ┌─→ LLM 深度推理假设生成（R1）+ 对抗审查（R2）
    │       ↓
    │   Block 系统安全执行 + 参数搜索（T1）
    │       ↓
    │   横截面评估 → OOS 验证 + 成本调整 IC
    │       ↓
    │   经验回路沉淀 + 正交性引导
    │       ↓
    └── 进化搜索（自适应方向 + 元学习调参）
              ↓
        多因子 IC 加权正交组合
              ↓
        因子库治理（生命周期 + 归档 + 拥挤度检测）
              ↓
        风控评估（仓位/行业/回撤/流动性）
              ↓
        WorldQuant 标准交付筛选
              ↓
        交付报告（JSON + Markdown）
              ↓
        告警通知（关键事件自动推送）
```

---

## Highlights

| 能力 | 说明 |
|------|------|
| **LLM 深度推理** | DeepSeek v4-pro thinking mode，R1 假设生成 + R2 对抗审查 + P1 架构设计 + P3 定制代码生成 |
| **Block 系统 v2** | 5 类 100+ 算子的完整因子 DSL，Data/Transform/Combine/Relational/Filter，可审计可组合 |
| **8 阶段每日调度器** | 增量刷新 → 衰减监控 → 进化搜索（自适应）→ OOS 验证 → 多因子组合 → 交付筛选 → 因子库治理 → 报告生成 |
| **多智能体协作** | 3 团队 6 Agent（R1+R2 → P1+P2+P3 → T1+T2），MessageBus + LLM 驱动 |
| **经验学习回路** | ExperienceLoop：成功/失败模式沉淀 → 指导下一轮假设生成 |
| **事前正交性引导** | OrthogonalityGuide：假设生成前分析因子库饱和度，避开已有空间 |
| **市场状态感知** | RegimeDetector：牛熊识别 → regime 调整 IC，避免跨市场状态误判 |
| **样本外验证** | 6 个月 OOS 切分，train/test IC 对比，衰减 > 50% 标记失败 |
| **成本调整 IC** | 周转率 → 交易成本冲击 → 成本调整后 IC，不是毛 IC |
| **幸存者偏差修正** | SurvivorshipFilter：新股/退市/极端收益过滤，回测更可信 |
| **因子库治理** | 自动归档低效因子 + 拥挤度集群检测 + 生命周期管理（DRAFT→OBSERVE→PAPER→PILOT→LIVE→RETIRED） |
| **风控层** | 仓位集中度 / 行业暴露 / 回撤熔断 / 流动性底线，输出 risk_score |
| **冷启动引导** | 7 个学术验证种子因子（动量/反转/价值/规模/低波/换手/质量）自动注入经验回路 |
| **告警系统** | AlertBus：info/warning/critical 分级，文件持久化 + Telegram 扩展点 |
| **纸交易券商** | BrokerInterface 抽象 + PaperBroker + OrderManager，接入真实成本模型 |
| **实盘券商** | 华泰 QMT/xtquant + 聚宽 JoinQuant，ConnectionManager 自动重连 + mock fallback |
| **多品种覆盖** | A 股 + 股指期货 + 商品期货 + 可转债，每类有独立因子模板 |
| **Windows 计划任务** | 一键安装每日无人值守运行 |

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                      DailyScheduler (8-Stage Pipeline)                    │
│  刷新 → 衰减 → 进化(自适应+元学习) → OOS验证 → 组合 → 筛选 → 治理 → 报告  │
├──────────────────────────────────────────────────────────────────────────┤
│                   FactorMultiAgentOrchestrator (3-Team, 6-Agent)          │
│    Research Team (R1+R2) → Programming Team (P1+P2+P3) → Testing (T1+T2)│
├──────────────────────────────────────────────────────────────────────────┤
│                         Factor Enhancements Layer                         │
│  ExperienceLoop │ RiskNeutralizer │ FactorCombiner │ OrthogonalityGuide  │
│  ParameterSearcher │ CustomCodeGenerator │ CrowdingDetector │ RegimeDetector│
├──────────────────────┬──────────────────────┬────────────────────────────┤
│   Factor Discovery   │   Trading Engine     │      Data Layer            │
│   ├ Hypothesis Gen   │   ├ CostModel        │   ├ DataHub                │
│   ├ Block System v2  │   ├ Portfolio        │   ├ TushareProProvider     │
│   ├ Evolution Loop   │   ├ Simulator        │   ├ AkShareIncremental     │
│   ├ Decay Monitor    │   ├ RiskManager      │   ├ SurvivorshipFilter     │
│   ├ Delivery Screener│   ├ PaperBroker      │   ├ MultiAssetContext      │
│   ├ Factor Report    │   └ OrderManager     │   └ Seed Factors           │
│   └ LifecycleManager │                      │                            │
├──────────────────────┴──────────────────────┴────────────────────────────┤
│   SafeFactorExecutor / PersistentFactorStore / FactorExperienceMemory    │
│   AlertBus / Notifier / BrokerInterface / FactorLifecycleManager        │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. `scheduler.py` — 8 阶段日常调度器

```bash
python -m quantlab.scheduler run_daily     # 执行一次完整管线
python -m quantlab.scheduler status        # 查看最近执行记录
python -m quantlab.scheduler install_cron  # 安装 Windows 计划任务
```

| 阶段 | 说明 |
|------|------|
| Phase 1 数据刷新 | 增量拉取 + 幸存者偏差过滤 |
| Phase 2 衰减监控 | 4 项检测，衰减因子自动标记再发掘 |
| Phase 3 进化搜索 | 自适应方向选择 + 经验回路引导 + 元学习调参 + 冷启动种子因子 |
| Phase 4 OOS 验证 | 6 个月 train/test 切分 + 成本调整 IC |
| Phase 5 多因子组合 | IC 加权 + 正交选择 + 组合 ICIR |
| Phase 6 交付筛选 | WorldQuant 8 项标准 |
| Phase 7 因子库治理 | 归档 + 市场状态 + 拥挤度 + 生命周期 + 风控 |
| Phase 8 交付报告 | JSON + Markdown 买方尽调报告 |
| → 告警汇总 | info/warning/critical 分级推送 |

### 2. `multi_agent.py` — 多智能体协作框架

3 团队 6 Agent 协作：R1 假设生成（LLM）+ R2 对抗审查（LLM）→ P1 架构设计（LLM）+ P2 Block 组装 + P3 定制编码（LLM）→ T1 回测运行（参数搜索 + 风险中性化）+ T2 结果验证。

LLMClient 支持 DeepSeek thinking mode（`reasoning_effort=max`）、模型自动降级、`.env` 配置。

### 3. `blocks.py` — Block 系统 v2

可序列化因子 DSL，100+ 注册算子：

| 类型 | 算子示例 |
|------|----------|
| Transform | rank, zscore, delta, ts_std, ts_mean, ts_rank, log, ema, group_neutralize, group_rank, rolling_ols_residual, constant |
| Combine | add, sub, mul, div, max, min, where |
| Relational | group_aggregate, cross_section_ols, event_filter |
| Filter | where_condition, sample_weight |

### 4. `factor_enhancements.py` — 8 大增强模块

| 模块 | 功能 |
|------|------|
| ExperienceLoop | 历史假设→结果映射，沉淀成功/失败模式，指导 R1 生成 |
| RiskNeutralizer | 行业/市值/动量中性化，管控风险暴露 |
| FactorCombiner | IC 加权 + 正交选择 + 组合 ICIR |
| ParameterSearcher | Grid/Random 参数搜索 |
| CustomCodeGenerator | LLM 生成 Python 因子代码 + 沙箱安全执行 |
| OrthogonalityGuide | 事前正交性分析，避开饱和方向 |
| CrowdingDetector | 两两相关性 → 集群 → 拥挤评分 |
| RegimeDetector | 牛熊识别 → regime 调整 IC |

### 5. `trading/` — 交易引擎

- **cost_model.py** — A 股真实成本：佣金万三 / 印花税千一 / 滑点万二 / 冲击成本 + `compute_turnover_cost_impact()`
- **risk_control.py** — RiskManager：仓位集中度 / 行业暴露 / 回撤熔断 / 流动性过滤
- **broker.py** — BrokerInterface（抽象）+ PaperBroker（纸交易）+ OrderManager（调仓）
- **portfolio.py** — 4 种权重方案（等权 / 评分 / IC 加权 / 行业中性）
- **simulator.py** — 日频组合模拟（换手 / 成本 / 净值 / 回撤 / Sharpe / IR）

### 6. `runtime.py` — 因子运行时

- SafeFactorExecutor — AST 白名单执行 + FactorNode → Block 转换
- PersistentFactorStore — JSON 文件库 + 因子面板持久化 + `archive_underperforming()` + `get_library_stats()`

### 7. `datahub.py` — 统一数据层

- TushareProProvider — 批量全市场日线（5000+ 行 < 1s）
- AkShareIncrementalProvider — 增量刷新
- MultiAssetContext — 多品种数据/因子库路由

### 8. `seed_factors.py` — 冷启动引导

7 个学术验证因子自动注入：20 日动量 / 5 日反转 / 账面市值比 / 小盘溢价 / 低波动率 / 低换手率 / 低市盈率。Block 系统构造，首次运行自动注入经验回路。

### 9. `survivorship.py` — 幸存者偏差修正

过滤新股（上市不足 60 日）、退市前数据、极端收益（ST/退市信号）。

### 10. `notifier.py` — 告警系统

AlertBus 收集关键事件（数据失败 / 衰减 / 拥挤 / 风控破限），Notifier 文件持久化 + Telegram 扩展点。

### 11. `futures_factors.py` — 期货因子模板

股指期货（基差/期限结构/持仓量动量）和商品期货（carry/动量/基差/OI 信号）因子模板，Block 系统构造。

### 12. `convertible_bond_factors.py` — 可转债因子模板

5 个 Block 模板：溢价率、delta、债底缓冲、发行规模、双低策略。

### 13. `broker.py` — 券商接口

BrokerInterface 抽象 + PaperBroker（纸交易）+ **XtQuantBroker**（华泰 QMT/xtquant）+ **JoinQuantBroker**（聚宽），ConnectionManager 自动重连，均支持 mock fallback。

---

## Project Structure

```text
quant-agent/
├── quantlab/
│   ├── scheduler.py                    # 8 阶段每日调度器 + CLI
│   ├── config.py                       # 全局配置
│   ├── pipeline.py                     # 主研究管线
│   ├── factor_discovery/               # 因子发掘核心
│   │   ├── blocks.py                   #   Block 系统 v2 (100+ 算子)
│   │   ├── hypothesis.py               #   假设生成器
│   │   ├── evolution.py                #   进化搜索循环
│   │   ├── pipeline.py                 #   横截面评估体系
│   │   ├── runtime.py                  #   安全执行器 + 因子库
│   │   ├── models.py                   #   数据模型 + 生命周期管理
│   │   ├── datahub.py                  #   统一数据层 + 多品种上下文
│   │   ├── multi_agent.py              #   3 团队 6 Agent 协作 + LLMClient
│   │   ├── factor_enhancements.py      #   8 大增强模块
│   │   ├── seed_factors.py             #   冷启动种子因子
│   │   ├── futures_factors.py          #   期货因子模板（股指+商品）
│   │   ├── convertible_bond_factors.py #   可转债因子模板
│   │   ├── survivorship.py             #   幸存者偏差修正
│   │   ├── decay_monitor.py            #   衰减监控
│   │   ├── delivery_screener.py        #   交付筛选
│   │   ├── sample_split.py             #   样本外拆分
│   │   └── factor_report.py            #   交付报告生成
│   ├── trading/                        # 交易引擎
│   │   ├── cost_model.py               #   成本模型 + 换手成本影响
│   │   ├── risk_control.py             #   风控层
│   │   ├── broker.py                   #   券商接口 + 纸交易
│   │   ├── portfolio.py                #   组合构建器
│   │   └── simulator.py                #   模拟交易引擎
│   ├── assistant/                      # Agent 运行时
│   │   ├── tools.py                    #   21 个工具
│   │   ├── notifier.py                 #   告警系统
│   │   ├── config.py                   #   配置
│   │   ├── evaluator.py                #   决策门控
│   │   ├── knowledge_base.py           #   知识库
│   │   ├── llm.py                      #   LLM 接入
│   │   ├── memory.py                   #   经验记忆
│   │   └── planner.py                  #   研究规划器
│   ├── research/                       # 研究任务框架
│   │   ├── executor.py                 #   14 类任务执行
│   │   ├── models.py                   #   研究模型
│   │   └── protocol.py                 #   任务协议
│   ├── analysis/                       # 回测分析
│   │   ├── grid_search.py              #   网格搜索
│   │   ├── validation.py               #   验证
│   │   └── history_store.py            #   实验历史
│   ├── strategies/                     # 策略
│   │   ├── ma_cross.py                 #   均线
│   │   ├── channel_breakout.py         #   通道突破
│   │   └── registry.py                 #   注册表
│   └── data/                           # 数据 Provider
│       ├── tushare_provider.py         #   Tushare Pro
│       ├── fetcher.py                  #   数据抓取
│       └── loader.py                   #   数据加载
├── tests/                              # 99 个测试（8 个模块）
├── data/                               # 本地数据
├── assistant_data/                     # 因子库 + 组合 + 告警 + 经验
├── reports/                            # 交付报告
├── refresh_data.py                     # 数据刷新
├── run_backtest.py                     # 回测入口
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

主要依赖：`pandas>=2.2.0`, `numpy>=1.26.0`, `akshare>=1.18.55`, `scipy`

### 2) Configure

项目根目录创建 `.env`：

```ini
TUSHARE_TOKEN=your_tushare_token

# DeepSeek LLM（必填，驱动智能体推理）
ASSISTANT_API_KEY=sk-xxx
ASSISTANT_BASE_URL=https://api.deepseek.com
ASSISTANT_MODEL=deepseek-v4-pro
ASSISTANT_FALLBACK_MODEL=deepseek-v4-pro
ASSISTANT_REASONING_EFFORT=max
```

### 3) Run

```bash
# 每日因子工厂（全链路无人值守）
python -m quantlab.scheduler run_daily

# 多智能体因子发掘（单方向深度推理）
python verify_multi_agent.py

# 查看执行记录
python -m quantlab.scheduler status

# 安装 Windows 计划任务
python -m quantlab.scheduler install_cron

# 运行测试
python -m pytest tests/ -v
```

### 4) Python API

```python
from quantlab.factor_discovery.multi_agent import (
    FactorMultiAgentOrchestrator, MultiAgentConfig,
)
from quantlab.factor_discovery.datahub import DataHub

hub = DataHub()
df = hub.load("data/zz800_cross_section.csv")

orchestrator = FactorMultiAgentOrchestrator(
    config=MultiAgentConfig(
        enable_factor_combination=True,
        enable_experience_loop=True,
        enable_orthogonality_guide=True,
    ),
)
result = orchestrator.run(direction="动量+反转交叉", market_df=df)
```

---

## Configuration Reference

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `TUSHARE_TOKEN` | Tushare Pro API token | — |
| `ASSISTANT_API_KEY` | LLM API key | — |
| `ASSISTANT_BASE_URL` | LLM API 端点 | — |
| `ASSISTANT_MODEL` | 主力模型 | `gpt-4o` |
| `ASSISTANT_FALLBACK_MODEL` | 降级模型 | — |
| `ASSISTANT_REASONING_EFFORT` | DeepSeek 推理深度 (high/max) | — |

---

## Design References

| Framework | Borrowed Ideas |
|-----------|---------------|
| [R&D-Agent-Quant](https://github.com/microsoft/RD-Agent) | Research → Development 两阶段迭代 |
| [AlphaAgent](https://arxiv.org/abs/2502.16789) | LLM + 正则化探索 + 抗衰减评估 |
| [QuantaAlpha](https://arxiv.org/abs/2602.07085) | 轨迹级 mutation/crossover 进化 |
| [FactorMiner](https://arxiv.org/abs/2602.14670) | Ralph Loop: Retrieve → Adapt → Learn → Plan → Harvest |
| [Hubble](https://arxiv.org/abs/2604.09601) | DSL 约束生成 + AST 校验 + 族感知多样性 |
| [WorldQuant](https://www.worldquant.com/) | Factor screening standards, delivery format, cost modeling |

---

## Roadmap

### 已完成
- [x] LLM 深度推理（DeepSeek v4-pro thinking mode + 模型降级）
- [x] Block 系统 v2（100+ 算子，可序列化因子 DSL）
- [x] 多智能体协作（3 团队 6 Agent）
- [x] 8 阶段日常调度器（含 OOS 验证 + 成本调整 IC）
- [x] 经验学习回路 + 事前正交性引导
- [x] 市场状态检测 + regime 调整 IC
- [x] 幸存者偏差修正
- [x] 因子生命周期管理（7 状态流转）
- [x] 因子库治理（归档 + 拥挤度检测）
- [x] 风控层（仓位 / 行业 / 回撤 / 流动性）
- [x] 冷启动种子因子引导（7 个学术因子）
- [x] 告警系统（分级 + 文件持久化 + 扩展点）
- [x] 纸交易券商接口
- [x] 多品种扩展架构

- [x] 多智能体路径接入 DailyScheduler（经验回路 >10 条自动走 LLM 6-Agent 路径）
- [x] 经验回路记录真实 block_tree 和 input_fields
- [x] 因子 IC 时间序列持久化 + 趋势检验
- [x] 组合 vs 基准对比（等权/市值加权超额 IC）
- [x] 拥挤度检测闭环联动（自动降权 + 方向引导）
- [x] R1/R2 跨运行上下文记忆
- [x] 因子表现曲线跟踪（IC decay curve + 参数敏感性）
- [x] 纸交易接入调度器
- [x] Ridge 滚动窗口组合优化
- [x] 券商完善（OrderValidator + TradeLogger + BrokerFactory）
- [x] 期货品种模板（基差/期限结构/持仓量动量）
- [x] 数据质量自动监控
- [x] 因子版本追踪（parent_factor_id + get_evolution_tree()）
- [x] 报告格式扩展（HTML + CSV/Parquet 导出）
- [x] 实盘券商接入（华泰 QMT/xtquant + 聚宽 JoinQuant，含 ConnectionManager + mock fallback）
- [x] 更多品种数据接入（商品期货 4 模板 + 可转债 5 模板 + 全品种注册）

### 待做
- [ ] Web 管理面板

---

## License

MIT
