# Quant Agent

<div align="center">

**AI-native Quant Factor Factory — Cross-sectional Factor Discovery & Delivery**

聚焦 **A股横截面因子发掘** 的半自动化量化 Agent 系统。围绕 **因子假设生成→安全执行评估→多轮进化搜索→衰减监控→交付筛选→报告输出** 构建完整的因子工厂闭环。

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.2%2B-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26%2B-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="Tushare" src="https://img.shields.io/badge/Data-Tushare_Pro-red?style=flat-square">
  <img alt="AkShare" src="https://img.shields.io/badge/Data-AkShare-orange?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active%20Development-8A2BE2?style=flat-square">
</p>

</div>

---

## Overview

Quant Agent 不是一个只会"跑回测"的脚本集合，而是一个朝 **成熟量化因子发掘 Agent** 演进的研究系统——目标是 **WorldQuant 模式卖因子**，不是自己交易。

它融合了 **R&D-Agent-Quant、AlphaAgent、QuantaAlpha、FactorMiner、Hubble** 等框架的关键思路，把传统的量化研究流程拆成一条可自动化执行的链路：

> **增量数据刷新 → 衰减监控 → 提出假设 → 生成因子表达 → 安全执行 → 横截面评估 → 进化搜索 → 交付筛选 → 报告输出 → 再发掘**

当前版本已实现 **半自动化因子工厂**：

- 每日自动调度（增量刷新 + 衰减监控 + 进化搜索 + 交付筛选 + 报告生成）
- 因子衰减检测与自动触发再发掘
- WorldQuant 标准交付筛选（ICIR > 1.0、正交性、容量、换手）
- 全市场日线秒级拉取（Tushare Pro）
- 因子模拟交易（佣金 + 印花税 + 滑点 + 冲击成本）
- WorldQuant 风格因子交付报告（JSON + Markdown）
- Windows 计划任务一键安装

---

## Highlights

| 能力 | 说明 |
|---|---|
| **Daily Scheduler** | 5 阶段日常管线一键运行，支持 Windows 计划任务无人值守 |
| **Decay Monitor** | 因子 IC 衰减自动检测，衰减因子自动触发再发掘 |
| **Delivery Screener** | WorldQuant 标准自动筛选：ICIR > 1.0、正交性 < 0.3、换手 < 50%、容量 > 500 万 |
| **Hypothesis Generation** | 基于族感知多样性、正则化探索、经验记忆引导，自动生成候选因子 |
| **Evolution Loop** | 支持 mutation / crossover / early-stop 的多轮因子进化搜索 |
| **Safe Execution** | 用 DSL + AST + 白名单算子保证因子表达可控、可算、可审计 |
| **Cross-sectional Evaluation** | 覆盖 Rank IC、ICIR、分位单调性、稳定性、可交易性等核心指标 |
| **Factor Simulation** | A 股真实成本建模：佣金万三 + 印花税千一 + 滑点 + 冲击成本 + T+1 |
| **Delivery Report** | WorldQuant 风格交付报告（IC 族、扣费绩效、容量、正交性、稳健性、风险提示） |
| **DataHub** | 统一数据访问层：Tushare Pro（批量秒级全市场）+ AkShare（增量刷新）|
| **Agent Runtime** | 21 个工具统一调用，覆盖因子发掘全生命周期 |

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                       DailyScheduler (无人值守)                      │
│  增量刷新 → 衰减监控 → 进化搜索 → 交付筛选 → 报告生成              │
├──────────────────────────────────────────────────────────────────────┤
│                      AssistantToolRuntime (21 Tools)                 │
│  Backtest │ Factor Discovery │ Evolution │ Trading │ Scheduling     │
├──────────────────────────────────────────────────────────────────────┤
│                      ResearchTaskExecutor                            │
├────────────────────┬───────────────────┬────────────────────────────┤
│ Factor Discovery   │  Trading Engine   │      Data Layer            │
│ ├ Hypothesis Gen   │  ├ Cost Model     │  ├ TushareProProvider      │
│ ├ Evolution Loop   │  ├ Portfolio      │  ├ AkShareIncremental      │
│ ├ Decay Monitor    │  └ Simulator      │  └ DataHub (cache+quality) │
│ ├ Delivery Screener│                   │                            │
│ └ Delivery Report  │                   │                            │
├────────────────────┴───────────────────┴────────────────────────────┤
│ SafeFactorExecutor / FactorStore / ExperienceMemory / Evaluation    │
└──────────────────────────────────────────────────────────────────────┘
```

### Daily Pipeline

```text
Incremental Data Refresh
      ↓
Factor Decay Monitor ──→ IC 衰减 > 50%? ──→ 触发再发掘
      ↓
Evolution Search (衰减方向 + 计划方向)
      ↓
Delivery Screener (WorldQuant 标准)
      ↓
Delivery Report (JSON + Markdown)
      ↓
Schedule Record (全流程日志)
```

### Research Loop

```text
Research Direction
      ↓
Hypothesis Generator (族感知 + 经验记忆)
      ↓
Factor Spec / Expression Tree
      ↓
Safe Execution on Cross-sectional Data
      ↓
Evaluation & Scoring (IC / ICIR / 单调性 / 稳定性)
      ↓
Approve / Observe / Reject
      ↓
Evolution Loop + Experience Memory
      ↓
Next-round Candidates
```

---

## Project Structure

```text
quant-agent/
├── quantlab/
│   ├── scheduler.py               # 每日自动调度器 + Windows 计划任务 + CLI
│   ├── config.py                  # 全局配置
│   ├── pipeline.py                # 主研究管线入口
│   ├── factor_discovery/          # 因子发掘核心模块
│   │   ├── hypothesis.py          #   因子假设生成器
│   │   ├── evolution.py           #   多轮进化搜索循环
│   │   ├── pipeline.py            #   横截面评估体系
│   │   ├── runtime.py             #   因子发掘运行时
│   │   ├── models.py              #   数据模型
│   │   ├── datahub.py             #   统一数据抽象层
│   │   ├── decay_monitor.py       #   因子衰减监控
│   │   ├── delivery_screener.py   #   交付标准自动筛选
│   │   └── factor_report.py       #   WorldQuant 风格交付报告
│   ├── trading/                   # 模拟交易引擎
│   │   ├── cost_model.py          #   A 股交易成本模型
│   │   ├── portfolio.py           #   因子组合构建器
│   │   └── simulator.py           #   模拟交易引擎
│   ├── analysis/                  # 回测验证与实验分析
│   │   ├── grid_search.py         #   参数网格搜索
│   │   ├── validation.py          #   训练-测试 / Walk-forward 验证
│   │   └── history_store.py       #   实验历史存储
│   ├── assistant/                 # Agent 运行时与工具注册
│   │   ├── tools.py               #   21 个工具定义与实现
│   │   ├── config.py              #   助手配置
│   │   ├── evaluator.py           #   决策门控
│   │   ├── knowledge_base.py      #   知识库
│   │   ├── llm.py                 #   LLM 接入层
│   │   ├── memory.py              #   经验记忆
│   │   └── planner.py             #   研究规划器
│   ├── research/                  # 研究任务执行框架
│   │   ├── executor.py            #   任务执行器
│   │   ├── models.py              #   研究模型
│   │   └── protocol.py            #   任务协议
│   ├── strategies/                # 策略实现与注册
│   │   ├── base.py                #   策略基类
│   │   ├── ma_cross.py            #   均线交叉
│   │   ├── channel_breakout.py    #   通道突破
│   │   └── registry.py            #   策略注册表
│   └── data/                      # 数据 Provider
│       ├── tushare_provider.py    #   Tushare Pro（批量全市场秒级）
│       ├── akshare_incremental.py #   AkShare 增量刷新
│       ├── fetcher.py             #   数据抓取
│       └── loader.py              #   数据加载
├── data/                          # 本地行情与元数据
├── assistant_data/                # Agent 持久化知识、记忆与运行记录
├── reports/                       # 因子交付报告输出
├── refresh_data.py                # 横截面数据刷新脚本
├── fetch_hs300_etf.py             # ETF 数据抓取脚本
├── run_backtest.py                # 回测入口
├── requirements.txt               # Python 依赖
└── README.md
```

---

## Core Modules

### 1. `scheduler.py` — 每日自动调度器

5 阶段日常管线，一键运行因子工厂：

1. **增量刷新**：只拉新交易日数据
2. **衰减监控**：检查已有因子 IC 是否衰减，标记衰减因子
3. **进化搜索**：对衰减方向 + 计划方向运行多轮进化
4. **交付筛选**：按 WorldQuant 标准过滤因子
5. **报告生成**：可交付因子输出 JSON + Markdown 报告

```bash
# 手动执行一次日常任务
python -m quantlab.scheduler run_daily

# 安装 Windows 计划任务（每日 18:30 执行）
python -m quantlab.scheduler install_cron

# 查看最近执行记录
python -m quantlab.scheduler status
```

### 2. `decay_monitor.py` — 因子衰减监控

对因子库全量因子进行 IC 衰减检测：

| 指标 | 衰减阈值 | 动作 |
|---|---|---|
| IC 衰减比 | > 50% | `decayed` → 建议再发掘 |
| 近期 \|IC\| | < 0.015 | `warning` → 建议重评估 |
| 近期 ICIR | < 0.15 | `warning` → 建议重评估 |
| IC 正比例 | < 50% | `warning` → 建议监控 |

### 3. `delivery_screener.py` — 交付标准自动筛选

按 WorldQuant 交付标准逐项检验，只输出可卖因子：

| 标准 | 门槛 |
|---|---|
| ICIR | > 1.0 |
| \|RankIC\| | > 0.02 |
| 库内最大相关性 | < 0.3 |
| 日均换手率 | < 50% |
| 日容量 | > 500 万 |
| 市值暴露 | < 0.3 |
| 20d 衰减比 | > 30% |

### 4. `hypothesis.py` — 因子假设生成器

融合 **AlphaAgent + Hubble + FactorMiner Ralph Loop**：

- 6 类因子族：`momentum / reversal / volatility / volume_price / liquidity / fundamental`
- 12+ DSL 算子：`rank / zscore / delta / lag / mean / std / ts_rank / add / sub / mul / div / clip ...`
- 3 类模板策略：基础时序、双特征交叉、波动率调整
- 经验记忆检索 + 族感知多样性惩罚 + 探索奖励

### 5. `evolution.py` — 自主搜索循环

融合 **QuantaAlpha + FactorMiner + R&D-Agent-Quant**：

- 多轮闭环：假设 → 执行 → 评估 → 进化 → 再假设
- 支持 `mutation` / `crossover`
- 支持 early stop
- 自动沉淀成功模式、失败约束和观察记录

### 6. `trading/` — 模拟交易引擎

A 股真实成本建模：

- **成本模型**：佣金万三（最低 5 元）+ 印花税千一 + 滑点万二 + 冲击成本（基于成交量参与率）
- **组合构建**：因子面板 → 每日多空权重，A 股 T+1 约束
- **模拟交易**：跟踪组合净值、换手率、成本明细，输出扣费后绩效

### 7. `factor_report.py` — 因子交付报告

WorldQuant 风格交付标准，买方尽调维度全覆盖：

- 因子定义与表达式（可复现）
- IC 指标族（RankIC, ICIR, 正 IC 占比, 衰减曲线）
- 扣费后组合绩效（净 Sharpe, 净收益, 换手率）
- 容量估算（能容纳多少资金）
- 正交性（与已知因子 / 风险因子的相关性）
- 稳健性（不同市场阶段、样本外、walk-forward）
- 风险提示（拥挤度、衰减速度、风格暴露）

输出格式：JSON（程序化接入）+ Markdown（人类可读）

### 8. `datahub.py` — 统一数据抽象层

- **TushareProProvider**：批量 API，全市场日线秒级拉取（5000+ 行 < 1s）
- **AkShareIncrementalProvider**：增量刷新，只拉新交易日
- 内置缓存复用 + 数据质量报告（覆盖率、NaN 比例、最近刷新时间）

---

## Tooling

当前已注册 **21 个工具**，覆盖因子发掘全生命周期：

### 因子发掘

| Tool | Description |
|---|---|
| `run_factor_discovery` | 单次因子发掘闭环 |
| `generate_factor_hypotheses` | 批量生成因子候选 |
| `run_factor_evolution` | 运行多轮进化搜索 |
| `check_factor_decay` | 检查因子库衰减状态 |
| `screen_deliverable_factors` | WorldQuant 标准自动筛选 |
| `generate_factor_delivery_report` | 生成交付报告（JSON + Markdown）|
| `simulate_factor_portfolio` | 因子模拟交易（含成本）|

### 数据操作

| Tool | Description |
|---|---|
| `view_current_config` | 查看当前配置与数据路径 |
| `refresh_cross_section_data` | 全量刷新横截面数据 |
| `incremental_refresh_data` | 增量刷新（只拉新交易日）|

### 回测验证

| Tool | Description |
|---|---|
| `run_single_backtest` | 运行单次回测 |
| `run_grid_experiment` | 运行参数网格搜索 |
| `run_train_test_validation` | 训练-测试验证 |
| `run_walk_forward_validation` | Walk-forward 验证 |
| `run_multi_strategy_compare` | 多策略比较 |
| `review_portfolio_construction` | 组合构建评审 |

### 调度与自动化

| Tool | Description |
|---|---|
| `run_daily_schedule` | 一键执行日常因子工厂管线 |
| `install_daily_cron` | 安装 Windows 计划任务 |

### 实验管理

| Tool | Description |
|---|---|
| `list_strategies` | 查看可用策略列表 |
| `list_experiment_history` | 查看实验历史 |
| `get_experiment_detail` | 查看实验详情 |

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

主要依赖：

- `pandas >= 2.2.0`
- `numpy >= 1.26.0`
- `akshare >= 1.18.55`
- `tushare`（可选，用于全市场批量数据）

### 2) Configure Tushare Pro (推荐)

```bash
# 设置环境变量（Windows PowerShell）
[System.Environment]::SetEnvironmentVariable('TUSHARE_TOKEN', 'your_token', 'User')

# 或在项目根目录创建 .env 文件
echo "TUSHARE_TOKEN=your_token" > .env
```

获取 Token：[Tushare Pro 注册](https://tushare.pro/register)

### 3) Run the Daily Pipeline

```bash
# 手动执行一次因子工厂
python -m quantlab.scheduler run_daily

# 安装每日自动任务（默认 18:30）
python -m quantlab.scheduler install_cron

# 查看执行记录
python -m quantlab.scheduler status
```

### 4) Or Step by Step

```bash
# 刷新横截面数据
python refresh_data.py

# 或只抓 ETF 数据
python fetch_hs300_etf.py

# 运行回测
python run_backtest.py --data "data/hs300_etf.csv" --strategy ma_cross
```

### 5) Use as an Agent Runtime

```python
from pathlib import Path
from quantlab.config import BacktestConfig
from quantlab.assistant.tools import AssistantToolRuntime

rt = AssistantToolRuntime(
    BacktestConfig(),
    Path("data/hs300_cross_section.csv")
)

# 生成因子假设
hypotheses = rt.execute("generate_factor_hypotheses", {
    "research_direction": "量价背离",
    "max_candidates": 5,
})

# 进化搜索
result = rt.execute("run_factor_evolution", {
    "direction": "波动率调整动量",
    "max_rounds": 3,
    "candidates_per_round": 5,
})

# 检查衰减
decay = rt.execute("check_factor_decay", {})

# 筛选可交付因子
screened = rt.execute("screen_deliverable_factors", {})

# 日常一键运行
daily = rt.execute("run_daily_schedule", {
    "directions": ["量价背离", "波动率调整动量"],
    "evolution_rounds": 3,
})
```

---

## Configuration

### Tushare Pro Token

优先级从低到高：

1. 项目根目录 `.env` 中的 `TUSHARE_TOKEN`
2. 系统环境变量 `TUSHARE_TOKEN`
3. 代码中显式传入

### AI / LLM Configuration

支持两层配置来源，优先级从低到高：

1. 项目根目录 `.env`
2. 系统环境变量

```env
ASSISTANT_BASE_URL=https://your-api-endpoint/v1
ASSISTANT_API_KEY=your_api_key
ASSISTANT_MODEL=gpt-5.4
```

---

## Decision Gate

系统在执行研究计划前，会先做可信度评估，避免"生成一个计划就直接跑"。

决策分流为：

- `pass`：直接执行
- `review_required`：降级为更保守的验证路径
- `fail`：自动重规划，补齐缺失链路

评估维度包括：计划可信度、执行可信度、结论可信度。

---

## Design References

本项目当前架构主要借鉴以下框架：

| Framework | Borrowed Ideas |
|---|---|
| [R&D-Agent-Quant](https://github.com/microsoft/RD-Agent) | Research → Development 两阶段迭代 |
| [AlphaAgent](https://arxiv.org/abs/2502.16789) | LLM + 正则化探索 + 抗衰减评估 |
| [QuantaAlpha](https://arxiv.org/abs/2602.07085) | 轨迹级 mutation / crossover 进化 |
| [FactorMiner](https://arxiv.org/abs/2602.14670) | Ralph Loop: Retrieve → Adapt → Learn → Plan → Harvest |
| [Hubble](https://arxiv.org/abs/2604.09601) | DSL 约束生成 + AST 校验 + 族感知多样性 |

---

## Changelog

### 2026-04-27 — 半自动化因子工厂
- 新增 `DailyScheduler`：5 阶段日常管线 + Windows 计划任务 + CLI
- 新增 `FactorDecayMonitor`：IC 衰减检测 + 自动触发再发掘
- 新增 `DeliveryScreener`：WorldQuant 交付标准自动筛选
- 新增 `FactorDeliveryReportGenerator`：WorldQuant 风格交付报告
- 新增 `TushareProProvider`：全市场日线秒级拉取
- 新增 `AkShareIncrementalProvider`：增量数据刷新
- 新增模拟交易引擎：`CostModel` + `FactorPortfolioConstructor` + `FactorPortfolioSimulator`
- 工具总数 17 → 21
- Tushare Pro token 环境变量配置（`.env` + 系统环境变量）

### 2026-04-27 — 主线统一重构
- `DEFAULT_DATA_PATH` 拆为 `DEFAULT_CROSS_SECTION_DATA_PATH` + `DEFAULT_PRICE_DATA_PATH`
- 所有回测函数默认走 ETF 兼容路径
- `SUPPORTED_TASK_TYPES` 补齐 `generate_factor_hypotheses` + `factor_evolution`
- Executor 新增假设生成与因子进化两个任务分支
- 单资产代理横截面默认禁止（`allow_proxy=False`）
- 助手系统提示改为"横截面因子优先、单资产回测兼容"双模式

---

## Roadmap

- [x] 因子假设生成器
- [x] 多轮进化搜索循环
- [x] 统一数据抽象层 DataHub
- [x] AssistantToolRuntime 工具接入
- [x] 真实横截面闭环验证
- [x] 主线统一：全局默认指向横截面因子发掘
- [x] Tushare Pro 全市场数据源
- [x] 模拟交易引擎（成本建模 + T+1）
- [x] 因子交付报告（WorldQuant 风格）
- [x] 每日自动调度器 + Windows 计划任务
- [x] 因子衰减监控 + 自动触发再发掘
- [x] 交付标准自动筛选
- [ ] 更长历史区间与更多标签期评估
- [ ] 多因子组合层优化
- [ ] 更强的经验记忆治理
- [ ] 面向完整 Quant Agent 的任务编排能力
- [ ] 自动化因子库清理与归档
- [ ] 因子拥挤度监控

---

## License

MIT
