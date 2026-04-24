# Quant Agent

<div align="center">

**AI-native Quant Research Agent for Cross-sectional Factor Discovery**

聚焦 **沪深300横截面因子研究** 的量化 Agent 系统，围绕 **因子假设生成、自动执行评估、多轮进化搜索、经验记忆沉淀** 构建完整研究闭环。

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.2%2B-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26%2B-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="AkShare" src="https://img.shields.io/badge/Data-AkShare-orange?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active%20Development-8A2BE2?style=flat-square">
</p>

</div>

---

## Overview

Quant Agent 不是一个只会“跑回测”的脚本集合，而是一个朝 **成熟量化因子发掘 Agent** 演进的研究系统。

它融合了 **R&D-Agent-Quant、AlphaAgent、QuantaAlpha、FactorMiner、Hubble** 等框架的关键思路，把传统的量化研究流程拆成一条可自动化执行的链路：

> **提出假设 → 生成因子表达 → 安全执行 → 横截面评估 → 进化搜索 → 经验沉淀 → 再发掘**

当前版本已经能在真实沪深300横截面数据上完成：

- 因子假设自动生成
- 多轮进化搜索
- 横截面评估与打分
- 因子库持久化与经验记忆
- 数据刷新与质量报告
- Agent 工具化调用

---

## Highlights

| 能力 | 说明 |
|---|---|
| **Hypothesis Generation** | 基于族感知多样性、正则化探索、经验记忆引导，自动生成候选因子 |
| **Evolution Loop** | 支持 mutation / crossover / early-stop 的多轮因子进化搜索 |
| **Safe Execution** | 用 DSL + AST + 白名单算子保证因子表达可控、可算、可审计 |
| **Cross-sectional Evaluation** | 覆盖 Rank IC、ICIR、分位单调性、稳定性、可交易性等核心指标 |
| **DataHub** | 统一数据访问层，支持缓存、质量报告与 Provider 扩展 |
| **Agent Runtime** | 已接入 `AssistantToolRuntime`，支持 14 个研究工具统一调用 |

---

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    AssistantToolRuntime                    │
│                统一的 Agent 工具运行时入口                 │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  Backtest    │ Factor Disc. │  Evolution   │   Data Ops    │
│   Tools      │    Tools     │    Tools     │     Tools     │
├──────────────┴──────────────┴──────────────┴───────────────┤
│                    ResearchTaskExecutor                    │
├──────────────────────────┬──────────────────┬──────────────┤
│ FactorDiscoveryPipeline  │ FactorEvolution  │   DataHub    │
│                          │      Loop        │              │
├──────────────────────────┴──────────────────┴──────────────┤
│ HypothesisGenerator / SafeFactorExecutor / FactorStore     │
│ ExperienceMemory / Evaluation Pipeline / Data Providers    │
└─────────────────────────────────────────────────────────────┘
```

### Research Loop

```text
Research Direction
      ↓
Hypothesis Generator
      ↓
Factor Spec / Expression Tree
      ↓
Safe Execution on Cross-sectional Data
      ↓
Evaluation & Scoring
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
│   ├── assistant/                 # Agent 运行时与工具注册
│   ├── factor_discovery/          # 因子发掘核心模块
│   ├── research/                  # 研究任务执行框架
│   ├── analysis/                  # 回测验证、参数搜索、实验分析
│   ├── strategies/                # 策略实现与注册
│   ├── data/                      # 数据抓取与加载
│   ├── config.py                  # 全局配置
│   └── pipeline.py                # 主研究管线入口
├── data/                          # 本地行情与元数据
├── assistant_data/                # Agent 持久化知识、记忆与运行记录
├── reports/                       # 报告输出目录
├── refresh_data.py                # 横截面数据刷新脚本
├── fetch_hs300_etf.py             # ETF 数据抓取脚本
├── run_backtest.py                # 回测入口
├── requirements.txt               # Python 依赖
└── README.md
```

---

## Core Modules

### 1. `hypothesis.py` — 因子假设生成器

融合 **AlphaAgent + Hubble + FactorMiner Ralph Loop** 的思路：

- 6 类因子族：`momentum / reversal / volatility / volume_price / liquidity / fundamental`
- 12+ DSL 算子：`rank / zscore / delta / lag / mean / std / ts_rank / add / sub / mul / div / clip ...`
- 3 类模板策略：基础时序、双特征交叉、波动率调整
- 经验记忆检索 + 族感知多样性惩罚 + 探索奖励
- 已替代旧版 `executor.py` 中的硬编码模板生成逻辑

### 2. `evolution.py` — 自主搜索循环

融合 **QuantaAlpha + FactorMiner + R&D-Agent-Quant** 的设计：

- 多轮闭环：假设 → 执行 → 评估 → 进化 → 再假设
- 支持 `mutation` / `crossover`
- 支持 early stop
- 自动沉淀成功模式、失败约束和观察记录

### 3. `datahub.py` — 统一数据抽象层

- Provider 抽象，当前支持本地 CSV
- 可扩展到 AkShare / Tushare / Wind 等数据源
- 内置缓存复用
- 输出数据质量报告：覆盖率、NaN 比例、最近刷新时间等

### 4. `pipeline.py` — 横截面评估体系

- Rank IC / ICIR
- 分位单调性
- 稳定性评分
- 可交易性评分
- 综合评分与决策分级：`approved / observe / rejected`

### 5. `AssistantToolRuntime` — Agent 工具运行时

当前已注册 **14 个工具**，可直接被上层 Agent 或自动化流程调用。

---

## Tooling

| Tool | Description |
|---|---|
| `view_current_config` | 查看当前配置与数据路径 |
| `list_strategies` | 查看可用策略列表 |
| `run_single_backtest` | 运行单次回测 |
| `run_grid_experiment` | 运行参数网格搜索 |
| `run_train_test_validation` | 训练-测试验证 |
| `run_walk_forward_validation` | Walk-forward 验证 |
| `run_multi_strategy_compare` | 多策略比较 |
| `review_portfolio_construction` | 组合构建评审 |
| `run_factor_discovery` | 单次因子发掘闭环 |
| `refresh_cross_section_data` | 刷新横截面数据 |
| `generate_factor_hypotheses` | 批量生成因子候选 |
| `run_factor_evolution` | 运行多轮进化搜索 |
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

### 2) Refresh market data

```bash
python refresh_data.py
```

如果你只想抓 ETF 数据：

```bash
python fetch_hs300_etf.py
```

### 3) Run a backtest

```bash
python run_backtest.py --data "data/hs300_etf.csv" --strategy ma_cross
```

### 4) Use as an Agent runtime

```python
from pathlib import Path
from quantlab.config import BacktestConfig
from quantlab.assistant.tools import AssistantToolRuntime

rt = AssistantToolRuntime(
    BacktestConfig(),
    Path("data/hs300_cross_section.csv")
)

hypotheses = rt.execute("generate_factor_hypotheses", {
    "research_direction": "量价背离",
    "max_candidates": 5,
})

result = rt.execute("run_factor_evolution", {
    "direction": "波动率调整动量",
    "max_rounds": 3,
    "candidates_per_round": 5,
})

single_run = rt.execute("run_factor_discovery", {
    "factor_prompt": "量价背离",
})
```

---

## Data Pipeline

当前数据链路面向 **沪深300横截面研究**：

- 行情主数据：横截面多资产价格序列
- 元数据补齐：简称 / 行业 / 市值 / 股本
- 主来源：AkShare
- 行业兜底：雪球个股资料接口
- 对 `unknown` 行业资产支持自动重新抓取

这意味着项目不只是“读一份 CSV”，而是已经具备了面向真实研究场景的数据刷新与修复能力。

---

## AI / LLM Configuration

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

系统在执行研究计划前，会先做可信度评估，避免“生成一个计划就直接跑”。

决策分流为：

- `pass`：直接执行
- `review_required`：降级为更保守的验证路径
- `fail`：自动重规划，补齐缺失链路

评估维度包括：

- 计划可信度
- 执行可信度
- 结论可信度

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

## Roadmap

- [x] 因子假设生成器
- [x] 多轮进化搜索循环
- [x] 统一数据抽象层 DataHub
- [x] AssistantToolRuntime 正式工具接入
- [x] 真实横截面闭环验证
- [ ] 更长历史区间与更多标签期评估
- [ ] 多因子筛选与组合层优化
- [ ] 更强的经验记忆治理
- [ ] 面向完整 Quant Agent 的任务编排能力

---

## License

MIT
