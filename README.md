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

核心设计：**LLM-first 深度推理 + 学术知识注入 + Block 系统安全执行 + 全链路无人值守自动化**。

```text
        冷启动种子因子引导（7 个学术因子）
              ↓
        每日增量数据刷新（Tushare/AkShare，含幸存者偏差修正）
              ↓
        市场状态检测（牛/熊/震荡）
              ↓
        因子衰减监控（4 项检测 → 自动触发再发掘）
              ↓
    ┌─→ LLM-first 深度推理假设生成（R1）+ 对抗审查（R2）
    │       │ ← 注入 12 个学术研究方向知识
    │       ↓
    │   Block 系统安全执行（28+ 算子可序列化）+ 参数搜索（T1）
    │       ↓
    │   统一 Rank IC 计算 → OOS 验证 + 成本调整 IC
    │       ↓
    │   经验回路沉淀 + 正交性引导
    │       ↓
    └── 进化搜索（自适应方向 + 元学习调参）+ LLM→模板自动回退
              ↓
        多因子 IC 加权正交组合 + 基准对比
              ↓
        因子库治理（生命周期 + 归档 + 拥挤度闭环）
              ↓
        风控评估（仓位/行业/回撤/流动性）
              ↓
        WorldQuant 标准交付筛选
              ↓
        纸交易启动 + 交付报告（JSON + Markdown）
              ↓
        告警通知（info/warning/critical 分级推送）
```

---

## Highlights

| 能力 | 说明 |
|------|------|
| **LLM-first 推理** | DeepSeek v4-pro thinking mode，LLM 为默认假设生成路径，模板为自动回退；R1 生成 + R2 审查 + P1 架构 + P3 定制代码 |
| **学术知识注入** | FactorKnowledgeBase：12 个经学术文献验证的研究方向知识（动量/反转/价值/质量/低波/流动性/波动率/情绪/资金流/事件驱动/低风险/截面异象），自动注入 R1 prompt |
| **Block 系统 v2** | 5 类 100+ 算子，28 个可通过 FactorNode→Block 序列化，Data/Transform/Combine/Relational/Filter |
| **8 阶段可组合调度器** | 每个阶段为独立 `PipelineStage`（`quantlab/pipeline_stages/`），可单独测试和复用 |
| **统一 Rank IC 计算** | `quantlab/metrics/ic_calculator.py` — 消除 7+ 处重复实现，确保评估一致性 |
| **多智能体协作** | 3 团队 6 Agent（R1+R2 → P1+P2+P3 → T1+T2），MessageBus + LLM 驱动 |
| **5 策略假设生成** | 时序算子 / 双特征交叉 / 波动率调整 / 截面算子 / 数学变换 — 覆盖 30 个算子 |
| **经验学习回路** | ExperienceLoop：成功/失败模式沉淀 → 指导下一轮假设生成 |
| **事前正交性引导** | OrthogonalityGuide：假设生成前分析因子库饱和度，避开已有空间 |
| **AST 沙箱安全执行** | 双重防御：AST 级别验证 + 子字符串黑名单，pandas/numpy 白名单 |
| **市场状态感知** | RegimeDetector：牛熊识别 → regime 调整 IC |
| **样本外验证** | 6 个月 OOS 切分，train/test IC 对比，衰减 > 50% 标记失败 |
| **成本调整 IC** | 周转率 → 交易成本冲击 → 成本调整后 IC |
| **幸存者偏差修正** | SurvivorshipFilter：新股/退市/极端收益过滤 |
| **因子库治理** | 归档 + 拥挤度集群检测 + 7 状态生命周期（DRAFT→...→RETIRED）|
| **风控层** | 仓位集中度 / 行业暴露 / 回撤熔断 / 流动性底线 |
| **冷启动引导** | 7 个学术种子因子自动注入经验回路 |
| **告警系统** | AlertBus：info/warning/critical 分级，文件持久化 |
| **纸交易 + 实盘券商** | PaperBroker + XtQuantBroker（华泰）+ JoinQuantBroker（聚宽），ConnectionManager 自动重连 |
| **多品种覆盖** | A 股 + 股指期货 + 商品期货 + 可转债 |
| **Windows 计划任务** | 一键安装每日无人值守运行 |
| **Web 管理面板** | Flask 仪表盘：因子库概览 + Pipeline 状态 + 告警列表，纯 HTML+JS，10 秒自动刷新 |
| **新闻知识摄入** | akshare 实时新闻 → LLM 智能提取因子研究方向 → 知识库自动追加 |
| **GPU 加速** | CuPy 加速 Rank IC 计算（自动 CPU 回退），批量因子并行评估 |
| **多因子实盘模拟** | LiveSimulator：每日跟踪 + 持仓/净值持久化 + 综合绩效报告 |
| **139 测试覆盖** | 19 个沙箱测试 + 99 个核心测试 + 21 个新增测试，覆盖所有模块 |

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                    DailyScheduler → PipelineStages (可组合编排)            │
│  刷新→衰减→进化(LLM-first+自适应)→OOS验证→组合+基准→筛选→治理→报告+纸交易  │
├──────────────────────────────────────────────────────────────────────────┤
│                 FactorMultiAgentOrchestrator (3-Team, 6-Agent)            │
│   Research (R1+R2,←知识注入) → Programming (P1+P2+P3) → Testing (T1+T2) │
├──────────────────────────────────────────────────────────────────────────┤
│                       Factor Enhancements Layer (9 模块)                   │
│  ExperienceLoop │ RiskNeutralizer │ FactorCombiner │ OrthogonalityGuide  │
│  ParameterSearcher │ CustomCodeGenerator(AST沙箱) │ CrowdingDetector     │
│  RegimeDetector │ FactorCurveAnalyzer                                     │
├──────────────────────┬──────────────────┬────────────────────────────────┤
│   Factor Discovery   │  Trading Engine  │      Data & Knowledge         │
│   ├ Hypothesis Gen   │  ├ CostModel     │  ├ DataHub（Tushare+AkShare） │
│   │  (5策略30算子)    │  ├ Portfolio     │  ├ FactorKnowledgeBase        │
│   ├ Block System v2  │  ├ Simulator     │  ├ SurvivorshipFilter         │
│   │  (28 FactorNode   │  ├ RiskManager   │  ├ MultiAssetContext          │
│   │   映射算子)        │  ├ PaperBroker   │  └ Seed Factors (7模板)      │
│   ├ Evolution Loop   │  └ OrderManager  │                               │
│   ├ Decay Monitor    │                  │   Shared Services             │
│   ├ Delivery Screener│                  │   ├ metrics/ic_calculator     │
│   ├ Factor Report    │                  │   └ knowledge/                │
│   └ LifecycleManager │                  │                               │
├──────────────────────┴──────────────────┴────────────────────────────────┤
│  SafeFactorExecutor / PersistentFactorStore / FactorExperienceMemory     │
│  AlertBus / Notifier / BrokerInterface / FactorLifecycleManager         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. `scheduler.py` + `pipeline_stages/` — 8 阶段可组合调度器

调度器已重构为可组合架构——`DailyScheduler`（~250 行）负责编排，每个阶段为独立的 `PipelineStage`（`quantlab/pipeline_stages/`），可单独测试和复用。

```bash
python -m quantlab.scheduler run_daily     # 执行一次完整管线
python -m quantlab.scheduler status        # 查看最近执行记录
python -m quantlab.scheduler install_cron  # 安装 Windows 计划任务
```

| 阶段 | 实现文件 | 说明 |
|------|----------|------|
| Phase 1 数据刷新 | `data_refresh.py` | AkShare 增量拉取 + Tushare 回退 + 幸存者偏差过滤 |
| Phase 2 衰减监控 | `decay_monitor.py` | 4 项检测，衰减因子自动标记再发掘 |
| Phase 3 进化搜索 | `evolution.py` | **LLM-first**（多智能体优先，模板自动回退）+ 自适应方向 + 元学习调参 |
| Phase 4 OOS 验证 | `oos_validation.py` | 6 个月 train/test 切分 + 成本调整 IC |
| Phase 5 多因子组合 | `combination.py` | IC 加权 + 正交选择 + 等权/市值加权基准对比 |
| Phase 6 交付筛选 | `delivery.py` | WorldQuant 8 项标准 |
| Phase 7 因子库治理 | `governance.py` | 归档 + 市场状态 + 拥挤度闭环 + 因子曲线 + 风控 |
| Phase 8 交付报告 + 纸交易 | `delivery.py` | JSON + Markdown 买方尽调报告 + PaperBroker 启动 |

### 2. `multi_agent.py` — 多智能体协作框架

3 团队 6 Agent 协作：R1 假设生成（LLM-first + 知识注入 + 经验回路 + 正交引导）+ R2 对抗审查 → P1 架构设计 + P2 Block 组装 + P3 定制编码（AST 沙箱）→ T1 回测运行（参数搜索 + 风险中性化）+ T2 结果验证。

LLMClient 支持 DeepSeek thinking mode（`reasoning_effort=max`）、模型自动降级、`.env` 配置。

### 3. `knowledge/` — 量化因子研究知识库

12 个经学术文献验证的研究方向知识，每个条目包含核心直觉、经典做法、潜在 Alpha 来源和关键文献。在 LLM 假设生成时自动注入 prompt。

| 研究方向 | 关键文献 |
|----------|----------|
| 动量与反转 | Jegadeesh & Titman (1993); Moskowitz, Ooi & Pedersen (2012) |
| 质量与盈利 | Sloan (1996); Novy-Marx (2013); Fama & French (2015) |
| 量价背离 | Gervais, Kaniel & Mingelgrin (2001); Lee & Swaminathan (2000) |
| 波动率与市场状态 | Ang et al. (2006); Moreira & Muir (2017) |
| 流动性溢价 | Amihud & Mendelson (1986); Pastor & Stambaugh (2003) |
| 价值与逆向 | Fama & French (1992); Lakonishok, Shleifer & Vishny (1994) |
| 规模效应 | Banz (1981); Asness et al. (2018) |
| 情绪与行为金融 | Baker & Wurgler (2006); Stambaugh, Yu & Yuan (2012) |
| 资金流向 | Lou (2012); Frazzini & Lamont (2008) |
| 事件驱动 | Ball & Brown (1968); Bernard & Thomas (1989) |
| 低风险异象 | Frazzini & Pedersen (2014) BAB 因子 |
| 截面异象综合 | McLean & Pontiff (2016); Harvey, Liu & Zhu (2016) |

### 4. `metrics/` — 统一 IC 计算

`compute_rank_ic()` 和 `compute_ic_sequence()` — 消除原本分布在 7+ 文件中的重复 Rank IC 计算，确保同一组 factor_values + market_df 无论走哪个管线都得到相同 IC。

### 5. `blocks.py` — Block 系统 v2

可序列化因子 DSL，100+ 注册算子：

| 类型 | 算子示例 |
|------|----------|
| Transform | rank, zscore, delta, ts_std, ts_mean, ts_rank, log, ema, group_neutralize, group_rank, rolling_ols_residual, constant |
| Combine | add, sub, mul, div, max, min, where |
| Relational | group_aggregate, cross_section_ols, event_filter |
| Filter | where_condition, sample_weight |

### 6. `factor_enhancements.py` — 9 大增强模块

| 模块 | 功能 |
|------|------|
| ExperienceLoop | 历史假设→结果映射，沉淀成功/失败模式，指导 R1 生成 |
| RiskNeutralizer | 行业/市值/动量中性化，管控风险暴露 |
| FactorCombiner | IC 加权 + 正交选择 + 组合 ICIR |
| ParameterSearcher | Grid/Random 参数搜索 |
| CustomCodeGenerator | LLM 生成 Python 因子代码 + **AST 沙箱双重验证** |
| OrthogonalityGuide | 事前正交性分析，避开饱和方向 |
| CrowdingDetector | 两两相关性 → 集群 → 拥挤评分 |
| RegimeDetector | 牛熊识别 → regime 调整 IC |
| FactorCurveAnalyzer | IC 衰减曲线 + 半衰期估计 |

### 7. `hypothesis.py` — 5 策略假设生成

覆盖 30 个 DSL 算子的模板假设生成，LLM 路径不可用时自动回退：
- 策略 1：时序算子（窗口 × 特征 × operator → rank 包裹）
- 策略 2：双特征交叉（zscore+delta 预处理 + 算术交叉）
- 策略 3：波动率调整（短 delta / 长 std）
- 策略 4：截面算子（group_neutralize / group_rank / quantile）
- 策略 5：数学变换（abs / log / sigmoid / sign）

### 8. `trading/` — 交易引擎

- **cost_model.py** — A 股真实成本：佣金万三 / 印花税千一 / 滑点万二 / 冲击成本
- **risk_control.py** — RiskManager：仓位集中度 / 行业暴露 / 回撤熔断 / 流动性过滤
- **broker.py** — BrokerInterface（抽象）+ PaperBroker + XtQuantBroker + JoinQuantBroker
- **portfolio.py** — 4 种权重方案（等权 / 评分 / IC 加权 / 行业中性）
- **simulator.py** — 日频组合模拟（换手 / 成本 / 净值 / 回撤 / Sharpe / IR）

### 9. `runtime.py` — 因子运行时

- SafeFactorExecutor — FactorNode → Block 转换 + 统一执行
- PersistentFactorStore — JSON 文件库 + 因子面板持久化

### 10. `seed_factors.py` — 冷启动引导

7 个学术种子因子自动注入（动量 / 反转 / 价值 / 规模 / 低波 / 换手 / 质量），Block 系统构造，首次运行注入经验回路。可用数据字段不足时自动跳过（如缺少 `pb` 时不注入价值因子）。

### 11-14. 其余模块

| # | 模块 | 功能 |
|---|------|------|
| 11 | `datahub.py` | TushareProProvider + AkShareIncrementalProvider + MultiAssetContext |
| 12 | `survivorship.py` | 新股/退市/极端收益过滤 |
| 13 | `notifier.py` | AlertBus + Notifier 分级告警 |
| 14 | `futures_factors.py` / `convertible_bond_factors.py` | 期货 + 可转债因子模板 |

---

## Project Structure

```text
quant-agent/
├── quantlab/
│   ├── scheduler.py                    # 调度编排器（~250 行）
│   ├── pipeline_stages/                # 可组合管线阶段
│   │   ├── base.py                     #   PipelineContext + PipelineStage ABC
│   │   ├── data_refresh.py             #   数据刷新
│   │   ├── decay_monitor.py            #   衰减监控
│   │   ├── evolution.py                #   进化搜索（LLM-first）
│   │   ├── oos_validation.py           #   样本外验证
│   │   ├── combination.py              #   多因子组合 + 基准对比
│   │   ├── governance.py               #   因子库治理
│   │   └── delivery.py                 #   交付筛选 + 纸交易 + 报告
│   ├── metrics/                        # 共享指标计算
│   │   ├── ic_calculator.py            #   统一 Rank IC（MultiIndex 兼容）
│   │   └── gpu_accelerator.py          #   GPU 加速 + 批量评估
│   ├── knowledge/                      # 因子研究知识库
│   │   ├── factor_research_knowledge.py #   12 个学术研究方向
│   │   └── news_ingestor.py            #   新闻实时摄入器（akshare+LLM）
│   ├── web/                            # Web 管理面板
│   │   ├── app.py                      #   Flask 仪表盘（API + HTML）
│   │   └── __init__.py
│   ├── config.py                       # 全局配置
│   ├── pipeline.py                     # 主研究管线
│   ├── factor_discovery/               # 因子发掘核心
│   │   ├── blocks.py                   #   Block 系统 v2 (100+ 算子，28 FactorNode 映射)
│   │   ├── hypothesis.py               #   5 策略假设生成器（30 算子）
│   │   ├── evolution.py                #   进化搜索循环
│   │   ├── pipeline.py                 #   横截面评估体系
│   │   ├── runtime.py                  #   安全执行器 + 因子库
│   │   ├── models.py                   #   数据模型 + 7 状态生命周期
│   │   ├── datahub.py                  #   统一数据层
│   │   ├── multi_agent.py              #   3 团队 6 Agent + 知识注入
│   │   ├── factor_enhancements.py      #   9 增强模块（AST 沙箱）
│   │   ├── seed_factors.py             #   冷启动种子因子
│   │   ├── survivorship.py             #   幸存者偏差修正
│   │   ├── decay_monitor.py            #   衰减监控
│   │   ├── delivery_screener.py        #   交付筛选
│   │   └── factor_report.py            #   交付报告生成
│   ├── trading/                        # 交易引擎
│   │   ├── cost_model.py               #   A股真实成本模型
│   │   ├── risk_control.py             #   风控层
│   │   ├── broker.py                   #   券商接口 + 纸交易
│   │   ├── portfolio.py                #   4 种权重方案
│   │   ├── simulator.py                #   日频组合模拟
│   │   └── live_simulator.py           #   实盘模拟 + 持仓跟踪
│   ├── assistant/                      # Agent 运行时
│   │   ├── tools.py                    #   21 个工具
│   │   ├── notifier.py                 #   告警系统
│   │   └── llm.py                      #   LLM 接入
│   ├── research/                       # 研究任务框架
│   │   ├── executor.py                 #   14 类任务执行
│   │   ├── models.py                   #   研究模型
│   │   └── protocol.py                 #   任务协议
│   └── data/                           # 数据 Provider
│       ├── tushare_provider.py         #   Tushare Pro
│       ├── fetcher.py                  #   数据抓取
│       └── loader.py                   #   数据加载
├── tests/                              # 139 个测试（12 个模块）
│   ├── test_blocks.py                  #   18 测试
│   ├── test_broker.py                  #   11 测试
│   ├── test_custom_code_sandbox.py     #   19 测试（AST 沙箱）
│   ├── test_data_survivor.py           #   14 测试
│   ├── test_decay_agent.py             #   13 测试
│   ├── test_enhancements.py            #   16 测试
│   ├── test_gpu_accelerator.py         #   12 测试（GPU 加速）
│   ├── test_live_simulator.py          #   9 测试（实盘模拟）
│   ├── test_portfolio.py               #   6 测试
│   ├── test_scheduler.py               #   11 测试
│   └── test_trading.py                 #   8 测试
├── data/                               # 本地数据（含 zz800_cross_section.csv）
├── assistant_data/                     # 因子库 + 组合 + 告警 + 经验
├── reports/                            # 交付报告
├── pull_zz800_cross_section.py         # 中证 800 数据拉取脚本
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1) Install

```bash
pip install -r requirements.txt
```

主要依赖：`pandas>=2.2.0`, `numpy>=1.26.0`, `akshare>=1.18.55`, `scipy`, `flask>=3.0.0`

GPU 加速（可选）：`pip install cupy-cuda12x`（需 NVIDIA GPU + CUDA 12.x）

### 2) Configure

项目根目录创建 `.env`：

```ini
TUSHARE_TOKEN=your_tushare_token

# DeepSeek LLM（驱动智能体推理，可选；不配则走模板生成路径）
ASSISTANT_API_KEY=sk-xxx
ASSISTANT_BASE_URL=https://api.deepseek.com
ASSISTANT_MODEL=deepseek-v4-pro
ASSISTANT_FALLBACK_MODEL=deepseek-v4-pro
ASSISTANT_REASONING_EFFORT=max
```

### 3) Pull Data

```bash
# 拉取中证 800 截面数据（使用 akshare，无需 Tushare token）
python pull_zz800_cross_section.py --start 20240101 --assets 150

# 或使用 Tushare Pro（更快，需要 token）
python pull_zz800_cross_section.py --start 20240101 --provider tushare
```

### 4) Run

```bash
# 每日因子工厂（全链路无人值守，LLM 优先 + 模板回退）
python -m quantlab.scheduler run_daily

# 查看执行记录
python -m quantlab.scheduler status

# 安装 Windows 计划任务
python -m quantlab.scheduler install_cron

# 启动 Web 管理面板（端口 8080）
python -m quantlab.web.app

# 从新闻提取因子研究知识
python -m quantlab.knowledge.news_ingestor --keywords "量化因子,A股" --max 10

# 运行测试（跳过依赖网络的测试）
python -m pytest tests/ -v --ignore=tests/test_scheduler.py --ignore=tests/test_decay_agent.py
```

### 5) Python API

```python
from quantlab.factor_discovery.multi_agent import (
    FactorMultiAgentOrchestrator, MultiAgentConfig,
)
from quantlab.factor_discovery.datahub import DataHub

hub = DataHub()
df = hub.load("data/zz800_cross_section.csv")

orchestrator = FactorMultiAgentOrchestrator(
    config=MultiAgentConfig(
        enable_knowledge_injection=True,   # 注入学术研究知识
        enable_experience_loop=True,
        enable_orthogonality_guide=True,
        enable_factor_combination=True,
        enable_custom_code_gen=True,
    ),
)
result = orchestrator.run(direction="momentum_reversal", market_df=df)
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
- [x] **LLM-first 默认路径**（移除经验 >10 门控，LLM 优先 + 模板自动回退）
- [x] Block 系统 v2（100+ 算子，28 FactorNode 映射）
- [x] 多智能体协作（3 团队 6 Agent）
- [x] **学术知识注入**（12 研究方向 FactorKnowledgeBase）
- [x] **统一 Rank IC 计算**（消除 7+ 处重复，MultiIndex 兼容）
- [x] **可组合调度器架构**（PipelineStage + PipelineContext）
- [x] 8 阶段日常调度器（含 OOS 验证 + 成本调整 IC）
- [x] **AST 沙箱双重验证**（AST 级别 + 子字符串黑名单，19 测试覆盖）
- [x] 5 策略 30 算子假设生成
- [x] 经验学习回路 + 事前正交性引导
- [x] 市场状态检测 + regime 调整 IC
- [x] 幸存者偏差修正
- [x] 因子生命周期管理（7 状态流转）
- [x] 因子库治理（归档 + 拥挤度闭环）
- [x] 风控层（仓位 / 行业 / 回撤 / 流动性）
- [x] 冷启动种子因子引导（7 个学术因子，缺数据字段自动跳过）
- [x] 告警系统（分级 + 文件持久化）
- [x] 纸交易 + 实盘券商（华泰 + 聚宽）
- [x] 多品种覆盖（A 股 + 期货 + 可转债）
- [x] 真实数据接入（Tushare/AkShare 中证 800 截面数据）
- [x] Web 管理面板（`quantlab/web/` — Flask 仪表盘，实时因子库/Pipeline/告警监控）
- [x] 研报/新闻实时知识摄入（`quantlab/knowledge/news_ingestor.py` — akshare+LLM 双路径提取）
- [x] GPU 加速因子搜索（`quantlab/metrics/gpu_accelerator.py` — CuPy 加速 IC 计算，自动 CPU 回退，批量评估 3.1x 加速）
- [x] 多因子组合实盘模拟（`quantlab/trading/live_simulator.py` — 每日跟踪 + 净值持久化 + 绩效报告）
- [x] **139 测试通过**（21 个新增：GPU 加速器 12 + 实盘模拟 9）

---

## License

MIT
