# Quant Agent

<div align="center">

**AI-native Quant Factor Factory — Fully Autonomous Cross-sectional Factor Discovery**

聚焦 **A股横截面 alpha 因子发掘** 的全自动 AI Agent 系统。围绕 **LLM 假设生成 → Block 系统安全执行 → 横截面评估 → 进化搜索 → Agent OOS 诊断 → 多因子组合 → 交付筛选 → Agent 治理解读 → Agent 叙事报告** 构建完整的 WorldQuant 风格因子工厂闭环。

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Pandas" src="https://img.shields.io/badge/Pandas-2.2%2B-150458?style=flat-square&logo=pandas&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-1.26%2B-013243?style=flat-square&logo=numpy&logoColor=white">
  <img alt="Tests" src="https://img.shields.io/badge/Tests-136%20passed-green?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green?style=flat-square">
</p>

</div>

---

## Overview

Quant Agent 是一个朝 **WorldQuant 卖因子** 目标演进的 AI Agent 系统——**发现因子、验证因子、筛选因子、交付因子**，不是跑回测的脚本集合。

核心设计：**LLM-first 深度推理 + LLM 输出监督 + 学术知识注入 + Block 系统安全执行 + Agent 驱动的诊断/解读/叙事 + 衰减→再发掘闭环 + 全链路无人值守自动化**。

```text
        冷启动种子因子引导（7 个学术因子）
              ↓
        每日增量数据刷新（Tushare/AkShare，含幸存者偏差修正）
              ↓
        因子衰减监控（4 项检测 → 自动触发再发掘）
              ↓
    ┌─→ LLM-first 深度推理假设生成（R1）+ 对抗审查（R2）
    │       │ ← 注入 12 个学术研究方向知识
    │       │ ← 注入上一轮 Agent 反馈（优先/避免方向）
    │       ↓
    │   Block 系统安全执行（49 算子可序列化）+ 参数搜索（T1）
    │       ↓
    │   统一 Rank IC 计算 → Agent OOS 诊断（逐因子分析 + 跨轮反馈）
    │       ↓
    │   经验回路沉淀 + 正交性引导
    │       ↓
    └── 进化搜索（自适应方向 + 元学习调参）+ LLM→模板自动回退
              ↓
        多因子 IC 加权正交组合 + 基准对比
              ↓
        WorldQuant 标准交付筛选 + 纸交易启动
              ↓
        Agent 因子库治理（市况解读 + 拥挤分析 + 衰减诊断 + 前瞻建议）
              ↓
        Agent 交付报告（数值报告 + LLM 买方叙事 + 尽调清单）
              ↓
        告警通知（info/warning/critical 分级推送）
```

---

## Highlights

| 能力 | 说明 |
|------|------|
| **LLM-first 推理** | LLM 为默认假设生成路径，模板为自动回退；R1 生成 + R2 审查 + P1 架构 + P3 定制代码 |
| **Agent 驱动后三阶段** | OOS 验证/治理解读/交付报告均由 AgentAnalyst 进行 LLM 定性分析，规则引擎负责数值计算 |
| **Agent 反馈闭环** | OOS 诊断 + 治理分析 → ctx._meta → 下一轮 EvolutionStage 注入 R1 prompt |
| **学术知识注入** | FactorKnowledgeBase：12 个经学术文献验证的研究方向知识，自动注入 R1 prompt |
| **Block 系统 v2** | 5 类 49 算子，含高阶矩(skew/kurt)、协方差、幂函数等，Data/Transform/Combine/Relational/Filter |
| **9 阶段可组合调度器** | 每个阶段为独立 `PipelineStage`（`quantlab/pipeline_stages/`），可单独测试和复用 |
| **统一 Rank IC 计算** | `quantlab/metrics/ic_calculator.py` — 消除 7+ 处重复实现，确保评估一致性 |
| **多智能体协作** | 3 团队 6 Agent（R1+R2 → P1+P2+P3 → T1+T2），MessageBus + LLM 驱动 |
| **5 策略假设生成** | 时序算子 / 双特征交叉 / 波动率调整 / 截面算子 / 数学变换 — 覆盖 30 个算子 |
| **经验学习回路** | ExperienceLoop：成功/失败模式沉淀 → 指导下一轮假设生成 |
| **事前正交性引导** | OrthogonalityGuide：假设生成前分析因子库饱和度，避开已有空间 |
| **AST 沙箱安全执行** | 双重防御：AST 级别验证 + 子字符串黑名单，pandas/numpy 白名单 |
| **市场状态感知** | RegimeDetector：牛熊识别 → regime 调整 IC |
| **厚数据集** | 280 HS300 成分股 × 5.3 年 × 16 列，PE/PB/换手率/市值/行业全填充 |
| **样本外验证** | 6 个月 OOS 切分，train/test IC 对比，Agent 逐因子诊断 |
| **成本调整 IC** | 周转率 → 交易成本冲击 → 成本调整后 IC |
| **幸存者偏差修正** | SurvivorshipFilter：新股/退市/极端收益过滤 |
| **因子库治理** | 归档 + 拥挤度集群检测 + 7 状态生命周期 + Agent 综合解读 |
| **风控层** | 仓位集中度 / 行业暴露 / 回撤熔断 / 流动性底线 |
| **冷启动引导** | 7 个学术种子因子自动注入经验回路 |
| **告警系统** | AlertBus：info/warning/critical 分级，Agent 分析告警增强 |
| **纸交易 + 实盘券商** | PaperBroker + XtQuantBroker（华泰）+ JoinQuantBroker（聚宽），ConnectionManager 自动重连 |
| **多品种覆盖** | A 股 + 股指期货 + 商品期货 + 可转债 |
| **Windows 计划任务** | 一键安装每日无人值守运行 |
| **Web 管理面板** | Flask 仪表盘：因子库概览 + Pipeline 状态 + 告警列表，纯 HTML+JS，10 秒自动刷新 |
| **新闻知识摄入** | akshare 实时新闻 → LLM 智能提取因子研究方向 → 知识库自动追加 |
| **GPU 加速** | CuPy 加速 Rank IC 计算（自动 CPU 回退），批量因子并行评估 |
| **多因子实盘模拟** | LiveSimulator：每日跟踪 + 持仓/净值持久化 + 综合绩效报告 |
| **LLM 输出监督** | LLMSupervisor：第二 LLM 自动纠正 JSON + 重试回退 |  
| **LLM 熔断保护** | 10min 超时 + 3次慢响应自动降级模板 + 1h 冷却 |
| **衰减→再发掘** | 衰减因子自动触发定向再进化，版本链保持 |
| **23 基准因子** | BenchmarkFactorRegistry：Block DSL 表达，相关性对比 |
| **异常检测** | AnomalyGuard：数据质量/拆股/停牌自动检测 |
| **FDR 校正** | Benjamini-Hochberg 多重检验校正 |
| **管线断点恢复** | 每阶段 checkpoint + `--resume` |
| **邮件推送** | QQ SMTP：告警 + 日度摘要 |
| **136 测试覆盖** | 17 Agent + 18 Block + 19 沙箱 + 4 集成 + 其余模块 |

---

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                  DailyScheduler → PipelineStages (9-stage composable)         │
│   Refresh→Decay→Evolution(LLM-first)→AgentOOS→Combine→Screen→AgentGov→Report│
│                    ↑ Agent Feedback Loop (ctx._meta) ↓                       │
├──────────────────────────────────────────────────────────────────────────────┤
│               FactorMultiAgentOrchestrator (3-Team, 6-Agent)                  │
│   Research (R1+R2,←知识注入+Agent反馈) → Programming (P1+P2+P3) → Testing (T1+T2)│
├──────────────────────────────────────────────────────────────────────────────┤
│                     Factor Enhancements Layer (9 modules)                     │
│  ExperienceLoop │ RiskNeutralizer │ FactorCombiner │ OrthogonalityGuide      │
│  ParameterSearcher │ CustomCodeGenerator(AST沙箱) │ CrowdingDetector         │
│  RegimeDetector │ FactorCurveAnalyzer                                         │
├─────────────────────────┬──────────────────┬─────────────────────────────────┤
│    Factor Discovery     │  Trading Engine  │   Data & Knowledge              │
│    ├ Hypothesis Gen     │  ├ CostModel     │ ├ DataHub（Tushare+AkShare）    │
│    │  (5策略30算子)      │  ├ Portfolio     │ ├ FactorKnowledgeBase (12方向)  │
│    ├ Block System v2    │  ├ Simulator     │ ├ SurvivorshipFilter            │
│    │  (49算子,可序列化)   │  ├ RiskManager   │ ├ MultiAssetContext (4 domain)  │
│    ├ Evolution Loop     │  ├ PaperBroker   │ └ Seed Factors (7模板)          │
│    ├ Decay Monitor      │  ├ LiveSimulator │                                 │
│    ├ Delivery Screener  │  └ OrderManager  │   Shared Services               │
│    ├ Factor Report      │                  │ ├ metrics/ic_calculator.py      │
│    └ LifecycleManager   │                  │ ├ metrics/gpu_accelerator.py    │
│                          │                  │ ├ knowledge/                    │
│    Agent Analysis Layer  │                  │ ├ web/ (Flask dashboard)        │
│    ├ AgentAnalyst        │                  │ └ build_dataset.py              │
│    │  ├ analyze_oos()    │                  │   (thick dataset builder)       │
│    │  ├ analyze_gov()    │                  │                                 │
│    │  ├ narrative_rpt()  │                  │                                 │
│    │  └ gen_feedback()   │                  │                                 │
├─────────────────────────┴──────────────────┴─────────────────────────────────┤
│  SafeFactorExecutor / PersistentFactorStore / FactorExperienceMemory         │
│  AlertBus / Notifier / BrokerInterface / FactorLifecycleManager              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Modules

### 1. `scheduler.py` + `pipeline_stages/` — 9 阶段可组合调度器

调度器已重构为可组合架构——`DailyScheduler`（~250 行）负责编排，每个阶段为独立的 `PipelineStage`，可单独测试和复用。后三阶段为 **Agent 驱动**（LLM 定性分析 + 规则引擎数值计算）。

```bash
python -m quantlab.scheduler run_daily     # 执行一次完整管线
python -m quantlab.scheduler status        # 查看最近执行记录
python -m quantlab.scheduler install_cron  # 安装 Windows 计划任务
```

| 阶段 | 实现文件 | 驱动方式 | 说明 |
|------|----------|----------|------|
| Phase 1 数据刷新 | `data_refresh.py` | 规则 | AkShare 增量拉取 + Tushare 回退 + 幸存者偏差过滤 |
| Phase 2 衰减监控 | `decay_monitor.py` | 规则 | 4 项检测，衰减因子自动标记再发掘 |
| Phase 3 进化搜索 | `evolution.py` | **LLM Agent** | LLM-first 多智能体 + 模板自动回退 + 自适应方向 + Agent 反馈注入 |
| Phase 4 OOS 验证 | `oos_validation.py` | **Agent 驱动** | 数值层(拆分+IC) + Agent 逐因子诊断 + 跨轮反馈 |
| Phase 5 多因子组合 | `combination.py` | 规则 | IC 加权 + 正交选择 + 等权/市值加权基准对比 |
| Phase 6 交付筛选 | `delivery.py` | 规则 | WorldQuant 8 项标准 |
| Phase 6.5 纸交易 | `delivery.py` | 规则 | PaperBroker 启动 + 日度再平衡 |
| Phase 7 因子库治理 | `governance.py` | **Agent 驱动** | 数值层(regime/crowding/curve/risk) + Agent 综合解读 + 前瞻建议 |
| Phase 8 交付报告 | `delivery.py` | **Agent 驱动** | 数值报告 + LLM 买方叙事 + 尽调清单 |

### 2. `agent_analyst.py` — Agent 分析引擎

共享 LLM 分析引擎，为后三阶段提供定性分析。LLM 不可用时自动回退到规则化摘要，不阻塞管线。

| 方法 | 触发阶段 | 输出 |
|------|---------|------|
| `analyze_oos()` | OOS 验证 | 逐因子诊断 (healthy/overfit/structural_break/weak_signal) + 跨因子汇总 |
| `analyze_governance()` | 治理 | 执行摘要 + 市况解读 + 拥挤分析 + 衰减解读 + 前瞻建议 |
| `generate_narrative_report()` | 交付报告 | 执行摘要 + 因子故事 + 优劣势 + 市况适配 + 买方尽调清单 |
| `generate_feedback()` | 跨轮反馈 | 优先/避免方向 + 有效/失败模式 → 注入下一轮 EvolutionStage |

### 3. `multi_agent.py` — 多智能体协作框架

3 团队 6 Agent 协作：R1 假设生成（LLM-first + 知识注入 + 经验回路 + 正交引导 + Agent 反馈）+ R2 对抗审查 → P1 架构设计 + P2 Block 组装 + P3 定制编码（AST 沙箱）→ T1 回测运行（参数搜索 + 风险中性化）+ T2 结果验证。

LLMClient 支持 thinking mode、模型自动降级、`.env` 配置、跨运行上下文记忆。

### 4. `knowledge/` — 量化因子研究知识库

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

### 5. `metrics/` — 统一 IC 计算

`compute_rank_ic()` 和 `compute_ic_sequence()` — 消除原本分布在 7+ 文件中的重复 Rank IC 计算。`GpuAccelerator.batch_compute_ic()` 提供 GPU 加速批量评估（500 因子约 3.1x 提升），自动 CPU 回退。

### 6. `blocks.py` — Block 系统 v2（49 算子）

可序列化因子 DSL：

| 类型 | 数量 | 算子 |
|------|------|------|
| Transform | 27 | rank, zscore, quantile, top_n, bottom_n, delta, lag, ts_mean, ts_std, ts_rank, ts_max, ts_min, ts_sum, ts_corr, ts_cov, ts_argmax, ts_argmin, ts_skew, ts_kurt, ema, rolling_ols_residual, abs, sign, log, sigmoid, clip, piecewise, power, sqrt, scale, constant |
| Group Transform | 5 | group_neutralize, group_rank, group_zscore, group_top_n, group_bottom_n |
| Combine | 7 | add, sub, mul, div, max, min, where |
| Relational | 3 | group_aggregate, cross_section_ols, event_filter |
| Filter | 2 | where_condition, sample_weight |

**新增算子**（v2.1）：`power`, `sqrt`, `scale`, `ts_skew`, `ts_kurt`, `ts_cov`

### 7. `factor_enhancements.py` — 9 大增强模块

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

### 8. `hypothesis.py` — 5 策略假设生成

覆盖 30+ DSL 算子的模板假设生成，LLM 路径不可用时自动回退。

### 9. `trading/` — 交易引擎

- **cost_model.py** — A 股真实成本：佣金万三 / 印花税千一 / 滑点万二 / 冲击成本
- **risk_control.py** — RiskManager：仓位集中度 / 行业暴露 / 回撤熔断 / 流动性过滤
- **broker.py** — BrokerInterface（抽象）+ PaperBroker + XtQuantBroker + JoinQuantBroker
- **portfolio.py** — 4 种权重方案（等权 / 评分 / IC 加权 / 行业中性）
- **simulator.py** — 日频组合模拟（换手 / 成本 / 净值 / 回撤 / Sharpe / IR）
- **live_simulator.py** — 每日实盘模拟（持仓跟踪 + NAV 持久化 + 绩效报告）

### 10. 数据管线

#### 数据集

| 文件 | 说明 |
|------|------|
| `data/cross_section_thick.csv` | **主力数据集**：280 HS300 成分股 × 1,288 天（5.3 年）× 16 列 |
| `data/hs300_cross_section.csv` | 原始 HS300 数据（1 年，含完整基本面，用于扩展基础）|
| `data/zz800_cross_section.csv` | 中证 800 数据（legacy）|

#### 主力数据列

| 列 | 填充率 | 来源 |
|----|--------|------|
| open/high/low/close/volume/amount | 100% | tushare daily() 向后扩展 |
| turnover | 100% | volume / outstanding_share 计算 |
| market_cap | 100% | close × outstanding_share 计算 |
| pb | 100% | akshare 百度估值 + 财报净资产补充 |
| pe | 97% | akshare 季度财报 TTM + merge_asof + ffill |
| industry | 100% | HS300 元数据静态填充 |

#### 数据脚本

```bash
# 全量构建（扩展 HS300 数据到 2021，含 checkpoint 断点续传）
python build_dataset.py --full

# 增量刷新（拉取新交易日数据）
python refresh_data.py

# 使用原始 HS300 数据（不扩展）
python build_dataset.py --use-snapshot
```

---

## Project Structure

```text
quant-agent/
├── quantlab/
│   ├── scheduler.py                    # 调度编排器
│   ├── pipeline_stages/                # 可组合管线阶段
│   │   ├── base.py                     #   PipelineContext + PipelineStage ABC
│   │   ├── data_refresh.py             #   数据刷新
│   │   ├── decay_monitor.py            #   衰减监控
│   │   ├── evolution.py                #   进化搜索（LLM-first + Agent反馈注入）
│   │   ├── oos_validation.py           #   Agent OOS 验证（数值+LLM诊断）
│   │   ├── combination.py              #   多因子组合 + 基准对比
│   │   ├── governance.py               #   Agent 治理（数值+LLM解读）
│   │   ├── delivery.py                 #   交付筛选 + 纸交易 + Agent 报告
│   │   └── agent_analyst.py            #   共享 Agent 分析引擎（4 分析域）
│   ├── metrics/                        # 共享指标计算
│   │   ├── ic_calculator.py            #   统一 Rank IC（MultiIndex 兼容）
│   │   └── gpu_accelerator.py          #   GPU 加速 + 批量评估
│   ├── knowledge/                      # 因子研究知识库
│   │   ├── factor_research_knowledge.py #   12 个学术研究方向
│   │   └── news_ingestor.py            #   新闻实时摄入器（akshare+LLM）
│   ├── web/                            # Web 管理面板
│   │   ├── app.py                      #   Flask 仪表盘（API + HTML）
│   │   └── __init__.py
│   ├── config.py                       # 全局配置 + DEFAULT_DATA_PATH
│   ├── pipeline.py                     # 主研究管线
│   ├── factor_discovery/               # 因子发掘核心
│   │   ├── blocks.py                   #   Block 系统 v2 (49 算子)
│   │   ├── hypothesis.py               #   5 策略假设生成器
│   │   ├── evolution.py                #   进化搜索循环
│   │   ├── pipeline.py                 #   横截面评估体系
│   │   ├── runtime.py                  #   安全执行器 + 因子库
│   │   ├── models.py                   #   数据模型 + 7 状态生命周期
│   │   ├── datahub.py                  #   统一数据层
│   │   ├── multi_agent.py              #   3 团队 6 Agent
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
├── tests/                              # 136 个测试（12 模块）
│   ├── test_agent_pipeline_stages.py   #   17 测试（Agent 阶段）
│   ├── test_blocks.py                  #   18 测试（Block DSL）
│   ├── test_broker.py                  #   12 测试（券商）
│   ├── test_custom_code_sandbox.py     #   19 测试（AST 沙箱）
│   ├── test_data_survivor.py           #   15 测试（幸存者偏差）
│   ├── test_enhancements.py            #   16 测试（增强模块）
│   ├── test_gpu_accelerator.py         #   12 测试（GPU 加速）
│   ├── test_live_simulator.py          #   9 测试（实盘模拟）
│   ├── test_portfolio.py               #   6 测试（组合）
│   ├── test_trading.py                 #   8 测试（交易引擎）
│   ├── test_scheduler.py               #   12 测试（网络依赖）
│   └── test_decay_agent.py             #   12 测试（网络依赖）
├── data/                               # 本地数据
│   ├── cross_section_thick.csv         #   主力厚数据集（43 MB）
│   ├── cross_section_thick_meta.json   #   数据集元信息
│   ├── hs300_cross_section.csv         #   原始 HS300 数据
│   ├── zz800_cross_section.csv         #   中证 800 数据（legacy）
│   ├── checkpoints/                    #   数据集构建断点
│   ├── scheduler/                      #   调度器运行记录
│   └── trading/                        #   纸交易日志
├── assistant_data/                     # 因子库 + 经验 + 知识库
├── reports/                            # 交付报告
├── build_dataset.py                    # 厚数据集构建器（含 checkpoint 断点续传）
├── refresh_data.py                     # 增量数据刷新
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

主要依赖：`pandas>=2.2.0`, `numpy>=1.26.0`, `akshare>=1.18.55`, `scipy`, `flask>=3.0`

GPU 加速（可选）：`pip install cupy-cuda12x`（需 NVIDIA GPU + CUDA 12.x）

### 2) Configure

项目根目录创建 `.env`：

```ini
TUSHARE_TOKEN=your_tushare_token

# LLM（驱动 Agent 推理，可选；不配则走规则回退 + 模板生成路径）
ASSISTANT_API_KEY=sk-xxx
ASSISTANT_BASE_URL=https://api.deepseek.com
ASSISTANT_MODEL=deepseek-v4-pro
ASSISTANT_FALLBACK_MODEL=deepseek-v4-pro
ASSISTANT_REASONING_EFFORT=max
```

### 3) Build Dataset

```bash
# 全量构建厚数据集（280 HS300 成分股 × 5.3 年 × 16 列，含断点续传）
python build_dataset.py --full

# 增量刷新（拉取最新交易日）
python refresh_data.py

# 使用已有 HS300 数据（不扩展历史）
python build_dataset.py --use-snapshot
```

### 4) Run

```bash
# 每日因子工厂（全链路无人值守，LLM Agent 优先 + 规则回退）
python -m quantlab.scheduler run_daily

# 查看执行记录
python -m quantlab.scheduler status

# 安装 Windows 计划任务（每日 18:30）
python -m quantlab.scheduler install_cron

# 启动 Web 管理面板（端口 8080）
python -m quantlab.web.app

# 从新闻提取因子研究知识
python -m quantlab.knowledge.news_ingestor --keywords "量化因子,A股" --max 10

# 运行测试（跳过网络依赖）
python -m pytest tests/ -v --ignore=tests/test_scheduler.py --ignore=tests/test_decay_agent.py
```

### 5) Python API

```python
from quantlab.factor_discovery.multi_agent import (
    FactorMultiAgentOrchestrator, MultiAgentConfig,
)
from quantlab.factor_discovery.datahub import DataHub

hub = DataHub()
df = hub.load("data/cross_section_thick.csv")

orchestrator = FactorMultiAgentOrchestrator(
    config=MultiAgentConfig(
        enable_knowledge_injection=True,
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
| `ASSISTANT_REASONING_EFFORT` | 推理深度 (high/max) | — |

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

- [x] LLM 深度推理（thinking mode + 模型降级）
- [x] **LLM-first 默认路径**（移除经验门控，LLM 优先 + 模板自动回退）
- [x] Block 系统 v2（49 算子，含高阶矩/协方差/幂函数）
- [x] 多智能体协作（3 团队 6 Agent）
- [x] **学术知识注入**（12 研究方向 FactorKnowledgeBase）
- [x] **Agent 驱动后三阶段**（OOS 诊断 + 治理解读 + 交付报告叙事）
- [x] **Agent 反馈闭环**（OOS/治理分析 → 下一轮 EvolutionStage）
- [x] **统一 Rank IC 计算**（消除 7+ 处重复，MultiIndex 兼容）
- [x] **可组合调度器架构**（PipelineStage + PipelineContext）
- [x] 9 阶段日常调度器（含 Agent 驱动阶段）
- [x] **厚数据集**（280 资产 × 5.3 年 × 16 列，PE/PB/市值/换手率全填充）
- [x] **数据集断点续传构建**（checkpoint + 超时保护）
- [x] **AST 沙箱双重验证**（AST 级别 + 子字符串黑名单，19 测试覆盖）
- [x] 5 策略 30 算子假设生成
- [x] 经验学习回路 + 事前正交性引导
- [x] 市场状态检测 + regime 调整 IC
- [x] 幸存者偏差修正
- [x] 因子生命周期管理（7 状态流转）
- [x] 因子库治理（归档 + 拥挤度闭环 + Agent 解读）
- [x] 风控层（仓位 / 行业 / 回撤 / 流动性）
- [x] 冷启动种子因子引导（7 个学术因子）
- [x] 告警系统（分级 + Agent 增强）
- [x] 纸交易 + 实盘券商（华泰 + 聚宽）
- [x] 多品种覆盖（A 股 + 期货 + 可转债）
- [x] Web 管理面板（Flask 仪表盘）
- [x] 研报/新闻实时知识摄入（akshare+LLM）
- [x] GPU 加速因子搜索（CuPy + 自动 CPU 回退）
- [x] 多因子组合实盘模拟（LiveSimulator）
- [x] **LLM 输出监督**（LLMSupervisor + 自动重试回退）
- [x] **LLM 熔断保护**（10min 超时 + 慢响应自动降级）
- [x] **衰减→再发掘闭环**（定向再进化 + 版本链）
- [x] **23 基准因子对比**（Block DSL 表达 + 相关性矩阵）
- [x] **异常检测系统**（AnomalyGuard：数据/拆股/停牌）
- [x] **多重检验校正**（Benjamini-Hochberg FDR）
- [x] **管线断点恢复**（checkpoint + --resume）
- [x] **邮件推送**（QQ SMTP：告警 + 日度报告）
- [x] **因子库自动备份**（保留 30 份）
- [x] **136 测试通过**（17 Agent + 4 集成测试）

---

## License

MIT
