# Quant Agent — AI 驱动的量化因子发掘系统

基于 LLM Agent 的量化因子自主发掘系统，聚焦沪深300横截面因子研究。系统融合了 R&D-Agent-Quant、AlphaAgent、QuantaAlpha、FactorMiner、Hubble 等开源框架的核心设计，实现了因子假设自动生成、多轮进化搜索、统一数据管理和研究流程闭环。

## 核心特性

- **因子假设自动生成**：基于族感知多样性 + 正则化探索 + 经验记忆引导，自动生成候选因子表达式树
- **多轮进化搜索**：假设→执行→评估→变异/交叉→再假设的自主闭环，支持早停与轨迹追踪
- **安全因子执行**：白名单算子逐节点递归求值 + 预处理 + 行业中性化，确保因子面板可计算
- **横截面评估体系**：Rank IC / ICIR / 分位单调性 / 稳定性 / 可交易性 / 综合评分
- **因子库管理**：持久化存储 + 库内相关性比较 + 经验记忆沉淀
- **统一数据层**：DataHub 抽象 + Provider 可扩展 + 质量报告 + 缓存复用
- **策略回测与验证**：单次回测 / 参数扫描 / 训练-测试验证 / Walk-forward 验证
- **决策可信度闸门**：计划评估→执行分流→自动重规划→结论验收

## 系统架构

```text
┌─────────────────────────────────────────────────┐
│               AssistantToolRuntime              │
│          （14 个工具的统一运行时入口）              │
├──────────┬──────────┬──────────┬────────────────┤
│ 策略研究  │ 因子发掘  │ 进化搜索  │   数据管理     │
│  6 tools │ 1 tool   │ 2 tools  │   2 tools     │
├──────────┴──────────┴──────────┴────────────────┤
│              Research Task Executor              │
├─────────────────────────────────────────────────┤
│  FactorDiscovery   │  FactorEvolution  │ DataHub │
│  Orchestrator      │  Loop             │         │
├────────────────────┼───────────────────┼─────────┤
│  HypothesisGenerator│ EvolutionStrategy│ Provider│
├────────────────────┴───────────────────┴─────────┤
│  SafeFactorExecutor │ PersistentFactorStore       │
│  FactorExperienceMemory │ FactorSpec/Models       │
└─────────────────────────────────────────────────┘
```

## 目录结构

```text
quant-agent/
├── quantlab/
│   ├── factor_discovery/          # 因子发掘核心模块
│   │   ├── models.py              # 因子数据模型（FactorSpec / FactorNode / FactorLabelSpec）
│   │   ├── hypothesis.py          # 因子假设生成器（族感知 + DSL约束 + Ralph Loop）
│   │   ├── evolution.py           # 自主搜索循环（mutation/crossover + 轨迹进化）
│   │   ├── pipeline.py            # 因子评估管线（横截面评估 + 评分卡 + 闭环编排）
│   │   ├── runtime.py             # 安全执行器 + 因子库 + 经验记忆
│   │   └── datahub.py             # DataHub 统一数据抽象层
│   ├── assistant/                 # AI Agent 运行时
│   │   ├── tools.py               # AssistantToolRuntime（14 工具）
│   │   ├── evaluator.py           # 决策可信度评估
│   │   ├── planner.py             # 研究计划生成
│   │   ├── llm.py                 # LLM 调用封装
│   │   ├── memory.py              # 会话记忆管理
│   │   └── knowledge_base.py      # 知识库检索
│   ├── research/                  # 研究任务执行框架
│   │   ├── executor.py            # ResearchTaskExecutor
│   │   ├── models.py              # 研究任务模型
│   │   └── protocol.py            # 研究协议定义
│   ├── strategies/                # 策略实现
│   │   ├── base.py                # 策略基类
│   │   ├── ma_cross.py            # 双均线交叉
│   │   └── registry.py            # 策略注册表
│   ├── analysis/                  # 分析工具
│   │   ├── grid_search.py         # 参数网格搜索
│   │   ├── validation.py          # 验证框架
│   │   └── history_store.py       # 实验历史
│   ├── data/                      # 数据获取层
│   │   ├── fetcher.py             # 数据抓取（akshare + 雪球兜底）
│   │   └── loader.py              # 数据加载
│   ├── config.py                  # 全局配置
│   └── pipeline.py                # 研究管线入口
├── data/                          # 数据文件
│   ├── hs300_cross_section.csv    # 沪深300横截面行情
│   ├── hs300_cross_section_asset_metadata.csv  # 元数据缓存
│   ├── hs300_cross_section_refresh_report.json # 刷新报告
│   └── hs300_etf.csv              # 沪深300ETF行情
├── assistant_data/                # Agent 持久化数据
│   ├── knowledge/                 # 知识库文档
│   └── memory/                    # 因子发掘运行记录
│       └── factor_discovery_runs/ # 因子历史 JSON
├── reports/                       # 研究报告输出
├── strategies/                    # 策略参数配置
├── fetch_hs300_etf.py             # 数据抓取脚本
├── refresh_data.py                # 数据刷新脚本
├── run_backtest.py                # 回测入口脚本
└── requirements.txt               # Python 依赖
```

## 工具清单

`AssistantToolRuntime` 当前注册 14 个工具：

| 工具名 | 说明 |
|--------|------|
| `view_current_config` | 查看当前配置与数据路径 |
| `list_strategies` | 查看可用策略列表 |
| `run_single_backtest` | 运行单次回测 |
| `run_grid_experiment` | 运行参数网格搜索 |
| `run_train_test_validation` | 训练-测试验证 |
| `run_walk_forward_validation` | Walk-forward 滚动验证 |
| `run_multi_strategy_compare` | 多策略横向比较 |
| `review_portfolio_construction` | 组合构建评审 |
| `run_factor_discovery` | 因子发掘闭环（假设→执行→评估→入库） |
| `refresh_cross_section_data` | 刷新横截面行情数据 |
| `generate_factor_hypotheses` | 生成因子假设候选 |
| `run_factor_evolution` | 运行多轮进化搜索循环 |
| `list_experiment_history` | 查看实验历史 |
| `get_experiment_detail` | 查看实验详情 |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：`pandas >= 2.2.0`、`numpy >= 1.26.0`、`akshare >= 1.18.55`

### 数据准备

```bash
# 刷新沪深300横截面数据（多资产行情 + 元数据）
python refresh_data.py

# 或只刷新 ETF 单资产行情
python fetch_hs300_etf.py
```

### 运行回测

```bash
python run_backtest.py --data "data/hs300_etf.csv" --strategy ma_cross
```

### 编程方式调用

```python
from pathlib import Path
from quantlab.config import BacktestConfig
from quantlab.assistant.tools import AssistantToolRuntime

rt = AssistantToolRuntime(BacktestConfig(), Path("data/hs300_cross_section.csv"))

# 生成因子假设
result = rt.execute("generate_factor_hypotheses", {
    "research_direction": "量价背离",
    "max_candidates": 5,
})

# 运行进化搜索
result = rt.execute("run_factor_evolution", {
    "direction": "波动率调整动量",
    "max_rounds": 3,
    "candidates_per_round": 5,
})

# 单次因子发掘
result = rt.execute("run_factor_discovery", {
    "factor_prompt": "量价背离",
})
```

## 因子发掘架构详解

### 因子假设生成器（hypothesis.py）

融合 AlphaAgent 正则化探索 + Hubble DSL 约束 + FactorMiner Ralph Loop：

- **6 个因子族**：momentum / reversal / volatility / volume_price / liquidity / fundamental
- **12 个 DSL 算子**：rank / zscore / delta / lag / mean / std / ts_rank / add / sub / mul / div / min / max / clip
- **3 种模板策略**：基础时序算子组合、双特征交叉组合、波动率调整组合
- **Retrieve-Adapt 循环**：从经验记忆和因子库提取相关模式，引导假设生成
- **族感知多样性**：对过度集中的族施加惩罚，保证搜索覆盖面
- **正则化探索**：对探索不足的方向给予奖励，避免同质化

### 自主搜索循环（evolution.py）

融合 QuantaAlpha 轨迹进化 + FactorMiner Ralph Loop + R&D-Agent-Quant 两阶段迭代：

- **轨迹级进化**：mutation（算子替换 / 窗口调整 / 子树交换）+ crossover（子树交叉）
- **多轮闭环**：假设→执行→评估→进化→再假设
- **早停机制**：连续 N 轮无改善自动终止
- **经验自动记录**：成功模式 / 失败约束 / 观察结果持久化

### 统一数据层（datahub.py）

借鉴 Qlib DataProvider 抽象：

- **Provider 模式**：`LocalCSVProvider`（当前）→ 可扩展 `AkshareProvider` / `TushareProvider` / `WindProvider`
- **数据质量报告**：资产数 / 行业覆盖率 / 市值覆盖率 / NaN 比率 / 最近刷新时间
- **缓存复用**：加载后缓存，刷新后自动失效

### 横截面评估体系（pipeline.py）

- **Rank IC / ICIR**：因子排序预测能力
- **分位单调性**：多空组合收益单调性
- **稳定性评分**：时间序列 IC 一致性
- **可交易性评分**：换手率与冲击成本考量
- **综合评分**：加权合成，决策阈值 approved(>0.55) / observe(>0.25) / rejected

### 数据获取层（data/fetcher.py）

- **主来源**：akshare（`stock_individual_info_em`）
- **兜底来源**：雪球个股资料（`stock_individual_basic_info_xq`）
- **元数据缓存**：行业 / 简称 / 市值 / 股本自动补齐
- **unknown 行业智能刷新**：缓存中行业仍为 unknown 的资产会自动重新抓取

## AI Agent 配置

支持两层配置来源（优先级从低到高）：

1. 项目根目录 `.env`
2. 系统环境变量

```env
ASSISTANT_BASE_URL=https://your-api-endpoint/v1
ASSISTANT_API_KEY=your_api_key
ASSISTANT_MODEL=gpt-5.4
```

## 决策可信度机制

研究计划在执行前会经过可信度评估，决定后续执行路径：

- **pass**：正常执行原计划
- **review_required**：降级为保守验证计划
- **fail**：触发自动重规划，补齐缺失研究链路

评估维度：计划可信度 / 执行可信度 / 结论可信度，每个维度都有验收阈值和闸门状态。

## 参考框架

本项目架构设计借鉴了以下开源框架的核心思路：

| 框架 | 借鉴点 |
|------|--------|
| [R&D-Agent-Quant](https://github.com/microsoft/RD-Agent) | Research→Development 两阶段迭代 |
| [AlphaAgent](https://arxiv.org/abs/2502.16789) | LLM + 正则化探索 + 抗衰减评估 |
| [QuantaAlpha](https://arxiv.org/abs/2602.07085) | 轨迹级 mutation/crossover 进化 |
| [FactorMiner](https://arxiv.org/abs/2602.14670) | Ralph Loop (Retrieve→Adapt→Learn→Plan→Harvest) |
| [Hubble](https://arxiv.org/abs/2604.09601) | DSL 约束生成 + AST 验证 + 族感知多样性 |

## License

MIT
