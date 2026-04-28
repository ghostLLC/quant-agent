# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

Following file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# Run the daily factor factory pipeline (5 stages: refresh → decay monitor → evolution → screening → report)
python -m quantlab.scheduler run_daily

# Run a specific module
python -m quantlab.scheduler status           # View recent execution records
python -m quantlab.scheduler install_cron     # Install Windows scheduled task (daily 18:30)

# Refresh data
python refresh_data.py                        # Cross-section data refresh
python fetch_hs300_etf.py                     # ETF data fetch

# Run backtest
python run_backtest.py --data data/hs300_etf.csv --strategy ma_cross

# Run test suite (32 tests, pytest)
python -m pytest tests/ -v

# Verification scripts (smoke tests for data + multi-agent pipeline)
python verify_multi_agent.py                  # Multi-agent framework end-to-end
python verify_zz800.py                        # zz800 data + Block system + IC check
```

## Environment

- `.env` at project root provides `TUSHARE_TOKEN` (Tushare Pro data) and optional LLM keys (`ASSISTANT_BASE_URL`, `ASSISTANT_API_KEY`, `ASSISTANT_MODEL`)
- Token priority: `.env` → system env var → code argument
- Python 3.10+, dependencies: `pandas>=2.2.0`, `numpy>=1.26.0`, `akshare>=1.18.55` (`tushare` is optional)

## Architecture Overview

This is a **quant factor factory** targeting A-share cross-sectional alpha factor discovery, modeled on the WorldQuant sell-factor business. The system does NOT trade — it discovers, validates, and produces deliverable factor reports.

### Core Pipeline (7 stages)

```
Hypothesis Generation → Safe Execution → Cross-sectional Evaluation
    → Evolution Search → Decay Monitoring → Delivery Screening → Report Output
```

### Key architectural layers

**Factor Discovery Core** (`quantlab/factor_discovery/`) — The heart of the system:

- **`pipeline.py`** — `FactorDiscoveryOrchestrator`: the single-factor closed loop. Builds a 6-stage research plan (spec review → sandbox guard → compute → evaluate → validate → screen), evaluates with 11-dimension scoring (IC, ICIR, monotonicity, turnover, coverage, decay, exposure, stability, tradability, novelty, complexity), and decides approved/observe/rejected.
- **`evolution.py`** — `FactorEvolutionLoop`: multi-round evolution search. Round 0 generates fresh hypotheses; subsequent rounds apply mutation/crossover to top-performing factors from the library (QuantaAlpha trajectory-level evolution). Early-stop when no improvement for N consecutive rounds.
- **`blocks.py`** — Block system v2: a complete serializable factor DSL. 5 block types (Data, Transform, Combine, Relational, Filter) with 100+ registered operators. `BlockExecutor` recursively evaluates block trees against market DataFrames. This replaces raw Python expression evaluation with structured, auditable computation.
- **`runtime.py`** — `SafeFactorExecutor` (AST whitelist enforcement, 18 operators, depth/window limits) + `PersistentFactorStore` (JSON file-based factor library) + `FactorExperienceMemory` (experience pattern tracking).
- **`hypothesis.py`** — `FactorHypothesisGenerator`: template-based candidate generation using 3 strategies (single-operator, cross-feature, volatility-adjusted) with family-aware diversity penalties.
- **`multi_agent.py`** — `FactorMultiAgentOrchestrator`: 3-team, 6-agent collaboration (R1 generation + R2 review → P1 architect + P2 block assembly + P3 custom code → T1 backtest + T2 validation) with a `MessageBus` and `LLMClient` for LLM-driven deep reasoning.
- **`factor_enhancements.py`** — 6 enhancement modules: ExperienceLoop, RiskNeutralizer, FactorCombiner, ParameterSearcher, CustomCodeGenerator (LLM code gen + sandbox), OrthogonalityGuide.

**Delivery & Monitoring** (`decay_monitor.py`, `delivery_screener.py`, `factor_report.py`):
- Decay monitor checks 4 criteria (IC decay >50%, |IC| <0.015, ICIR <0.15, IC positive ratio <50%) and triggers re-discovery.
- Delivery screener enforces 8 WorldQuant standards before a factor is considered sellable.
- Report generator produces JSON + Markdown with full buy-side due diligence coverage.

**Trading Engine** (`quantlab/trading/`):
- A-share cost model: commission 0.03% (min ¥5), stamp tax 0.1% (sell only), slippage 0.02%, impact cost based on participation rate.
- Portfolio constructor supports 4 weight schemes (equal, score, IC-weighted, sector-neutral).
- Simulator tracks daily turnover, cost, net/gross returns, drawdown, Sharpe, IR.

**Data Layer** (`quantlab/data/`, `quantlab/factor_discovery/datahub.py`):
- `DataHub` provides unified access with provider abstraction: `LocalCSVProvider`, `TushareProProvider` (batch daily API, 5000+ rows in <1s), `AkShareIncrementalProvider` (incremental refresh).
- Default data path is `zz800_cross_section.csv` (see `config.py` `DEFAULT_DATA_PATH`).

**Agent Runtime** (`quantlab/assistant/tools.py`):
- `AssistantToolRuntime` registers 21 tools covering the full factor lifecycle.
- `ResearchTaskExecutor` (`quantlab/research/executor.py`) routes 14 task types: single backtest, grid search, train-test validation, walk-forward, multi-strategy compare, portfolio review, factor discovery, hypothesis generation, factor evolution, multi-agent discovery, data refresh, experiment history/detail.

**Scheduler** (`quantlab/scheduler.py`):
- `DailyScheduler` runs the 5-stage pipeline: incremental refresh → decay monitor → evolution search (per direction) → delivery screening → report generation.
- Supports Windows Task Scheduler integration via `schtasks`.

### Important Design Patterns

1. **FactorSpec is the universal currency** — Everything flows through `FactorSpec` (factor_id, expression_tree, dependencies, preprocess config, label spec, universe spec, execution policy). It serializes to/from JSON for persistence and agent communication.

2. **Cross-section only, no proxy** — The system explicitly rejects single-asset proxy data for factor discovery. `_load_factor_market_frame` in `executor.py` checks for `{date, close, volume}` + (`asset` or `ts_code`) columns and raises if not genuine cross-section data (`allow_proxy=False`).

3. **Unified expression system via FactorNode → Block conversion** — `SafeFactorExecutor.execute()` converts `FactorNode` trees to `Block` trees via `factor_node_to_block()`, then executes through `BlockExecutor`. This unifies the two expression systems: the older `FactorNode` (used by hypothesis/evolution/pipeline) and the richer `Block` system (100+ operators). Conversion is bidirectional (`block_to_factor_node()` also available). Both are JSON-serializable.

4. **Persistent JSON storage** — Factor library, experience registry, run records, and scheduler logs all use JSON files under `assistant_data/` and `data/scheduler/`. No database dependency.

5. **Template → LLM fallback** — Most generators try LLM first (when configured) and fall back to template-based generation when LLM is unavailable or fails.

6. **`ts_code` → `asset` normalization at data boundary** — `load_cross_section_data()` and all `DataProvider.load_cross_section()` implementations normalize `ts_code` (e.g. `000001.SZ`) to `asset` (e.g. `000001`) at load time. Downstream code should always use `asset`.

### File Organization Notes

- `quantlab/factor_discovery/__init__.py` re-exports all public classes — it's the canonical import path.
- `quantlab/config.py` defines `BacktestConfig` (dataclass with 20+ fields) and global path constants.
- `quantlab/pipeline.py` is the main research pipeline entry, wrapping all backtest/validation/refresh functions.
- `run_backtest.py`, `refresh_data.py`, `fetch_hs300_etf.py` are top-level entry scripts.
- Uncommitted scripts (`*.py` in .gitignore as `test_*.py`) are verification tools, not formal tests.
