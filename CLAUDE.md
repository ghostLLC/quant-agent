# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**
- No features beyond what was asked.
- No abstractions for single-use code.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

---

## Common Commands

```bash
# Activate project venv (Python 3.13, project-isolated)
source .venv/Scripts/activate

# Run the daily factor factory pipeline (8 stages, LLM-first)
python -m quantlab.scheduler run_daily

# View recent execution records
python -m quantlab.scheduler status

# Install Windows scheduled task (daily 18:30)
python -m quantlab.scheduler install_cron

# Build/refresh the unified thick dataset (HS300, 2021+, 15 cols, incremental)
python build_dataset.py --full      # First time: full build
python refresh_data.py              # Incremental refresh (new dates only)

# Pull zz800 cross-section data (legacy)
python pull_zz800_cross_section.py --start 20240101 --assets 150

# Run backtest
python run_backtest.py --data data/hs300_etf.csv --strategy ma_cross

# Run test suite (136 tests); skip network-dependent scheduler/decay tests
D:\quant-agent\.venv\Scripts\python.exe -m pytest tests/ -v --ignore=tests/test_scheduler.py --ignore=tests/test_decay_agent.py

# Run a single test file
D:\quant-agent\.venv\Scripts\python.exe -m pytest tests/test_blocks.py -v

# Start web dashboard (port 8080)
D:\quant-agent\.venv\Scripts\python.exe -m quantlab.web.app

# Ingest news into knowledge base
D:\quant-agent\.venv\Scripts\python.exe -m quantlab.knowledge.news_ingestor --keywords "量化因子,A股" --max 10

# Lint
D:\quant-agent\.venv\Scripts\python.exe -m ruff check quantlab/
```

## Environment

- `.env` at project root provides `TUSHARE_TOKEN` and LLM keys (`ASSISTANT_API_KEY`, `ASSISTANT_BASE_URL`, `ASSISTANT_MODEL`)
- Token priority: `.env` → system env var → code argument
- Python 3.10+, key dependencies: `pandas>=2.2.0`, `numpy>=1.26.0`, `akshare>=1.18.55`, `flask>=3.0.0`
- GPU acceleration (optional): `pip install cupy-cuda12x` (NVIDIA GPU + CUDA 12.x required)
- Project venv: `D:\quant-agent\.venv`

## Architecture

This is an **AI-native quant factor factory** for A-share cross-sectional alpha factor discovery, modeled on WorldQuant's sell-factor business. It discovers, validates, and produces deliverable factor reports — it does NOT trade.

### Pipeline (9 stages)

```
Data Refresh → Decay Monitor → Evolution Search (LLM-first) → Agent OOS Validation
    → Factor Combination + Benchmark → Delivery Screening → Paper Trading
    → Agent Governance → Agent Delivery Report
```

Agent-driven stages (OOS/Governance/Report): numerical computation by rules, qualitative analysis by `AgentAnalyst` LLM. LLM unavailable → rule-based fallback, pipeline never blocks.
Agent feedback loop: OOS diagnosis → `ctx._meta["discovery_feedback"]` → next EvolutionStage R1 prompt injection.

### Top-level organization

**`quantlab/pipeline_stages/`** — Composable `PipelineStage` classes, one per pipeline phase. Each stage takes a `PipelineContext` and returns `dict[str, Any]`. Stages are independently testable. `DailyScheduler` (~250 lines in `scheduler.py`) orchestrates them.

**`quantlab/factor_discovery/`** — Factor discovery core:

- **`multi_agent.py`** — `FactorMultiAgentOrchestrator`: 3-team 6-agent LLM collaboration (R1 generation + R2 review → P1 architect + P2 block assembly + P3 custom code → T1 backtest + T2 validation). LLM is the DEFAULT path; template evolution is the fallback. R1 prompt includes:
  - Experience context (from `ExperienceLoop`)
  - Orthogonality constraints (from `OrthogonalityGuide`)
  - Research knowledge (from `FactorKnowledgeBase`, 12 academic directions)
  - Cross-run context memory (from `LLMClient.get_context_summary()`)
- **`blocks.py`** — Block system v2: serializable factor DSL. 5 block types (Data, Transform, Combine, Relational, Filter) with **49 operators** (27 transform, 7 combine, 3 relational, 2 filter, 5 group transform, 6 math/higher-moment). `Block.from_dict(d)` is the canonical deserialization. Operators include: power, sqrt, ts_skew, ts_kurt, scale, ts_cov (v2.1 additions).
- **`hypothesis.py`** — 5-strategy template hypothesis generation covering 30 DSL operators. Template is the fallback when LLM is unavailable. `_build_op_params()` handles dynamic parameter construction.
- **`evolution.py`** — `FactorEvolutionLoop`: multi-round evolution search with mutation/crossover.
- **`factor_enhancements.py`** — 9 enhancement modules: ExperienceLoop, RiskNeutralizer, FactorCombiner, ParameterSearcher, CustomCodeGenerator (AST-level sandbox + substring blacklist dual defense), OrthogonalityGuide, CrowdingDetector, RegimeDetector, FactorCurveAnalyzer.
- **`pipeline.py`** — `FactorDiscoveryOrchestrator`: single-factor closed loop with 11-dimension scoring.
- **`runtime.py`** — `SafeFactorExecutor` (FactorNode → Block conversion + execution) + `PersistentFactorStore` (JSON file-based factor library).
- **`models.py`** — `FactorSpec` (universal currency), `FactorLibraryEntry`, `FactorStatus` (7-state lifecycle: DRAFT → CANDIDATE → OBSERVE → PAPER → PILOT → LIVE → RETIRED), `FactorScorecard`.
- **`datahub.py`** — `DataHub` with `LocalCSVProvider`, `TushareProProvider`, `AkShareIncrementalProvider`.
- **`seed_factors.py`** — 7 academic seed factors. `bootstrap_seed_factors()` skips factors whose required data fields are missing from the market DataFrame.

**`quantlab/metrics/`** — Shared computation:
- `compute_rank_ic(factor_values, market_df) → dict` — Unified Rank IC, used by all 7+ pipeline paths. Handles both flat Series and MultiIndex (date, asset) factor values.
- `GpuAccelerator.compute_rank_ic_gpu()` — GPU-accelerated IC with automatic CPU fallback (threshold: >10k assets per cross-section). CuPy required for GPU path.
- `GpuAccelerator.batch_compute_ic(factor_panels, market_df)` — Batch multi-factor evaluation. **Shares fwd_ret pre-computation across all factors** — for 500 factors, ~3.1x speedup over sequential `compute_rank_ic()` calls.

**`quantlab/knowledge/`** — External research knowledge:
- `FactorKnowledgeBase.get_knowledge_context(direction) → str` — Returns markdown-formatted academic factor research knowledge for LLM prompt injection.
- `NewsIngestor` — Fetches A-share news via akshare, extracts factor research knowledge via LLM (rule-based fallback), deduplicates and persists to `assistant_data/news_knowledge.json`.

**`quantlab/trading/`** — Trading engine: cost model, portfolio construction, simulator, risk control, broker interface.
- `LiveSimulator.run_daily(date, factor_panels, market_df)` — Daily paper trading: multi-factor signal combination → OrderManager rebalance → NAV/persistence. Integration hook: `run_live_simulation(ctx, factor_panels)` for pipeline delivery stage.

**`quantlab/web/`** — Web dashboard (Flask):
- `GET /api/status` — factor count, pipeline runs, alerts
- `GET /api/factors` — factor library list with IC stats
- `GET /api/runs` — recent 20 run records
- `GET /api/alerts` — recent alerts sorted by severity
- `GET /` — HTML dashboard with auto-refresh

**`quantlab/strategies/`** — Strategy implementations (MA cross, channel breakout) with a registry pattern. Separate from the factor pipeline — these are traditional trading strategies for backtesting.
**`quantlab/analysis/`** — Grid search, history store, validation utilities for backtest optimization.

**New production modules (v2 hardening):**

- **`pipeline_stages/agent_analyst.py`** — `AgentAnalyst`: shared LLM analysis engine. 4 domains: `analyze_oos()` (per-factor diagnosis), `analyze_governance()` (holistic interpretation), `generate_narrative_report()` (buyer narrative), `generate_feedback()` (cross-round feedback). LLM unavailable → rule-based fallback.
- **`factor_discovery/llm_supervisor.py`** — `LLMSupervisor`: second LLM monitors first, corrects JSON errors with retry feedback. Max 2 retries before template fallback. Logs to `supervisor_log.json`.
- **`pipeline_stages/anomaly_guard.py`** — `AnomalyGuard`: data sanity (NaN, zero vol, price gaps, duplicates), corporate actions (split/dividend detection), suspension handling. Results in `ctx._meta["anomalies"]`.
- **`pipeline_stages/factor_monitor.py`** — `FactorMonitor`: continuous health tracking. `detect_ic_drift()` (60d rolling IC vs peak), `detect_crowding_trend()` (consecutive increase count), `monitor_rolling_sharpe()` (60d rolling Sharpe from NAV).
- **`factor_discovery/benchmark_factors.py`** — `BenchmarkFactorRegistry`: **44 academic/industry factors** in Block DSL across 10 categories (Value, Momentum, Size, Quality, LowVol, Liquidity, Reversal, Behavioral, EventDriven, A-Share specific). `evaluate_all()` + `compare_to_benchmarks()`.
- **`factor_discovery/factor_namer.py`** — `FactorNamer` (LLM semantic naming + timestamp) + `FactorVersionManager` (parent→child lineage, auto-increment versions).
- **`factor_discovery/real_return.py`** — `RealReturnEvaluator`: portfolio-level backtesting (monthly rebalance, A-share costs). `compare_to_ic()` validates IC→returns translation.
- **`metrics/fdr.py`** — `apply_fdr_correction()` (Benjamini-Hochberg), `screen_factors_with_fdr()` (IC→p-value→FDR).
- **`pipeline_stages/experiment_tracker.py`** — `ExperimentTracker`: full run provenance (factor→direction→agent→prompt→evaluation). Stored in `experiments/{run_id}.json`.
- **`assistant/email_notifier.py`** — `EmailNotifier`: QQ SMTP (smtp.qq.com:587). Critical alerts immediate, daily summary batched. Non-blocking (background thread).

### Critical Design Patterns

1. **FactorSpec is the universal currency** — Everything flows through `FactorSpec`. Serializes to/from JSON for persistence and agent communication.

2. **Block.from_dict() for deserialization** — Not `executor._deserialize_block()` (removed). Always use `Block.from_dict(block_tree_dict)`.

3. **`ts_code` → `asset` normalization at data boundary** — `load_cross_section_data()` normalizes `ts_code` (e.g. `000001.SZ`) to `asset` (e.g. `000001`). Downstream code always uses `asset`.

4. **LLM-first, template fallback** — Evolution stage tries LLM multi-agent per direction. If LLM is unavailable or fails, falls back to `FactorEvolutionLoop` (template). No experience gate.

5. **MultiIndex factor values** — `BlockExecutor.execute()` returns `pd.Series` with `MultiIndex(date, asset)`. `compute_rank_ic()` handles this via `_align_factor()`.

6. **JSON persistence, no database** — Factor library, experience registry, run records use JSON files under `assistant_data/` and `data/scheduler/`.

7. **Canonical import path** — `quantlab/factor_discovery/__init__.py` re-exports all public classes.

### Important Implementation Details

- **CustomCodeGenerator sandbox**: `_safety_check()` uses AST validation first (`_ast_validate()`), substring blacklist second. `_sandbox_execute()` runs with restricted `__builtins__` (must include `__import__` for pandas/numpy imports). `SAFE_BUILTINS` controls the whitelist. Code must define `def compute_factor(df):`.
- **Seed factor bootstrap**: `bootstrap_seed_factors()` requires `FactorLibraryEntry` with `latest_report` (minimal `FactorEvaluationReport` with empty `FactorScorecard`) and `retention_reason`. It calls `store.upsert_library_entry(entry, factor_panel=...)` to persist.
- **PipelineContext._meta**: Used to pass deliverable factor IDs between stages (e.g., `ctx._meta = {"deliverable_factor_ids": ids}`).
- **Config path**: `DEFAULT_DATA_PATH` points to `data/cross_section_thick.csv` (280 HS300 assets × 1,288 trading days × 16 columns, 2021-2026).
- **Data freshness**: `build_dataset.py --refresh` skips tushare pull if last data date ≤ 2 trading days behind.
- **Pipeline recovery**: `python -m quantlab.scheduler run_daily --resume` skips completed stages via `pipeline_checkpoint.json`. Checkpoint saved after each stage, cleared on success.
- **Circuit breaker**: LLMClient tracks last 5 call latencies. 3+ calls >30s → auto-degrade to template mode, 1h cooldown.
- **Experience systems**: Two separate stores. `ExperienceLoop` → `quantlab/assistant_data/experience_loop/outcomes.json` (LLM multi-agent). `PersistentFactorStore` → `assistant_data/memory/factor_discovery/experience_registry.json` (template evolution). Both must be written for full coverage.
- **Factor naming**: `FactorNamer` produces `{family}_{direction}_{ops}_{ic}_{timestamp}` via LLM (fallback: ops/windows extraction). `FactorVersionManager` tracks parent→child lineage in `factor_versions.json`.
- **Benchmark factors**: 44 known factors in Block DSL across 10 categories (including A-share specific). `BenchmarkFactorRegistry.compare_to_benchmarks()` for correlation analysis.
- **Data paths**: `DEFAULT_THICK_DATA_PATH` = `cross_section_thick.csv`. `DEFAULT_DATA_PATH` ⊥ `DEFAULT_THICK_DATA_PATH`.
