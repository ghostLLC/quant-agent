"""A股新闻实时摄入模块 —— 从财经新闻中提取量化因子研究知识。

将财经新闻转化为结构化的因子研究知识条目，支持 LLM 智能提取和规则引擎两种模式。
提取的知识可注入已有知识库，供 LLM 假设生成使用。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── 默认数据目录 ──────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path(r"D:\quant-agent\assistant_data")
DEFAULT_NEWS_KNOWLEDGE_FILE = "news_knowledge.json"

# ── 规则引擎关键词映射 ─────────────────────────────────────────────

_KEYWORD_RULES: list[tuple[list[str], str, str, str, str]] = [
    (
        ["资金流", "北向", "外资"],
        "capital_flow",
        "资金流向",
        "新闻监测到资金流相关动态，北向/外资动向可能反映机构投资者观点和资金配置变化。",
        "北向资金被称为聪明钱，其净买入方向有显著的收益预测能力；机构大额资金流入流出反映专业投资者观点。",
    ),
    (
        ["波动", "震荡", "恐慌"],
        "volatility_regime",
        "波动率与市场状态",
        "新闻提及市场波动、震荡或恐慌情绪，波动率状态转换可能影响因子表现。",
        "低波动率异象——低波动率股票风险调整后收益更高；波动率聚类性和状态转换时的因子轮动机会。",
    ),
    (
        ["业绩", "盈利", "利润"],
        "quality_earnings",
        "质量与盈利",
        "新闻涉及公司业绩、盈利能力或利润变化，盈利质量是截面收益的重要预测变量。",
        "高盈利、低应计、稳定增长的公司应获更高估值；盈利超预期后股价持续漂移（PEAD）。",
    ),
    (
        ["涨", "跌", "反弹", "趋势"],
        "momentum_reversal",
        "动量与反转",
        "新闻描述市场涨跌、反弹或趋势变化，价格趋势延续或反转是量化因子的核心方向。",
        "过去表现优异的股票倾向于继续优异（动量），但极短期和极长期会出现反转。",
    ),
    (
        ["情绪", "信心", "散户"],
        "sentiment_behavioral",
        "情绪与行为金融",
        "新闻反映投资者情绪、市场信心或散户行为，情绪驱动的错误定价是 Alpha 来源之一。",
        "投资者情绪影响资产定价——高情绪期投机性股票被推高随后反转；A 股个人投资者占比高，情绪因子特别有效。",
    ),
    (
        ["估值", "市盈", "市净", "便宜"],
        "value_contrarian",
        "价值与逆向",
        "新闻涉及估值讨论、市盈率/市净率分析或价值判断，价值因子是最稳健的因子之一。",
        "便宜的股票（低市盈率、低市净率）长期跑赢贵的股票；需要结合质量过滤避免价值陷阱。",
    ),
    (
        ["流动", "换手", "成交"],
        "liquidity_premium",
        "流动性溢价",
        "新闻提及流动性、换手率或成交量异常，流动性是影响收益的重要维度。",
        "流动性差的股票需要提供更高的期望收益来补偿交易成本；换手率因子在 A 股表现尤为显著。",
    ),
]

# ── LLM 提取 prompt ──────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = (
    "你是一个专业的量化因子研究分析师。你的任务是从财经新闻中提取与量化因子研究相关的信息。"
    "你需要识别新闻是否包含因子研究价值，并输出结构化的 JSON。"
    "不要编造信息，只提取新闻中实际存在的内容。"
)

_LLM_EXTRACT_PROMPT_TEMPLATE = """以下是一则A股市场新闻，请判断是否包含量化因子研究的相关信息。如果是，提取：
1. 研究方向 (direction, 英文snake_case)
2. 研究方向名称 (name, 中文)
3. 核心直觉 (intuition)
4. 潜在Alpha来源 (alpha_sources)

如果不相关，返回 'NOT_RELEVANT'。

请仅输出 JSON 格式，不要包含其他文字：
{"direction": "...", "name": "...", "intuition": "...", "alpha_sources": "..."}

新闻标题: {title}
新闻内容: {content}
来源: {source}
发布时间: {publish_time}
"""


class NewsIngestor:
    """A股新闻实时摄入器。

    从 akshare 获取 A 股相关新闻，通过 LLM 或规则引擎提取因子研究知识，
    持久化到 JSON 文件并可供现有知识库使用。

    Attributes:
        data_dir: 数据存储目录。
        llm: 可选的 LLM 客户端，用于智能提取。
        news_file: 新闻知识 JSON 文件路径。
    """

    def __init__(
        self,
        llm_client: Any = None,
        data_dir: Path | None = None,
        news_filename: str = DEFAULT_NEWS_KNOWLEDGE_FILE,
    ) -> None:
        """初始化新闻摄入器。

        Args:
            llm_client: 可选的 LLM 客户端实例，需提供 chat(system_prompt, user_prompt) -> str 接口。
                        若为 None，则使用关键词规则引擎。
            data_dir: 数据存储目录，默认为 assistant_data。
        """
        self.llm = llm_client
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.news_file = self.data_dir / news_filename

    # ── 新闻获取 ──────────────────────────────────────────────────

    def fetch_news(
        self,
        keywords: list[str] | None = None,
        max_items: int = 20,
    ) -> list[dict]:
        """获取 A 股相关新闻。

        优先使用 akshare.stock_news_em() 获取东方财富新闻，
        失败时回退到 akshare.stock_info_global_em()。

        Args:
            keywords: 可选的关键词列表，用于过滤新闻。
            max_items: 最大返回新闻数量。

        Returns:
            新闻条目列表，每条包含 title, content, source, publish_time, url。
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("akshare 未安装，无法获取新闻")
            return []

        news_list: list[dict] = []

        # 优先尝试东方财富新闻
        try:
            df = ak.stock_news_em()
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    if len(news_list) >= max_items:
                        break
                    title = str(row.get("标题", row.get("title", "")))
                    if keywords:
                        if not any(kw in title for kw in keywords):
                            continue
                    news_list.append({
                        "title": title,
                        "content": str(row.get("内容", row.get("content", ""))),
                        "source": str(row.get("来源", row.get("source", "东方财富"))),
                        "publish_time": str(row.get("发布时间", row.get("publish_time", ""))),
                        "url": str(row.get("链接", row.get("url", ""))),
                    })
                if news_list:
                    logger.info("从东方财富获取 %d 条新闻", len(news_list))
                    return news_list
        except Exception as exc:
            logger.debug("stock_news_em() 失败: %s，尝试回退", exc)

        # 回退：全球财经新闻
        try:
            df = ak.stock_info_global_em()
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    if len(news_list) >= max_items:
                        break
                    title = str(row.get("标题", row.get("title", "")))
                    if keywords:
                        if not any(kw in title for kw in keywords):
                            continue
                    news_list.append({
                        "title": title,
                        "content": str(row.get("内容", row.get("content", ""))),
                        "source": str(row.get("来源", row.get("source", "全球财经"))),
                        "publish_time": str(row.get("发布时间", row.get("publish_time", ""))),
                        "url": str(row.get("链接", row.get("url", ""))),
                    })
                if news_list:
                    logger.info("从全球财经获取 %d 条新闻", len(news_list))
                    return news_list
        except Exception as exc:
            logger.debug("stock_info_global_em() 失败: %s", exc)

        if not news_list:
            logger.warning("未能获取任何新闻")
        return news_list

    # ── 知识摄入 ──────────────────────────────────────────────────

    def ingest(self, news_items: list[dict]) -> list[dict]:
        """处理新闻条目，提取因子研究知识。

        1. 对每条新闻提取因子研究方向、直觉和 Alpha 来源
        2. LLM 优先，不可用时使用关键词规则引擎
        3. 去重：与已有知识条目的 direction 字段比对
        4. 持久化到 JSON 文件
        5. 返回本次新增的知识条目

        Args:
            news_items: fetch_news() 返回的新闻条目列表。

        Returns:
            本次新增的知识条目列表。
        """
        if not news_items:
            return []

        # 加载已有知识
        existing_entries = self._load_existing_entries()
        existing_directions = {e.get("direction", "") for e in existing_entries}

        new_entries: list[dict[str, str]] = []

        for item in news_items:
            extracted = self._extract_from_news(item)
            if extracted is None:
                continue

            direction = extracted.get("direction", "")
            if not direction or direction in existing_directions:
                continue

            existing_directions.add(direction)
            new_entries.append(extracted)

            logger.info("新增因子知识: direction=%s, name=%s", direction, extracted.get("name", ""))

        if new_entries:
            existing_entries.extend(new_entries)
            self._save_entries(existing_entries)
            logger.info("新增 %d 条因子研究知识，总计 %d 条", len(new_entries), len(existing_entries))

        return new_entries

    def _extract_from_news(self, item: dict) -> dict[str, str] | None:
        """从单条新闻中提取因子知识。LLM 优先，规则引擎回退。"""
        if self.llm is not None:
            result = self._extract_with_llm(item)
            if result is not None:
                return result

        return self._extract_with_rules(item)

    def _extract_with_llm(self, item: dict) -> dict[str, str] | None:
        """使用 LLM 从新闻中提取因子知识。"""
        try:
            user_prompt = _LLM_EXTRACT_PROMPT_TEMPLATE.format(
                title=item.get("title", ""),
                content=item.get("content", "")[:1000],
                source=item.get("source", ""),
                publish_time=item.get("publish_time", ""),
            )
            response = self.llm.chat(_LLM_SYSTEM_PROMPT, user_prompt)
            return self._parse_llm_response(response)
        except Exception as exc:
            logger.debug("LLM 提取失败: %s，回退到规则引擎", exc)
            return None

    def _parse_llm_response(self, response: str) -> dict[str, str] | None:
        """解析 LLM 响应，提取 JSON 或判断不相关。"""
        text = response.strip()

        if "NOT_RELEVANT" in text:
            return None

        # 提取 JSON
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("无法解析 LLM 响应为 JSON: %s", text[:200])
            return None

        direction = data.get("direction", "")
        name = data.get("name", "")
        intuition = data.get("intuition", "")
        alpha_sources = data.get("alpha_sources", "")

        if not direction or not name:
            return None

        return {
            "direction": str(direction).strip().lower().replace(" ", "_"),
            "name": str(name).strip(),
            "intuition": str(intuition).strip(),
            "alpha_sources": str(alpha_sources).strip(),
            "source_news": item.get("title", ""),
        }

    def _extract_with_rules(self, item: dict) -> dict[str, str] | None:
        """使用关键词规则引擎从新闻中提取因子知识。"""
        title = item.get("title", "")
        content = item.get("content", "")[:500]
        combined = f"{title} {content}"

        for keywords, direction, name, intuition, alpha_sources in _KEYWORD_RULES:
            if any(kw in combined for kw in keywords):
                return {
                    "direction": direction,
                    "name": name,
                    "intuition": intuition,
                    "alpha_sources": alpha_sources,
                    "source_news": item.get("title", ""),
                }

        return None

    # ── 便捷方法 ──────────────────────────────────────────────────

    def run(self, keywords: list[str] | None = None, max_items: int = 20) -> dict[str, Any]:
        """便捷方法：获取新闻并摄入，返回摘要。

        Args:
            keywords: 可选的关键词过滤。
            max_items: 最大获取新闻数。

        Returns:
            包含 fetch_count, new_entries_count, entries 的摘要字典。
        """
        news_items = self.fetch_news(keywords=keywords, max_items=max_items)
        new_entries = self.ingest(news_items)
        return {
            "fetch_count": len(news_items),
            "new_entries_count": len(new_entries),
            "entries": new_entries,
        }

    # ── 持久化 ────────────────────────────────────────────────────

    def _load_existing_entries(self) -> list[dict[str, str]]:
        """从 JSON 文件加载已有的新闻知识条目。"""
        if not self.news_file.exists():
            return []
        try:
            data = json.loads(self.news_file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
            return []
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("加载新闻知识文件失败: %s", exc)
            return []

    def _save_entries(self, entries: list[dict[str, str]]) -> None:
        """保存新闻知识条目到 JSON 文件。"""
        try:
            self.news_file.write_text(
                json.dumps(entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("保存新闻知识文件失败: %s", exc)

    @property
    def entry_count(self) -> int:
        """返回当前持久化的知识条目数量。"""
        return len(self._load_existing_entries())


# ── CLI ───────────────────────────────────────────────────────────

def main() -> None:
    """CLI 入口：python -m quantlab.knowledge.news_ingestor"""
    parser = argparse.ArgumentParser(
        description="A股新闻实时摄入 —— 从财经新闻提取量化因子研究知识",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        default="量化因子,A股,alpha",
        help="新闻过滤关键词，逗号分隔（默认: 量化因子,A股,alpha）",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=10,
        dest="max_items",
        help="最大获取新闻数（默认: 10）",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="强制使用规则引擎，不使用 LLM",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="数据存储目录（默认: assistant_data）",
    )

    args = parser.parse_args()

    keywords: list[str] = [kw.strip() for kw in args.keywords.split(",") if kw.strip()]

    # 构建 LLM 客户端（可选）
    llm_client = None
    if not args.no_llm:
        try:
            from quantlab.factor_discovery.multi_agent import LLMClient
            llm_client = LLMClient()
            logger.info("LLM 客户端已初始化，将使用智能提取模式")
        except Exception as exc:
            logger.warning("LLM 客户端初始化失败（%s），将使用规则引擎", exc)

    ingestor = NewsIngestor(llm_client=llm_client, data_dir=args.data_dir)
    result = ingestor.run(keywords=keywords, max_items=args.max_items)

    print(f"\n获取新闻: {result['fetch_count']} 条")
    print(f"新增因子知识: {result['new_entries_count']} 条")
    for entry in result["entries"]:
        print(f"  - [{entry['direction']}] {entry['name']}: {entry['intuition'][:80]}...")
    print(f"\n知识文件: {ingestor.news_file}")


if __name__ == "__main__":
    main()
