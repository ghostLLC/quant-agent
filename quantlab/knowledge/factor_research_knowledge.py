"""量化因子研究知识库 —— 注入学术因子发现方向的外部知识。

每条知识包含：研究方向、核心直觉、经典做法、潜在 alpha 来源。
LLM 假设生成时将这些知识拼接进 prompt，让 AI 不只是枚举模板。
"""

from __future__ import annotations

from typing import Any

# ── 研究方向知识条目 ──────────────────────────────────────────────

FACTOR_RESEARCH_KNOWLEDGE: list[dict[str, str]] = [
    {
        "direction": "momentum_reversal",
        "name": "动量与反转",
        "intuition": "过去表现优异的股票倾向于继续优异（动量），但在极短期和极长期会出现反转。Jegadeesh & Titman (1993) 发现 3-12 个月的动量效应，而 De Bondt & Thaler (1985) 发现 3-5 年的长期反转。",
        "classic_approaches": "12-1 个月动量、5 日反转、52 周高点接近度、截面动量、残差动量（对 Fama-French 因子回归取残差后的动量）",
        "alpha_sources": "投资者反应不足（信息缓慢扩散）、处置效应（过早卖出赢家）、机构羊群效应。在流动性枯竭期动量会崩溃（momentum crash），需要考虑波动率缩放。",
        "key_papers": "Jegadeesh & Titman (1993); Moskowitz, Ooi & Pedersen (2012) 时间序列动量; Blitz, Huij & Martens (2011) 残差动量",
    },
    {
        "direction": "quality_earnings",
        "name": "质量与盈利",
        "intuition": "高盈利、低应计、稳定增长的公司应该获得更高估值，但市场经常低估质量溢价。Novy-Marx (2013) 发现毛利率比净利润更能预测截面收益。",
        "classic_approaches": "ROE/ROA 排序、应计利润（accruals）、毛利率（gross profitability）、Piotroski F-Score、盈利稳定性（earnings variability）、资产周转率",
        "alpha_sources": "投资者过度关注短期盈利公告而忽视盈利质量；分析师覆盖不足的小盘股质量溢价更强；应计异象（Sloan 1996）是最稳健的 alpha 来源之一。",
        "key_papers": "Sloan (1996) 应计异象; Novy-Marx (2013) 毛利率; Fama & French (2015) 五因子模型中的 RMW（稳健减弱势）",
    },
    {
        "direction": "volume_price_divergence",
        "name": "量价背离",
        "intuition": "价格变化没有成交量配合时不可持续。放量上涨是真实买入需求，缩量上涨可能是诱多。Gervais, Kaniel & Mingelgrin (2001) 发现高成交量日的股票后续有正收益。",
        "classic_approaches": "量价相关性（price-volume correlation）、OBV 背离、成交量加权 Alpha、成交额/流通市值比异常、大单资金流向",
        "alpha_sources": "信息不对称——知情交易者通过成交量暴露意图；散户追涨杀跌导致量价背离可预测反转；大宗交易冲击后的价格修复。",
        "key_papers": "Gervais, Kaniel & Mingelgrin (2001) 高成交量溢价; Lee & Swaminathan (2000) 量价互动与动量生命周期",
    },
    {
        "direction": "volatility_regime",
        "name": "波动率与市场状态",
        "intuition": "低波动率股票的风险调整后收益高于高波动率股票（低波动率异象，Ang et al. 2006）。波动率本身有聚类性，波动率状态转换时因子表现会变化。",
        "classic_approaches": "低波动率因子（排序过去 1-3 年日收益标准差）、已实现波动率 vs 隐含波动率差、波动率-换手率交互、VIX 状态下的因子轮动",
        "alpha_sources": "杠杆约束——机构不能加杠杆所以追逐高波动率股票推高价格压缩收益；散户博彩偏好（lottery preference）也推高高波动率价格。低波因子的 alpha 在牛市中会被压抑。",
        "key_papers": "Ang, Hodrick, Xing & Zhang (2006) 低波动率异象; Baker, Bradley & Wurgler (2011) 低波动率与投资者行为; Moreira & Muir (2017) 波动率管理策略",
    },
    {
        "direction": "liquidity_premium",
        "name": "流动性溢价",
        "intuition": "流动性差的股票需要提供更高的期望收益来补偿交易成本（Amihud & Mendelson 1986）。不仅是买卖价差，市场深度和价格冲击也是流动性维度。",
        "classic_approaches": "Amihud 非流动性指标（|return|/volume）、换手率排序、买卖价差、Roll 模型价差估计、Pastor-Stambaugh 流动性 Beta",
        "alpha_sources": "市场流动性冲击时流动性差的股票跌幅更大（flight-to-quality）；流动性提供者获取风险溢价；换手率因子在 A 股环境下表现尤为显著（散户主导）。",
        "key_papers": "Amihud & Mendelson (1986) 流动性溢价; Amihud (2002) 非流动性指标; Pastor & Stambaugh (2003) 流动性风险因子",
    },
    {
        "direction": "value_contrarian",
        "name": "价值与逆向",
        "intuition": "便宜的股票（低市盈率、低市净率）长期跑赢贵的股票。Fama-French HML 因子是最稳健的因子之一。但价值陷阱（低估值因为基本面恶化）需要结合质量过滤。",
        "classic_approaches": "账面市值比、E/P 收益率、现金流/价格、股息率、52 周低点接近度、综合价值得分（combining multiple value metrics）",
        "alpha_sources": "投资者过度外推近期增长导致成长股高估；价值股的 alpha 在通胀上升期和利率上升期更强；A 股银行/地产的持续低估是结构性价值机会。",
        "key_papers": "Fama & French (1992, 1993) 三因子模型; Lakonishok, Shleifer & Vishny (1994) 逆向投资; Asness, Moskowitz & Pedersen (2013) 跨资产价值与动量",
    },
    {
        "direction": "size_effect",
        "name": "规模效应",
        "intuition": "小盘股长期跑赢大盘股（Banz 1981）。但规模效应在 1980 年代后在美国减弱，在 A 股市场仍在。小盘股的 alpha 来源更可能是流动性补偿和信息不对称。",
        "classic_approaches": "对数市值排序、小盘-大盘多空、市值 + 质量过滤（避免小盘垃圾股）、市值 + 动量交互（小盘动量更强）",
        "alpha_sources": "分析师覆盖不足导致错误定价；散户参与度高导致过度反应；壳价值在 A 股也贡献一部分小盘溢价。",
        "key_papers": "Banz (1981); Fama & French (1992); Asness et al. (2018) 规模效应是否消失",
    },
    {
        "direction": "sentiment_behavioral",
        "name": "情绪与行为金融",
        "intuition": "投资者情绪影响资产定价——高情绪期投机性股票被推高随后反转，低情绪期安全资产受追捧。Baker & Wurgler (2006) 构建了综合情绪指数。",
        "classic_approaches": "融资融券余额变化、北向资金净流入、涨停/跌停比例、新股首日收益（IPO 首日溢价）、换手率异动",
        "alpha_sources": "散户情绪驱动的错误定价；涨停板制度下的磁吸效应和次日溢价；融资盘集中度作为拥挤信号。A 股由于个人投资者占比高，情绪因子特别有效。",
        "key_papers": "Baker & Wurgler (2006) 投资者情绪; Stambaugh, Yu & Yuan (2012) 情绪与异象; 中国 A 股情绪因子研究 (Liu, Stambaugh & Yuan 2019)",
    },
    {
        "direction": "capital_flow",
        "name": "资金流向与机构行为",
        "intuition": "机构大额资金流入流出反映专业投资者的观点。Lou (2012) 发现资金流入驱动的买入会推高股价，因为需求曲线向下倾斜。",
        "classic_approaches": "北向资金（沪股通/深股通）净买入、主力资金净流入、公募基金重仓股变化、大宗交易折溢价、股东增持/减持",
        "alpha_sources": "北向资金被称为聪明钱，其净买入方向有显著的收益预测能力；公募季报披露后调仓有滞后跟随效应；大股东减持是负面信号。",
        "key_papers": "Lou (2012) 资金流驱动的收益; Frazzini & Lamont (2008) 基金资金流与股票收益; 北向资金 alpha 研究",
    },
    {
        "direction": "event_driven",
        "name": "事件驱动",
        "intuition": "公司特定事件（财报公告、分红、回购、定增、并购）会引发价格跳跃和漂移。PEAD（Post-Earnings Announcement Drift）是最稳健的异象之一——盈利超预期后股价持续漂移数月。",
        "classic_approaches": "盈利超预期（SUE）、财报公告后漂移（PEAD）、分析师上调/下调、分红预案公告效应、回购公告效应、ST/摘帽事件",
        "alpha_sources": "信息缓慢扩散——市场不能立即充分消化盈利公告的信息含量；套利限制阻止专业投资者充分套利 PEAD；A 股高送转行情的独特文化溢价。",
        "key_papers": "Ball & Brown (1968) PEAD; Bernard & Thomas (1989) SUE; Fama (1998) 事件研究",
    },
    {
        "direction": "low_risk_anomaly",
        "name": "低风险异象",
        "intuition": "Beta 与收益的关系平坦甚至为负，挑战 CAPM 模型（Black, Jensen & Scholes 1972; Frazzini & Pedersen 2014）。低 Beta 股票的风险调整后收益（Sharpe ratio）高于高 Beta 股票。",
        "classic_approaches": "历史 Beta 排序、协方差 Beta、下行 Beta（只计算负市场收益时的 Beta）、Betting Against Beta (BAB) 因子——做多低 Beta 做空高 Beta，杠杆调整至 Beta=1",
        "alpha_sources": "杠杆约束——受约束的投资者追逐高 Beta 来获取更高绝对收益，推高高 Beta 价格压缩其 alpha；低 Beta 股票在避险期表现更好。",
        "key_papers": "Black (1972) 零 Beta CAPM; Frazzini & Pedersen (2014) BAB 因子; Baker, Bradley & Wurgler (2011) 低风险异象的投资者行为解释",
    },
    {
        "direction": "cross_section_anomalies",
        "name": "截面异象综合",
        "intuition": "大量截面异象之间存在相关性，单一异象可能只是少数共同因子的表现。McLean & Pontiff (2016) 发现异象在学术发表后衰减约 30%。需要识别异象背后的共同驱动因子。",
        "classic_approaches": "多异象综合评分（composite anomaly score）、异象拥挤度监控、异象条件表现（在不同市场状态下分别评估）、机器学习的异象选择",
        "alpha_sources": "未被学术界广泛研究的异象 alpha 衰减较慢；A 股市场由于制度差异（T+1、涨跌停）存在独特截面异象；不同市场状态下的异象轮动。",
        "key_papers": "McLean & Pontiff (2016) 学术发表后的异象衰减; Green, Hand & Zhang (2017) 异象数量; Harvey, Liu & Zhu (2016) 多重检验与因子动物园",
    },
]


class FactorKnowledgeBase:
    """量化因子研究知识库。

    为 LLM 假设生成提供学术因子研究的外部知识，
    让 AI 不只是枚举模板算子，而是基于学术文献和业界实践的深入理解来生成假设。
    """

    def __init__(self, custom_entries: list[dict[str, str]] | None = None) -> None:
        self._entries = list(FACTOR_RESEARCH_KNOWLEDGE)
        if custom_entries:
            self._entries.extend(custom_entries)

    def get_knowledge_context(self, direction: str) -> str:
        """获取与 direction 最相关的知识作为 prompt 上下文。

        Args:
            direction: 研究方向关键字（如 "momentum_reversal"）

        Returns:
            格式化的 markdown 文本，可直接拼接进 LLM prompt
        """
        matches: list[dict[str, str]] = []
        direction_lower = direction.lower()

        for entry in self._entries:
            entry_dir = entry.get("direction", "").lower()
            entry_name = entry.get("name", "").lower()

            # 精确匹配 direction key 或 name
            if direction_lower == entry_dir:
                matches.append(entry)
            elif direction_lower in entry_dir or entry_dir in direction_lower:
                matches.append(entry)

        # 如果没有精确匹配，返回前 3 个关联度最高的条目
        if not matches:
            # 关键词模糊匹配
            keywords = direction_lower.replace("_", " ").split()
            for entry in self._entries:
                text = f"{entry.get('direction', '')} {entry.get('name', '')} {entry.get('intuition', '')} {entry.get('classic_approaches', '')}".lower()
                score = sum(1 for kw in keywords if kw in text)
                if score > 0:
                    matches.append((score, entry))
            matches.sort(key=lambda x: x[0], reverse=True)
            matches = [m[1] for m in matches[:3]]

        if not matches:
            return "无特定研究方向知识可用。请基于量化因子发现的通用原则进行假设生成。"

        return self._format_knowledge(matches)

    def get_all_knowledge_summary(self) -> str:
        """获取所有知识条目的简要摘要（供 R1 初始 prompt 使用）。"""
        lines = ["## 量化因子研究知识库摘要\n"]
        lines.append("以下是已知的因子研究方向及其核心直觉：\n")
        for entry in self._entries:
            lines.append(f"- **{entry['name']}**: {entry['intuition'][:120]}...")
        lines.append("\n在生成因子假设时，请考虑综合上述研究方向，不要局限于单一范式。")
        return "\n".join(lines)

    def _format_knowledge(self, entries: list[dict[str, str]]) -> str:
        """格式化知识条目为 prompt 文本。"""
        parts = ["## 量化因子研究知识（来自学术文献和业界实践）\n"]
        parts.append("以下是与当前研究方向相关的学术知识，请在生成因子假设时参考：\n")

        for i, entry in enumerate(entries, 1):
            parts.append(f"### {i}. {entry.get('name', '研究方向')}")
            parts.append(f"**核心直觉**: {entry.get('intuition', '')}")
            parts.append(f"**经典做法**: {entry.get('classic_approaches', '')}")
            parts.append(f"**潜在 Alpha 来源**: {entry.get('alpha_sources', '')}")
            refs = entry.get("key_papers", "")
            if refs:
                parts.append(f"**关键文献**: {refs}")
            parts.append("")

        parts.append("请在上述知识的基础上，提出新颖的、能够捕捉到未被充分定价的信息的因子假设。")
        parts.append("避免直接复制经典做法——尝试将不同研究方向的思路组合或应用到新的数据特征上。")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"total_entries": len(self._entries), "directions": [e["direction"] for e in self._entries]}
