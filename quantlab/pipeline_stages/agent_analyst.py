"""Agent Analyst — 共享 LLM 分析引擎。

为管线后三阶段（OOS 诊断、治理解读、交付报告）提供定性分析能力。
始终保留数值计算（规则引擎精确可靠），LLM 仅负责定性解读与叙事生成。
LLM 不可用时回退到规则化摘要，不阻塞管线。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class AgentAnalyst:
    """管线后三阶段的共享 LLM 分析师。

    封装 LLMClient，为每个分析域提供专用 prompt 模板。
    LLM 仅做定性分析——所有数值计算在调用前完成。
    """

    def __init__(self, timeout: int = 90) -> None:
        self._llm = None
        self._available: bool | None = None
        self.timeout = timeout

    # ------------------------------------------------------------------
    # LLM 生命周期
    # ------------------------------------------------------------------

    def _get_llm(self):
        if self._llm is None:
            from quantlab.factor_discovery.multi_agent import LLMClient

            self._llm = LLMClient(timeout=self.timeout)
        return self._llm

    def _check_available(self) -> bool:
        if self._available is None:
            try:
                llm = self._get_llm()
                llm._load_from_env()
                self._available = bool(llm.api_key)
            except Exception:
                self._available = False
        return self._available

    def _call(self, system_prompt: str, user_prompt: str, expect_json: bool = True) -> dict[str, Any]:
        """调用 LLM，失败时返回空 dict。"""
        if not self._check_available():
            logger.info("AgentAnalyst: LLM 未配置，跳过定性分析")
            return {}
        try:
            llm = self._get_llm()
            if expect_json:
                return llm.chat_json(system_prompt, user_prompt)
            text = llm.chat(system_prompt, user_prompt)
            return {"text": text}
        except Exception as exc:
            logger.warning("AgentAnalyst 调用失败: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # 1. OOS 诊断分析
    # ------------------------------------------------------------------

    OOS_SYSTEM_PROMPT = """你是一位量化因子分析专家，负责诊断因子的样本外（OOS）验证结果。

你的任务：
1. 对每个因子，基于训练集IC和测试集IC的对比，判断其OOS表现是"健康"、"过拟合"、"结构性断裂"还是"信号不足"
2. 分析测试集IC衰减率的含义（衰减快=信号短暂，衰减慢/甚至提升=信号稳健）
3. 对fail的因子，给出具体失败原因的诊断，以及是否需要继续迭代
4. 对pass的因子，评估其稳健性等级（勉强通过/稳健/非常稳健）
5. 跨因子汇总：哪些因子类型在OOS中表现最好/最差？是否存在系统性的方向失效？

输出JSON格式：
{
  "per_factor": [
    {
      "factor_id": "...",
      "diagnosis": "healthy|overfit|structural_break|weak_signal",
      "diagnosis_cn": "一句话中文诊断",
      "robustness": "marginal|solid|strong",
      "recommendation": "promote|iterate|observe|abandon",
      "reasoning": "详细分析理由，2-3句话",
      "risk_note": "需要关注的风险点，若无则为null"
    }
  ],
  "cross_factor_summary": {
    "overall_pass_rate": 0.0,
    "top_performing_families": ["..."],
    "worst_performing_families": ["..."],
    "systemic_issues": "跨因子的系统性问题描述，若无可为null",
    "market_context_hypothesis": "基于OOS表现的市况假设，1-2句话",
    "recommendations_for_next_round": "对下一轮因子发现的建议，1-2句话"
  }
}"""

    def analyze_oos(self, oos_checks: list[dict], market_summary: dict | None = None) -> dict[str, Any]:
        """对 OOS 验证结果进行 LLM 诊断分析。

        Args:
            oos_checks: OOSValidationStage 输出的 checks 列表
            market_summary: 市场数据概览（日期范围、资产数等）
        """
        if not oos_checks:
            return {"status": "skipped", "reason": "无待分析因子"}

        user_prompt = self._build_oos_user_prompt(oos_checks, market_summary)
        result = self._call(self.OOS_SYSTEM_PROMPT, user_prompt)
        if not result:
            return self._rule_based_oos_fallback(oos_checks)
        result["status"] = "llm_analyzed"
        return result

    def _build_oos_user_prompt(self, checks: list[dict], market_summary: dict | None) -> str:
        lines = ["## OOS 验证数据\n"]
        if market_summary:
            lines.append(f"市场数据: {json.dumps(market_summary, ensure_ascii=False)}")
            lines.append("")

        for c in checks:
            fid = c.get("factor_id", "?")
            train_ic = c.get("train_ic", 0)
            test_ic = c.get("test_ic", 0)
            oos_decay = c.get("oos_decay", 0)
            cost_adj_ic = c.get("cost_adj_ic", test_ic)
            turnover = c.get("turnover", 0)
            passed = c.get("passed", False)
            error = c.get("error", "")

            lines.append(f"--- {fid} ---")
            lines.append(f"  训练集RankIC: {train_ic:.6f}")
            lines.append(f"  测试集RankIC: {test_ic:.6f}")
            lines.append(f"  OOS衰减率: {oos_decay:.2%}")
            lines.append(f"  扣费后IC: {cost_adj_ic:.6f}")
            lines.append(f"  换手率: {turnover:.4f}")
            lines.append(f"  通过: {'是' if passed else '否'}")
            if error:
                lines.append(f"  错误: {error}")
            lines.append("")

        return "\n".join(lines)

    def _rule_based_oos_fallback(self, checks: list[dict]) -> dict[str, Any]:
        """LLM 不可用时的规则化 OOS 诊断回退。"""
        per_factor = []
        families_pass = {}
        families_fail = {}

        for c in checks:
            train_ic = c.get("train_ic", 0)
            test_ic = c.get("test_ic", 0)
            oos_decay = c.get("oos_decay", 0)
            passed = c.get("passed", False)
            fid = c.get("factor_id", "")

            # 简单规则诊断
            if test_ic > train_ic * 1.1:
                diagnosis = "healthy"
                diag_cn = "样本外表现优于样本内，信号稳健"
                robustness = "strong"
            elif oos_decay < 0.3 and passed:
                diagnosis = "healthy"
                diag_cn = "OOS衰减可控，信号稳定"
                robustness = "solid"
            elif passed and oos_decay > 0.5:
                diagnosis = "overfit"
                diag_cn = "勉强通过但衰减明显，疑似部分过拟合"
                robustness = "marginal"
            elif test_ic < 0 and train_ic > 0.02:
                diagnosis = "structural_break"
                diag_cn = "信号方向反转，可能存在结构性断裂"
                robustness = "marginal"
            elif not passed:
                diagnosis = "weak_signal"
                diag_cn = "信号强度不足或衰减过快"
                robustness = "marginal"
            else:
                diagnosis = "weak_signal"
                diag_cn = "信号质量一般"
                robustness = "marginal"

            rec = "promote" if robustness == "strong" else ("iterate" if diagnosis == "overfit" else "observe")

            # 按因子族分组统计
            family = fid.split("_")[0] if "_" in fid else fid[:8]
            if passed:
                families_pass[family] = families_pass.get(family, 0) + 1
            else:
                families_fail[family] = families_fail.get(family, 0) + 1

            per_factor.append({
                "factor_id": fid,
                "diagnosis": diagnosis,
                "diagnosis_cn": diag_cn,
                "robustness": robustness,
                "recommendation": rec,
                "reasoning": f"训练IC={train_ic:.4f}, 测试IC={test_ic:.4f}, OOS衰减={oos_decay:.1%}",
                "risk_note": "OOS衰减>50%，信号持续性存疑" if oos_decay > 0.5 else None,
            })

        total = len(checks)
        passed_count = sum(1 for c in checks if c.get("passed"))
        top_families = sorted(families_pass, key=families_pass.get, reverse=True)[:3]
        worst_families = sorted(families_fail, key=families_fail.get, reverse=True)[:3]

        return {
            "status": "rule_fallback",
            "per_factor": per_factor,
            "cross_factor_summary": {
                "overall_pass_rate": round(passed_count / max(total, 1), 2),
                "top_performing_families": top_families,
                "worst_performing_families": worst_families,
                "systemic_issues": None,
                "market_context_hypothesis": None,
                "recommendations_for_next_round": None,
            },
        }

    # ------------------------------------------------------------------
    # 2. 治理解读分析
    # ------------------------------------------------------------------

    GOVERNANCE_SYSTEM_PROMPT = """你是一位量化因子库治理专家，负责对因子工厂的治理检查结果进行整体解读。

你的任务：
1. **市场状态解读**: 当前市场处于什么状态（牛/熊/震荡）？这对因子库整体意味着什么？
2. **拥挤度分析**: 拥挤因子集中在哪些方向？它们在捕捉什么共同特征？拥挤度是否达到危险水平？
3. **衰减曲线解读**: 哪些因子的衰减曲线健康？哪些有问题？有没有系统性的衰减模式？
4. **风控总结**: 当前组合的主要风险点是什么？需要立即关注什么？
5. **前瞻性建议**: 综合以上，对因子库的未来管理提出2-3条具体建议。

输出JSON格式：
{
  "executive_summary": "2-3句话的总览",
  "regime_analysis": {
    "interpretation": "市况解读，1-2句话",
    "factor_implications": "对因子表现的影响分析",
    "watch_points": ["需关注的点"]
  },
  "crowding_analysis": {
    "interpretation": "拥挤度解读",
    "cluster_descriptions": ["集群1描述", "集群2描述"],
    "severity": "low|moderate|high|critical",
    "recommended_actions": ["建议操作"]
  },
  "decay_analysis": {
    "interpretation": "衰减模式解读",
    "healthy_factors": ["因子ID"],
    "concerning_factors": ["因子ID"],
    "systemic_pattern": "系统性的衰减模式，若无则为null"
  },
  "risk_summary": {
    "overall_level": "low|moderate|high|critical",
    "top_risks": ["风险1", "风险2"],
    "immediate_actions": ["需立即采取的行动"],
    "forward_guidance": "前瞻性建议，2-3条"
  }
}"""

    def analyze_governance(self, governance_data: dict[str, Any]) -> dict[str, Any]:
        """对治理阶段的结果进行 LLM 解读。

        Args:
            governance_data: GovernanceStage.run() 的完整输出
        """
        if governance_data.get("status") == "failed":
            return {"status": "skipped", "reason": "治理阶段执行失败"}

        user_prompt = self._build_governance_user_prompt(governance_data)
        result = self._call(self.GOVERNANCE_SYSTEM_PROMPT, user_prompt)
        if not result:
            return self._rule_based_governance_fallback(governance_data)
        result["status"] = "llm_analyzed"
        return result

    def _build_governance_user_prompt(self, data: dict[str, Any]) -> str:
        lines = ["## 因子库治理数据\n"]

        # Regime
        regime = data.get("regime", {})
        if regime:
            lines.append("### 市场状态")
            lines.append(json.dumps(regime, ensure_ascii=False, default=str))
            lines.append("")

        # Crowding
        crowding = data.get("crowding", {})
        if crowding:
            lines.append("### 拥挤度检测")
            lines.append(json.dumps(crowding, ensure_ascii=False, default=str))
            lines.append("")

        # Lifecycle changes
        lifecycle = data.get("lifecycle_changes", [])
        if lifecycle:
            lines.append("### 生命周期变更")
            lines.append(json.dumps(lifecycle, ensure_ascii=False))
            lines.append("")

        # Curves
        curves = data.get("curves", {})
        if curves:
            lines.append(f"### 因子衰减曲线 (共{curves.get('total',0)}个因子)")
            lines.append(f"有半衰期的因子数: {curves.get('factors_with_half_life', 0)}")
            lines.append("")

        # Risk
        risk = data.get("risk", {})
        if risk:
            lines.append("### 风控评估")
            lines.append(json.dumps(risk, ensure_ascii=False, default=str))
            lines.append("")

        # Stats
        before = data.get("stats_before", {})
        after = data.get("stats_after", {})
        if before or after:
            lines.append(f"### 库统计变化")
            lines.append(f"治理前: {json.dumps(before, ensure_ascii=False)}")
            lines.append(f"治理后: {json.dumps(after, ensure_ascii=False)}")
            lines.append("")

        lines.append("请基于以上数据进行综合分析。")
        return "\n".join(lines)

    def _rule_based_governance_fallback(self, data: dict[str, Any]) -> dict[str, Any]:
        """LLM 不可用时的治理回退摘要。"""
        regime = data.get("regime", {})
        crowding = data.get("crowding", {})
        risk = data.get("risk", {})
        curves = data.get("curves", {})

        regime_current = regime.get("current", "unknown") if isinstance(regime, dict) else "unknown"

        crowded_count = len(crowding.get("crowded_factor_ids", [])) if isinstance(crowding, dict) else 0
        crowding_severity = "high" if crowded_count > 5 else ("moderate" if crowded_count > 2 else "low")

        risk_score = risk.get("risk_score", 0) if isinstance(risk, dict) else 0
        risk_level = "high" if risk_score > 0.7 else ("moderate" if risk_score > 0.4 else "low")

        lifecycle_count = len(data.get("lifecycle_changes", []))

        return {
            "status": "rule_fallback",
            "executive_summary": f"市场状态={regime_current}, 拥挤因子={crowded_count}个, 生命周期变更={lifecycle_count}个, 风险水平={risk_level}",
            "regime_analysis": {
                "interpretation": f"当前市场处于{regime_current}状态" if regime_current != "unknown" else "市况未知",
                "factor_implications": None,
                "watch_points": [],
            },
            "crowding_analysis": {
                "interpretation": f"检测到{crowded_count}个拥挤因子" if crowded_count > 0 else "未检测到显著拥挤",
                "cluster_descriptions": [],
                "severity": crowding_severity,
                "recommended_actions": ["降低拥挤因子权重"] if crowded_count > 0 else [],
            },
            "decay_analysis": {
                "interpretation": f"已分析{curves.get('total', 0)}个因子衰减曲线" if isinstance(curves, dict) else "衰减分析未执行",
                "healthy_factors": [],
                "concerning_factors": [],
                "systemic_pattern": None,
            },
            "risk_summary": {
                "overall_level": risk_level,
                "top_risks": risk.get("breaches", [])[:3] if isinstance(risk, dict) else [],
                "immediate_actions": [],
                "forward_guidance": "定期监控因子库拥挤度和IC衰减",
            },
        }

    # ------------------------------------------------------------------
    # 3. 交付报告叙事生成
    # ------------------------------------------------------------------

    REPORT_SYSTEM_PROMPT = """你是一位量化因子研究员，负责为买方撰写因子交付报告的研究叙事部分。

你的读者是专业机构投资者（基金公司、资管公司），他们关心：
- 这个因子捕捉了什么经济逻辑？
- 数据的说服力如何？
- 有什么他们需要在意的风险？
- 这个因子在什么市场环境下有效/失效？
- 相比同类因子有什么优势？

你需要在保持专业性的同时，让报告具有可读性。

输出JSON格式：
{
  "executive_summary": "2-3段话的总体概述，包括：因子做什么、核心数据亮点、关键风险提示",
  "factor_story": {
    "economic_intuition": "这个因子背后的经济学直觉，1-2句话",
    "mechanism_explanation": "因子为什么有效的机制解释",
    "academic_background": "涉及的相关学术研究方向（如果适用）"
  },
  "strengths": ["优势1", "优势2", "优势3"],
  "weaknesses": ["劣势1", "劣势2"],
  "market_context": {
    "best_environments": ["最有效的市场环境"],
    "worst_environments": ["最无效的市场环境"],
    "current_suitability": "当前市场环境下该因子的适用性评估，1-2句话"
  },
  "risk_assessment": {
    "primary_risk": "首要风险",
    "secondary_risks": ["次要风险"],
    "mitigation_suggestions": ["风险缓释建议"]
  },
  "peer_comparison": "与库内同族因子的比较分析，1-2句话（若无同族因子则为null）",
  "buyer_checklist": ["买方尽调时需关注的3-5个要点"]
}"""

    def generate_narrative_report(
        self,
        report_dict: dict[str, Any],
        library_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """基于数值报告生成 LLM 研究叙事。

        Args:
            report_dict: FactorDeliveryReport.to_dict() 的输出
            library_context: 因子库统计（用于 peer comparison）
        """
        if not report_dict:
            return {"status": "skipped", "reason": "报告数据为空"}

        user_prompt = self._build_report_user_prompt(report_dict, library_context)
        result = self._call(self.REPORT_SYSTEM_PROMPT, user_prompt)
        if not result:
            return self._rule_based_report_fallback(report_dict)
        result["status"] = "llm_generated"
        return result

    def _build_report_user_prompt(
        self, report: dict[str, Any], library_ctx: dict[str, Any] | None
    ) -> str:
        lines = ["## 因子交付报告数据\n"]

        lines.append(f"### 基本信息")
        lines.append(f"因子ID: {report.get('factor_id', '?')}")
        lines.append(f"因子名称: {report.get('factor_name', '?')}")
        lines.append(f"因子族: {report.get('factor_family', '?')}")
        lines.append(f"方向: {report.get('direction', '?')}")
        lines.append(f"表达式: {report.get('expression', '?')}")
        lines.append(f"假设: {report.get('hypothesis', '?')}")
        lines.append("")

        lines.append("### IC 指标族")
        lines.append(f"RankIC均值: {report.get('rank_ic_mean', 0):.6f}")
        lines.append(f"RankIC标准差: {report.get('rank_ic_std', 0):.6f}")
        lines.append(f"ICIR: {report.get('icir', 0):.4f}")
        lines.append(f"IC正比例: {report.get('ic_positive_ratio', 0):.1%}")
        decay = report.get("decay_profile", {})
        if decay:
            lines.append(f"衰减曲线: {json.dumps(decay)}")
        lines.append("")

        sim = report.get("simulation", {})
        if sim:
            lines.append("### 扣费后组合绩效")
            lines.append(f"年化毛收益: {sim.get('gross_return', 0):.2%}")
            lines.append(f"年化净收益: {sim.get('net_return', 0):.2%}")
            lines.append(f"Sharpe: {sim.get('sharpe_ratio', 0):.3f}")
            lines.append(f"最大回撤: {sim.get('max_drawdown', 0):.2%}")
            lines.append(f"日均换手: {sim.get('avg_daily_turnover', 0):.2%}")
            lines.append(f"IR: {sim.get('information_ratio', 0):.3f}")
            lines.append("")

        cap = report.get("capacity", {})
        if cap:
            lines.append("### 容量估算")
            lines.append(json.dumps(cap, ensure_ascii=False))
            lines.append("")

        lines.append(f"### 暴露度")
        lines.append(f"市值暴露: {report.get('market_cap_exposure', 0):.4f}")
        lines.append(f"行业暴露: {json.dumps(report.get('industry_exposure', {}), ensure_ascii=False)}")
        lines.append(f"与已知因子相关性: {json.dumps(report.get('correlation_to_known_factors', {}), ensure_ascii=False)}")
        lines.append("")

        lines.append(f"### 稳健性")
        lines.append(f"稳定性评分: {report.get('stability_score', 0):.3f}")
        lines.append(f"样本外超额: {report.get('sample_out_performance', 0):.2%}")
        lines.append("")

        risk_flags = report.get("risk_flags", [])
        if risk_flags:
            lines.append("### 风险标记")
            for rf in risk_flags:
                lines.append(f"- {rf}")
            lines.append("")

        if library_ctx:
            lines.append("### 因子库背景")
            lines.append(json.dumps(library_ctx, ensure_ascii=False))
            lines.append("")

        lines.append("请基于以上数据生成买方交付报告的研究叙事部分。")
        return "\n".join(lines)

    def _rule_based_report_fallback(self, report: dict[str, Any]) -> dict[str, Any]:
        """LLM 不可用时的报告回退（使用数值指标生成规则化摘要）。"""
        rank_ic = report.get("rank_ic_mean", 0)
        icir = report.get("icir", 0)
        sim = report.get("simulation", {})
        sharpe = sim.get("sharpe_ratio", 0) if sim else 0
        turnover = report.get("avg_daily_turnover", 0)
        risk_flags = report.get("risk_flags", [])

        # 规则化强度判断
        ic_strength = "强" if abs(rank_ic) > 0.05 else ("中等" if abs(rank_ic) > 0.02 else "偏弱")
        icir_strength = "优秀" if abs(icir) > 2.0 else ("良好" if abs(icir) > 1.0 else ("一般" if abs(icir) > 0.5 else "偏低"))

        strengths = []
        weaknesses = []
        if abs(rank_ic) > 0.03:
            strengths.append(f"RankIC={rank_ic:.4f}，信号区分力{ic_strength}")
        else:
            weaknesses.append(f"RankIC={rank_ic:.4f}，信号区分力{ic_strength}")

        if abs(icir) > 1.0:
            strengths.append(f"ICIR={icir:.3f}，信号稳定性{icir_strength}")
        else:
            weaknesses.append(f"ICIR={icir:.3f}，信号稳定性{icir_strength}")

        if sharpe > 1.0:
            strengths.append(f"扣费后Sharpe={sharpe:.3f}，风险调整收益优秀")
        if turnover > 0.5:
            weaknesses.append(f"日均换手={turnover:.1%}，换手偏高")

        for rf in (risk_flags or []):
            weaknesses.append(rf)

        return {
            "status": "rule_fallback",
            "executive_summary": (
                f"该因子（{report.get('factor_name', '?')}）属于{report.get('factor_family', '?')}族，"
                f"RankIC均值为{rank_ic:.4f}，ICIR为{icir:.3f}。"
                f"扣费后Sharpe={sharpe:.3f}。{'存在' + str(len(risk_flags)) + '个风险提示需关注。' if risk_flags else '当前未检测到显著风险。'}"
            ),
            "factor_story": {
                "economic_intuition": report.get("hypothesis", ""),
                "mechanism_explanation": f"因子表达式: {report.get('expression', '')}",
                "academic_background": None,
            },
            "strengths": strengths or ["指标计算完成，详见数值报告"],
            "weaknesses": weaknesses or ["无明显弱点"],
            "market_context": {
                "best_environments": [],
                "worst_environments": [],
                "current_suitability": "需结合当前市场状态判断",
            },
            "risk_assessment": {
                "primary_risk": risk_flags[0] if risk_flags else "无明显风险",
                "secondary_risks": risk_flags[1:] if len(risk_flags) > 1 else [],
                "mitigation_suggestions": ["定期监控IC衰减", "关注因子的拥挤度变化"],
            },
            "peer_comparison": None,
            "buyer_checklist": [
                "检查因子在样本外的IC稳定性",
                "确认因子与库内已有因子的正交性",
                "评估扣费后可交易性",
                "验证因子在不同市场状态下的表现一致性",
                "确认因子容量满足策略需求",
            ],
        }

    # ------------------------------------------------------------------
    # 4. 跨轮反馈摘要
    # ------------------------------------------------------------------

    FEEDBACK_SYSTEM_PROMPT = """你是一位量化因子发现系统的元分析师。基于本轮OOS验证和治理分析的结果，
生成针对下一轮因子发现（Factor Discovery Agent）的反馈建议。

你的建议将被注入到下一轮因子发现的LLM prompt中，帮助R1（假设生成器）和R2（假设审查器）做出更好的决策。

输出JSON格式：
{
  "summary": "1-2句话的本轮关键发现摘要",
  "directions_to_prioritize": ["应优先探索的方向"],
  "directions_to_avoid": ["应避免的方向"],
  "structural_patterns": {
    "what_worked": "本轮验证有效的结构模式",
    "what_failed": "本轮失败的结构模式"
  },
  "data_quality_notes": "数据层面的注意事项（若有）",
  "methodology_suggestions": "方法论层面的改进建议"
}"""

    def generate_feedback(
        self,
        oos_analysis: dict[str, Any],
        governance_analysis: dict[str, Any],
    ) -> dict[str, Any]:
        """生成跨轮反馈，供下一轮因子发现使用。

        Args:
            oos_analysis: analyze_oos() 的输出
            governance_analysis: analyze_governance() 的输出
        """
        if not self._check_available():
            return self._rule_based_feedback_fallback(oos_analysis, governance_analysis)

        user_prompt = self._build_feedback_user_prompt(oos_analysis, governance_analysis)
        result = self._call(self.FEEDBACK_SYSTEM_PROMPT, user_prompt)
        if not result:
            return self._rule_based_feedback_fallback(oos_analysis, governance_analysis)
        result["status"] = "llm_generated"
        result["generated_at"] = datetime.now().isoformat()
        return result

    def _build_feedback_user_prompt(
        self, oos: dict[str, Any], gov: dict[str, Any]
    ) -> str:
        lines = ["## 本轮分析结果\n"]

        lines.append("### OOS 验证分析")
        cross = oos.get("cross_factor_summary", {})
        if cross:
            lines.append(json.dumps(cross, ensure_ascii=False))
        lines.append("")

        lines.append("### 治理分析")
        regime = gov.get("regime_analysis", {})
        crowding = gov.get("crowding_analysis", {})
        decay = gov.get("decay_analysis", {})
        if regime:
            lines.append(f"市况: {json.dumps(regime, ensure_ascii=False)}")
        if crowding:
            lines.append(f"拥挤: {json.dumps(crowding, ensure_ascii=False)}")
        if decay:
            lines.append(f"衰减: {json.dumps(decay, ensure_ascii=False)}")
        lines.append("")

        lines.append("请生成下一轮因子发现的反馈建议。")
        return "\n".join(lines)

    def _rule_based_feedback_fallback(
        self, oos: dict[str, Any], gov: dict[str, Any]
    ) -> dict[str, Any]:
        """LLM 不可用时的回退反馈。"""
        cross = oos.get("cross_factor_summary", {})
        crowding = gov.get("crowding_analysis", {})
        regime = gov.get("regime_analysis", {})

        top_families = cross.get("top_performing_families", [])
        worst_families = cross.get("worst_performing_families", [])
        avoid_dirs = (crowding.get("cluster_descriptions", []) or [])[:3]

        return {
            "status": "rule_fallback",
            "summary": f"本轮OOS通过率{cross.get('overall_pass_rate', 0):.0%}，"
                       f"表现最佳族: {', '.join(top_families) if top_families else '无'}",
            "directions_to_prioritize": top_families[:3],
            "directions_to_avoid": avoid_dirs,
            "structural_patterns": {
                "what_worked": None,
                "what_failed": ", ".join(worst_families) if worst_families else None,
            },
            "data_quality_notes": None,
            "methodology_suggestions": None,
            "generated_at": datetime.now().isoformat(),
        }
