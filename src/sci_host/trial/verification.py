"""验证复现引擎 — 对通过初筛的假设进行多轮复现验证.

科学发现的核心要求: 可复现.
一个假设通过一次孪生仿真不算发现，必须:
    1. 重跑复现: 同一假设用不同随机种子/扰动参数重跑 N 次，看通过率
    2. 交叉验证: 用不同算法子集分别验证，看结论是否一致
    3. 稳定性评分: 综合复现率和交叉一致性给出稳定性分数

只有稳定性达标的假设才被认证为"科学发现".

本系统的 VerificationEngine 对科研假设做可复现性检查。
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..hypothesis.generator import Hypothesis


@dataclass
class VerificationResult:
    """单次验证结果."""
    run_id: int
    passed: bool
    score: float
    consistency: float
    algo_predictions: Dict[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    perturbation: str = ""


@dataclass
class VerificationReport:
    """完整验证报告."""
    hypothesis_id: str
    statement: str
    hypothesis_type: str
    keywords: List[str]
    total_runs: int = 0
    passed_runs: int = 0
    reproduce_rate: float = 0.0
    cross_validation_passed: bool = False
    cross_validation_detail: Dict[str, bool] = field(default_factory=dict)
    stability_score: float = 0.0
    is_discovery: bool = False
    runs: List[VerificationResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def discovery_level(self) -> str:
        """候选发现级别.

        注意: 本系统的"发现"是内部算法验证的候选发现,
        不是 DFT/实验/真实仿真验证的正式科学发现.
        """
        if self.metadata.get("agent_override"):
            return "Agent override candidate (未经复现验证)"
        if not self.is_discovery:
            return "未通过"
        if self.stability_score < 0.55:
            return "数据异常"
        if self.stability_score >= 0.8:
            return "强候选发现 (Gold)"
        elif self.stability_score >= 0.6:
            return "中候选发现 (Silver)"
        else:
            return "弱候选发现 (Bronze)"


class VerificationEngine:
    """验证复现引擎.

    对通过初筛的假设进行严格的多轮复现验证:

    第一层 — 重跑复现:
        同一假设在孪生环境中重跑 N 次，每次扰动不同参数:
        - 随机种子变化
        - 算子权重微扰
        - 阈值微调
        复现率 = 通过次数 / 总次数

    第二层 — 交叉验证:
        用不同算法子集分别验证同一假设:
        - 子集 A: {cross_domain, causal, uncertainty}
        - 子集 B: {knowledge_graph, llm_reasoning}
        - 子集 C: {multi_fidelity_fusion} (融合视角)
        交叉验证通过 = 所有子集结论一致

    第三层 — 稳定性评分:
        stability = 0.5 * reproduce_rate + 0.5 * cross_consistency
        is_discovery = (stability >= threshold) and (reproduce_rate >= 0.6)
    """

    N_REPRODUCE_RUNS: int = 5
    REPRODUCE_THRESHOLD: float = 0.6
    STABILITY_THRESHOLD: float = 0.55
    _REPRODUCE_SCORE_THRESHOLDS: List[float] = [0.50, 0.65, 0.55, 0.70, 0.60]

    def __init__(self, twin: Optional[Any] = None) -> None:
        self.twin = twin
        self._verification_history: List[VerificationReport] = []

    def set_twin(self, twin: Any) -> None:
        self.twin = twin

    def verify(
        self,
        hypothesis: Hypothesis,
        original_result: Optional[Any] = None,
    ) -> VerificationReport:
        """对假设进行完整的多层验证.

        Args:
            hypothesis: 待验证的假设
            original_result: 初筛时的试错结果 (可选，用于对比)

        Returns:
            VerificationReport 完整验证报告
        """
        if self.twin is None:
            return self._no_twin_report(hypothesis)

        reproduce_results = self._reproduce_runs(hypothesis)

        cross_detail = self._cross_validate(hypothesis)

        passed_count = sum(1 for r in reproduce_results if r.passed)
        reproduce_rate = passed_count / len(reproduce_results) if reproduce_results else 0.0

        cross_passed = all(cross_detail.values()) if cross_detail else False
        cross_consistency = sum(1 for v in cross_detail.values() if v) / max(len(cross_detail), 1)

        import numpy as np
        _scores = [r.score for r in reproduce_results if r.score > 0]
        _score_std = float(np.std(_scores)) if len(_scores) >= 2 else 0.0
        variance_penalty = 0.0
        if _score_std < 0.01 and len(_scores) >= 3:
            variance_penalty = 0.10
        elif _score_std < 0.03 and len(_scores) >= 3:
            variance_penalty = 0.05

        stability = 0.5 * reproduce_rate + 0.5 * cross_consistency - variance_penalty

        is_known_principle = hypothesis.metadata.get("is_known_principle", False)
        known_principle_desc = hypothesis.metadata.get("known_principle_desc", "")
        if is_known_principle:
            stability = min(stability, 0.59)

        is_fallback = hypothesis.metadata.get("fallback", False)
        if is_fallback:
            stability = min(stability, 0.59)

        has_prediction = hypothesis.metadata.get("has_numerical_prediction", False)
        if not has_prediction:
            stability = min(stability, 0.79)

        csp_keywords = hypothesis.keywords or []
        has_low_quality_csp = (
            "unknown" in csp_keywords or
            "general" in csp_keywords or
            any(kw == "general" for kw in csp_keywords)
        )
        if has_low_quality_csp:
            stability = min(stability, 0.79)

        statement_lower = hypothesis.statement.lower()
        has_chemical_formula = any(
            any(c.isdigit() for c in kw) and any(c.isupper() for c in kw)
            for kw in csp_keywords
        )
        has_property_name = any(
            prop in statement_lower
            for prop in ["bandgap", "conductivity", "mobility", "magnetization",
                        "coercivity", "dielectric", "piezoelectric", "seebeck",
                        "zt", "young", "hardness", "thermal", "formation",
                        "cohesive", "curie", "debye"]
        )
        if not has_chemical_formula and not has_property_name and is_fallback:
            stability = min(stability, 0.59)

        is_pending = hypothesis.metadata.get("is_pending_computation", False)
        if is_pending:
            stability = min(stability, 0.79)

        range_check_passed = hypothesis.metadata.get("range_check_passed", True)
        if not range_check_passed:
            stability = 0.0

        unit_compatible = hypothesis.metadata.get("unit_compatible", True)
        if not unit_compatible:
            stability = 0.0

        _stmt = hypothesis.statement
        if "从" in _stmt and "迁移到" in _stmt:
            _parts = _stmt.split("从")
            if len(_parts) >= 2:
                _before = _parts[0].replace("将", "").strip().lower()
                _after = _parts[1]
                if "迁移到" in _after:
                    _dp = _after.split("迁移到")
                    if len(_dp) >= 2:
                        _src = _dp[0].strip().strip("，").lower()
                        _dst = _dp[1].split("，")[0].strip().lower()
                        if (_src == _dst or _before == _src or _before == _dst or
                            (len(_src) > 3 and len(_dst) > 3 and (_src in _dst or _dst in _src))):
                            stability = 0.0
        if "与" in _stmt and "组合" in _stmt:
            _parts = _stmt.split("与")
            if len(_parts) >= 2:
                _before = _parts[0].split("将")[-1].strip().lower()
                _after = _parts[1].split("组合")[0].strip().lower()
                if _before and _after and (_before == _after or
                    (len(_before) > 3 and len(_after) > 3 and (_before in _after or _after in _before))):
                    stability = 0.0

        stability = max(0.0, stability)

        is_discovery = (
            stability >= self.STABILITY_THRESHOLD
            and reproduce_rate >= self.REPRODUCE_THRESHOLD
            and cross_passed
        )

        report = VerificationReport(
            hypothesis_id=hypothesis.hypothesis_id,
            statement=hypothesis.statement,
            hypothesis_type=hypothesis.hypothesis_type,
            keywords=hypothesis.keywords,
            total_runs=len(reproduce_results),
            passed_runs=passed_count,
            reproduce_rate=round(reproduce_rate, 3),
            cross_validation_passed=cross_passed,
            cross_validation_detail=cross_detail,
            stability_score=round(stability, 3),
            is_discovery=is_discovery,
            runs=reproduce_results,
            metadata={
                "original_score": getattr(original_result, "score", 0.0),
                "original_consistency": getattr(original_result, "consistency", 0.0),
                "paper_a_id": hypothesis.paper_a_id,
                "paper_a_title": hypothesis.paper_a_title,
                "paper_b_id": hypothesis.paper_b_id,
                "paper_b_title": hypothesis.paper_b_title,
                "source_pair": hypothesis.source_pair,
                "has_numerical_prediction": has_prediction,
                "materials_mode": hypothesis.metadata.get("materials_mode", False),
                "is_known_principle": is_known_principle,
                "known_principle_desc": known_principle_desc,
                "is_fallback": is_fallback,
                "has_low_quality_csp": has_low_quality_csp,
                "is_pending_computation": is_pending,
                "range_check_passed": range_check_passed,
                "unit_compatible": unit_compatible,
                "hypothesis_metadata": dict(hypothesis.metadata),
            },
        )

        self._verification_history.append(report)
        if len(self._verification_history) > 500:
            self._verification_history = self._verification_history[-500:]

        return report

    def record_agent_override(
        self,
        hypothesis: Hypothesis,
        original_result: Optional[Any] = None,
        reasoning: str = "",
    ) -> VerificationReport:
        """记录 Agent force_pass 结果 (不作为正式候选发现).

        Agent override 是人机协作中的干预决策, 不是复现验证.
        它不进入正式候选发现列表, 而是单独记录为
        "Agent override candidate", 以明确区分:
          - 正式候选发现: 经过三层验证 (重跑+交叉+稳定性)
          - Agent override candidate: Agent 判断通过, 未经复现验证
        """
        report = VerificationReport(
            hypothesis_id=hypothesis.hypothesis_id,
            statement=hypothesis.statement,
            hypothesis_type=hypothesis.hypothesis_type,
            keywords=hypothesis.keywords,
            total_runs=0,
            passed_runs=0,
            reproduce_rate=0.0,
            cross_validation_passed=False,
            stability_score=0.0,
            is_discovery=False,
            metadata={
                "agent_override": True,
                "verification_basis": "agent_override",
                "reasoning": reasoning,
                "original_score": getattr(original_result, "score", 0.0),
                "original_consistency": getattr(original_result, "consistency", 0.0),
                "paper_a_id": hypothesis.paper_a_id,
                "paper_a_title": hypothesis.paper_a_title,
                "paper_b_id": hypothesis.paper_b_id,
                "paper_b_title": hypothesis.paper_b_title,
                "source_pair": hypothesis.source_pair,
                "hypothesis_metadata": dict(hypothesis.metadata),
            },
        )
        self._verification_history.append(report)
        if len(self._verification_history) > 500:
            self._verification_history = self._verification_history[-500:]
        return report

    def _reproduce_runs(self, hypothesis: Hypothesis) -> List[VerificationResult]:
        """第一层: 重跑复现验证.

        同一假设在孪生环境中重跑 N 次，每次微扰参数.
        """
        results: List[VerificationResult] = []
        rng = random.Random(hash(hypothesis.hypothesis_id) % 2**31)

        for run_id in range(self.N_REPRODUCE_RUNS):
            perturbation = self._apply_perturbation(run_id, rng)

            observation: Dict[str, Any] = {
                "hypothesis": hypothesis,
                "twin": self.twin,
                "papers": [],
                "hypothesis_id": hypothesis.hypothesis_id,
                "_perturbation": perturbation,
            }
            instruction = f"复现验证 run {run_id}: {hypothesis.statement[:60]}"

            try:
                sim_result = self.twin.awake_step(observation, instruction)
                predictions = sim_result.get("predictions", [])
                safe = sim_result.get("safe", False)
                violations = sim_result.get("violations", [])

                if predictions:
                    pred = predictions[0]
                    raw_score = pred.get("weighted_score", 0.0)
                    score_threshold = self._REPRODUCE_SCORE_THRESHOLDS[
                        run_id % len(self._REPRODUCE_SCORE_THRESHOLDS)
                    ]
                    passed = safe and raw_score >= score_threshold
                    if not passed and safe and raw_score < score_threshold:
                        violations = violations + [
                            f"分数 {raw_score:.3f} < 阈值 {score_threshold:.2f}"
                        ]
                    results.append(VerificationResult(
                        run_id=run_id,
                        passed=passed,
                        score=raw_score,
                        consistency=pred.get("consistency", 0.0),
                        algo_predictions=pred.get("algo_predictions", {}),
                        violations=violations,
                        perturbation=perturbation,
                    ))
                else:
                    results.append(VerificationResult(
                        run_id=run_id, passed=False, score=0.0,
                        consistency=0.0, perturbation=perturbation,
                        violations=["孪生未产生预测"],
                    ))
            except Exception as e:
                results.append(VerificationResult(
                    run_id=run_id, passed=False, score=0.0,
                    consistency=0.0, perturbation=perturbation,
                    violations=[f"仿真异常: {e}"],
                ))

            self._restore_perturbation()

        return results

    def _apply_perturbation(self, run_id: int, rng: random.Random) -> str:
        """对孪生环境施加参数扰动，模拟不同实验条件.

        每轮扰动不同的参数:
        - run 0: 基线 (无扰动)
        - run 1: 算子权重微扰
        - run 2: 一致性阈值微调
        - run 3: 随机种子变化 (通过 VM 预测微扰)
        - run 4: 综合扰动
        """
        vm = self.twin.vm if self.twin and self.twin.vm else None
        if vm is None:
            return "no_vm"

        if run_id == 0:
            return "baseline (无扰动)"

        elif run_id == 1:
            for algo_name in list(vm._algorithm_weights.keys()):
                delta = rng.uniform(-0.2, 0.2)
                vm._algorithm_weights[algo_name] = max(0.1, vm._algorithm_weights[algo_name] + delta)
            return "算子权重微扰 ±20%"

        elif run_id == 2:
            old = vm.parameters.get("consistency_threshold", 0.6)
            vm.parameters["consistency_threshold"] = max(0.3, old + rng.uniform(-0.15, 0.15))
            return f"一致性阈值 {old:.2f} → {vm.parameters['consistency_threshold']:.2f}"

        elif run_id == 3:
            old = vm.parameters.get("min_algorithms_agree", 3)
            vm.parameters["min_algorithms_agree"] = max(2, old - 1)
            return f"最少算子数 {old} → {vm.parameters['min_algorithms_agree']}"

        else:
            for algo_name in list(vm._algorithm_weights.keys()):
                delta = rng.uniform(-0.15, 0.15)
                vm._algorithm_weights[algo_name] = max(0.1, vm._algorithm_weights[algo_name] + delta)
            old_ct = vm.parameters.get("consistency_threshold", 0.6)
            vm.parameters["consistency_threshold"] = max(0.3, old_ct + rng.uniform(-0.1, 0.1))
            return "综合扰动 (权重+阈值)"

    def _restore_perturbation(self) -> None:
        """恢复孪生环境的原始参数.

        通过 EWC knowledge_base 中存储的权重恢复.
        """
        if self.twin is None or self.twin.vm is None:
            return

        vm = self.twin.vm
        if self.twin.cl and hasattr(self.twin.cl, "knowledge_base"):
            ewc_weights = self.twin.cl.knowledge_base.get("algorithm_weights", {})
            if ewc_weights:
                vm._algorithm_weights = dict(ewc_weights)

        vm.parameters["consistency_threshold"] = 0.6
        vm.parameters["min_algorithms_agree"] = 3

    def _cross_validate(self, hypothesis: Hypothesis) -> Dict[str, bool]:
        """第二层: 交叉验证.

        用不同算法子集分别验证同一假设，看结论是否一致.
        """
        if self.twin is None or self.twin.vm is None:
            return {}

        vm = self.twin.vm
        all_algos = self.twin.algorithms

        subsets: Dict[str, List[str]] = {
            "subset_A_迁移因果不确定性": [
                "cross_domain_transfer", "causal_discovery", "uncertainty_quantifier",
            ],
            "subset_B_图谱语义融合": [
                "semantic_knowledge_graph", "llm_reasoning", "multi_fidelity_fusion",
            ],
            "subset_C_全量算子": list(all_algos.keys()),
        }

        results: Dict[str, bool] = {}

        original_algos = dict(all_algos)

        for subset_name, algo_names in subsets.items():
            try:
                hidden = {}
                for algo_name in list(all_algos.keys()):
                    if algo_name not in algo_names:
                        hidden[algo_name] = all_algos[algo_name]
                        del all_algos[algo_name]

                observation = {
                    "hypothesis": hypothesis,
                    "twin": self.twin,
                    "papers": [],
                }
                predictions = vm.simulate(observation, horizon=1)

                if predictions:
                    pred = predictions[0]
                    safe, _ = vm.check_constraints(pred)
                    results[subset_name] = safe
                else:
                    results[subset_name] = False

            except Exception:
                results[subset_name] = False
            finally:
                all_algos.update(hidden)

        return results

    def _no_twin_report(self, hypothesis: Hypothesis) -> VerificationReport:
        """无孪生时的降级报告."""
        return VerificationReport(
            hypothesis_id=hypothesis.hypothesis_id,
            statement=hypothesis.statement,
            hypothesis_type=hypothesis.hypothesis_type,
            keywords=hypothesis.keywords,
            is_discovery=False,
            metadata={"error": "孪生环境未初始化"},
        )

    @property
    def stats(self) -> Dict[str, Any]:
        """验证引擎统计.

        明确区分:
          - formal_candidate_discoveries: 经过三层验证的正式候选发现
          - agent_override_candidates: Agent force_pass 的候选 (未经复现验证)
        """
        if not self._verification_history:
            return {
                "total_verified": 0,
                "total_candidate_discoveries": 0,
                "formal_candidate_discoveries": 0,
                "agent_override_candidates": 0,
                "gold_candidates": 0,
                "silver_candidates": 0,
                "bronze_candidates": 0,
                "avg_stability": 0.0,
                "avg_reproduce_rate": 0.0,
                "discovery_rate": 0.0,
            }

        total = len(self._verification_history)
        formal_discoveries = sum(
            1 for r in self._verification_history
            if r.is_discovery and not r.metadata.get("agent_override")
        )
        agent_overrides = sum(
            1 for r in self._verification_history
            if r.metadata.get("agent_override")
        )
        avg_stability = sum(r.stability_score for r in self._verification_history) / total
        avg_reproduce = sum(r.reproduce_rate for r in self._verification_history) / total

        gold = sum(1 for r in self._verification_history
                   if r.is_discovery and not r.metadata.get("agent_override")
                   and r.stability_score >= 0.8)
        silver = sum(1 for r in self._verification_history
                     if r.is_discovery and not r.metadata.get("agent_override")
                     and 0.6 <= r.stability_score < 0.8)
        bronze = sum(1 for r in self._verification_history
                     if r.is_discovery and not r.metadata.get("agent_override")
                     and r.stability_score < 0.6)

        return {
            "total_verified": total,
            "total_candidate_discoveries": formal_discoveries,
            "formal_candidate_discoveries": formal_discoveries,
            "agent_override_candidates": agent_overrides,
            "gold_candidates": gold,
            "silver_candidates": silver,
            "bronze_candidates": bronze,
            "avg_stability": round(avg_stability, 3),
            "avg_reproduce_rate": round(avg_reproduce, 3),
            "discovery_rate": round(formal_discoveries / max(total, 1), 3),
        }

    def get_discoveries(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取已认证的候选发现 (仅正式候选发现, 不含 Agent override).

        按稳定性排序, 仅返回经过三层验证的候选发现.
        Agent override candidates 请用 get_agent_override_candidates() 获取.
        """
        discoveries = [
            r for r in self._verification_history
            if r.is_discovery
            and not r.metadata.get("agent_override")
            and r.stability_score >= self.STABILITY_THRESHOLD
        ]
        discoveries.sort(key=lambda r: r.stability_score, reverse=True)
        return [
            {
                "hypothesis_id": r.hypothesis_id,
                "statement": r.statement[:120],
                "hypothesis_type": r.hypothesis_type,
                "keywords": r.keywords[:5],
                "stability_score": r.stability_score,
                "reproduce_rate": r.reproduce_rate,
                "discovery_level": r.discovery_level,
                "total_runs": r.total_runs,
                "passed_runs": r.passed_runs,
                "cross_validation": r.cross_validation_detail,
                "timestamp": r.timestamp,
                "metadata": r.metadata,
            }
            for r in discoveries[:n]
        ]

    def get_agent_override_candidates(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取 Agent override candidates (未经复现验证).

        这些是 Agent (Claude/LLM) 通过 force_pass 标记的假设,
        没有经过三层验证 (重跑+交叉+稳定性), 仅作为人机协作的候选.
        """
        candidates = [
            r for r in self._verification_history
            if r.metadata.get("agent_override")
        ]
        candidates.sort(key=lambda r: r.timestamp, reverse=True)
        return [
            {
                "hypothesis_id": r.hypothesis_id,
                "statement": r.statement[:120],
                "hypothesis_type": r.hypothesis_type,
                "keywords": r.keywords[:5],
                "discovery_level": r.discovery_level,
                "reasoning": r.metadata.get("reasoning", ""),
                "original_score": r.metadata.get("original_score", 0.0),
                "timestamp": r.timestamp,
                "metadata": r.metadata,
            }
            for r in candidates[:n]
        ]

    def get_recent_verifications(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的验证报告 (包括未通过的)."""
        recent = self._verification_history[-n:]
        recent.reverse()
        return [
            {
                "hypothesis_id": r.hypothesis_id,
                "statement": r.statement[:100],
                "is_discovery": r.is_discovery,
                "stability_score": r.stability_score,
                "reproduce_rate": r.reproduce_rate,
                "discovery_level": r.discovery_level,
                "passed_runs": f"{r.passed_runs}/{r.total_runs}",
                "cross_validation_passed": r.cross_validation_passed,
            }
            for r in recent
        ]

    def get_verification_detail(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """获取某个假设的完整验证报告."""
        for r in self._verification_history:
            if r.hypothesis_id == hypothesis_id:
                return {
                    "hypothesis_id": r.hypothesis_id,
                    "statement": r.statement,
                    "hypothesis_type": r.hypothesis_type,
                    "keywords": r.keywords,
                    "is_discovery": r.is_discovery,
                    "discovery_level": r.discovery_level,
                    "stability_score": r.stability_score,
                    "reproduce_rate": r.reproduce_rate,
                    "total_runs": r.total_runs,
                    "passed_runs": r.passed_runs,
                    "cross_validation_passed": r.cross_validation_passed,
                    "cross_validation_detail": r.cross_validation_detail,
                    "runs": [
                        {
                            "run_id": run.run_id,
                            "passed": run.passed,
                            "score": round(run.score, 3),
                            "consistency": round(run.consistency, 3),
                            "perturbation": run.perturbation,
                            "violations": run.violations,
                            "algo_predictions": {
                                k: round(v, 3) for k, v in run.algo_predictions.items()
                            },
                        }
                        for run in r.runs
                    ],
                    "metadata": r.metadata,
                }
        return None
