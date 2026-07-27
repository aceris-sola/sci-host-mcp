"""试错引擎 — 在数字孪生虚拟环境中对假设进行仿真试错.

核心思想 (用户原话):
    "试错应该在孪生环境里仿真：把假设放进孪生跑模拟，看预测结果是否自洽"

工作流程:
    1. 把假设转为孪生观测 (hypothesis + papers → observation)
    2. 调用 twin.awake_step() 跑仿真:
       - PE 读取论文语料状态 (传感器)
       - VM (HypothesisEvaluatorModel) 用算法算子做预测
       - check_constraints() 检查多算法预测的自洽性
    3. 仿真结果 → TrialResult
    4. 试错结果写入孪生 DD 维度 (DigitalTwinData)
    5. EWC 持续学习: 保护高置信度知识，校准算子权重

与原项目的关系:
    原项目: LearningOperator.apply_local() 在 Sleep Cycle 中训练
    本系统: twin.awake_step() 在连续循环中仿真试错 (无睡眠)
    原项目: EWC 防遗忘保护已学模型参数
    本系统: EWC 保护已验证科研知识的算子权重
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import TrialConfig
from ..hypothesis.generator import Hypothesis




MATERIAL_PROPERTY_CONSTRAINTS: Dict[str, tuple] = {
    "bandgap": (0.0, 12.0),            
    "conductivity": (0.0, 1e8),        
    "mobility": (0.0, 1e6),            
    "tc": (0.0, 2000.0),              
    "tn": (0.0, 2000.0),              
    "magnetization": (0.0, 300.0),    
    "coercivity": (0.0, 1e6),         # Oe
    "dielectric_constant": (1.0, 1e5), 
    "dielectric_loss": (0.0, 10.0),   # tan δ
    "piezoelectric_coeff": (0.0, 2000.0),  # pC/N
    "seebeck": (-1000.0, 1000.0),     
    "zt": (0.0, 10.0),                
    "youngs_modulus": (0.0, 1000.0),  # GPa
    "hardness": (0.0, 100.0),         # GPa
    "thermal_conductivity": (0.0, 2000.0),  # W/mK
    "specific_heat": (0.0, 1000.0),   # J/molK
    "debye_temp": (0.0, 2000.0),      # K
    "formation_energy": (-10.0, 5.0), 
    "cohesive_energy": (-20.0, 0.0),  
}


@dataclass
class TrialResult:
    """试错结果."""
    hypothesis_id: str
    statement: str
    hypothesis_type: str
    source_pair: str
    paper_a_id: str
    paper_b_id: str
    paper_a_title: str
    paper_b_title: str
    keywords: List[str]
    passed: bool                        
    score: float                        
    consistency: float                  
    algo_predictions: Dict[str, float] = field(default_factory=dict)
    failure_reason: str = ""
    retry_count: int = 0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    novelty: float = 0.0
    confidence: float = 0.0
    strategy_scores: Dict[str, float] = field(default_factory=dict)
    
    
    credibility: str = ""              # "simulated_only" / "llm_augmented" / "agent_certified"
    credibility_note: str = ""         

    def __post_init__(self) -> None:
        """自动计算可信度标签."""
        if not self.credibility:
            self.credibility = self._compute_credibility()
        if not self.credibility_note:
            self.credibility_note = self._compute_credibility_note()

    def _compute_credibility(self) -> str:
        """根据元数据自动判定可信度级别."""
        if self.metadata.get("agent_override"):
            return "agent_certified"
        if self.metadata.get("llm_reasoning"):
            return "llm_augmented"
        if self.algo_predictions and self.algo_predictions.get("fallback") is not None:
            return "heuristic_only"
        return "simulated_only"

    def _compute_credibility_note(self) -> str:
        """生成人可读的可信度说明."""
        level = self.credibility
        if level == "agent_certified":
            return ("Agent 人工认证, 仿真分数仅供参考. "
                    "需独立实验验证后方可作为工程结论.")
        elif level == "llm_augmented":
            return ("仿真 + LLM 推理, 算子一致性较高但仍为模拟环境结果. "
                    "不可替代真实扭矩/效率/寿命测试.")
        elif level == "heuristic_only":
            return ("无孪生仿真, 仅后发式评分. 可信度最低, "
                    "不应作为任何决策依据.")
        else:
            return ("孪生仿真验证, 算子自洽但不代表物理可行. "
                    "仿真分数与真实实验结果可能存在显著偏差.")


class TrialEngine:
    """孪生仿真试错引擎.

    把假设放进数字孪生虚拟环境跑仿真:
    - 多个算法算子 (CrossDomainTransfer, CausalDiscovery, UncertaintyQuantifier,
      SemanticKnowledgeGraph, LLMReasoning, MultiFidelityFusion) 各自给出预测
    - 检查各算子预测的自洽性 (一致性)
    - 自洽且评分达标 → 通过; 否则 → 失败

    EWC 持续学习:
    - 每轮试错后，通过验证的结果用于校准算子权重
    - EWC 保护高准确率算子的权重不被新数据冲刷
    - 系统越跑越聪明 (算子权重越来越准)
    """

    def __init__(
        self,
        config: TrialConfig,
        twin: Optional[Any] = None,
        knowledge_graph: Optional[Any] = None,
        materials_mode: bool = False,
    ) -> None:
        self.config = config
        self.twin = twin
        self.knowledge_graph = knowledge_graph
        self.materials_mode = materials_mode
        self._trial_history: Dict[str, int] = {}
        self._total_trials: int = 0
        self._total_passed: int = 0
        self._total_failed: int = 0
        
        self._calibration_buffer: List[Dict[str, Any]] = []

    def set_twin(self, twin: Any) -> None:
        """设置数字孪生实例."""
        self.twin = twin

    def run_trials(self, hypotheses: List[Hypothesis]) -> List[TrialResult]:
        """对假设列表进行孪生仿真试错."""
        if not hypotheses:
            return []

        results: List[TrialResult] = []

        for hypo in hypotheses:
            if len(results) >= self.config.max_trials_per_round:
                break
            result = self._simulate_trial(hypo)
            results.append(result)

        
        if results and self.twin:
            self._ewc_update(results)

        return results

    def _simulate_trial(self, hypo: Hypothesis) -> TrialResult:
        """把假设放进孪生环境跑仿真.

        流程:
        1. 构造孪生观测 (hypothesis → observation)
        2. twin.awake_step() → 算法算子做预测
        3. 检查自洽性 → 通过/失败
        4. 内容质量证伪检查 (新增)
        5. 记录到 DD 维度
        """
        self._total_trials += 1

        
        
        quality_violations = self._check_hypothesis_quality(hypo)
        if quality_violations:
            return self._make_fail_result(
                hypo, "; ".join(quality_violations),
            )

        if self.twin is None:
            
            return self._fallback_trial(hypo)

        
        observation: Dict[str, Any] = {
            "hypothesis": hypo,
            "twin": self.twin,
            "papers": [],  
            "hypothesis_id": hypo.hypothesis_id,
        }
        instruction = f"验证假设: {hypo.statement[:80]}"

        
        try:
            sim_result = self.twin.awake_step(observation, instruction)
        except Exception as e:
            return self._make_fail_result(hypo, f"孪生仿真异常: {e}")

        
        predictions = sim_result.get("predictions", [])
        safe = sim_result.get("safe", False)
        violations = sim_result.get("violations", [])

        if not predictions:
            return self._make_fail_result(hypo, "孪生未产生预测")

        pred = predictions[0]
        algo_predictions = pred.get("algo_predictions", {})
        weighted_score = pred.get("weighted_score", 0.5)
        consistency = pred.get("consistency", 0.5)
        n_algorithms = pred.get("n_algorithms", 0)

        
        llm_reasoning_text = ""
        llm_streaming_error = ""
        try:
            from ..materials.llm_streaming import get_streaming_reasoner
            reasoner = get_streaming_reasoner()
            if reasoner.is_available:
                
                csp_context = []
                if hasattr(hypo, 'metadata') and hypo.metadata.get('csp_triples'):
                    csp_context = hypo.metadata['csp_triples']

                llm_result = reasoner.evaluate(hypo, algo_predictions, csp_context)
                if llm_result:
                    
                    algo_predictions["llm_streaming"] = llm_result["score"]
                    llm_reasoning_text = llm_result["reasoning"]

                    
                    import numpy as np
                    scores = list(algo_predictions.values())
                    weights = [1.5 if k == "llm_streaming" else 1.0 for k in algo_predictions]
                    weighted_score = float(np.average(scores, weights=weights))

                    
                    if len(algo_predictions) >= 2:
                        scores_arr = np.array(list(algo_predictions.values()))
                        std = float(np.std(scores_arr))
                        consistency = max(0.0, 1.0 - std * 3.0)

                    n_algorithms = len(algo_predictions)
                else:
                    llm_streaming_error = "LLM returned no result"
            # else: LLM not configured, silently skip
        except Exception as e:
            llm_streaming_error = f"LLM streaming failed: {e}"
            
            import logging
            logging.getLogger(__name__).warning(llm_streaming_error)

        
        
        passed = safe and weighted_score >= self.config.pass_threshold

        
        if self.materials_mode:
            constraint_violations = self._check_material_constraints(hypo)
            if constraint_violations:
                passed = False
                violations.extend(constraint_violations)
                weighted_score *= 0.5  
            elif hypo.metadata.get("has_numerical_prediction"):
                
                weighted_score = min(1.0, weighted_score * 1.15)
                if not passed and weighted_score >= self.config.pass_threshold:
                    passed = safe  

        
        retry_count = self._trial_history.get(hypo.hypothesis_id, 0)
        if not passed:
            if retry_count < self.config.max_retries:
                self._trial_history[hypo.hypothesis_id] = retry_count + 1
                failure_reason = (
                    f"仿真未通过 (第{retry_count+1}次), "
                    f"自洽性={consistency:.3f}, 评分={weighted_score:.3f}, "
                    f"将重试"
                )
            else:
                failure_reason = self._diagnose_failure(
                    algo_predictions, consistency, weighted_score, violations,
                )
            self._total_failed += 1
        else:
            failure_reason = ""
            self._total_passed += 1

        
        self._calibration_buffer.append({
            "hypothesis_id": hypo.hypothesis_id,
            "algo_predictions": algo_predictions,
            "actual_passed": passed,
            "weighted_score": weighted_score,
            "consistency": consistency,
        })
        if len(self._calibration_buffer) > 500:
            self._calibration_buffer = self._calibration_buffer[-500:]

        
        
        strategy_scores = dict(algo_predictions)
        strategy_scores["consistency"] = consistency
        strategy_scores["weighted_score"] = weighted_score

        
        result_metadata = dict(hypo.metadata)
        if llm_reasoning_text:
            result_metadata["llm_reasoning"] = llm_reasoning_text
        if llm_streaming_error:
            result_metadata["llm_streaming_error"] = llm_streaming_error

        return TrialResult(
            hypothesis_id=hypo.hypothesis_id,
            statement=hypo.statement,
            hypothesis_type=hypo.hypothesis_type,
            source_pair=hypo.source_pair,
            paper_a_id=hypo.paper_a_id,
            paper_b_id=hypo.paper_b_id,
            paper_a_title=hypo.paper_a_title,
            paper_b_title=hypo.paper_b_title,
            keywords=hypo.keywords,
            passed=passed,
            score=round(weighted_score, 4),
            consistency=round(consistency, 4),
            algo_predictions={k: round(v, 3) for k, v in algo_predictions.items()},
            failure_reason=failure_reason,
            retry_count=retry_count,
            metadata=result_metadata,
            novelty=hypo.novelty,
            confidence=hypo.confidence,
            strategy_scores={k: round(v, 3) for k, v in strategy_scores.items()},
        )

    def _check_material_constraints(self, hypo: Hypothesis) -> List[str]:
        """检查假设中的数值预测是否符合材料物理约束.

        赛题: "可证伪性" — 预测值必须在物理合理范围内.
        如果假设预测 bandgap=50 eV (物理不可能), 直接拒绝.

        新增检查:
            - 晶体化学可行性: 化合物不能存在于单元素结构 (如 Nd2Fe14B 在 diamond)
            - 相态一致性: YBCO 四方相不具备 Tc 属性
            - 已知规律: 层状→低热导率等不算新发现 (仅降级不拒绝)

        Returns:
            违规列表, 空列表=通过
        """
        import re
        violations: List[str] = []
        statement = hypo.statement.lower()

        
        for prop_name, (min_val, max_val) in MATERIAL_PROPERTY_CONSTRAINTS.items():
            
            
            search_names = [prop_name, prop_name.replace("_", " ")]
            
            if prop_name == "bandgap":
                search_names.extend(["带隙", "band gap"])
            elif prop_name == "conductivity":
                search_names.extend(["电导率", "离子电导率"])
            elif prop_name == "tc":
                search_names.extend(["居里温度", "curie"])

            found_name = None
            for sn in search_names:
                if sn in statement:
                    found_name = sn
                    break

            if found_name is None:
                continue

            
            idx = statement.find(found_name)
            search_region = statement[max(0, idx-50):idx+150]

            
            value_pattern = re.compile(r'(-?[\d.]+(?:e[+-]?\d+)?)')
            matches = value_pattern.findall(search_region)

            for val_str in matches:
                try:
                    val = float(val_str)
                    
                    if val < 0.001 and min_val > 0:
                        continue
                    
                    if val < min_val or val > max_val:
                        violations.append(
                            f"物理约束违反: {prop_name}={val} 超出合理范围 "
                            f"[{min_val}, {max_val}]"
                        )
                except ValueError:
                    continue

        
        
        
        try:
            from ..materials import MaterialPhysicsValidator
            
            keywords = hypo.keywords or []
            structure_types = [
                "diamond", "graphite", "rocksalt", "zincblende", "wurtzite",
                "rutile", "fluorite", "perovskite", "spinel", "garnet",
                "ilmenite", "scheelite", "pyrochlore", "olivine",
            ]
            
            found_structures = [s for s in structure_types if s in statement]
            
            found_compositions = []
            for kw in keywords:
                has_upper = any(c.isupper() for c in kw)
                has_digit = any(c.isdigit() for c in kw)
                has_second_upper = sum(1 for c in kw if c.isupper()) >= 2
                if has_upper and (has_digit or has_second_upper) and len(kw) >= 2:
                    found_compositions.append(kw)

            
            for comp in found_compositions:
                for struct in found_structures:
                    v = MaterialPhysicsValidator.validate_structure_compatibility(comp, struct)
                    violations.extend(v)

                
                for struct in found_structures:
                    
                    prop = "general"
                    for p_name in MATERIAL_PROPERTY_CONSTRAINTS:
                        if p_name in statement or p_name.replace("_", " ") in statement:
                            prop = p_name
                            break
                    v = MaterialPhysicsValidator.check_phase_consistency(comp, struct, prop)
                    violations.extend(v)
        except Exception:
            pass  

        return violations

    def _ewc_update(self, results: List[TrialResult]) -> None:
        """EWC 持续学习: 用试错结果校准算子权重.

        流程:
        1. 用试错结果校准 VM 的算子权重
        2. EWC 保护高准确率算子 (防止权重被新数据冲刷)
        3. 将校准后的权重存入 CL 维度的 knowledge_base
        """
        if not self.twin or not self.twin.vm:
            return

        vm = self.twin.vm
        cl = self.twin.cl

        
        if hasattr(vm, "calibrate") and self._calibration_buffer:
            recent = self._calibration_buffer[-50:]
            rmse = vm.calibrate(recent)
        else:
            rmse = 0.0

        
        
        
        ewc_weights: Dict[str, float] = {}
        if hasattr(cl, "knowledge_base"):
            old_weights = cl.knowledge_base.get("algorithm_weights", {})
        else:
            old_weights = {}

        new_weights = getattr(vm, "_algorithm_weights", {})

        ewc_lambda = getattr(cl, "ewc_lambda", 0.4) if cl else 0.4

        for algo_name, new_w in new_weights.items():
            old_w = old_weights.get(algo_name, new_w)
            
            
            ewc_weights[algo_name] = (1 - ewc_lambda) * new_w + ewc_lambda * old_w

        
        if cl and hasattr(cl, "knowledge_base"):
            cl.knowledge_base["algorithm_weights"] = ewc_weights
            cl.knowledge_base["calibration_rmse"] = rmse
            cl.knowledge_base["total_trials"] = self._total_trials
            cl.knowledge_base["total_passed"] = self._total_passed

        
        if cl and hasattr(cl, "add_skill"):
            if self._total_passed > 0:
                cl.add_skill(f"trial_round_{self._total_trials}", {
                    "passed": self._total_passed,
                    "failed": self._total_failed,
                    "rmse": rmse,
                    "weights": ewc_weights,
                })

    def _diagnose_failure(
        self,
        algo_predictions: Dict[str, float],
        consistency: float,
        score: float,
        violations: List[str],
    ) -> str:
        """诊断仿真失败原因."""
        reasons: List[str] = []

        if violations:
            reasons.append("; ".join(violations))

        if algo_predictions:
            
            sorted_algos = sorted(algo_predictions.items(), key=lambda x: x[1])
            weakest_algo = sorted_algos[0]
            algo_names = {
                "cross_domain_transfer": "跨域迁移",
                "semantic_knowledge_graph": "知识图谱",
                "uncertainty_quantifier": "不确定性量化",
                "causal_discovery": "因果发现",
                "llm_reasoning": "语义推理",
                "multi_fidelity_fusion": "多保真度融合",
            }
            weakest_name = algo_names.get(weakest_algo[0], weakest_algo[0])
            reasons.append(f"最弱算子: {weakest_name}={weakest_algo[1]:.3f}")

        if consistency < 0.4:
            reasons.append(f"算子间分歧大 (自洽性={consistency:.3f})")

        return " | ".join(reasons) if reasons else f"综合评分不足 ({score:.3f})"

    def _check_hypothesis_quality(self, hypo: Hypothesis) -> List[str]:
        """检查假设内容质量，返回违规列表 (空=通过).

        这是独立于孪生仿真的内容级证伪机制，用于拦截:
        1. 自指假设: "Creep 中的 Creep 可能与...共享原理"
        2. 摘要碎片填槽: "numerical investigation protective" 等无意义片段
        3. 材料模式下的空 CSP 假设: 无化学式、无性能名的话题级类比
        4. 同域类比: domain_a == domain_b 的 analogy 假设 (不是跨域迁移)
        5. 重复关键词: 关键词列表中同一词重复出现
        6. unknown/general 泄漏: CSP 抽取失败产物混入假设陈述
        7. 无意义数值预测: property_value=None 时用 1.0 硬算的随机数
        """
        violations: List[str] = []
        statement = hypo.statement
        statement_lower = statement.lower()
        hypo_type = hypo.hypothesis_type
        keywords = hypo.keywords or []

        import re as _re

        
        
        if hypo_type in ("analogy", "combination"):
            
            
            if "从" in statement and "迁移到" in statement:
                
                parts = statement.split("从")
                if len(parts) >= 2:
                    after_cong = parts[1]
                    if "迁移到" in after_cong:
                        domain_parts = after_cong.split("迁移到")
                        if len(domain_parts) >= 2:
                            src = domain_parts[0].strip().strip("，")
                            dst = domain_parts[1].split("，")[0].strip()
                            
                            if src and dst and src.lower() == dst.lower():
                                violations.append(
                                    f"自指假设: 迁移源和目标相同 ('{src}')"
                                )
                            
                            before_cong = parts[0].replace("将", "").strip()
                            if before_cong and src and before_cong.lower() == src.lower():
                                violations.append(
                                    f"自指假设: 方法名与迁移源域名相同 ('{src}')"
                                )
            
            self_ref_pattern = _re.compile(r'(.{2,20}?) 中的 \1', _re.IGNORECASE)
            if self_ref_pattern.search(statement):
                violations.append("自指假设: 方法/概念自引用")
            
            if len(keywords) >= 2:
                
                unique_kw = []
                for kw in keywords:
                    if kw.lower() not in [k.lower() for k in unique_kw]:
                        unique_kw.append(kw)
                if len(unique_kw) < len(keywords) * 0.5:
                    violations.append("关键词高度重复: 假设缺乏多样性")

        
        
        
        fragment_patterns = [
            r'\b(numerical|investigation|experimental|theoretical|comprehensive)\s+'
            r'(study|investigation|analysis|research|approach)\s+'
            r'(of|on|for|protective|enhanced|novel)\b',
            r'\b(preliminary|detailed|systematic)\s+'
            r'(study|investigation|analysis)\s+[a-z]{4,}\b',
        ]
        for pattern in fragment_patterns:
            if _re.search(pattern, statement_lower):
                violations.append("摘要碎片填槽: 检测到无意义标题片段")
                break

        
        
        
        if self.materials_mode:
            
            
            
            if _re.search(r'\(unknown\)', statement_lower) or\
               _re.search(r'unknown\)', statement_lower) or\
               _re.search(r'unknown的', statement_lower) or\
               _re.search(r'unknown结构', statement_lower):
                violations.append(
                    "CSP泄漏: 'unknown' 出现在假设陈述中 "
                    "(CSP抽取未识别晶体结构, 不应填入模板)"
                )
            
            
            if _re.search(r'的general[≈=]', statement_lower) or\
               _re.search(r'general≈', statement_lower) or\
               _re.search(r'general性能', statement_lower) or\
               _re.search(r'\bgeneral\b.*≈', statement_lower):
                violations.append(
                    "CSP泄漏: 'general' 作为性能名出现在陈述中 "
                    "(CSP抽取未识别具体性能, 不应填入模板)"
                )
            
            
            
            if hypo.metadata.get("fallback", False):
                meaningless_num = _re.search(
                    r'≈\s*(0\.\d{3,}|1\.\d{3,})\s*(?:；|,|。|$|\s)',
                    statement_lower,
                )
                if meaningless_num:
                    violations.append(
                        f"无意义数值预测: ≈{meaningless_num.group(1)} "
                        "是 property_value=None 时用默认值1.0硬算的随机数, 不具可证伪性"
                    )

        
        
        
        
        he_pattern = _re.compile(r'([\w]{2,20})和\1')
        if he_pattern.search(statement):
            violations.append("自指假设: 'X和X' 模式 — 相同实体并列")

        
        
        if hypo_type == "analogy":
            
            cross_domain = hypo.metadata.get("cross_domain", False)
            if not cross_domain:
                
                
                if "迁移到" in statement and not cross_domain:
                    violations.append(
                        "伪跨域: 声称跨域迁移但两篇论文属于同一领域"
                    )

        
        if len(statement.strip()) < 20:
            violations.append("陈述过短: 缺乏科学内容")

        return violations

    def _fallback_trial(self, hypo: Hypothesis) -> TrialResult:
        """无孪生时的降级处理 (启发式评分).

        修复: 原版评分过于宽松 (base=0.4 + bonus=0.35 = 0.75, 几乎必过)
        新版: 降低 base 分, 增加内容质量扣分, 让低质量假设被拒绝
        二次修复: base 从 0.3→0.25, fallback 扣分从 -0.10→-0.20,
                 确保 fallback 假设 (CSP 抽取失败) 被拒绝而非压线通过
        """
        self._total_trials += 1

        
        score = 0.25

        
        if hypo.keywords and len(hypo.keywords) >= 2:
            score += 0.08

        
        if len(hypo.statement) > 30:
            score += 0.04

        
        if hypo.metadata.get("pair_similarity", 0) > 0.2:
            score += 0.04

        
        quality_violations = self._check_hypothesis_quality(hypo)
        if quality_violations:
            
            score -= 0.25 * len(quality_violations)
            score = max(0.0, score)

        
        if self.materials_mode:
            if hypo.metadata.get("has_numerical_prediction", False):
                score += 0.08  
            if hypo.metadata.get("fallback", False):
                score -= 0.20  
            if hypo.metadata.get("is_known_principle", False):
                score -= 0.15  
            
            if hypo.metadata.get("cross_domain", False):
                score += 0.04  

        passed = score >= self.config.pass_threshold
        if passed:
            self._total_passed += 1
            failure_reason = ""
        else:
            self._total_failed += 1
            failure_reason = f"降级模式: 评分不足 ({score:.3f})"

        
        if quality_violations:
            if not passed:
                failure_reason = (
                    f"降级模式: 评分不足 ({score:.3f}) | "
                    f"质量违规: {'; '.join(quality_violations)}"
                )
            else:
                failure_reason = ""  
        else:
            if not passed:
                failure_reason = f"降级模式: 评分不足 ({score:.3f})"
            else:
                failure_reason = ""

        return TrialResult(
            hypothesis_id=hypo.hypothesis_id,
            statement=hypo.statement,
            hypothesis_type=hypo.hypothesis_type,
            source_pair=hypo.source_pair,
            paper_a_id=hypo.paper_a_id,
            paper_b_id=hypo.paper_b_id,
            paper_a_title=hypo.paper_a_title,
            paper_b_title=hypo.paper_b_title,
            keywords=hypo.keywords,
            passed=passed,
            score=round(score, 4),
            consistency=0.5,
            algo_predictions={"fallback": round(score, 3)},
            failure_reason=failure_reason,
            metadata=hypo.metadata,
            novelty=hypo.novelty,
            confidence=hypo.confidence,
            strategy_scores={"fallback": round(score, 3)},
        )

    def _make_fail_result(self, hypo: Hypothesis, reason: str) -> TrialResult:
        """生成失败结果."""
        self._total_failed += 1
        return TrialResult(
            hypothesis_id=hypo.hypothesis_id,
            statement=hypo.statement,
            hypothesis_type=hypo.hypothesis_type,
            source_pair=hypo.source_pair,
            paper_a_id=hypo.paper_a_id,
            paper_b_id=hypo.paper_b_id,
            paper_a_title=hypo.paper_a_title,
            paper_b_title=hypo.paper_b_title,
            keywords=hypo.keywords,
            passed=False,
            score=0.0,
            consistency=0.0,
            failure_reason=reason,
            metadata=hypo.metadata,
            novelty=hypo.novelty,
            confidence=hypo.confidence,
        )

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_trials": self._total_trials,
            "total_passed": self._total_passed,
            "total_failed": self._total_failed,
        }

    @property
    def ewc_info(self) -> Dict[str, Any]:
        """EWC 持续学习状态."""
        if self.twin and self.twin.cl and hasattr(self.twin.cl, "knowledge_base"):
            kb = self.twin.cl.knowledge_base
            return {
                "algorithm_weights": kb.get("algorithm_weights", {}),
                "calibration_rmse": kb.get("calibration_rmse", 0.0),
                "total_trials": kb.get("total_trials", 0),
                "total_passed": kb.get("total_passed", 0),
                "ewc_lambda": getattr(self.twin.cl, "ewc_lambda", 0.4),
                "sleep_cycles": getattr(self.twin.cl, "sleep_cycles", 0),
                "skill_count": getattr(self.twin.cl, "skill_count", 0),
            }
        return {}
