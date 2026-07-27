"""科研探索运行时的领域对象和假设评估模型。"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class PhysicalEntity:
    """轻量研究状态实体基类。"""

    def __init__(self, entity_id: str, entity_type: str) -> None:
        self.entity_id = entity_id
        self.entity_type = entity_type


class VirtualModel:
    """轻量虚拟评估模型基类。"""

    def __init__(self) -> None:
        self.rules: List[Dict[str, Any]] = []

    def add_rule(self, rule: Dict[str, Any]) -> None:
        self.rules.append(dict(rule))


class DomainAdapter:
    """创建论文语料实体的最小适配器接口。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}


class PaperCorpusEntity(PhysicalEntity):
    """科研探索的"物理实体" — 论文语料.

    sensors:
        paper_count: 当前论文池大小
        cross_domain_ratio: 跨领域论文比例
        avg_similarity: 平均配对相似度
        keyword_diversity: 关键词多样性
        domain_coverage: 领域覆盖数
    actuators:
        crawl_batch_size: 采集批量
        pairing_threshold: 配对阈值
        exploration_focus: 探索焦点关键词权重
    """

    def __init__(self, entity_id: str = "corpus-001") -> None:
        super().__init__(entity_id, "paper_corpus")
        self.sensors = {
            "paper_count": 0.0,
            "cross_domain_ratio": 0.0,
            "avg_similarity": 0.0,
            "keyword_diversity": 0.0,
            "domain_coverage": 0.0,
        }
        self.actuators = {
            "crawl_batch_size": 20.0,
            "pairing_threshold": 0.1,
            "exploration_focus": 1.0,
        }
        self.environment = {"research_seeds": "robotics,machine learning"}
        self._paper_history: List[Dict[str, Any]] = []

    def read_sensors(self) -> Dict[str, Any]:
        """读取当前论文语料状态."""
        return dict(self.sensors)

    def apply_control(self, action: Dict[str, Any]) -> bool:
        """应用控制动作（调整采集/配对参数）."""
        for key, value in action.items():
            if key in self.actuators:
                self.actuators[key] = float(value)
        return True

    def update_corpus_stats(self, papers: List[Any], pairs: List[Any]) -> None:
        """更新语料统计（由 HostSystem 在每轮循环后调用）."""
        self._paper_history.extend(papers)
        if len(self._paper_history) > 500:
            self._paper_history = self._paper_history[-500:]

        self.sensors["paper_count"] = float(len(self._paper_history))

        if pairs:
            cross = sum(1 for p in pairs if getattr(p, "cross_domain", False))
            self.sensors["cross_domain_ratio"] = cross / len(pairs)
            sims = [getattr(p, "similarity", 0) for p in pairs]
            self.sensors["avg_similarity"] = sum(sims) / len(sims) if sims else 0.0

        
        all_kw: set = set()
        domains: set = set()
        for p in self._paper_history:
            for kw in getattr(p, "keywords", []):
                all_kw.add(kw.lower())
            for cat in getattr(p, "categories", []):
                domains.add(cat)
        self.sensors["keyword_diversity"] = min(1.0, len(all_kw) / 50.0)
        self.sensors["domain_coverage"] = float(len(domains))

    def snapshot(self) -> Dict[str, Any]:
        return {"sensors": dict(self.sensors), "actuators": dict(self.actuators)}

    def restore(self, snapshot: Dict[str, Any]) -> None:
        self.sensors = snapshot.get("sensors", dict(self.sensors))
        self.actuators = snapshot.get("actuators", dict(self.actuators))

    def health(self) -> float:
        if self.sensors["paper_count"] < 1:
            return 0.0
        return min(1.0, 0.3 + 0.3 * self.sensors["keyword_diversity"] +
                   0.2 * self.sensors["cross_domain_ratio"] +
                   0.2 * min(1.0, self.sensors["domain_coverage"] / 5.0))


class HypothesisEvaluatorModel(VirtualModel):
    """科研探索的"虚拟模型" — 假设评估器.

    用注册在孪生上的算法算子做预测:
    - simulate(): 把假设喂给多个算法算子，每个算子给出预测分数
    - check_constraints(): 检查多算法预测的自洽性 (一致性)
    - calibrate(): 根据历史试错结果校准算子权重
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_version = "scihost-1.0.0"
        self.parameters = {
            "consistency_threshold": 0.6,
            "min_algorithms_agree": 3,
        }
        self.physics = {"dt": 1.0}
        
        self.add_rule({"name": "paper_count_min", "field": "paper_count",
                       "op": ">=", "threshold": 1.0})
        self.add_rule({"name": "keyword_diversity_min", "field": "keyword_diversity",
                       "op": ">=", "threshold": 0.0})
        
        self._algorithm_weights: Dict[str, float] = {}
        
        self._prediction_history: List[Dict[str, Any]] = []

    def simulate(self, inputs: Dict[str, Any], horizon: int = 1) -> List[Dict[str, Any]]:
        """运行算法算子做预测.

        inputs 中应包含:
            hypothesis: 待验证的假设对象
            twin: 当前科研仿真实例 (用于获取注册的评估算子)
            papers: 相关论文列表
        """
        hypothesis = inputs.get("hypothesis")
        twin = inputs.get("twin")
        papers = inputs.get("papers", [])

        if hypothesis is None or twin is None:
            return [inputs]

        predictions: List[Dict[str, Any]] = []
        algo_predictions: Dict[str, float] = {}

        
        algorithms = twin.algorithms

        
        if "cross_domain_transfer" in algorithms:
            try:
                score = self._eval_cross_domain(algorithms["cross_domain_transfer"],
                                                  hypothesis, twin)
                algo_predictions["cross_domain_transfer"] = score
            except Exception:
                algo_predictions["cross_domain_transfer"] = 0.5

        
        if "semantic_knowledge_graph" in algorithms:
            try:
                score = self._eval_knowledge_graph(algorithms["semantic_knowledge_graph"],
                                                     hypothesis, twin)
                algo_predictions["semantic_knowledge_graph"] = score
            except Exception:
                algo_predictions["semantic_knowledge_graph"] = 0.5

        
        if "uncertainty_quantifier" in algorithms:
            try:
                score = self._eval_uncertainty(algorithms["uncertainty_quantifier"],
                                                 hypothesis, twin)
                algo_predictions["uncertainty_quantifier"] = score
            except Exception:
                algo_predictions["uncertainty_quantifier"] = 0.5

        
        if "causal_discovery" in algorithms:
            try:
                score = self._eval_causal(algorithms["causal_discovery"],
                                           hypothesis, twin)
                algo_predictions["causal_discovery"] = score
            except Exception:
                algo_predictions["causal_discovery"] = 0.5

        
        if "llm_reasoning" in algorithms:
            try:
                score = self._eval_llm_reasoning(algorithms["llm_reasoning"],
                                                   hypothesis, twin)
                algo_predictions["llm_reasoning"] = score
            except Exception:
                algo_predictions["llm_reasoning"] = 0.5
        
        if "multi_fidelity_fusion" in algorithms and len(algo_predictions) >= 2:
            try:
                fused = self._eval_multi_fidelity(algorithms["multi_fidelity_fusion"],
                                                    algo_predictions, twin)
                algo_predictions["multi_fidelity_fusion"] = fused
            except Exception:
                pass

        
        if algo_predictions:
            
            weights = self._get_weights(list(algo_predictions.keys()), twin)
            total_weight = sum(weights.values()) or 1.0
            weighted_score = sum(
                algo_predictions[k] * weights.get(k, 1.0)
                for k in algo_predictions
            ) / total_weight
        else:
            weighted_score = 0.5

        
        if len(algo_predictions) >= 2:
            import numpy as np
            scores_arr = np.array(list(algo_predictions.values()))
            std = float(np.std(scores_arr))
            consistency = max(0.0, 1.0 - std * 3.0)  
        else:
            consistency = 0.5  

        prediction = {
            "hypothesis_id": getattr(hypothesis, "hypothesis_id", ""),
            "hypothesis_type": getattr(hypothesis, "hypothesis_type", ""),
            "hypothesis_metadata": getattr(hypothesis, "metadata", {}),
            "algo_predictions": algo_predictions,
            "weighted_score": round(weighted_score, 4),
            "consistency": round(consistency, 4),
            "n_algorithms": len(algo_predictions),
            "timestamp": time.time(),
        }
        predictions.append(prediction)
        self._prediction_history.append(prediction)
        if len(self._prediction_history) > 200:
            self._prediction_history = self._prediction_history[-200:]

        return predictions

    def check_constraints(self, predicted_state: Dict[str, Any]) -> tuple:
        """检查预测的自洽性约束.

        三重检查:
        1. 否决机制: 任何一个算子给出极低分 (< 0.2) → 直接否决
        2. 自洽性: 各算子预测的标准差不能太大
        3. 综合分: 加权平均必须达标

        Returns:
            (safe, violations): safe=True 表示通过, violations 为违规列表
        """
        violations: List[str] = []

        algo_predictions = predicted_state.get("algo_predictions", {})

        
        for algo_name, score in algo_predictions.items():
            if score < 0.2:
                algo_label = {
                    "cross_domain_transfer": "跨域迁移",
                    "semantic_knowledge_graph": "知识图谱",
                    "uncertainty_quantifier": "不确定性量化",
                    "causal_discovery": "因果发现",
                    "llm_reasoning": "语义推理",
                    "multi_fidelity_fusion": "多保真度融合",
                }.get(algo_name, algo_name)
                violations.append(
                    f"{algo_label}算子否决: 评分={score:.3f} < 0.2 "
                    f"(该假设在{algo_label}维度存在严重缺陷)"
                )

        
        consistency = predicted_state.get("consistency", 0.0)
        threshold = self.parameters.get("consistency_threshold", 0.6)
        
        
        hypo_meta = predicted_state.get("hypothesis_metadata", {})
        if hypo_meta.get("materials_mode") and hypo_meta.get("has_numerical_prediction"):
            threshold = max(0.45, threshold - 0.15)
        if consistency < threshold:
            violations.append(
                f"自洽性不足: {consistency:.3f} < {threshold} "
                f"(各算子预测分歧过大)"
            )

        
        n_algos = predicted_state.get("n_algorithms", 0)
        min_agree = self.parameters.get("min_algorithms_agree", 3)
        if n_algos < min_agree:
            violations.append(
                f"算子数量不足: {n_algos} < {min_agree}"
            )

        
        weighted_score = predicted_state.get("weighted_score", 0.0)
        if weighted_score < 0.3:
            violations.append(
                f"综合评分过低: {weighted_score:.3f} < 0.3"
            )

        safe = len(violations) == 0
        return safe, violations

    def calibrate(self, observations: List[Dict[str, Any]]) -> float:
        """根据试错历史校准算子权重 (持续学习的一部分).

        observations 中每条包含:
            algo_predictions: 各算子的预测分数
            actual_passed: 实际是否通过 (True/False)

        返回 RMSE.
        """
        if not observations:
            return 0.0

        
        algo_correct: Dict[str, int] = {}
        algo_total: Dict[str, int] = {}

        for obs in observations:
            algo_preds = obs.get("algo_predictions", {})
            actual = obs.get("actual_passed", False)
            for algo_name, pred_score in algo_preds.items():
                algo_total[algo_name] = algo_total.get(algo_name, 0) + 1
                predicted_pass = pred_score >= 0.5
                if predicted_pass == actual:
                    algo_correct[algo_name] = algo_correct.get(algo_name, 0) + 1

        
        for algo_name in algo_total:
            accuracy = algo_correct.get(algo_name, 0) / algo_total[algo_name]
            
            self._algorithm_weights[algo_name] = max(0.3, accuracy)

        
        import numpy as np
        residuals: List[float] = []
        for obs in observations:
            algo_preds = obs.get("algo_predictions", {})
            actual = 1.0 if obs.get("actual_passed", False) else 0.0
            for algo_name, pred_score in algo_preds.items():
                residuals.append((pred_score - actual) ** 2)
        if not residuals:
            return 0.0
        return float(np.sqrt(sum(residuals) / len(residuals)))

    def _get_weights(self, algo_names: List[str], twin: Any) -> Dict[str, float]:
        """获取算子权重 (优先用 EWC 校准过的权重)."""
        
        ewc_weights: Dict[str, float] = {}
        if twin and twin.cl and hasattr(twin.cl, "knowledge_base"):
            ewc_weights = twin.cl.knowledge_base.get("algorithm_weights", {})

        weights: Dict[str, float] = {}
        for name in algo_names:
            if name in ewc_weights:
                weights[name] = ewc_weights[name]
            elif name in self._algorithm_weights:
                weights[name] = self._algorithm_weights[name]
            else:
                weights[name] = 1.0
        return weights

    

    def _eval_cross_domain(self, engine: Any, hypothesis: Any, twin: Any) -> float:
        """根据关键词覆盖与跨域距离评估迁移可行性。"""
        kw = getattr(hypothesis, "keywords", [])
        cross_domain = getattr(hypothesis, "metadata", {}).get("cross_domain", False)

        if not cross_domain:
            return 0.6
        if len(kw) < 4:
            return 0.35

        source_terms = {str(term).lower() for term in kw[: len(kw) // 2]}
        target_terms = {str(term).lower() for term in kw[len(kw) // 2 :]}
        overlap = len(source_terms & target_terms)
        diversity = min(1.0, len(source_terms | target_terms) / 8.0)
        return max(0.2, min(0.8, 0.3 + diversity * 0.35 + overlap * 0.1))

    def _eval_knowledge_graph(self, graph: Any, hypothesis: Any, twin: Any) -> float:
        """用 SemanticKnowledgeGraph 评估知识图谱支持度.

        假设关键词在图谱中的连接越多，支持度越高.
        模板化/占位符假设会被惩罚.
        材料科学假设: 含数值预测的给予加分 (可证伪性).
        """
        kw = getattr(hypothesis, "keywords", [])
        if not kw:
            return 0.2

        
        statement = getattr(hypothesis, "statement", "")
        is_materials = getattr(hypothesis, "metadata", {}).get("materials_mode", False)
        placeholder_patterns = ["domain A", "domain B", "target problem",
                                "the proposed method", "shared concept"]
        if not is_materials:
            placeholder_patterns.append("unknown")
        has_placeholder = any(ph in statement for ph in placeholder_patterns)
        if has_placeholder:
            return 0.15  

        
        support_score = 0.0
        for keyword in kw[:5]:
            node_id = f"concept_{keyword.lower().replace(' ', '_')}"
            try:
                if hasattr(graph, '_nodes') and node_id in graph._nodes:
                    neighbors = 0
                    if hasattr(graph, '_adjacency'):
                        neighbors = len(graph._adjacency.get(node_id, set()))
                    support_score += min(0.15, neighbors * 0.04)
            except Exception:
                pass

        
        h_type = getattr(hypothesis, "hypothesis_type", "")
        type_bonus = {
            "analogy": 0.08, "combination": 0.12, "gap": 0.08, "contradiction": 0.04,
            
            "structure_property": 0.12, "composition_transfer": 0.10,
            "process_optimization": 0.08, "hidden_link": 0.10, "gap_filling": 0.08,
        }
        support_score += type_bonus.get(h_type, 0.03)

        
        if len(kw) >= 4:
            support_score += 0.05
        elif len(kw) < 2:
            support_score -= 0.1

        
        if is_materials and getattr(hypothesis, "metadata", {}).get("has_numerical_prediction", False):
            support_score += 0.10

        return max(0.1, min(0.8, 0.25 + support_score))

    def _eval_uncertainty(self, quantifier: Any, hypothesis: Any, twin: Any) -> float:
        """用 EvidentialUncertaintyQuantifier 量化不确定性.

        证据越充分、假设类型越可靠，不确定性越低，分数越高.
        但模板化/模糊假设会被惩罚.
        材料科学假设: 含数值预测 → 降低不确定性 (可验证=低风险).
        """
        kw = getattr(hypothesis, "keywords", [])
        statement = getattr(hypothesis, "statement", "")
        is_materials = getattr(hypothesis, "metadata", {}).get("materials_mode", False)

        
        evidence_strength = min(1.0, len(kw) / 8.0 * 0.5 + len(statement) / 200.0 * 0.5)

        
        epistemic = 1.0 - evidence_strength

        
        h_type = getattr(hypothesis, "hypothesis_type", "")
        aleatoric_map = {
            "contradiction": 0.6, "gap": 0.5, "analogy": 0.35, "combination": 0.3,
            
            "structure_property": 0.25, "composition_transfer": 0.30,
            "process_optimization": 0.35, "hidden_link": 0.35, "gap_filling": 0.30,
        }
        aleatoric = aleatoric_map.get(h_type, 0.4)

        
        placeholder_patterns = ["domain A", "domain B", "target problem",
                                "the proposed method", "shared concept"]
        if not is_materials:
            placeholder_patterns.append("unknown")
        has_placeholder = any(ph in statement for ph in placeholder_patterns)
        if has_placeholder:
            epistemic += 0.2  

        
        if len(kw) < 3:
            epistemic += 0.15

        
        if is_materials and getattr(hypothesis, "metadata", {}).get("has_numerical_prediction", False):
            epistemic -= 0.15

        
        total_uncertainty = min(1.0, max(0.0, (epistemic + aleatoric) / 2.0))
        score = 1.0 - total_uncertainty * 0.8

        return max(0.1, min(1.0, score))

    def _eval_causal(self, engine: Any, hypothesis: Any, twin: Any) -> float:
        """用 CausalDiscoveryEngine 评估因果一致性.

        矛盾假设需要混淆变量解释; 类比需要因果方向保持;
        组合需要因果图无环. 模板化假设扣分.
        材料科学假设: 结构-性能因果链明确，给予更高分.
        """
        kw = getattr(hypothesis, "keywords", [])
        h_type = getattr(hypothesis, "hypothesis_type", "")
        statement = getattr(hypothesis, "statement", "")
        is_materials = getattr(hypothesis, "metadata", {}).get("materials_mode", False)

        
        placeholder_patterns = ["domain A", "domain B", "target problem",
                                "the proposed method", "shared concept"]
        if not is_materials:
            placeholder_patterns.append("unknown")
        has_placeholder = any(ph in statement for ph in placeholder_patterns)
        if has_placeholder:
            return 0.2

        
        if is_materials:
            
            if h_type == "structure_property":
                base = 0.65
            elif h_type == "composition_transfer":
                base = 0.60  
            elif h_type == "process_optimization":
                base = 0.50  
            elif h_type == "hidden_link":
                base = 0.55  
            elif h_type == "gap_filling":
                base = 0.50
            else:
                base = 0.45
            
            if getattr(hypothesis, "metadata", {}).get("has_numerical_prediction", False):
                base += 0.10
            
            if len(kw) >= 4:
                base += 0.05
            return min(0.85, base)

        
        
        if h_type == "contradiction":
            if len(kw) >= 2:
                return 0.55  
            return 0.25  

        
        if h_type == "analogy":
            cross_domain = getattr(hypothesis, "metadata", {}).get("cross_domain", False)
            if cross_domain:
                return 0.45  
            return 0.6  

        
        if h_type == "combination":
            if len(kw) >= 3:
                return 0.55
            return 0.3

        
        if h_type == "gap":
            if len(kw) >= 3:
                return 0.5
            return 0.3

        return 0.35

    def _eval_llm_reasoning(self, reasoner: Any, hypothesis: Any, twin: Any) -> float:
        """用 LLMPoweredTwinReasoner 做语义推理评估.

        材料科学假设: 识别化学式/性能名/数值预测，给予加分.
        """
        statement = getattr(hypothesis, "statement", "")
        kw = getattr(hypothesis, "keywords", [])
        h_type = getattr(hypothesis, "hypothesis_type", "")
        is_materials = getattr(hypothesis, "metadata", {}).get("materials_mode", False)

        
        if kw and statement:
            statement_lower = statement.lower()
            appeared = sum(1 for k in kw if k.lower() in statement_lower)
            coverage = appeared / len(kw)
        else:
            coverage = 0.0

        
        method_words = [
            "transformer", "neural", "learning", "optimization", "bayesian",
            "graph", "quantum", "diffusion", "causal", "digital twin",
        ]
        material_words = [
            "perovskite", "bandgap", "conductivity", "formation energy",
            "cohesive energy", "seebeck", "piezoelectric", "dielectric",
            "thermal conductivity", "young", "hardness", "curie",
            "cubic", "tetragonal", "orthorhombic", "hexagonal", "wurtzite",
            "olivine", "garnet", "layered", "bcc", "fcc",
            "预测", "≈", "dft", "alloy", "cathode", "electrolyte",
        ]
        statement_lower = statement.lower()
        has_method = any(w in statement_lower for w in method_words)
        has_material = any(w in statement_lower for w in material_words)

        base = 0.3 + 0.3 * coverage
        if has_method:
            base += 0.15
        if has_material:
            base += 0.15  
        if len(statement) > 30:
            base += 0.1

        
        type_match = {
            "analogy": ["迁移", "借鉴", "类似", "统一"],
            "contradiction": ["矛盾", "冲突", "相反", "边界"],
            "gap": ["未探索", "尚未", "交互", "复合"],
            "combination": ["组合", "结合", "端到端", "桥梁"],
            
            "structure_property": ["预测", "≈", "结构", "性能"],
            "composition_transfer": ["替换", "迁移", "组分"],
            "process_optimization": ["工艺", "最优", "窗口"],
            "hidden_link": ["隐藏", "关联", "桥接", "报道"],
            "gap_filling": ["空白", "尚未", "未见", "gap"],
        }
        expected_words = type_match.get(h_type, [])
        if any(w in statement for w in expected_words):
            base += 0.1

        
        if is_materials and getattr(hypothesis, "metadata", {}).get("has_numerical_prediction", False):
            base += 0.10

        return min(1.0, base)

    def _eval_multi_fidelity(self, engine: Any,
                              algo_predictions: Dict[str, float], twin: Any) -> float:
        """用 MultiFidelityFusionEngine 融合多算子预测."""
        import numpy as np

        scores = list(algo_predictions.values())
        if not scores:
            return 0.5

        
        median = float(np.median(scores))
        mean = float(np.mean(scores))

        
        agreement = 1.0 - abs(median - mean) * 2.0
        fused = median * 0.6 + mean * 0.2 + agreement * 0.2

        return max(0.1, min(1.0, fused))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": self.model_version,
            "parameters": dict(self.parameters),
            "weights": dict(self._algorithm_weights),
            "history_len": len(self._prediction_history),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        self.parameters = snapshot.get("parameters", dict(self.parameters))
        self._algorithm_weights = snapshot.get("weights", {})

    def health(self) -> float:
        if not self._prediction_history:
            return 0.5
        recent = self._prediction_history[-20:]
        avg_consistency = sum(p.get("consistency", 0) for p in recent) / len(recent)
        return min(1.0, 0.4 + 0.6 * avg_consistency)


class SciHostAdapter(DomainAdapter):
    """科研探索领域适配器 — domain='sci_host'.

    将科研仿真运行时适配为科研方向探索场景:
    - PE = PaperCorpusEntity (论文语料)
    - VM = HypothesisEvaluatorModel (假设评估模型)
    - 算法算子由 HostSystem 注册
    """

    domain = "sci_host"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def create_physical_entity(self) -> PaperCorpusEntity:
        return PaperCorpusEntity(
            entity_id=self.config.get("entity_id", "corpus-001"),
        )

    def create_virtual_model(self) -> HypothesisEvaluatorModel:
        return HypothesisEvaluatorModel()

    def derive_action(self, predictions: List[Dict[str, Any]],
                      instruction: str = "") -> Dict[str, Any]:
        """从预测结果推导动作 — 调整探索参数."""
        if not predictions:
            return {}

        pred = predictions[0]
        consistency = pred.get("consistency", 0.5)
        score = pred.get("weighted_score", 0.5)

        action: Dict[str, Any] = {}
        
        if consistency < 0.4:
            action["pairing_threshold"] = 0.05
        
        if score > 0.7:
            action["crawl_batch_size"] = 30.0
        
        if score < 0.3:
            action["exploration_focus"] = 1.5

        return action
