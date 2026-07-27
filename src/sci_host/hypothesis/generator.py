"""假设生成器 — 从论文配对中生成科学假设.

核心思想:
    从隐性配对中发现的两篇论文，推导出可验证的科学假设。
    不同配对类型生成不同类型的假设:

    1. cross_domain_bridge → 类比假设
       "A 领域的方法 X 在 B 领域可能有类似效果"
    2. method_transfer → 迁移假设
       "方法 X 从 A 迁移到 B 可解决 B 中的问题 Y"
    3. concept_complement → 组合假设
       "A 的概念 α 与 B 的概念 β 组合可产生新能力 γ"
    4. contradiction → 矛盾假设
       "A 的结论与 B 的结论矛盾，可能存在隐藏变量"
    5. direct_similarity → 空白假设
       "A 和 B 都研究 X，但都没探索方向 Z"

与原项目 LLMPoweredTwinReasoner 的关系:
    原项目: 基于 6D 状态的 CoT 推理（模板化）
    本系统: 基于论文配对的假设生成（模板 + 随机探索）
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config import HypothesisConfig, ResearchQualityConfig
from ..pairing.implicit_pairer import PaperPair
from ..research_quality import EVIDENCE_TERMS, ResearchQualityGate
from ..materials import (
    CSPExtractor, CSPTriple, MaterialHypothesisTemplates, MaterialEntity,
    MaterialPhysicsValidator,
)
from ..materials.llm_hypothesis_gen import get_hypothesis_generator as _get_llm_hypo_gen


_TITLE_PREFIXES = {
    "a novel", "towards", "enhanced", "adaptive", "scalable",
    "robust", "efficient", "unified", "deep", "progressive",
}


def _strip_title_prefix(title: str) -> str:
    """去除离线语料添加的前缀（如 'A Novel', 'Adaptive' 等）."""
    if not title:
        return title
    lower = title.lower()
    for prefix in _TITLE_PREFIXES:
        if lower.startswith(prefix + " "):
            return title[len(prefix) + 1:]
    return title


def _category_to_domain(category: str) -> str:
    """将 arXiv 分类代码转换为可读的领域名称."""
    if not category or category == "unknown":
        return "the target domain"
    mapping = {
        "cs.AI": "artificial intelligence",
        "cs.LG": "machine learning",
        "cs.RO": "robotics",
        "cs.CL": "natural language processing",
        "cs.CV": "computer vision",
        "cs.MA": "multi-agent systems",
        "cs.GT": "game theory",
        "cs.CR": "cryptography and security",
        "cs.DB": "databases",
        "cs.IR": "information retrieval",
        "cs.DC": "distributed computing",
        "cs.NI": "networking",
        "cs.NA": "numerical analysis",
        "stat.ML": "statistical machine learning",
        "stat.ME": "statistical methodology",
        "q-bio.QM": "quantitative biology",
        "q-bio.NC": "computational neuroscience",
        "physics": "physics",
        "physics.chem-ph": "chemical physics",
        "physics.flu-dyn": "fluid dynamics",
        "physics.comp-ph": "computational physics",
        "quant-ph": "quantum computing",
        "cond-mat.mtrl-sci": "materials science",
        "eess.SP": "signal processing",
    }
    return mapping.get(category, category)


@dataclass
class Hypothesis:
    """科学假设."""
    hypothesis_id: str
    statement: str
    hypothesis_type: str
    source_pair: str
    paper_a_id: str
    paper_b_id: str
    paper_a_title: str
    paper_b_title: str
    rationale: str
    testable: bool = True
    novelty: float = 0.0
    confidence: float = 0.0
    keywords: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HypothesisGenerator:
    """假设生成器.

    从论文配对中自动生成可验证的科学假设。
    使用模板库 + 随机探索，模拟"试错"的科学发现过程。

    与原项目 Sleep Cycle 的区别:
        原项目: 梦境合成（反事实/噩梦/回声）在睡眠期生成经验
        本系统: 假设生成在"清醒"期持续进行，不需要睡眠
    """

    TEMPLATES: Dict[str, List[str]] = {
        "analogy": [
            "将 {method_a} 从 {domain_a} 迁移到 {domain_b}，预期可在 {problem_b} 上获得类似效果",
            "借鉴 {domain_a} 中 {method_a} 的思想，在 {domain_b} 中设计类似的 {concept_b} 机制",
            "{domain_a} 中的 {method_a} 可能与 {domain_b} 中的 {method_b} 共享底层原理，"
            "统一框架可能同时改善两个领域",
        ],
        "contradiction": [
            "《{title_a}》的结论与《{title_b}》的方法存在潜在矛盾，"
            "可能存在未被发现的调节变量 '{bridge}'",
            "当 '{bridge}' 同时出现在两个场景中时，{domain_a} 和 {domain_b} 的预测可能相反，"
            "需探索边界条件",
        ],
        "gap": [
            "《{title_a}》和《{title_b}》都涉及 '{bridge}'，但均未探索 "
            "'{keyword_a}' 与 '{keyword_b}' 的交互效应",
            "两篇论文在 '{bridge}' 上取得进展，但将 {method_a} 与 {method_b} "
            "结合的复合方案尚未被研究",
        ],
        "combination": [
            "将 {method_a} 与 {method_b} 组合，"
            "利用 '{bridge}' 作为桥梁，可能产生优于单一方法的效果",
            "以 {concept_a} 为前端提取特征，以 {concept_b} 为后端做决策，"
            "端到端组合可能在 {domain_b} 上取得突破",
        ],
    }

    def __init__(
        self,
        config: HypothesisConfig,
        materials_mode: bool = False,
        domain_keywords: Optional[List[str]] = None,
        quality_config: Optional[ResearchQualityConfig] = None,
    ) -> None:
        self.config = config
        self.materials_mode = materials_mode
        self._quality_gate = ResearchQualityGate(quality_config)
        self._generated_count: int = 0
        self._csp_knowledge: Dict[str, CSPTriple] = {}
        self._domain_keywords: Set[str] = set()
        if domain_keywords:
            for kw in domain_keywords:
                for token in kw.lower().split():
                    if len(token) >= 3:
                        self._domain_keywords.add(token)
        self._llm_gen = _get_llm_hypo_gen()
        self._llm_success_count: int = 0
        self._llm_fallback_count: int = 0
        self._domain_context: str = ""
        if domain_keywords:
            self._domain_context = f"Research focus: {', '.join(domain_keywords[:5])}"

    def _is_domain_relevant(self, pair: PaperPair) -> bool:
        """检查配对中的论文是否与目标研究领域相关.

        如果未设置 domain_keywords, 返回 True (不做过滤).
        否则要求至少一篇论文的标题/关键词/摘要中包含至少 1 个领域词.
        """
        if not self._domain_keywords:
            return True

        for side in ("a", "b"):
            title = getattr(pair, f"paper_{side}_title", "") or ""
            kws = getattr(pair, f"paper_{side}_keywords", []) or []
            abstract = getattr(pair, f"paper_{side}_abstract", "") or ""
            text = f"{title} {' '.join(kws)} {abstract}".lower()
            if any(dw in text for dw in self._domain_keywords):
                return True

        return False

    def generate(self, pairs: List[PaperPair]) -> List[Hypothesis]:
        """从配对列表生成假设."""
        if not pairs:
            return []

        if self.materials_mode:
            return self._generate_materials(pairs)
        return self._generate_general(pairs)

    def _generate_general(self, pairs: List[PaperPair]) -> List[Hypothesis]:
        """通用模式: 从论文配对中生成假设.

        优先使用 LLM 因果推理生成; LLM 不可用或失败时降级为模板生成.
        """
        hypotheses: List[Hypothesis] = []
        rng = random.Random(int(time.time()) % 2**31)

        for pair in pairs:
            if len(hypotheses) >= self.config.max_per_round:
                break

            if not self._is_domain_relevant(pair):
                continue
            if self._quality_gate.enabled and not self._pair_has_quality_signal(pair):
                continue

            if pair.similarity < self.config.min_pair_similarity:
                continue

            if self._llm_gen.is_available:
                llm_hypo = self._try_llm_generate(pair, None, None)
                if llm_hypo is not None:
                    hypotheses.append(llm_hypo)
                    continue
                self._llm_fallback_count += 1

            hypo = self._template_generate_one(pair, rng)
            if hypo is not None:
                hypotheses.append(hypo)

        return hypotheses

    def _template_generate_one(
        self, pair: PaperPair, rng: random.Random,
    ) -> Optional[Hypothesis]:
        """模板填空式生成单个假设 (原有逻辑, 降级用)."""
        hypo_type = self._map_pair_type_to_hypo(pair.pair_type)
        if hypo_type is None:
            return None

        templates = self.TEMPLATES.get(hypo_type, [])
        if not templates:
            return None

        min_falsifiability = (
            self._quality_gate.config.min_falsifiability_score
            if self._quality_gate.enabled else 0.3
        )
        shuffled_templates = list(templates)
        rng.shuffle(shuffled_templates)
        statement = ""
        for template in shuffled_templates:
            candidate = self._fill_template(template, pair)
            if self._validate_statement(candidate, pair, hypo_type):
                continue
            candidate_score = self._falsifiability_score(candidate, pair, hypo_type)
            if candidate_score < min_falsifiability:
                continue
            statement = candidate
            falsifiability = candidate_score
            break
        if not statement:
            return None

        self._generated_count += 1
        raw_id = f"{pair.pair_id}{self._generated_count}"
        hypo_id = f"hypo_{hashlib.md5(raw_id.encode()).hexdigest()[:8]}"

        novelty = self._estimate_novelty(pair, hypo_type)
        confidence = self._estimate_confidence(pair, hypo_type)

        return Hypothesis(
            hypothesis_id=hypo_id,
            statement=statement,
            hypothesis_type=hypo_type,
            source_pair=pair.pair_id,
            paper_a_id=pair.paper_a_id,
            paper_b_id=pair.paper_b_id,
            paper_a_title=pair.paper_a_title,
            paper_b_title=pair.paper_b_title,
            rationale=pair.connection_reason,
            testable=True,
            novelty=round(novelty, 3),
            confidence=round(confidence, 3),
            keywords=self._build_keywords(pair),
            metadata={
                "pair_similarity": pair.similarity,
                "pair_type": pair.pair_type,
                "cross_domain": pair.cross_domain,
                "generation_method": "template",
            },
        )

    def _pair_has_quality_signal(self, pair: PaperPair) -> bool:
        """生成前再次确认两篇论文仍满足专项质量门槛."""
        if not self._quality_gate.enabled:
            return True
        for side in ("a", "b"):
            title = getattr(pair, f"paper_{side}_title", "") or ""
            abstract = getattr(pair, f"paper_{side}_abstract", "") or ""
            keywords = getattr(pair, f"paper_{side}_keywords", []) or []
            if not self._quality_gate.assess_text(title, abstract, keywords).accepted:
                return False
        return True

    def _try_llm_generate(
        self,
        pair: PaperPair,
        csp_a: Optional[List[CSPTriple]],
        csp_b: Optional[List[CSPTriple]],
    ) -> Optional[Hypothesis]:
        """尝试用 LLM 生成假设, 返回 Hypothesis 对象或 None.

        LLM 生成的假设具有:
        - 因果推理链 (causal_chain)
        - 可证伪性判据 (falsification_criterion)
        - 具体预测变量和方向 (predicted_variable, predicted_direction)
        """
        llm_result = self._llm_gen.generate_hypothesis(
            pair, csp_a, csp_b, self._domain_context,
        )
        if llm_result is None:
            return None

        statement = llm_result.get("statement", "")
        hypo_type = llm_result.get("hypothesis_type", "unknown")
        if self._validate_statement(statement, pair, hypo_type):
            return None
        falsifiability = self._falsifiability_score(
            statement, pair, hypo_type,
        )
        min_falsifiability = (
            self._quality_gate.config.min_falsifiability_score
            if self._quality_gate.enabled else 0.3
        )
        if falsifiability < min_falsifiability:
            return None

        self._generated_count += 1
        self._llm_success_count += 1
        raw_id = f"{pair.pair_id}{self._generated_count}"
        hypo_id = f"hypo_{hashlib.md5(raw_id.encode()).hexdigest()[:8]}"

        confidence = llm_result.get("confidence", 0.5)
        novelty = self._estimate_novelty(pair, hypo_type)
        novelty = min(1.0, novelty + 0.1)

        keywords = self._build_keywords(pair)
        predicted_var = llm_result.get("predicted_variable", "")
        if predicted_var and predicted_var not in keywords:
            keywords.insert(0, predicted_var)

        rationale = llm_result.get("rationale", "")
        if not rationale:
            rationale = pair.connection_reason

        return Hypothesis(
            hypothesis_id=hypo_id,
            statement=statement,
            hypothesis_type=hypo_type,
            source_pair=pair.pair_id,
            paper_a_id=pair.paper_a_id,
            paper_b_id=pair.paper_b_id,
            paper_a_title=pair.paper_a_title,
            paper_b_title=pair.paper_b_title,
            rationale=rationale,
            testable=True,
            novelty=round(novelty, 3),
            confidence=round(confidence, 3),
            keywords=keywords[:10],
            metadata={
                "pair_similarity": pair.similarity,
                "pair_type": pair.pair_type,
                "cross_domain": pair.cross_domain,
                "generation_method": "llm_causal",
                "causal_chain": llm_result.get("causal_chain", []),
                "predicted_variable": llm_result.get("predicted_variable", ""),
                "predicted_direction": llm_result.get("predicted_direction", ""),
                "predicted_magnitude": llm_result.get("predicted_magnitude"),
                "comparison_baseline": llm_result.get("comparison_baseline", ""),
                "falsification_criterion": llm_result.get("falsification_criterion", ""),
                "llm_falsifiability": round(falsifiability, 3),
            },
        )

    DOMAIN_VAGUE_TERMS: Set[str] = {
        "joint", "joints", "stiffness", "torque", "force", "actuator",
        "actuation", "transmission", "gear", "gears", "bearing",
        "linkage", "mechanism", "spring", "motor", "servo",
        "gripper", "end-effector", "kinematic", "dynamic",
        "compliance", "damping", "friction", "backlash",
        "network", "neural", "model", "learning", "training",
        "gradient", "embedding", "layer", "encoder", "decoder",
        "alloy", "composite", "ceramic", "polymer", "crystal",
        "thin film", "nanostructure", "phase",
        "spectroscopy", "microscopy", "diffraction", "characterization",
        "ferroelectric", "piezoelectric", "thermoelectric",
        "superconducting", "magnetic", "dielectric",
        "bandgap", "conductivity", "crystals", "films",
        "continuum", "topological", "anomalous", "disorder",
        "charge", "electron", "spin", "lattice", "phonon",
        "energy", "power", "field", "wave", "interface",
        "contacts", "surface", "boundary", "waals",
        "waveguide", "resonator", "cavity", "antiphase",
        "spinel", "perovskite", "emergence", "twist",
    }

    def _falsifiability_score(
        self, statement: str, pair: PaperPair, hypo_type: str,
    ) -> float:
        """评估假设的可证伪性 — 能否被实验检验.

        可证伪的假设需要:
        1. 包含具体的技术方法/机制名称 (不是 "approach" "method")
        2. 包含可观测的预测 (性能指标、效果指标)
        3. 方法名与领域名不同 (避免 "把 robot 从 robotics 迁移到 robotics")
        4. 不应是领域泛词的机械拼接 (如 "将 joint 从 robotics 迁移到 stiffness")

        返回 [0, 1] 的分数, 越高越可证伪.
        """
        score = 0.0
        statement_lower = statement.lower()

        technical_terms = self._extract_technical_terms(statement)
        if technical_terms:
            substantive_terms = [
                t for t in technical_terms
                if t not in self.DOMAIN_VAGUE_TERMS
            ]
            if substantive_terms:
                score += 0.3
                if len(substantive_terms) >= 2:
                    score += 0.1
            elif technical_terms:
                score += 0.05

        observable_indicators = [
            "性能", "效果", "效率", "精度", "稳定性", "强度",
            "扭矩", "刚度", "阻尼", "寿命", "成本", "重量",
            "bandgap", "conductivity", "piezoelectric", "thermoelectric",
            "hardness", "modulus", "fatigue", "wear",
            "≈", "预测", "expected", "improve",
        ]
        if any(ind in statement_lower for ind in observable_indicators):
            score += 0.3

        import re as _re
        if _re.search(r'≈\s*\d', statement):
            score += 0.2

        template_filler = [
            "预期可在", "可能产生", "可能存在", "可能共享",
            "可能具有", "可能有", "尚未被研究", "尚未探索",
        ]
        filler_count = sum(1 for f in template_filler if f in statement)
        score -= 0.1 * filler_count

        if len(statement) > 50:
            score += 0.1

        vague_in_statement = [
            t for t in self.DOMAIN_VAGUE_TERMS if t in statement_lower
        ]
        if len(vague_in_statement) >= 2:
            multi_word_phrases = _re.findall(
                r'[a-z]{3,}[- ][a-z]{3,}', statement_lower,
            )
            substantive_multi = [
                p for p in multi_word_phrases
                if not any(v in p for v in self.DOMAIN_VAGUE_TERMS)
            ]
            if not substantive_multi and not _re.search(r'≈\s*\d', statement):
                score -= 0.25

        return max(0.0, min(1.0, score))

    @staticmethod
    def _extract_technical_terms(text: str) -> List[str]:
        """从文本中提取有技术语义的术语 (过滤通用词)."""
        _generic = {
            "the", "a", "an", "for", "of", "to", "in", "on", "with",
            "and", "or", "via", "from", "based", "using", "through",
            "novel", "enhanced", "adaptive", "scalable", "robust",
            "efficient", "unified", "deep", "progressive", "towards",
            "approach", "method", "framework", "model", "system",
            "design", "performance", "analysis", "study", "research",
            "investigation", "experiment", "experimental", "simulation",
            "numerical", "theoretical", "comprehensive", "systematic",
            "preliminary", "detailed", "general", "proposed",
            "conference", "ieee", "international", "proceedings",
            "symposium", "journal", "transactions", "workshop",
            "robot", "robotics", "learning", "machine", "intelligence",
            "control", "optimization", "algorithm", "data", "result",
            "evaluation", "assessment", "comparison", "application",
            "implementation", "development", "measurement", "testing",
            "validation", "micro", "nano", "macro", "large", "small",
            "high", "low", "new", "recent", "advanced", "modern",
            "calibration", "将", "从", "迁移到", "借鉴", "中的",
            "组合", "利用", "作为", "前端", "后端", "端到端",
            "可能", "预期", "获得", "产生", "存在", "共享",
            "统一", "矛盾", "调节", "变量", "边界", "条件",
            "涉及", "交互", "效应", "结合", "方案",
            "joint", "joints", "stiffness", "torque", "force",
            "actuator", "actuation", "transmission", "gear", "gears",
            "bearing", "linkage", "spring", "motor", "servo",
            "gripper", "kinematic", "dynamic", "compliance",
            "damping", "friction", "backlash",
            "network", "neural", "layer", "encoder", "decoder",
            "alloy", "composite", "ceramic", "polymer", "crystal",
            "spectroscopy", "microscopy", "diffraction", "characterization",
            "ferroelectric", "piezoelectric", "thermoelectric",
            "superconducting", "magnetic", "dielectric",
            "bandgap", "conductivity", "crystals", "films",
            "continuum", "topological", "anomalous", "disorder",
            "charge", "electron", "spin", "lattice", "phonon",
            "energy", "power", "field", "wave", "interface",
            "contacts", "surface", "boundary", "waals",
            "waveguide", "resonator", "cavity", "antiphase",
            "spinel", "perovskite", "emergence", "twist",
            "using", "problem", "negative", "second", "first",
            "states", "phase", "phases", "transition", "properties",
            "property", "synthesis", "fabrication", "deposition",
            "growth", "annealing", "sintering", "doping",
        }
        words = []
        for w in text.split():
            w_clean = w.lower().strip(".,;:!?()[]\"'")
            if len(w_clean) >= 4 and w_clean not in _generic:
                if any('\u4e00' <= c <= '\u9fff' for c in w_clean):
                    if len(w_clean) >= 2:
                        words.append(w_clean)
                else:
                    words.append(w_clean)
        return words[:5]

    def _validate_statement(
        self, statement: str, pair: PaperPair, hypo_type: str,
    ) -> List[str]:
        """检查生成的假设陈述是否有质量问题.

        Returns:
            问题列表, 空=质量合格
        """
        issues: List[str] = []
        statement_lower = statement.lower()

        import re as _re

        if hypo_type in ("analogy", "combination"):
            if "从" in statement and "迁移到" in statement:
                parts = statement.split("从")
                if len(parts) >= 2:
                    before_cong = parts[0].replace("将", "").strip()
                    after_cong = parts[1]
                    if "迁移到" in after_cong:
                        domain_parts = after_cong.split("迁移到")
                        if len(domain_parts) >= 2:
                            src = domain_parts[0].strip().strip("，")
                            dst = domain_parts[1].split("，")[0].strip()
                            if src and dst and src.lower() == dst.lower():
                                issues.append("自指: 迁移源=目标")
                            if before_cong and src and before_cong.lower() == src.lower():
                                issues.append("自指: 方法名=迁移源域名")
                            if before_cong and dst and before_cong.lower() == dst.lower():
                                issues.append("自指: 方法名=迁移目标域名")
                            if src and dst and len(src) > 3 and len(dst) > 3:
                                if src.lower() in dst.lower() or dst.lower() in src.lower():
                                    issues.append("自指: 迁移源与目标高度相似")

            self_ref = _re.compile(r'(.{2,20}?) 中的 \\1', _re.IGNORECASE)
            if self_ref.search(statement):
                issues.append("自指: 概念自引用")

            if "与" in statement and "组合" in statement:
                parts = statement.split("与")
                if len(parts) >= 2:
                    before_yu = parts[0].split("将")[-1].strip()
                    after_yu = parts[1].split("组合")[0].strip()
                    if before_yu and after_yu:
                        bl = before_yu.lower()
                        al = after_yu.lower()
                        if bl == al:
                            issues.append("自指: 组合源=目标")
                        elif len(bl) > 3 and len(al) > 3:
                            if bl in al or al in bl:
                                issues.append("自指: 组合源与目标高度相似")
                        elif len(bl) >= 3 and len(al) >= 3:
                            common = 0
                            for c1, c2 in zip(bl, al):
                                if c1 == c2:
                                    common += 1
                                else:
                                    break
                            if common >= 4 and common >= min(len(bl), len(al)) - 2:
                                issues.append("自指: 组合源与目标前缀重叠")

            if "以" in statement and "为" in statement:
                _parts = statement.split("以")
                if len(_parts) >= 3:
                    first_seg = _parts[1].split("为")[0].strip() if "为" in _parts[1] else ""
                    second_seg = _parts[2].split("为")[0].strip() if "为" in _parts[2] else ""
                    if first_seg and second_seg and first_seg.lower() == second_seg.lower():
                        issues.append("自指: 组合源=目标")

        fragment_patterns = [
            r'\b(numerical|investigation|experimental|theoretical)\s+'
            r'(study|investigation|analysis|research)\s+'
            r'(of|on|for|protective|enhanced)\b',
        ]
        for pattern in fragment_patterns:
            if _re.search(pattern, statement_lower):
                issues.append("摘要碎片填槽")
                break

        if hypo_type == "analogy" and not pair.cross_domain:
            if "迁移到" in statement:
                issues.append("伪跨域类比")

        if len(statement.strip()) < 20:
            issues.append("陈述过短")

        if "将  从" in statement or "将 从" in statement:
            issues.append("空方法名")
        if "the proposed method" in statement_lower:
            issues.append("回退方法名")

        return issues

    def _map_pair_type_to_hypo(self, pair_type: str) -> Optional[str]:
        """配对类型 → 假设类型映射."""
        mapping = {
            "cross_domain_bridge": "analogy",
            "method_transfer": "analogy",
            "concept_complement": "combination",
            "contradiction": "contradiction",
            "direct_similarity": "gap",
        }
        return mapping.get(pair_type)

    def _fill_template(self, template: str, pair: PaperPair) -> str:
        """填充假设模板."""
        clean_title_a = _strip_title_prefix(pair.paper_a_title)
        clean_title_b = _strip_title_prefix(pair.paper_b_title)

        bridge = pair.bridge_keywords[0] if pair.bridge_keywords else "shared concept"
        keywords = pair.bridge_keywords[:3] if pair.bridge_keywords else ["research"]

        all_kw_a = pair.paper_a_keywords if pair.paper_a_keywords else []
        all_kw_b = pair.paper_b_keywords if pair.paper_b_keywords else []

        method_a = self._extract_method(clean_title_a, keywords, all_kw_a)
        method_b = self._extract_method(clean_title_b, keywords, all_kw_b)

        domain_a = _category_to_domain(pair.paper_a_category)
        domain_b = _category_to_domain(pair.paper_b_category)

        if domain_a == domain_b and "迁移到" in template:
            template = self.TEMPLATES["combination"][0]

        if method_a.lower() == method_b.lower() and "迁移到" in template:
            template = self.TEMPLATES["combination"][0]

        problem_b = keywords[-1] if len(keywords) > 1 else domain_b

        statement = template.format(
            method_a=method_a,
            method_b=method_b,
            concept_a=method_a,
            concept_b=method_b,
            domain_a=domain_a,
            domain_b=domain_b,
            problem_b=problem_b,
            title_a=clean_title_a,
            title_b=clean_title_b,
            bridge=bridge,
            keyword_a=keywords[0],
            keyword_b=keywords[-1],
        )

        if self._quality_gate.enabled:
            source_text = " ".join([
                statement,
                " ".join(all_kw_a),
                " ".join(all_kw_b),
                getattr(pair, "paper_a_abstract", "") or "",
                getattr(pair, "paper_b_abstract", "") or "",
            ])
            metrics = ResearchQualityGate.match_terms(source_text, EVIDENCE_TERMS)
            if metrics and not ResearchQualityGate.match_terms(statement, EVIDENCE_TERMS):
                statement += f"，以 {metrics[0]} 作为主要可观测指标"

        return statement

    @staticmethod
    def _extract_method(title: str, bridge_keywords: List[str],
                        paper_keywords: List[str]) -> str:
        """从标题和关键词中提取方法名.

        优先级:
        1. bridge_keywords 中出现在标题里的词 (过滤通用词)
        2. paper_keywords 中出现在标题里的词 (过滤通用词)
        3. 标题中的方法类关键词 (如 transformer, learning, optimization)
        4. 标题中最长的方法类词 (而非前3个实词, 避免摘要碎片)
        """
        _generic = {
            "novel", "enhanced", "adaptive", "scalable", "robust", "efficient",
            "unified", "deep", "progressive", "towards", "based",
            "approach", "method", "framework", "model", "system",
            "design", "performance", "analysis", "study", "research",
            "investigation", "experiment", "experimental", "simulation",
            "numerical", "theoretical", "comprehensive", "systematic",
            "preliminary", "detailed", "general", "proposed",
            "conference", "ieee", "international", "proceedings",
            "symposium", "journal", "transactions", "workshop",
            "robot", "robotics", "learning", "machine", "intelligence",
            "control", "optimization", "algorithm", "data", "result",
            "performance", "evaluation", "assessment", "comparison",
            "application", "implementation", "development",
            "measurement", "testing", "validation",
            "micro", "nano", "macro", "large", "small", "high", "low",
            "new", "recent", "advanced", "modern",
            "layered", "amorphous", "crystalline", "bulk", "thin",
            "nanoscale", "mesoporous", "porous", "dense",
            "single", "polycrystalline", "epitaxial", "powder",
            "calibration",
            "joint", "joints", "stiffness", "torque", "force",
            "actuator", "actuation", "transmission", "gear", "gears",
            "bearing", "linkage", "spring", "motor", "servo",
            "gripper", "kinematic", "dynamic", "compliance",
            "damping", "friction", "backlash",
            "network", "neural", "layer", "encoder", "decoder",
            "alloy", "composite", "ceramic", "polymer", "crystal",
            "spectroscopy", "microscopy", "diffraction", "characterization",
            "measurement", "imaging", "scattering", "spectrometry",
            "ferroelectric", "piezoelectric", "thermoelectric",
            "superconducting", "superconductor", "magnetic", "dielectric",
            "bandgap", "conductivity", "resistivity", "permittivity",
            "crystals", "films", "film", "substrate", "wafer",
            "bulk", "powder", "nanoparticles", "nanostructure",
            "using", "problem", "negative", "positive", "second",
            "first", "third", "emergence", "emerging", "novel",
            "continuum", "topological", "anomalous", "disorder",
            "disordered", "ordered", "states", "state", "phase",
            "phases", "transition", "transitions", "properties",
            "property", "synthesis", "fabrication", "deposition",
            "growth", "annealing", "sintering", "doping",
            "charge", "electron", "electrons", "hole", "holes",
            "spin", "spins", "lattice", "phonon", "phonons",
            "photon", "photons", "wave", "waves", "field", "fields",
            "energy", "power", "voltage", "current", "frequency",
            "temperature", "pressure", "density", "velocity",
            "contacts", "interface", "interfaces", "surface", "surfaces",
            "bulk", "edge", "edges", "boundary", "boundaries",
            "waals", "waveguide", "waveguides", "resonator", "resonators",
            "cavity", "cavities", "antiphase", "spinel", "perovskite",
            "harvesting", "twisted", "twist", "mechanisms", "mechanism",
        }

        title_lower = title.lower()

        for kw in bridge_keywords:
            kw_lower = kw.lower()
            if kw_lower in title_lower and kw_lower not in _generic and len(kw_lower) >= 4:
                return kw

        for kw in paper_keywords:
            kw_lower = kw.lower()
            if kw_lower in title_lower and kw_lower not in _generic and len(kw_lower) >= 4:
                return kw

        method_words = {
            "transformer", "neural", "network", "learning", "optimization",
            "gradient", "embedding", "attention", "diffusion", "gaussian",
            "bayesian", "graph", "federated", "quantum", "blockchain",
            "reinforcement", "evolutionary", "meta-learning", "transfer",
            "digital twin", "physics-informed", "causal", "counterfactual",
            "evidential", "knowledge graph", "game theory", "maml",
            "perovskite", "dft", "alloy", "composite", "catalyst",
            "sintering", "annealing", "deposition", "doping",
        }
        for mw in method_words:
            if mw in title_lower:
                return mw

        stop = _generic | {
            "the", "a", "an", "for", "of", "to", "in", "on", "with",
            "and", "or", "via", "from", "based", "using", "through",
            "study", "investigation", "analysis", "research", "approach",
            "novel", "enhanced", "adaptive", "towards",
        }
        words = [w for w in title.split()
                 if w.lower().strip(".,;:!?()[]") not in stop
                 and len(w.strip(".,;:!?()[]")) >= 4]
        return words[0].strip(".,;:!?()[]") if words else ""

    def _build_keywords(self, pair: PaperPair) -> List[str]:
        """构建假设关键词列表 — bridge + 论文关键词去重."""
        result: List[str] = []
        seen: set = set()
        for kw in pair.bridge_keywords + pair.paper_a_keywords + pair.paper_b_keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                result.append(kw)
        return result[:10] if result else ["research"]

    def _estimate_novelty(self, pair: PaperPair, hypo_type: str) -> float:
        """估计假设新颖性.

        跨领域 + 低相似度 = 高新颖性
        同领域 + 高相似度 = 低新颖性
        """
        base = 0.5
        if pair.cross_domain:
            base += 0.2
        if pair.similarity < 0.3:
            base += 0.15
        if hypo_type == "contradiction":
            base += 0.1
        if hypo_type == "combination":
            base += 0.05
        return min(1.0, base)

    def _estimate_confidence(self, pair: PaperPair, hypo_type: str) -> float:
        """估计初始置信度.

        高相似度 = 高置信度
        低相似度 = 低置信度（但更具新颖性）
        """
        base = pair.similarity * 0.5
        if hypo_type == "direct_similarity" or hypo_type == "gap":
            base += 0.1
        if pair.cross_domain:
            base -= 0.05
        return max(0.1, min(0.9, base + 0.3))

    @property
    def total_generated(self) -> int:
        return self._generated_count

    @property
    def llm_stats(self) -> Dict[str, Any]:
        """LLM 假设生成统计."""
        return {
            "llm_available": self._llm_gen.is_available,
            "llm_success": self._llm_success_count,
            "llm_fallback": self._llm_fallback_count,
            "llm_internal_stats": self._llm_gen.stats if self._llm_gen else {},
        }


    def _generate_materials(self, pairs: List[PaperPair]) -> List[Hypothesis]:
        """材料科学模式: 从论文配对 + CSP 抽取生成可证伪假设.

        流程:
        1. 从每对论文中抽取 CSP 三元组
        2. 更新 CSP 知识库
        3. 优先使用 LLM 因果推理生成假设; 失败则降级为模板生成
        """
        hypotheses: List[Hypothesis] = []
        rng = random.Random(int(time.time()) % 2**31)

        for pair in pairs:
            if len(hypotheses) >= self.config.max_per_round:
                break

            if pair.similarity < self.config.min_pair_similarity:
                continue

            if not self._is_domain_relevant(pair):
                continue
            if self._quality_gate.enabled and not self._pair_has_quality_signal(pair):
                continue

            text_a = f"{pair.paper_a_title}. {pair.paper_a_abstract}" if pair.paper_a_abstract else f"{pair.paper_a_title}. {' '.join(pair.paper_a_keywords[:8])}"
            text_b = f"{pair.paper_b_title}. {pair.paper_b_abstract}" if pair.paper_b_abstract else f"{pair.paper_b_title}. {' '.join(pair.paper_b_keywords[:8])}"

            csp_a = CSPExtractor.extract(
                text_a, paper_id=pair.paper_a_id, paper_title=pair.paper_a_title,
            )
            csp_b = CSPExtractor.extract(
                text_b, paper_id=pair.paper_b_id, paper_title=pair.paper_b_title,
            )

            for triple in csp_a + csp_b:
                if triple.key not in self._csp_knowledge:
                    self._csp_knowledge[triple.key] = triple

            if self._llm_gen.is_available:
                llm_hypo = self._try_llm_generate(pair, csp_a, csp_b)
                if llm_hypo is not None:
                    llm_hypo.metadata["materials_mode"] = True
                    llm_hypo.metadata["csp_count"] = len(csp_a) + len(csp_b)
                    if llm_hypo.metadata.get("predicted_magnitude") is not None:
                        llm_hypo.metadata["has_numerical_prediction"] = True
                    hypotheses.append(llm_hypo)
                    continue
                self._llm_fallback_count += 1

            hypo = self._generate_material_hypothesis(pair, csp_a, csp_b, rng)
            if hypo is not None:
                hypotheses.append(hypo)

        return hypotheses

    @staticmethod
    def _make_prediction_range(
        known_val: float,
        uncertainty: float = 0.15,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> Tuple[float, float]:
        """基于已知值生成预测区间 (P0-3).

        不再输出随机单点, 而是输出 [lo, hi] 区间,
        区间宽度由 uncertainty 决定 (如 0.15 = ±15%).

        Args:
            known_val: 参考已知值
            uncertainty: 相对不确定度 (0.1-0.3)
            min_val/max_val: 物理下/上限 (可选, 截断用)

        Returns:
            (low, high) 预测区间
        """
        low = known_val * (1.0 - uncertainty)
        high = known_val * (1.0 + uncertainty)
        if low > high:
            low, high = high, low
        if min_val is not None:
            low = max(low, min_val)
        if max_val is not None:
            high = min(high, max_val)
        return round(low, 6), round(high, 6)

    def _generate_material_hypothesis(
        self,
        pair: PaperPair,
        csp_a: List[CSPTriple],
        csp_b: List[CSPTriple],
        rng: random.Random,
    ) -> Optional[Hypothesis]:
        """从一对论文 + CSP 三元组生成材料科学假设."""

        clean_title_a = _strip_title_prefix(pair.paper_a_title)
        clean_title_b = _strip_title_prefix(pair.paper_b_title)
        bridge = pair.bridge_keywords[0] if pair.bridge_keywords else "shared material"

        def _is_valid_triple(t: CSPTriple) -> bool:
            if t.property_name == "general":
                return False
            if not t.structure or t.structure == "unknown":
                return False
            if not t.composition:
                return False
            return True

        valid_a = [t for t in csp_a if _is_valid_triple(t)]
        valid_b = [t for t in csp_b if _is_valid_triple(t)]

        if not valid_a and not valid_b:
            return self._fallback_general_hypothesis(pair, rng)

        props_a = {t.property_name for t in valid_a}
        props_b = {t.property_name for t in valid_b}
        shared_props = props_a & props_b

        structs_a = {t.structure for t in valid_a}
        structs_b = {t.structure for t in valid_b}
        shared_structs = structs_a & structs_b

        is_pending_computation = False

        if shared_props and valid_a and valid_b:
            prop_name = rng.choice(list(shared_props))
            triple_a = next((t for t in valid_a if t.property_name == prop_name), None)
            triple_b = next((t for t in valid_b if t.property_name == prop_name), None)

            if triple_a and triple_b and triple_a.property_value is not None:
                if shared_structs:
                    hypo_type = "structure_property"
                else:
                    hypo_type = "hidden_link"

                known_val = triple_a.property_value
                _unc = 0.10 if shared_structs else 0.15
                pred_low, pred_high = self._make_prediction_range(known_val, _unc)
                predicted_val_str = f"{pred_low:.4g}–{pred_high:.4g}"

                template = MaterialHypothesisTemplates.get_template(hypo_type, rng)
                if template is None:
                    return None

                if triple_a.composition == triple_b.composition:
                    return self._fallback_general_hypothesis(pair, rng)

                statement = template.format(
                    composition_a=triple_a.composition,
                    composition_b=triple_b.composition,
                    structure_a=triple_a.structure,
                    structure_b=triple_b.structure,
                    property=prop_name,
                    property_a=prop_name,
                    property_b=triple_b.property_name if triple_b.property_name != prop_name else prop_name,
                    known_value=f"{known_val:.4g}",
                    value_b=f"{triple_b.property_value:.4g}" if triple_b.property_value else "?",
                    predicted_value=predicted_val_str,
                    unit=triple_a.property_unit,
                    unit_a=triple_a.property_unit,
                    unit_b=triple_b.property_unit,
                    bridge=bridge,
                    title_a=clean_title_a,
                    title_b=clean_title_b,
                    element_a=triple_a.composition[:2] if len(triple_a.composition) > 2 else "A",
                    element_b=triple_b.composition[:2] if len(triple_b.composition) > 2 else "B",
                    structure=triple_a.structure,
                    process="sol-gel",
                )

            elif triple_a and triple_a.property_value is not None and triple_b:
                hypo_type = "gap_filling"
                known_val = triple_a.property_value
                pred_low, pred_high = self._make_prediction_range(known_val, 0.25)
                predicted_val_str = f"{pred_low:.4g}–{pred_high:.4g}"

                template = MaterialHypothesisTemplates.get_template(hypo_type, rng)
                if template is None:
                    return None

                statement = template.format(
                    composition_a=triple_a.composition,
                    composition_b=triple_b.composition,
                    structure_a=triple_a.structure,
                    structure_b=triple_b.structure,
                    property=prop_name,
                    known_value=f"{known_val:.4g}",
                    predicted_value=predicted_val_str,
                    unit=triple_a.property_unit,
                )
            else:
                return self._fallback_general_hypothesis(pair, rng)

        elif valid_a and valid_b and not shared_props:
            triple_a = next((t for t in valid_a if t.property_value is not None), None)
            triple_b = next((t for t in valid_b if t.property_value is not None), None)

            if not triple_a or not triple_b:
                return self._fallback_general_hypothesis(pair, rng)

            if triple_a.composition == triple_b.composition:
                return self._fallback_general_hypothesis(pair, rng)

            hypo_type = "hidden_link"
            known_val = triple_a.property_value
            pred_low, pred_high = self._make_prediction_range(known_val, 0.40)
            predicted_val_str = f"{pred_low:.4g}–{pred_high:.4g}"
            is_pending_computation = True

            template = MaterialHypothesisTemplates.get_template(hypo_type, rng)
            if template is None:
                return self._fallback_general_hypothesis(pair, rng)

            try:
                statement = template.format(
                    composition_a=triple_a.composition,
                    composition_b=triple_b.composition,
                    structure_a=triple_a.structure,
                    structure_b=triple_b.structure,
                    property_a=triple_a.property_name,
                    property_b=triple_b.property_name,
                    known_value=f"{known_val:.4g}",
                    value_b=f"{triple_b.property_value:.4g}" if triple_b.property_value else "?",
                    predicted_value=predicted_val_str,
                    unit_a=triple_a.property_unit,
                    unit_b=triple_b.property_unit,
                    bridge=bridge,
                    title_a=clean_title_a,
                    title_b=clean_title_b,
                )
            except KeyError:
                return self._fallback_general_hypothesis(pair, rng)

        elif valid_a and not valid_b:
            triple = next((t for t in valid_a if t.property_value is not None), None)
            if not triple:
                return self._fallback_general_hypothesis(pair, rng)

            hypo_type = "gap_filling"
            known_val = triple.property_value
            pred_low, pred_high = self._make_prediction_range(known_val, 0.30)
            predicted_val_str = f"{pred_low:.4g}–{pred_high:.4g}"
            is_pending_computation = True

            template = MaterialHypothesisTemplates.get_template(hypo_type, rng)
            if template is None:
                return None

            statement = (
                f"待计算假设: {triple.composition}在{triple.structure}结构下的"
                f"{triple.property_name}已有报道(={known_val:.4g}{triple.property_unit})，"
                f"推测其在其他可行结构下的{triple.property_name}"
                f"可能在{predicted_val_str}{triple.property_unit}范围内，"
                f"但具体结构需DFT计算确认"
            )
            alt_struct = "unknown"
        else:
            return self._fallback_general_hypothesis(pair, rng)

        self._generated_count += 1
        raw_id = f"{pair.pair_id}{self._generated_count}"
        hypo_id = f"hypo_{hashlib.md5(raw_id.encode()).hexdigest()[:8]}"

        all_csp = valid_a + valid_b
        csp_keywords: List[str] = []
        for t in all_csp:
            if t.composition and t.composition not in csp_keywords:
                csp_keywords.append(t.composition)
            if t.structure and t.structure not in csp_keywords:
                csp_keywords.append(t.structure)
            if t.property_name and t.property_name not in csp_keywords:
                csp_keywords.append(t.property_name)
        csp_keywords.extend(pair.bridge_keywords[:3])

        novelty = self._estimate_material_novelty(pair, csp_a, csp_b)
        confidence = self._estimate_confidence(pair, hypo_type)

        comp_a = triple_a.composition if 'triple_a' in dir() and triple_a else (triple.composition if 'triple' in dir() and triple else "")
        comp_b = triple_b.composition if 'triple_b' in dir() and triple_b else comp_a
        struct_a_val = triple_a.structure if 'triple_a' in dir() and triple_a else (triple.structure if 'triple' in dir() and triple else "unknown")
        struct_b_val = triple_b.structure if 'triple_b' in dir() and triple_b else (alt_struct if 'alt_struct' in dir() else "unknown")
        prop_val = prop_name if 'prop_name' in dir() else (triple.property_name if 'triple' in dir() and triple else "general")

        _pred_val = None
        _pred_unit = ""
        if 'predicted_val_str' in dir():
            try:
                _parts = predicted_val_str.replace('–', '-').split('-')
                if len(_parts) == 2:
                    _pred_val = (float(_parts[0]) + float(_parts[1])) / 2.0
                else:
                    _pred_val = float(_parts[0])
                _pred_unit = (triple_a.property_unit if 'triple_a' in dir() and triple_a
                              else (triple.property_unit if 'triple' in dir() and triple else ""))
            except (ValueError, IndexError):
                pass

        physics = MaterialPhysicsValidator.validate_hypothesis(
            composition_a=comp_a or "",
            composition_b=comp_b or "",
            structure_a=struct_a_val or "unknown",
            structure_b=struct_b_val or "unknown",
            property_name=prop_val or "general",
            hypo_type=hypo_type,
            predicted_value=_pred_val,
            predicted_unit=_pred_unit,
        )

        if physics["violations"]:
            return self._fallback_general_hypothesis(pair, rng)

        if physics["is_known_principle"]:
            novelty *= 0.4
            confidence *= 0.7

        has_prediction = "predicted_val_str" in dir() or "≈" in statement or "预测" in statement

        return Hypothesis(
            hypothesis_id=hypo_id,
            statement=statement,
            hypothesis_type=hypo_type,
            source_pair=pair.pair_id,
            paper_a_id=pair.paper_a_id,
            paper_b_id=pair.paper_b_id,
            paper_a_title=pair.paper_a_title,
            paper_b_title=pair.paper_b_title,
            rationale=pair.connection_reason,
            testable=True,
            novelty=round(novelty, 3),
            confidence=round(confidence, 3),
            keywords=csp_keywords[:10] if csp_keywords else ["materials"],
            metadata={
                "pair_similarity": pair.similarity,
                "pair_type": pair.pair_type,
                "cross_domain": pair.cross_domain,
                "csp_count": len(all_csp),
                "has_numerical_prediction": has_prediction,
                "materials_mode": True,
                "physics_validation": physics,
                "is_known_principle": physics.get("is_known_principle", False),
                "known_principle_desc": physics.get("known_principle_desc", ""),
                "generation_method": "template",
                "is_pending_computation": is_pending_computation,
                "range_check_passed": physics.get("range_check", {}).get("in_range", True),
                "unit_compatible": physics.get("unit_compatible", True),
            },
        )

    def _fallback_general_hypothesis(
        self, pair: PaperPair, rng: random.Random,
    ) -> Optional[Hypothesis]:
        """当 CSP 抽取失败时，降级为通用假设生成."""
        hypo_type = self._map_pair_type_to_hypo(pair.pair_type)
        if hypo_type is None:
            return None

        templates = self.TEMPLATES.get(hypo_type, [])
        if not templates:
            return None

        for template in templates:
            statement = self._fill_template(template, pair)
            issues = self._validate_statement(statement, pair, hypo_type)
            if not issues:
                break
        else:
            return None

        min_falsifiability = (
            self._quality_gate.config.min_falsifiability_score
            if self._quality_gate.enabled else 0.3
        )
        if self._falsifiability_score(statement, pair, hypo_type) < min_falsifiability:
            return None

        self._generated_count += 1
        raw_id = f"{pair.pair_id}{self._generated_count}"
        hypo_id = f"hypo_{hashlib.md5(raw_id.encode()).hexdigest()[:8]}"

        return Hypothesis(
            hypothesis_id=hypo_id,
            statement=statement,
            hypothesis_type=hypo_type,
            source_pair=pair.pair_id,
            paper_a_id=pair.paper_a_id,
            paper_b_id=pair.paper_b_id,
            paper_a_title=pair.paper_a_title,
            paper_b_title=pair.paper_b_title,
            rationale=pair.connection_reason,
            testable=True,
            novelty=round(self._estimate_novelty(pair, hypo_type), 3),
            confidence=round(self._estimate_confidence(pair, hypo_type), 3),
            keywords=self._build_keywords(pair),
            metadata={
                "pair_similarity": pair.similarity,
                "pair_type": pair.pair_type,
                "cross_domain": pair.cross_domain,
                "materials_mode": True,
                "fallback": True,
                "generation_method": "template",
            },
        )

    def _estimate_material_novelty(
        self, pair: PaperPair,
        csp_a: List[CSPTriple], csp_b: List[CSPTriple],
    ) -> float:
        """估计材料科学假设的新颖性."""
        base = 0.5
        if pair.cross_domain:
            base += 0.2
        comps_a = {t.composition for t in csp_a}
        comps_b = {t.composition for t in csp_b}
        if comps_a and comps_b and not (comps_a & comps_b):
            base += 0.15
        if any(t.property_value is not None for t in csp_a + csp_b):
            base += 0.1
        return min(1.0, base)
