"""研究方向追踪器 — 追踪和评估科学研究方向.

替代原项目的 SkillGenomeRouter (技能基因组路由)。
原项目: 将 ML 模型参数映射为 8 维能力基因向量
本系统: 将验证通过的假设聚类为研究方向，追踪置信度变化

方向生命周期:
    1. 发现: 新的验证假设创建新方向
    2. 提升: 后续验证支持该方向 → 置信度提升
    3. 衰减: 每轮自然衰减（无新证据支持）
    4. 淘汰: 置信度低于阈值 → 淘汰
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..config import KnowledgeConfig
from ..config import ResearchQualityConfig
from ..research_quality import EVIDENCE_TERMS, MECHANISM_TERMS, ResearchQualityGate




_DIRECTION_VAGUE_KEYWORDS = {
    "robot", "robotic", "robotics", "joint", "joints", "stiffness",
    "torque", "force", "actuator", "motor", "control", "system",
    "design", "method", "model", "approach", "performance",
    "learning", "algorithm", "optimization", "network", "neural",
    "machine", "intelligence", "data", "result", "analysis",
    "study", "research", "experiment", "simulation", "test",
    "evaluation", "application", "development", "implementation",
    "general", "general", "research",
}


@dataclass
class ResearchDirection:
    """研究方向."""
    direction_id: str
    label: str                    
    keywords: List[str]           
    hypothesis_type: str          
    confidence: float             
    support_count: int = 0        
    failure_count: int = 0        
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    source_hypotheses: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_promising(self) -> bool:
        """是否有前景.

        需要满足以下条件才标记为 promising:
        1. 置信度 >= 0.5 (需要至少 2 次验证提升才达到)
        2. 支持证据数 >= 2 (单次试错不足以认定有前景)
        3. 失败不能远多于支持 (失败率 < 60%)
        """
        if self.confidence < 0.5:
            return False
        if self.support_count < 2:
            return False
        total = self.support_count + self.failure_count
        if total > 0 and self.failure_count / total > 0.6:
            return False
        return True

    @property
    def age(self) -> float:
        return time.time() - self.created_at


class DirectionTracker:
    """研究方向追踪器.

    从试错结果中提取和维护研究方向，实现:
    - 新方向发现（从验证假设中聚类）
    - 置信度提升（验证支持）
    - 置信度衰减（无新证据）
    - 方向淘汰（置信度过低）

    与原项目 SkillGenomeRouter 的类比:
        SkillGenome: 8 维能力基因 → 加权欧氏距离 → 系统树
        Direction: 关键词集合 → Jaccard 相似度 → 方向聚类
    """

    
    MERGE_THRESHOLD: float = 0.25

    def __init__(
        self,
        config: KnowledgeConfig,
        quality_config: Optional[ResearchQualityConfig] = None,
    ) -> None:
        self.config = config
        self._directions: Dict[str, ResearchDirection] = {}
        self._quality_gate = ResearchQualityGate(quality_config)

    @property
    def direction_count(self) -> int:
        return len(self._directions)

    @property
    def promising_count(self) -> int:
        return sum(1 for d in self._directions.values() if d.is_promising)

    def update_from_trial(self, trial_result: Any) -> Optional[ResearchDirection]:
        """从通过验证的试错结果中更新方向."""
        keywords = getattr(trial_result, "keywords", [])
        hypo_type = getattr(trial_result, "hypothesis_type", "")
        score = getattr(trial_result, "score", 0.0)
        novelty = getattr(trial_result, "novelty", 0.0)
        statement = getattr(trial_result, "statement", "")
        hypo_id = getattr(trial_result, "hypothesis_id", "")
        paper_a_title = getattr(trial_result, "paper_a_title", "")
        paper_b_title = getattr(trial_result, "paper_b_title", "")

        if self._quality_gate.enabled:
            keywords = self._technical_keywords(
                statement, paper_a_title, paper_b_title, keywords,
            )
            if not keywords:
                return None

        
        if not keywords:
            keywords = self._extract_fallback_keywords(
                statement, hypo_type, paper_a_title, paper_b_title,
            )

        
        substantive_kws = [
            kw for kw in keywords
            if kw.lower().strip() not in _DIRECTION_VAGUE_KEYWORDS
        ]
        if not substantive_kws:
            return None

        
        existing = self._find_similar_direction(keywords)
        if existing is not None:
            
            existing.support_count += 1
            existing.confidence = min(1.0, existing.confidence + self.config.confidence_boost)
            existing.last_updated = time.time()
            if hypo_id not in existing.source_hypotheses:
                existing.source_hypotheses.append(hypo_id)
            
            for kw in keywords:
                if kw not in existing.keywords:
                    existing.keywords.append(kw)
            return existing
        else:
            
            dir_id = f"dir_{len(self._directions)}_{int(time.time()) % 100000}"
            label = self._generate_label(keywords, hypo_type)
            direction = ResearchDirection(
                direction_id=dir_id,
                label=label,
                keywords=list(keywords),
                hypothesis_type=hypo_type,
                
                confidence=min(1.0, 0.15 + score * 0.3 + novelty * 0.2),
                support_count=1,
                source_hypotheses=[hypo_id],
                metadata={
                    "initial_score": score,
                    "initial_novelty": novelty,
                    "statement": statement[:100],
                },
            )
            self._directions[dir_id] = direction
            return direction

    def record_failure(self, trial_result: Any) -> None:
        """记录试错失败（降低相关方向置信度）."""
        keywords = getattr(trial_result, "keywords", [])
        if not keywords:
            
            keywords = self._extract_fallback_keywords(
                getattr(trial_result, "statement", ""),
                getattr(trial_result, "hypothesis_type", ""),
                getattr(trial_result, "paper_a_title", ""),
                getattr(trial_result, "paper_b_title", ""),
            )
        if self._quality_gate.enabled:
            keywords = self._technical_keywords(
                getattr(trial_result, "statement", ""),
                getattr(trial_result, "paper_a_title", ""),
                getattr(trial_result, "paper_b_title", ""),
                keywords,
            )
            if not keywords:
                return
        existing = self._find_similar_direction(keywords)
        if existing is not None:
            existing.failure_count += 1
            existing.confidence = max(0.0, existing.confidence - 0.05)

    def decay_and_eliminate(self) -> List[str]:
        """置信度衰减 + 淘汰低置信度方向.

        Returns:
            被淘汰的方向 ID 列表
        """
        eliminated: List[str] = []
        for dir_id in list(self._directions.keys()):
            direction = self._directions[dir_id]
            
            direction.confidence *= self.config.confidence_decay
            direction.last_updated = time.time()

            
            if direction.confidence < self.config.elimination_threshold:
                eliminated.append(dir_id)
                del self._directions[dir_id]

        
        if len(self._directions) > self.config.max_directions:
            
            sorted_dirs = sorted(
                self._directions.items(),
                key=lambda x: x[1].confidence,
            )
            excess = len(self._directions) - self.config.max_directions
            for dir_id, _ in sorted_dirs[:excess]:
                eliminated.append(dir_id)
                del self._directions[dir_id]

        return eliminated

    def top_directions(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最有前景的方向."""
        sorted_dirs = sorted(
            self._directions.values(),
            key=lambda d: (d.confidence * 0.6 + d.support_count * 0.05 + d.is_promising * 0.3),
            reverse=True,
        )
        return [
            {
                "direction_id": d.direction_id,
                "label": d.label,
                "keywords": d.keywords[:5],
                "confidence": round(d.confidence, 3),
                "support_count": d.support_count,
                "failure_count": d.failure_count,
                "hypothesis_type": d.hypothesis_type,
                "is_promising": d.is_promising,
                "age_hours": round(d.age / 3600, 1),
                "statement": d.metadata.get("statement", ""),
            }
            for d in sorted_dirs[:n]
        ]

    def add_manual_direction(self, hint: str) -> str:
        """手动注入研究方向 (来自 Agent/Claude 的 direction_hint).

        Args:
            hint: 方向描述文字

        Returns:
            direction_id
        """
        import hashlib as _hashlib
        direction_id = f"manual_{_hashlib.md5(hint[:50].encode()).hexdigest()[:8]}"

        
        if self._quality_gate.enabled:
            hint_keywords = self._technical_keywords(hint, "", "", [])
            if not hint_keywords:
                return ""
        else:
            hint_keywords = [w for w in hint.split() if len(w) > 2][:5]
        similar = self._find_similar_direction(hint_keywords)
        if similar:
            similar.confidence = min(1.0, similar.confidence + 0.15)
            similar.support_count += 1
            similar.metadata["agent_hint"] = hint[:200]
            return similar.direction_id

        
        import time as _time
        direction = ResearchDirection(
            direction_id=direction_id,
            label=hint[:80],
            keywords=hint_keywords,
            confidence=0.2,  
            hypothesis_type="agent_suggested",
        )
        direction.support_count = 0
        direction.metadata["agent_hint"] = hint[:200]
        direction.metadata["statement"] = hint[:120]
        self._directions[direction_id] = direction
        return direction_id

    def _find_similar_direction(self, keywords: List[str]) -> Optional[ResearchDirection]:
        """找到与给定关键词最相似的方向（Jaccard 相似度）."""
        if not self._directions or not keywords:
            return None

        kw_set = set(k.lower() for k in keywords)
        best_dir: Optional[ResearchDirection] = None
        best_sim = 0.0

        for direction in self._directions.values():
            dir_kw_set = set(k.lower() for k in direction.keywords)
            sim = self._jaccard(kw_set, dir_kw_set)
            if sim > best_sim:
                best_sim = sim
                best_dir = direction

        if best_sim >= self.MERGE_THRESHOLD:
            return best_dir
        return None

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        """Jaccard 相似度."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _generate_label(keywords: List[str], hypo_type: str) -> str:
        """生成方向标签 — 过滤碎片词，保留有语义的关键词."""
        type_label = {
            "analogy": "类比迁移",
            "contradiction": "矛盾探究",
            "gap": "空白填补",
            "combination": "方法组合",
            "structure_property": "构效关系",
            "hidden_link": "隐藏关联",
            "gap_filling": "性能预测",
        }.get(hypo_type, "探索方向")

        
        _STOP = {
            "the", "a", "an", "for", "of", "to", "in", "on", "with", "and",
            "or", "via", "from", "based", "using", "through", "by", "as",
            "is", "are", "was", "were", "be", "been", "this", "that",
            "study", "investigation", "analysis", "research", "approach",
            "novel", "enhanced", "adaptive", "towards", "method", "result",
            "show", "showed", "shown", "demonstrate", "propose", "present",
            "use", "used", "using", "employ", "apply", "applied",
            "paper", "work", "study", "article", "we", "our",
            
            "conference", "ieee", "international", "proceedings",
            "symposium", "journal", "transactions", "workshop",
            "acm", "springer", "elsevier", "wiley", "doi", "http",
            "https", "www", "com", "org", "pdf", "arxiv", "copyright",
            "license", "volume", "issue", "pages", "vol",
            "author", "authors", "published", "accepted",
            
            "preliminary", "comprehensive", "systematic", "detailed",
            "numerical", "experimental", "theoretical",
        }
        clean_kws = []
        seen = set()
        for kw in keywords:
            kw_lower = kw.lower().strip(".,;:!?()[]")
            if len(kw_lower) < 3:
                continue
            if kw_lower in _STOP:
                continue
            if kw_lower in seen:
                continue
            seen.add(kw_lower)
            clean_kws.append(kw)
            if len(clean_kws) >= 3:
                break

        kw_str = " + ".join(clean_kws) if clean_kws else "general"
        return f"{type_label}: {kw_str}"

    def _technical_keywords(
        self,
        statement: str,
        title_a: str,
        title_b: str,
        keywords: List[str],
    ) -> List[str]:
        """专项模式下只保留机制/性能词，避免出版元数据成为方向."""
        text = " ".join([statement, title_a, title_b, " ".join(keywords)])
        terms = list(MECHANISM_TERMS) + list(EVIDENCE_TERMS)
        found = ResearchQualityGate.match_terms(text, terms, phrase_aware=True)
        
        return list(dict.fromkeys(found))[:10]

    @staticmethod
    def _extract_fallback_keywords(
        statement: str, hypo_type: str,
        title_a: str, title_b: str,
    ) -> List[str]:
        """当 keywords 为空时，从陈述和标题中提取保底关键词."""
        result: List[str] = []
        seen: set = set()

        
        tech_terms = [
            "transformer", "neural", "network", "learning", "optimization",
            "gradient", "embedding", "attention", "diffusion", "gaussian",
            "bayesian", "graph", "federated", "quantum", "blockchain",
            "reinforcement", "digital twin", "physics-informed",
            "causal", "counterfactual", "evidential", "knowledge graph",
            "meta-learning", "transfer", "evolutionary", "game theory",
            
            "perovskite", "catalyst", "alloy", "composite", "ceramic",
            "polymer", "semiconductor", "crystal", "nanostructure",
            "thin film", "battery", "electrode", "electrolyte",
            "bandgap", "conductivity", "magnetization", "dielectric",
            "piezoelectric", "thermoelectric", "photocatalysis",
            "corrosion", "coating", "sintering", "annealing",
            "doping", "deposition", "crystallization", "phase transition",
            "high-entropy alloy", "metal-organic framework", "graphene",
            "nanotube", "quantum dot", "superconductor", "ferroelectric",
            "antiferromagnetic", "paramagnetic", "piezoresistive",
        ]
        combined_lower = (statement + " " + title_a + " " + title_b).lower()
        for term in tech_terms:
            if term in combined_lower and term not in seen:
                seen.add(term)
                result.append(term)

        
        if hypo_type and hypo_type not in seen:
            seen.add(hypo_type)
            result.append(hypo_type)

        
        stop = {"the", "a", "an", "for", "of", "to", "in", "on", "with",
                "and", "or", "via", "from", "based", "using", "through",
                
                "conference", "ieee", "international", "proceedings",
                "symposium", "journal", "transactions", "workshop",
                "acm", "springer", "elsevier", "wiley", "doi", "http",
                "https", "www", "com", "org", "pdf", "arxiv",
                "copyright", "license", "volume", "issue", "pages",
                "vol", "author", "authors", "published", "accepted",
                
                "preliminary", "comprehensive", "systematic", "detailed",
                "numerical", "experimental", "theoretical",
                "novel", "enhanced", "adaptive", "towards",
                "study", "investigation", "analysis", "research",
                "approach", "method", "result", "results",
                "show", "showed", "shown", "demonstrate", "propose",
                "present", "paper", "work", "article"}
        for title in [title_a, title_b]:
            for word in title.split():
                w = word.lower().strip(".,;:!?()[]")
                if len(w) > 3 and w not in stop and w not in seen:
                    seen.add(w)
                    result.append(w)
                if len(result) >= 5:
                    break

        return result[:5] if result else ["research"]

    def summary(self) -> Dict[str, Any]:
        """方向追踪摘要."""
        type_counts: Dict[str, int] = defaultdict(int)
        for d in self._directions.values():
            type_counts[d.hypothesis_type] += 1

        avg_confidence = (
            sum(d.confidence for d in self._directions.values()) /
            max(len(self._directions), 1)
        )

        return {
            "total_directions": self.direction_count,
            "promising": self.promising_count,
            "type_distribution": dict(type_counts),
            "avg_confidence": round(avg_confidence, 3),
        }
