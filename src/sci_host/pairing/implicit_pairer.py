"""隐性配对引擎 — 发现论文间的隐性关联.

核心思想:
    不是简单地找"相似"论文（那是显性配对），
    而是发现看似不相关但存在深层关联的论文对。

配对策略:
    1. 语义相似度: 嵌入向量余弦相似度
    2. 跨领域发现: 不同 category 间的低相似度配对（隐性关联）
    3. 关键词桥接: 共享少量但关键的桥接词
    4. 概念互补: A 的方法可解决 B 的问题

与原项目 SemanticKnowledgeGraph 的区别:
    原项目: 用 Jaccard 相似度自动建图 (显性)
    本系统: 用嵌入空间 + 跨领域策略发现隐性关联
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..config import PairingConfig, ResearchQualityConfig
from ..research_quality import ResearchQualityGate
from .embedding import TextEmbedder


@dataclass
class PaperPair:
    """论文配对结果."""
    pair_id: str
    paper_a_id: str
    paper_b_id: str
    paper_a_title: str
    paper_b_title: str
    similarity: float              
    cross_domain: bool             
    pair_type: str                 # pairing type
    bridge_keywords: List[str]     
    connection_reason: str         
    paper_a_category: str = ""     
    paper_b_category: str = ""     
    paper_a_keywords: List[str] = field(default_factory=list)
    paper_b_keywords: List[str] = field(default_factory=list)
    paper_a_abstract: str = ""    
    paper_b_abstract: str = ""    
    timestamp: float = field(default_factory=time.time)

    @property
    def is_implicit(self) -> bool:
        """是否为隐性配对（相似度不高但有关联）."""
        return self.similarity < 0.5 and self.pair_type != "direct_similarity"


class ImplicitPairer:
    """隐性配对引擎.

    持续接收新论文，发现论文间的隐性关联。

    配对类型:
        - direct_similarity: 直接语义相似（同领域，高相似度）
        - cross_domain_bridge: 跨领域桥接（不同领域，中等相似度）
        - method_transfer: 方法迁移（A 的方法可应用于 B 的问题）
        - concept_complement: 概念互补（两篇论文的概念可组合）
        - contradiction: 矛盾发现（两篇论文的结论可能冲突）
    """

    PAIR_TYPES = {
        "direct_similarity", "cross_domain_bridge",
        "method_transfer", "concept_complement", "contradiction",
    }

    
    METHOD_KEYWORDS = {
        "transformer", "neural", "network", "learning", "optimization",
        "gradient", "embedding", "attention", "diffusion", "gaussian",
        "bayesian", "graph", "federated", "quantum", "blockchain",
        "reinforcement", "evolutionary", "meta-learning", "transfer",
        
        "cable-driven", "tendon-driven", "series-elastic", "compliant",
        "pneumatic", "hydraulic", "planetary", "harmonic",
        "actuator", "transmission", "flexure", "morphing",
        
        "perovskite", "dft", "alloy", "composite", "catalyst",
        "sintering", "annealing", "deposition", "doping",
    }

    
    PROBLEM_KEYWORDS = {
        "robot", "manipulation", "diagnosis", "prediction", "planning",
        "scheduling", "allocation", "discovery", "generation", "calibration",
        "maintenance", "coordination", "detection", "tracking", "validation",
        "joint", "torque", "efficiency", "fatigue", "wear",
        "stiffness", "damping", "backlash", "friction",
    }

    
    
    GENERIC_WORDS = {
        
        "novel", "enhanced", "adaptive", "scalable", "robust", "efficient",
        "unified", "deep", "progressive", "towards", "based",
        "novel", "approach", "method", "framework", "model", "system",
        "design", "performance", "analysis", "study", "research",
        "investigation", "experiment", "experimental", "simulation",
        "numerical", "theoretical", "comprehensive", "systematic",
        "preliminary", "detailed", "general", "proposed",
        
        "conference", "ieee", "international", "proceedings",
        "symposium", "journal", "transactions", "workshop",
        "acm", "springer", "elsevier", "wiley",
        
        "robot", "robotics", "learning", "machine", "intelligence",
        "control", "optimization", "algorithm", "data", "result",
        "performance", "evaluation", "assessment", "comparison",
        "application", "implementation", "development",
        "measurement", "testing", "validation",
        
        "micro", "nano", "macro", "large", "small", "high", "low",
        "new", "recent", "advanced", "modern",
    }

    def __init__(
        self,
        embedder: TextEmbedder,
        config: PairingConfig,
        quality_config: Optional[ResearchQualityConfig] = None,
    ) -> None:
        self.embedder = embedder
        self.config = config
        self._quality_gate = ResearchQualityGate(quality_config)
        self.quality_rejected_total = 0
        self._paper_pool: Dict[str, Any] = {}  # paper_id → Paper
        self._embeddings: Dict[str, np.ndarray] = {}  # paper_id → embedding
        self._recent_pairs: List[PaperPair] = []
        self._pair_history: Set[str] = set()  
        
        
        self._paper_pair_count: Dict[str, int] = {}  
        self._max_pairs_per_paper: int = 5  

    def add_papers(self, papers: List[Any]) -> None:
        """将新论文加入配对池."""
        if self._quality_gate.enabled:
            papers, rejected = self._quality_gate.filter_papers(papers)
            self.quality_rejected_total += rejected
        if not papers:
            return

        
        if not self.embedder.is_fitted and papers:
            texts = [p.text for p in papers]
            self.embedder.fit(texts)
            
            for p in papers:
                emb = self.embedder.embed(p.text)
                self._paper_pool[p.paper_id] = p
                self._embeddings[p.paper_id] = emb
        else:
            
            if papers:
                texts = [p.text for p in papers]
                self.embedder.partial_fit(texts)
                
                if self._paper_pool:
                    for pid, p in self._paper_pool.items():
                        self._embeddings[pid] = self.embedder.embed(p.text)
                
                for p in papers:
                    emb = self.embedder.embed(p.text)
                    self._paper_pool[p.paper_id] = p
                    self._embeddings[p.paper_id] = emb

    def find_pairs(self) -> List[PaperPair]:
        """在论文池中寻找隐性配对."""
        if len(self._paper_pool) < 2:
            return []

        paper_ids = list(self._paper_pool.keys())
        pairs: List[PaperPair] = []

        
        self._paper_pair_count = {pid: 0 for pid in paper_ids}

        
        n = len(paper_ids)
        total_possible = n * (n - 1) // 2

        candidates: List[Tuple[str, str, float]] = []

        if total_possible <= 200:
            
            for i in range(n):
                for j in range(i + 1, n):
                    id_a, id_b = paper_ids[i], paper_ids[j]
                    key = frozenset({id_a, id_b})
                    if key in self._pair_history:
                        continue
                    sim = self._cosine_similarity(id_a, id_b)
                    if sim >= 0:
                        candidates.append((id_a, id_b, sim))
        else:
            
            max_candidates = min(
                self.config.max_pairs_per_round * 5,
                total_possible,
            )
            rng = np.random.RandomState(int(time.time()) % 2**31)
            seen_pairs: set = set()

            for _ in range(max_candidates * 2):
                i, j = rng.randint(0, n), rng.randint(0, n)
                if i == j:
                    continue
                pair_key = (min(i, j), max(i, j))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                id_a, id_b = paper_ids[i], paper_ids[j]
                key = frozenset({id_a, id_b})
                if key in self._pair_history:
                    continue

                sim = self._cosine_similarity(id_a, id_b)
                if sim >= 0:
                    candidates.append((id_a, id_b, sim))

                if len(candidates) >= max_candidates:
                    break

        
        candidates.sort(key=lambda x: -x[2])

        
        for id_a, id_b, sim in candidates:
            
            
            if (self._paper_pair_count.get(id_a, 0) >= self._max_pairs_per_paper or
                self._paper_pair_count.get(id_b, 0) >= self._max_pairs_per_paper):
                continue

            pair = self._evaluate_pair(id_a, id_b, sim)
            if pair is not None:
                pairs.append(pair)
                self._pair_history.add(frozenset({id_a, id_b}))
                
                self._paper_pair_count[id_a] = self._paper_pair_count.get(id_a, 0) + 1
                self._paper_pair_count[id_b] = self._paper_pair_count.get(id_b, 0) + 1

            if len(pairs) >= self.config.max_pairs_per_round:
                break

        self._recent_pairs.extend(pairs)
        if len(self._recent_pairs) > 200:
            self._recent_pairs = self._recent_pairs[-200:]

        return pairs

    def _cosine_similarity(self, id_a: str, id_b: str) -> float:
        """计算两篇论文的余弦相似度."""
        emb_a = self._embeddings.get(id_a)
        emb_b = self._embeddings.get(id_b)
        if emb_a is None or emb_b is None:
            return -1.0
        if len(emb_a) != len(emb_b):
            return -1.0
        norm = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        if norm < 1e-10:
            return 0.0
        return float(np.dot(emb_a, emb_b) / norm)

    def _evaluate_pair(self, id_a: str, id_b: str, sim: float) -> Optional[PaperPair]:
        """评估候选配对，返回 PaperPair 或 None."""
        paper_a = self._paper_pool[id_a]
        paper_b = self._paper_pool[id_b]

        if self._quality_gate.enabled:
            quality_a = self._quality_gate.assess_paper(paper_a)
            quality_b = self._quality_gate.assess_paper(paper_b)
            if not quality_a.accepted or not quality_b.accepted:
                return None

        cross_domain = paper_a.primary_category != paper_b.primary_category
        threshold = (self.config.cross_domain_threshold if cross_domain
                     else self.config.similarity_threshold)

        if (
            self._quality_gate.enabled
            and self._quality_gate.config.require_technical_pair_bridge
            and not cross_domain
            and self._quality_gate.technical_overlap(paper_a, paper_b)
        ):
            
            
            threshold = max(
                self.config.cross_domain_threshold,
                self.config.similarity_threshold * 0.75,
            )

        if sim < threshold:
            return None

        
        pair_type, reason, bridge = self._detect_pair_type(paper_a, paper_b, sim, cross_domain)

        if pair_type is None:
            return None

        
        
        if self._quality_gate.enabled and self._quality_gate.config.require_technical_pair_bridge:
            technical_bridge = self._quality_gate.technical_overlap(paper_a, paper_b)
            if not technical_bridge:
                return None
            bridge = list(dict.fromkeys(technical_bridge + bridge))[:5]
            reason = self._regenerate_reason(
                pair_type, paper_a, paper_b, sim, cross_domain, bridge,
            )

        
        if pair_type == "direct_similarity" and sim < self.config.similarity_threshold:
            return None

        
        if not bridge:
            bridge = self._fallback_bridge_keywords(paper_a, paper_b)
            
            if not bridge:
                return None
            
            reason = self._regenerate_reason(pair_type, paper_a, paper_b, sim, cross_domain, bridge)

        return PaperPair(
            pair_id=f"pair_{id_a.split('_')[-1][:6]}_{id_b.split('_')[-1][:6]}",
            paper_a_id=id_a,
            paper_b_id=id_b,
            paper_a_title=paper_a.title,
            paper_b_title=paper_b.title,
            similarity=round(sim, 4),
            cross_domain=cross_domain,
            pair_type=pair_type,
            bridge_keywords=bridge,
            connection_reason=reason,
            paper_a_category=paper_a.primary_category,
            paper_b_category=paper_b.primary_category,
            paper_a_keywords=list(paper_a.keywords),
            paper_b_keywords=list(paper_b.keywords),
            paper_a_abstract=getattr(paper_a, 'abstract', ''),
            paper_b_abstract=getattr(paper_b, 'abstract', ''),
        )

    def _detect_pair_type(
        self, paper_a: Any, paper_b: Any, sim: float, cross_domain: bool,
    ) -> Tuple[Optional[str], str, List[str]]:
        """检测配对类型.

        Returns:
            (pair_type, reason, bridge_keywords)
        """
        kw_a = set(k.lower() for k in paper_a.keywords)
        kw_b = set(k.lower() for k in paper_b.keywords)
        raw_bridge = list(kw_a & kw_b)

        
        bridge = [kw for kw in raw_bridge if kw not in self.GENERIC_WORDS]

        
        if cross_domain and sim >= self.config.cross_domain_threshold:
            if not bridge:
                
                return None, "", []
            reason = (f"跨领域关联: [{paper_a.primary_category}] 与 "
                      f"[{paper_b.primary_category}] 通过 '{', '.join(bridge[:3])}' 桥接")
            return "cross_domain_bridge", reason, bridge[:5]

        
        a_methods = kw_a & self.METHOD_KEYWORDS
        b_problems = kw_b & self.PROBLEM_KEYWORDS
        b_methods = kw_b & self.METHOD_KEYWORDS
        a_problems = kw_a & self.PROBLEM_KEYWORDS

        if a_methods and b_problems and not cross_domain:
            reason = (f"方法迁移: '{', '.join(list(a_methods)[:2])}' 方法 "
                      f"可应用于 '{', '.join(list(b_problems)[:2])}' 问题")
            return "method_transfer", reason, list(a_methods | b_problems)[:5]

        if b_methods and a_problems and not cross_domain:
            reason = (f"方法迁移: '{', '.join(list(b_methods)[:2])}' 方法 "
                      f"可应用于 '{', '.join(list(a_problems)[:2])}' 问题")
            return "method_transfer", reason, list(a_methods | b_problems)[:5]

        
        
        if bridge and sim < 0.5:
            unique_a = kw_a - kw_b
            unique_b = kw_b - kw_a
            if unique_a and unique_b:
                reason = (f"概念互补: 两篇论文共享 '{', '.join(bridge[:2])}' 但各自贡献 "
                          f"'{', '.join(list(unique_a)[:2])}' 和 '{', '.join(list(unique_b)[:2])}'")
                return "concept_complement", reason, bridge[:5]

        
        if sim >= self.config.similarity_threshold:
            reason = f"语义相似 (相似度={sim:.3f})"
            return "direct_similarity", reason, bridge[:5] if bridge else []

        return None, "", []

    def _regenerate_reason(
        self, pair_type: str, paper_a: Any, paper_b: Any,
        sim: float, cross_domain: bool, bridge: List[str],
    ) -> str:
        """用保底关键词重新生成配对理由（确保 reason 中不出现空桥接词）."""
        bridge_str = ", ".join(bridge[:3]) if bridge else "research"
        if pair_type == "cross_domain_bridge":
            return (f"跨领域关联: [{paper_a.primary_category}] 与 "
                    f"[{paper_b.primary_category}] 通过 '{bridge_str}' 桥接")
        elif pair_type == "method_transfer":
            return f"方法迁移: '{bridge_str}' 可跨论文应用 (相似度={sim:.3f})"
        elif pair_type == "concept_complement":
            return f"概念互补: 两篇论文通过 '{bridge_str}' 产生互补关联"
        elif pair_type == "direct_similarity":
            return f"语义相似 (相似度={sim:.3f}), 共享关键词: '{bridge_str}'"
        return f"关联发现 (类型={pair_type}, 桥接='{bridge_str}')"

    def _fallback_bridge_keywords(self, paper_a: Any, paper_b: Any) -> List[str]:
        """当交集为空时，从两篇论文的关键词中选取最可能有桥接价值的词.

        策略: 优先取方法类关键词 (METHOD_KEYWORDS)，
        其次取问题类关键词 (PROBLEM_KEYWORDS)，
        最后取两篇论文各自的前 3 个关键词的并集 (过滤通用词).
        """
        kw_a = [k.lower() for k in paper_a.keywords]
        kw_b = [k.lower() for k in paper_b.keywords]

        
        a_methods = [k for k in kw_a if k in self.METHOD_KEYWORDS]
        b_methods = [k for k in kw_b if k in self.METHOD_KEYWORDS]
        if a_methods or b_methods:
            return list(dict.fromkeys(a_methods + b_methods))[:5]

        
        a_problems = [k for k in kw_a if k in self.PROBLEM_KEYWORDS]
        b_problems = [k for k in kw_b if k in self.PROBLEM_KEYWORDS]
        if a_problems or b_problems:
            return list(dict.fromkeys(a_problems + b_problems))[:5]

        
        combined = [
            k for k in dict.fromkeys(kw_a[:3] + kw_b[:3])
            if k not in self.GENERIC_WORDS
        ]
        return combined[:5] if combined else []

    def recent_pairs(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的配对."""
        return [
            {
                "pair_id": p.pair_id,
                "paper_a": p.paper_a_title[:60],
                "paper_b": p.paper_b_title[:60],
                "similarity": p.similarity,
                "type": p.pair_type,
                "cross_domain": p.cross_domain,
                "reason": p.connection_reason,
            }
            for p in self._recent_pairs[-n:]
        ]

    @property
    def pool_size(self) -> int:
        return len(self._paper_pool)

    @property
    def total_pairs_found(self) -> int:
        return len(self._pair_history)
