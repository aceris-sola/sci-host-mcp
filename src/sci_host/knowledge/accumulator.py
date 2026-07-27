"""知识累积器 — 持续积累试错验证的知识.

替代原项目的 EWC (Elastic Weight Consolidation) 机制。
原项目: 用 Fisher 信息矩阵约束重要参数漂移，防止灾难性遗忘
本系统: 用置信度衰减/提升机制管理研究方向，淘汰无前景方向

知识类型:
    - validated: 通过试错验证的知识（假设被支持）
    - failed: 试错失败的知识（假设被否定）
    - pending: 待验证的知识
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..config import KnowledgeConfig


@dataclass
class KnowledgeEntry:
    """知识条目."""
    entry_id: str
    statement: str
    entry_type: str          # validated / failed / pending
    hypothesis_type: str
    score: float
    novelty: float
    keywords: List[str]
    source_papers: List[str]
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeAccumulator:
    """知识累积器.

    持续积累试错验证的知识，区分"已验证"和"已否定"。

    与原项目 EWC 的类比:
        EWC: Fisher 信息矩阵标记重要参数 → 约束更新
        本系统: 置信度分数标记有前景方向 → 提升/衰减
    """

    def __init__(self, config: KnowledgeConfig) -> None:
        self.config = config
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._validated: List[KnowledgeEntry] = []
        self._failed: List[KnowledgeEntry] = []
        self._keyword_index: Dict[str, List[str]] = defaultdict(list)  # keyword → entry_ids

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def validated_count(self) -> int:
        return len(self._validated)

    @property
    def failed_count(self) -> int:
        return len(self._failed)

    def add_validated(self, trial_result: Any) -> KnowledgeEntry:
        """添加通过验证的知识."""
        keywords = getattr(trial_result, "keywords", [])
        statement = getattr(trial_result, "statement", "")
        hypo_type = getattr(trial_result, "hypothesis_type", "")

        
        if not keywords:
            keywords = self._extract_keywords_from_text(statement, hypo_type)

        entry = KnowledgeEntry(
            entry_id=getattr(trial_result, "hypothesis_id", ""),
            statement=statement,
            entry_type="validated",
            hypothesis_type=hypo_type,
            score=getattr(trial_result, "score", 0.0),
            novelty=getattr(trial_result, "novelty", 0.0),
            keywords=keywords,
            source_papers=[
                getattr(trial_result, "paper_a_id", ""),
                getattr(trial_result, "paper_b_id", ""),
            ],
            retry_count=getattr(trial_result, "retry_count", 0),
            metadata=getattr(trial_result, "metadata", {}),
        )
        self._entries[entry.entry_id] = entry
        self._validated.append(entry)
        for kw in entry.keywords:
            self._keyword_index[kw.lower()].append(entry.entry_id)
        
        if hypo_type:
            self._keyword_index[f"type:{hypo_type}"].append(entry.entry_id)
        return entry

    def add_failed(self, trial_result: Any) -> KnowledgeEntry:
        """添加试错失败的知识."""
        keywords = getattr(trial_result, "keywords", [])
        statement = getattr(trial_result, "statement", "")
        hypo_type = getattr(trial_result, "hypothesis_type", "")

        
        if not keywords:
            keywords = self._extract_keywords_from_text(statement, hypo_type)

        entry = KnowledgeEntry(
            entry_id=getattr(trial_result, "hypothesis_id", ""),
            statement=statement,
            entry_type="failed",
            hypothesis_type=hypo_type,
            score=getattr(trial_result, "score", 0.0),
            novelty=getattr(trial_result, "novelty", 0.0),
            keywords=keywords,
            source_papers=[
                getattr(trial_result, "paper_a_id", ""),
                getattr(trial_result, "paper_b_id", ""),
            ],
            retry_count=getattr(trial_result, "retry_count", 0),
            metadata=getattr(trial_result, "metadata", {}),
        )
        self._entries[entry.entry_id] = entry
        self._failed.append(entry)
        for kw in entry.keywords:
            self._keyword_index[kw.lower()].append(entry.entry_id)
        if hypo_type:
            self._keyword_index[f"type:{hypo_type}"].append(entry.entry_id)
        return entry

    def search_by_keyword(self, keyword: str) -> List[KnowledgeEntry]:
        """按关键词搜索知识.

        支持两种匹配方式:
        1. 精确匹配: 搜索 'digital twin' 命中 keyword='digital twin' 的条目
        2. 子串匹配: 搜索 'learning' 命中 keyword='machine learning' / 'continual learning' 等条目
        """
        kw_lower = keyword.lower()
        
        entry_ids = self._keyword_index.get(kw_lower, [])
        results = {eid for eid in entry_ids if eid in self._entries}
        
        for indexed_kw, eids in self._keyword_index.items():
            if indexed_kw == kw_lower:
                continue  
            
            if indexed_kw.startswith("type:"):
                continue
            if kw_lower in indexed_kw or indexed_kw in kw_lower:
                for eid in eids:
                    if eid in self._entries:
                        results.add(eid)
        return [self._entries[eid] for eid in results]

    @staticmethod
    def _extract_keywords_from_text(statement: str, hypo_type: str) -> List[str]:
        """当关键词为空时从陈述文本中提取保底关键词."""
        result: List[str] = []
        seen: set = set()

        
        tech_terms = [
            "transformer", "neural", "network", "learning", "optimization",
            "gradient", "embedding", "attention", "diffusion", "gaussian",
            "bayesian", "graph", "federated", "quantum", "blockchain",
            "reinforcement", "digital twin", "physics-informed",
            "causal", "counterfactual", "evidential", "knowledge graph",
            "meta-learning", "transfer", "evolutionary", "game theory",
            "maml", "few-shot", "continual learning", "domain randomization",
            "sim-to-real", "multi-robot", "multi-agent", "predictive maintenance",
        ]
        text_lower = statement.lower()
        for term in tech_terms:
            if term in text_lower and term not in seen:
                seen.add(term)
                result.append(term)

        
        if hypo_type and hypo_type not in seen:
            seen.add(hypo_type)
            result.append(hypo_type)

        return result[:5] if result else ["research"]

    def get_top_validated(self, n: int = 10) -> List[KnowledgeEntry]:
        """获取评分最高的已验证知识."""
        sorted_validated = sorted(
            self._validated,
            key=lambda e: (e.score * 0.6 + e.novelty * 0.4),
            reverse=True,
        )
        return sorted_validated[:n]

    def get_recent_failures(self, n: int = 10) -> List[KnowledgeEntry]:
        """获取最近的失败知识（用于学习避免重复错误）."""
        return self._failed[-n:]

    def summary(self) -> Dict[str, Any]:
        """知识库摘要."""
        
        type_counts: Dict[str, int] = defaultdict(int)
        for e in self._validated:
            type_counts[e.hypothesis_type] += 1

        
        avg_score = (sum(e.score for e in self._validated) / len(self._validated)
                     if self._validated else 0.0)
        avg_novelty = (sum(e.novelty for e in self._validated) / len(self._validated)
                       if self._validated else 0.0)

        
        kw_freq: Dict[str, int] = defaultdict(int)
        for e in self._validated:
            for kw in e.keywords:
                kw_freq[kw] += 1
        top_keywords = sorted(kw_freq.items(), key=lambda x: -x[1])[:10]

        return {
            "total_entries": self.entry_count,
            "validated": self.validated_count,
            "failed": self.failed_count,
            "type_distribution": dict(type_counts),
            "avg_score": round(avg_score, 3),
            "avg_novelty": round(avg_novelty, 3),
            "top_keywords": top_keywords,
            "pass_rate": (self.validated_count / max(self.entry_count, 1)),
        }

    def get_learnings(self) -> Dict[str, Any]:
        """从失败中提取学习教训."""
        if not self._failed:
            return {"learnings": [], "total": 0}

        
        failure_patterns: Dict[str, int] = defaultdict(int)
        for e in self._failed:
            pattern = e.metadata.get("pair_type", "unknown")
            failure_patterns[pattern] += 1

        
        low_score_keywords: Dict[str, int] = defaultdict(int)
        for e in self._failed:
            if e.score < 0.3:
                for kw in e.keywords:
                    low_score_keywords[kw] += 1

        return {
            "total_failures": len(self._failed),
            "failure_by_pattern": dict(failure_patterns),
            "common_low_score_keywords": sorted(
                low_score_keywords.items(), key=lambda x: -x[1],
            )[:5],
            "retry_success_rate": self._compute_retry_success_rate(),
        }

    def _compute_retry_success_rate(self) -> float:
        """计算重试后的成功率."""
        retried_validated = sum(1 for e in self._validated if e.retry_count > 0)
        total_retried = retried_validated + sum(1 for e in self._failed if e.retry_count > 0)
        if total_retried == 0:
            return 0.0
        return retried_validated / total_retried
