"""假设评估器 — 对生成的假设进行快速预评估.

在试错引擎之前进行快速筛选，避免对低质量假设浪费试错资源。

评估维度:
    1. 可验证性: 假设是否可被文献/逻辑验证
    2. 新颖性: 假设是否足够新颖
    3. 相关性: 假设是否与当前研究方向相关
    4. 清晰度: 假设陈述是否足够清晰
"""
from __future__ import annotations

from typing import Dict

from .generator import Hypothesis


class HypothesisEvaluator:
    """假设快速评估器.

    对假设进行多维度评分，输出综合分数。
    用于在试错前快速筛选。
    """

    def __init__(self) -> None:
        self._evaluated_count: int = 0

    def evaluate(self, hypothesis: Hypothesis) -> Dict[str, float]:
        """评估假设，返回各维度分数."""
        self._evaluated_count += 1

        scores: Dict[str, float] = {}

        
        scores["testability"] = self._score_testability(hypothesis)

        
        scores["novelty"] = hypothesis.novelty

        
        scores["relevance"] = self._score_relevance(hypothesis)

        
        scores["clarity"] = self._score_clarity(hypothesis)

        
        scores["source_confidence"] = hypothesis.confidence

        
        scores["overall"] = (
            0.25 * scores["testability"] +
            0.25 * scores["novelty"] +
            0.20 * scores["relevance"] +
            0.15 * scores["clarity"] +
            0.15 * scores["source_confidence"]
        )

        return scores

    def _score_testability(self, h: Hypothesis) -> float:
        """可验证性评分."""
        score = 0.35
        if h.testable:
            score += 0.15
        
        if h.keywords:
            score += 0.12
        
        if h.hypothesis_type != "contradiction":
            score += 0.08
        
        if 20 < len(h.statement) < 300:
            score += 0.05
        return min(1.0, score)

    def _score_relevance(self, h: Hypothesis) -> float:
        """相关性评分."""
        score = 0.3
        
        if h.keywords:
            score += 0.2
        
        if h.metadata.get("cross_domain", False):
            score += 0.15
        
        sim = h.metadata.get("pair_similarity", 0)
        score += min(0.2, sim * 0.3)
        return min(1.0, score)

    def _score_clarity(self, h: Hypothesis) -> float:
        """清晰度评分."""
        score = 0.35
        
        length = len(h.statement)
        if 30 < length < 200:
            score += 0.15
        elif length >= 200:
            score += 0.08  
        else:
            score -= 0.1  
        
        if h.rationale and len(h.rationale) > 10:
            score += 0.1
        
        if h.hypothesis_type:
            score += 0.08
        return max(0.1, min(1.0, score))

    def quick_filter(self, hypothesis: Hypothesis, threshold: float = 0.4) -> bool:
        """快速过滤：返回是否值得进入试错."""
        scores = self.evaluate(hypothesis)
        return scores["overall"] >= threshold

    @property
    def total_evaluated(self) -> int:
        return self._evaluated_count
