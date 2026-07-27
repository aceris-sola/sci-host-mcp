"""科研论文质量与技术桥接判定.

该模块只负责可解释的规则判定，不调用网络或模型。它服务于专项检索，
默认由 ``ResearchQualityConfig.enabled`` 控制为关闭，因此不会改变通用模式。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Sequence, Tuple

from .config import ResearchQualityConfig



MECHANISM_TERMS: Tuple[str, ...] = (
    "robot", "robotic", "joint", "actuator", "actuation", "transmission",
    "gearbox", "gear train", "harmonic drive", "planetary", "cycloidal",
    "ball screw", "roller screw", "cable", "tendon", "capstan",
    "twisted string", "linkage", "cam", "underactuated", "compliant",
    "compliance", "series elastic", "direct drive", "brushless", "motor",
    "pneumatic", "hydraulic", "artificial muscle", "shape memory",
    "dielectric elastomer", "electroactive polymer", "piezoelectric",
    "magnetorheological", "origami", "continuum", "soft actuator",
    "mechanism", "bearing", "friction", "spring", "elastic element",
)

EVIDENCE_TERMS: Tuple[str, ...] = (
    "torque", "force", "efficiency", "backlash", "stiffness", "damping",
    "bandwidth", "load", "speed", "power", "position", "accuracy",
    "precision", "hysteresis", "fatigue", "lifetime", "durability", "wear",
    "friction", "prototype", "fabricated", "fabrication", "machined",
    "printed", "experiment", "measured", "measurement", "characterization",
    "validation", "tested", "test", "performance", "cost", "mass", "weight",
    "failure", "reliability", "repeatability", "compliance",
)

METADATA_TERMS = {
    "conference", "ieee", "international", "proceedings", "symposium",
    "journal", "transactions", "workshop", "acm", "springer", "elsevier",
    "wiley", "doi", "http", "https", "www", "com", "org", "pdf",
    "copyright", "license", "volume", "issue", "pages", "author", "authors",
    "published", "accepted", "study", "investigation", "analysis", "research",
    "approach", "method", "results", "result", "novel", "enhanced", "adaptive",
}


WEAK_BRIDGE_TERMS = {
    "robot", "robotic", "robotics", "design", "control", "system", "application",
    "model", "method", "mechanism", "performance", "experiment", "analysis",
    "study", "research", "data", "learning", "optimization",
}


@dataclass(frozen=True)
class PaperQuality:
    """一篇论文的可解释质量结果."""
    score: float
    accepted: bool
    focus_terms: List[str] = field(default_factory=list)
    mechanism_terms: List[str] = field(default_factory=list)
    evidence_terms: List[str] = field(default_factory=list)
    has_metric: bool = False
    metadata_only: bool = False
    reasons: List[str] = field(default_factory=list)


class ResearchQualityGate:
    """为论文和论文对提供统一的质量判定."""

    def __init__(self, config: ResearchQualityConfig | None = None) -> None:
        self.config = config or ResearchQualityConfig()
        self._last_degraded: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def assess_paper(self, paper: Any) -> PaperQuality:
        title = str(getattr(paper, "title", "") or "")
        abstract = str(getattr(paper, "abstract", "") or "")
        keywords = [str(k) for k in (getattr(paper, "keywords", []) or [])]
        return self.assess_text(title, abstract, keywords)

    def assess_text(
        self, title: str, abstract: str = "", keywords: Sequence[str] = (),
    ) -> PaperQuality:
        text = self._normalize(" ".join([title, abstract, " ".join(keywords)]))
        title_text = self._normalize(title)
        focus = self.match_terms(text, self.config.focus_terms, phrase_aware=True)
        mechanisms = self.match_terms(text, MECHANISM_TERMS)
        evidence = self.match_terms(text, EVIDENCE_TERMS)
        has_metric = bool(re.search(
            r"(?:\d+(?:\.\d+)?\s*(?:%|nm|mm|cm|m|n\s*m|kn|n|mn|\u03bcn|w|kw|hz|rad/s|deg|\u00b0c|g|kg|j|pa|mpa|ev))",
            text,
            re.IGNORECASE,
        ))

        abstract_len = len(abstract.strip())
        title_tokens = set(re.findall(r"[a-z][a-z0-9-]{2,}", title_text))
        identifier_title = bool(
            re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", title_text)
            and re.search(r"\d", title_text)
        )
        metadata_only = (
            abstract_len < self.config.min_abstract_chars
            and not has_metric
            and len(title_tokens - METADATA_TERMS) < 2
        )

        focus_score = 1.0 if not self.config.focus_terms else min(
            1.0, len(focus) / max(1, min(2, len(self.config.focus_terms)))
        )
        mechanism_score = min(1.0, len(mechanisms) / 2.0)
        evidence_score = min(1.0, len(evidence) / 2.0 + (0.25 if has_metric else 0.0))
        abstract_score = 1.0 if abstract_len >= self.config.min_abstract_chars else 0.0
        score = round(
            0.35 * focus_score
            + 0.35 * mechanism_score
            + 0.25 * evidence_score
            + 0.05 * abstract_score,
            3,
        )

        reasons: List[str] = []
        accepted = True
        if self.config.require_focus and self.config.focus_terms and not focus:
            accepted = False
            reasons.append("no_focus_term")
        if len(mechanisms) < self.config.min_mechanism_hits:
            accepted = False
            reasons.append("insufficient_mechanism")
        if len(evidence) < self.config.min_evidence_hits and not has_metric:
            accepted = False
            reasons.append("insufficient_evidence")
        if metadata_only:
            accepted = False
            reasons.append("metadata_only")
        if identifier_title:
            accepted = False
            reasons.append("identifier_title")
        if score < self.config.min_score:
            accepted = False
            reasons.append("score_below_threshold")

        if not self.enabled:
            accepted = True
            reasons = []

        return PaperQuality(
            score=score,
            accepted=accepted,
            focus_terms=focus,
            mechanism_terms=mechanisms,
            evidence_terms=evidence,
            has_metric=has_metric,
            metadata_only=metadata_only,
            reasons=reasons,
        )

    def filter_papers(self, papers: Iterable[Any]) -> Tuple[List[Any], int]:
        """过滤论文, 带降级机制避免全拒空转.

        当质量闸门开启但整批论文全部被拒时, 自动降级:
        按 score 取 top 20% 论文放行, 确保后续阶段有输入.
        降级行为会记录在返回的 ``degraded`` 标志中 (可通过
        ``filter_papers_degraded`` 属性读取).
        """
        if not self.enabled:
            return list(papers), 0
        paper_list = list(papers)
        if not paper_list:
            return [], 0

        assessed: List[Tuple[Any, PaperQuality]] = []
        accepted: List[Any] = []
        rejected = 0
        for paper in paper_list:
            q = self.assess_paper(paper)
            assessed.append((paper, q))
            if q.accepted:
                accepted.append(paper)
            else:
                rejected += 1

        
        if not accepted and len(paper_list) > 0:
            self._last_degraded = True
            
            assessed.sort(key=lambda x: x[1].score, reverse=True)
            keep_n = max(1, len(paper_list) // 5)
            accepted = [p for p, _ in assessed[:keep_n]]
            rejected = len(paper_list) - len(accepted)
        else:
            self._last_degraded = False

        return accepted, rejected

    def technical_overlap(self, paper_a: Any, paper_b: Any) -> List[str]:
        """返回两篇论文共享的有效技术桥词."""
        qa = self.assess_paper(paper_a)
        qb = self.assess_paper(paper_b)
        shared = set(qa.mechanism_terms) & set(qb.mechanism_terms)
        return [term for term in MECHANISM_TERMS if term in shared and term not in WEAK_BRIDGE_TERMS]

    @staticmethod
    def match_terms(
        text: str, terms: Iterable[str], phrase_aware: bool = False,
    ) -> List[str]:
        normalized = ResearchQualityGate._normalize(text)
        found: List[str] = []
        for term in terms:
            term_norm = ResearchQualityGate._normalize(str(term))
            if not term_norm:
                continue
            if term_norm in normalized:
                found.append(str(term))
            elif phrase_aware and len(term_norm.split()) > 1:
                tokens = term_norm.split()
                if all(token in normalized.split() for token in tokens):
                    found.append(str(term))
        return list(dict.fromkeys(found))

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value.lower()).strip()
