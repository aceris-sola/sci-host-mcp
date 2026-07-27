"""专项科研质量闸门回归测试."""
from __future__ import annotations

from sci_host.config import (
    HypothesisConfig,
    KnowledgeConfig,
    PairingConfig,
    ResearchQualityConfig,
)
from sci_host.crawler.paper_crawler import Paper
from sci_host.hypothesis.generator import HypothesisGenerator
from sci_host.knowledge.direction_tracker import DirectionTracker
from sci_host.pairing.embedding import TextEmbedder
from sci_host.pairing.implicit_pairer import ImplicitPairer, PaperPair
from sci_host.research_quality import ResearchQualityGate


def quality_config() -> ResearchQualityConfig:
    return ResearchQualityConfig(
        enabled=True,
        focus_terms=["twisted string actuator", "robot joint"],
        min_score=0.40,
        min_mechanism_hits=1,
        min_evidence_hits=1,
    )


def make_paper(paper_id: str, title: str, abstract: str, category: str) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        categories=[category],
        keywords=["twisted string", "actuator", "robot joint", "torque"],
    )


def test_metadata_noise_is_rejected_and_engineering_paper_is_kept():
    gate = ResearchQualityGate(quality_config())
    noisy = Paper(
        paper_id="noise",
        title="Robot Design IEEE Conference 2020",
        abstract="",
        keywords=["robot", "design", "IEEE", "conference"],
    )
    useful = make_paper(
        "useful",
        "Twisted string actuator for a robot joint",
        "A fabricated prototype was tested and measured torque, efficiency, backlash, "
        "and fatigue under repeated load cycles.",
        "cs.RO",
    )

    noisy_result = gate.assess_paper(noisy)
    useful_result = gate.assess_paper(useful)
    assert not noisy_result.accepted
    assert "insufficient_evidence" in noisy_result.reasons
    assert useful_result.accepted
    assert useful_result.evidence_terms

    identifier = Paper(
        paper_id="identifier",
        title="06-1745-b24",
        abstract="A robotic actuator experiment measured force and efficiency.",
        keywords=["actuator", "force", "experiment"],
    )
    assert not gate.assess_paper(identifier).accepted
    assert "identifier_title" in gate.assess_paper(identifier).reasons


def test_quality_pair_requires_a_real_technical_bridge():
    cfg = quality_config()
    embedder = TextEmbedder(max_features=200, target_dim=32)
    pairer = ImplicitPairer(
        embedder,
        config=PairingConfig(max_pairs_per_round=10),
        quality_config=cfg,
    )
    papers = [
        make_paper(
            "a",
            "Twisted string actuator for a robot joint",
            "A prototype measured torque and efficiency of a cable transmission.",
            "cs.RO",
        ),
        make_paper(
            "b",
            "Compliant actuator joint with low-cost transmission",
            "Experiments measured torque, stiffness, backlash, and fatigue of the actuator.",
            "eess.SY",
        ),
    ]
    pairer.add_papers(papers)
    assert pairer._cosine_similarity("a", "b") > 0.0
    pairs = pairer.find_pairs()
    pair = pairs[0] if pairs else None
    assert pair is not None
    assert "actuator" in pair.bridge_keywords
    assert "IEEE" not in pair.bridge_keywords

    
    pairer._paper_pool["b"].categories = ["cs.RO"]
    assert pairer._evaluate_pair("a", "b", 0.08) is not None


def test_quality_mode_rejects_non_falsifiable_template_signal():
    cfg = quality_config()
    generator = HypothesisGenerator(HypothesisConfig(), quality_config=cfg)
    pair = PaperPair(
        pair_id="weak",
        paper_a_id="a",
        paper_b_id="b",
        paper_a_title="Robot design",
        paper_b_title="Control design",
        similarity=0.3,
        cross_domain=True,
        pair_type="cross_domain_bridge",
        bridge_keywords=["robot"],
        connection_reason="weak",
        paper_a_keywords=["robot", "design"],
        paper_b_keywords=["control", "design"],
        paper_a_abstract="robot design",
        paper_b_abstract="control design",
    )
    assert generator.generate([pair]) == []


def test_manual_direction_is_candidate_and_technical_only():
    tracker = DirectionTracker(KnowledgeConfig(), quality_config())
    assert tracker.add_manual_direction("IEEE conference design 2020") == ""
    direction_id = tracker.add_manual_direction(
        "Use a cable actuator joint and measure torque efficiency"
    )
    assert direction_id
    result = tracker.top_directions(1)[0]
    assert result["confidence"] == 0.2
    assert result["is_promising"] is False
    assert "ieee" not in result["label"].lower()
