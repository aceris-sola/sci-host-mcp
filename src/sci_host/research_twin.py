"""Self-contained runtime for research-hypothesis simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .twin_adapter import HypothesisEvaluatorModel, PaperCorpusEntity


@dataclass
class ResearchDataBuffer:
    """Stores auditable simulation outputs for the active host."""

    records: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_samples(self) -> int:
        return len(self.records)

    def append(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        if len(self.records) > 10_000:
            self.records = self.records[-10_000:]


@dataclass
class LearningState:
    """Keeps calibration weights and compact learning summaries."""

    ewc_lambda: float = 0.4
    mode: str = "online_calibration"
    sleep_cycles: int = 0
    knowledge_base: Dict[str, Any] = field(default_factory=dict)
    _skills: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def skill_count(self) -> int:
        return len(self._skills)

    def add_skill(self, name: str, payload: Dict[str, Any]) -> None:
        self._skills[name] = dict(payload)


class ResearchTwin:
    """Small, transparent research simulation runtime used by Sci Host only."""

    VERSION = "SciHostResearchTwin/1.0"

    def __init__(
        self,
        twin_id: str,
        domain: str = "materials_research",
        ewc_lambda: float = 0.4,
        buffer_size: int = 10_000,
    ) -> None:
        self.twin_id = twin_id
        self.domain = domain
        self.version = self.VERSION
        self.buffer_size = buffer_size
        self.pe = PaperCorpusEntity()
        self.vm = HypothesisEvaluatorModel()
        self.dd = ResearchDataBuffer()
        self.cl = LearningState(ewc_lambda=ewc_lambda)
        self.algorithms: Dict[str, object] = {}

    def awake_step(self, observation: Dict[str, Any], instruction: str = "") -> Dict[str, Any]:
        predictions = self.vm.simulate(observation, horizon=1)
        if predictions:
            safe, violations = self.vm.check_constraints(predictions[0])
        else:
            safe, violations = False, ["research runtime produced no prediction"]
        self.dd.append({
            "instruction": instruction,
            "hypothesis_id": observation.get("hypothesis_id", ""),
            "predictions": predictions,
            "safe": safe,
            "violations": list(violations),
        })
        return {"predictions": predictions, "safe": safe, "violations": violations}


def register_research_operator(twin: ResearchTwin, name: str) -> None:
    """Register a named evaluator consumed by ``HypothesisEvaluatorModel``."""
    twin.algorithms[name] = object()
