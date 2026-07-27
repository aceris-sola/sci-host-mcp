"""StructurePropertySearchEngine — evolutionary search for CSP relationships.

Route A requires "search/optimization algorithm + LLM deep fusion".
This module provides a lightweight evolutionary search engine that:
  1. Seeds population from LLM/Agent-generated hypotheses + CSP triples
  2. Scores candidates with a multi-criteria fitness function
  3. Mutates top candidates (element substitution, structure swap, etc.)
  4. Returns top-N candidates with full provenance

Usage:
    engine = StructurePropertySearchEngine()
    result = engine.search(
        seed_hypotheses=hypos,
        csp_triples=triples,
        papers=papers,
        iterations=5,
        population_size=20,
    )
"""
from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from . import CSPExtractor, MaterialPhysicsValidator


@dataclass
class SearchCandidate:
    """A single candidate in the search space."""
    candidate_id: str
    composition: str
    structure: str
    property_name: str
    predicted_range: str                   # "3.0-3.5 eV"
    processing: str = ""
    dopant: str = ""
    source_hypothesis_id: str = ""
    source_paper_ids: List[str] = field(default_factory=list)

    # Scores
    score: float = 0.0
    literature_evidence: float = 0.0
    novelty: float = 0.0
    physical_plausibility: float = 0.0
    falsifiability: float = 0.0
    synthesizability: float = 0.0
    database_gap: float = 0.0

    # Metadata
    generation: int = 0                    # 0=seed, 1+=mutated
    mutation_history: List[str] = field(default_factory=list)
    parent_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "composition": self.composition,
            "structure": self.structure,
            "property_name": self.property_name,
            "predicted_range": self.predicted_range,
            "processing": self.processing,
            "dopant": self.dopant,
            "source_hypothesis_id": self.source_hypothesis_id,
            "source_paper_ids": self.source_paper_ids,
            "score": round(self.score, 3),
            "literature_evidence": round(self.literature_evidence, 3),
            "novelty": round(self.novelty, 3),
            "physical_plausibility": round(self.physical_plausibility, 3),
            "falsifiability": round(self.falsifiability, 3),
            "synthesizability": round(self.synthesizability, 3),
            "database_gap": round(self.database_gap, 3),
            "generation": self.generation,
            "mutation_history": self.mutation_history,
        }


class StructurePropertySearchEngine:
    """Evolutionary search for structure-property relationships.

    Fitness function:
        score = 0.25 * literature_evidence
              + 0.20 * novelty
              + 0.20 * physical_plausibility
              + 0.15 * falsifiability
              + 0.10 * synthesizability
              + 0.10 * database_gap
    """

    # Fitness weights
    W_EVIDENCE = 0.25
    W_NOVELTY = 0.20
    W_PHYSICS = 0.20
    W_FALSIFIABILITY = 0.15
    W_SYNTHESIS = 0.10
    W_GAP = 0.10

    # Element groups for mutation (same-group substitution)
    ELEMENT_GROUPS: Dict[str, List[str]] = {
        "alkali": ["Li", "Na", "K", "Rb", "Cs"],
        "alkaline": ["Be", "Mg", "Ca", "Sr", "Ba"],
        "transition_3d": ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"],
        "transition_4d": ["Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd"],
        "transition_5d": ["La", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"],
        "lanthanide": ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Yb", "Lu"],
        "chalcogen": ["O", "S", "Se", "Te"],
        "pnictogen": ["N", "P", "As", "Sb", "Bi"],
        "halogen": ["F", "Cl", "Br", "I"],
        "post_transition": ["Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi"],
        "metalloid": ["B", "Si", "Ge", "As", "Sb", "Te"],
    }

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng or random.Random(int(time.time()) % (2**31))
        self._candidates: List[SearchCandidate] = []
        self._history: List[Dict[str, Any]] = []

    def search(
        self,
        seed_hypotheses: List[Dict[str, Any]],
        csp_triples: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
        iterations: int = 5,
        population_size: int = 20,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run evolutionary search.

        Args:
            seed_hypotheses: hypothesis dicts with composition, structure, property
            csp_triples: CSP triple dicts from knowledge base
            papers: paper dicts for evidence scoring
            iterations: number of evolution iterations
            population_size: max population size
            top_k: number of top candidates to return

        Returns:
            {
                "top_candidates": List[SearchCandidate.to_dict()],
                "total_generated": int,
                "total_evaluated": int,
                "iterations": int,
                "best_score": float,
                "method": "evolutionary search",
                "history": List[Dict],
            }
        """
        self._candidates = []
        self._history = []

        # 1. Initialize population from seeds
        seeds = self._seed_population(seed_hypotheses, csp_triples, papers)
        self._candidates = seeds

        # 2. Score initial population
        for c in self._candidates:
            self._score_candidate(c, csp_triples, papers)

        total_evaluated = len(self._candidates)

        # 3. Evolution loop
        for gen in range(iterations):
            # Select top-K
            self._candidates.sort(key=lambda c: c.score, reverse=True)
            parents = self._candidates[:top_k]

            if not parents:
                break

            # Generate mutations
            children: List[SearchCandidate] = []
            for parent in parents:
                mutations = self._mutate(parent, csp_triples)
                children.extend(mutations)

            # Score children
            for c in children:
                self._score_candidate(c, csp_triples, papers)
                total_evaluated += 1

            # Combine and select
            combined = self._candidates + children
            combined.sort(key=lambda c: c.score, reverse=True)
            self._candidates = combined[:population_size]

            # Record history
            best = self._candidates[0] if self._candidates else None
            self._history.append({
                "generation": gen + 1,
                "population": len(self._candidates),
                "best_score": best.score if best else 0,
                "best_candidate": best.composition if best else "",
                "avg_score": sum(c.score for c in self._candidates) / max(len(self._candidates), 1),
            })

        # 4. Return top-N
        self._candidates.sort(key=lambda c: c.score, reverse=True)
        top = self._candidates[:top_k]

        return {
            "top_candidates": [c.to_dict() for c in top],
            "total_generated": len(self._candidates),
            "total_evaluated": total_evaluated,
            "iterations": iterations,
            "best_score": top[0].score if top else 0.0,
            "method": "evolutionary search",
            "history": self._history,
        }

    def _seed_population(
        self,
        hypotheses: List[Dict[str, Any]],
        csp_triples: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
    ) -> List[SearchCandidate]:
        """Create initial population from hypotheses and CSP triples."""
        seeds: List[SearchCandidate] = []
        seen_keys: Set[str] = set()

        # From hypotheses
        for h in hypotheses:
            stmt = h.get("statement", "")
            kws = h.get("keywords", [])

            # Extract composition, structure, property from keywords
            comp = self._find_composition(kws, csp_triples)
            struct = self._find_structure(kws, csp_triples)
            prop = self._find_property(kws, csp_triples)

            if not comp:
                continue

            key = f"{comp}|{struct}|{prop}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Find predicted range from statement
            pred_range = self._extract_range(stmt)

            cand = SearchCandidate(
                candidate_id=f"cand_{hashlib.md5(key.encode()).hexdigest()[:8]}",
                composition=comp,
                structure=struct or "unknown",
                property_name=prop or "general",
                predicted_range=pred_range or "TBD",
                source_hypothesis_id=h.get("hypothesis_id", ""),
                source_paper_ids=[
                    h.get("paper_a_id", ""),
                    h.get("paper_b_id", ""),
                ],
                generation=0,
            )
            seeds.append(cand)

        # From CSP triples (fill in gaps)
        for t in csp_triples:
            comp = t.get("composition", "")
            struct = t.get("structure", "")
            prop = t.get("property_name", "")
            val = t.get("property_value")
            unit = t.get("property_unit", "")

            if not comp or not prop:
                continue

            key = f"{comp}|{struct}|{prop}"
            if key in seen_keys:
                continue
            seen_keys.add(key)

            if val is not None:
                pred_range = f"{val:.4g}"
                if unit:
                    pred_range += f" {unit}"
            else:
                pred_range = "TBD"

            seeds.append(SearchCandidate(
                candidate_id=f"cand_{hashlib.md5(key.encode()).hexdigest()[:8]}",
                composition=comp,
                structure=struct or "unknown",
                property_name=prop,
                predicted_range=pred_range,
                source_paper_ids=[t.get("source_paper_id", "")],
                generation=0,
            ))

            if len(seeds) >= 50:
                break

        return seeds

    def _mutate(
        self,
        parent: SearchCandidate,
        csp_triples: List[Dict[str, Any]],
    ) -> List[SearchCandidate]:
        """Generate mutations of a parent candidate."""
        mutations: List[SearchCandidate] = []

        # Mutation 1: Element substitution
        mutated_comp = self._mutate_element(parent.composition)
        if mutated_comp and mutated_comp != parent.composition:
            key = f"{mutated_comp}|{parent.structure}|{parent.property_name}"
            cand = SearchCandidate(
                candidate_id=f"cand_{hashlib.md5(key.encode()).hexdigest()[:8]}",
                composition=mutated_comp,
                structure=parent.structure,
                property_name=parent.property_name,
                predicted_range=parent.predicted_range,
                source_hypothesis_id=parent.source_hypothesis_id,
                source_paper_ids=parent.source_paper_ids,
                generation=parent.generation + 1,
                parent_id=parent.candidate_id,
                mutation_history=parent.mutation_history + ["element_substitution"],
            )
            mutations.append(cand)

        # Mutation 2: Structure replacement (compatible only)
        alt_struct = self._mutate_structure(parent.composition, parent.structure)
        if alt_struct:
            key = f"{parent.composition}|{alt_struct}|{parent.property_name}"
            cand = SearchCandidate(
                candidate_id=f"cand_{hashlib.md5(key.encode()).hexdigest()[:8]}",
                composition=parent.composition,
                structure=alt_struct,
                property_name=parent.property_name,
                predicted_range=parent.predicted_range,
                source_hypothesis_id=parent.source_hypothesis_id,
                source_paper_ids=parent.source_paper_ids,
                generation=parent.generation + 1,
                parent_id=parent.candidate_id,
                mutation_history=parent.mutation_history + ["structure_replacement"],
            )
            mutations.append(cand)

        # Mutation 3: Property target swap
        alt_prop = self._mutate_property(parent.property_name, csp_triples)
        if alt_prop and alt_prop != parent.property_name:
            key = f"{parent.composition}|{parent.structure}|{alt_prop}"
            cand = SearchCandidate(
                candidate_id=f"cand_{hashlib.md5(key.encode()).hexdigest()[:8]}",
                composition=parent.composition,
                structure=parent.structure,
                property_name=alt_prop,
                predicted_range="TBD",
                source_hypothesis_id=parent.source_hypothesis_id,
                source_paper_ids=parent.source_paper_ids,
                generation=parent.generation + 1,
                parent_id=parent.candidate_id,
                mutation_history=parent.mutation_history + ["property_swap"],
            )
            mutations.append(cand)

        return mutations

    def _mutate_element(self, composition: str) -> Optional[str]:
        """Substitute one element with a same-group alternative."""
        if not composition:
            return None

        # Parse elements from composition
        import re
        elements = re.findall(r'[A-Z][a-z]?', composition)
        if not elements:
            return None

        # Pick a random element to substitute
        target_elem = self._rng.choice(elements)

        # Find its group
        target_group = None
        for group_name, members in self.ELEMENT_GROUPS.items():
            if target_elem in members:
                target_group = members
                break

        if not target_group or len(target_group) < 2:
            return None

        # Pick a replacement
        alternatives = [e for e in target_group if e != target_elem]
        if not alternatives:
            return None
        replacement = self._rng.choice(alternatives)

        # Replace in composition
        return composition.replace(target_elem, replacement, 1)

    def _mutate_structure(self, composition: str, current_struct: str) -> Optional[str]:
        """Find a compatible alternative structure."""
        # Pick from compatible structures
        compatible = []
        for s in CSPExtractor.STRUCTURE_TYPES:
            if s == current_struct:
                continue
            violations = MaterialPhysicsValidator.validate_structure_compatibility(composition, s)
            if not violations:
                compatible.append(s)

        if not compatible:
            return None
        return self._rng.choice(compatible[:20])

    def _mutate_property(
        self, current_prop: str, csp_triples: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Swap to a different property found in CSP triples."""
        all_props = list({t.get("property_name", "") for t in csp_triples if t.get("property_name")})
        all_props = [p for p in all_props if p and p != current_prop and p != "general"]
        if not all_props:
            return None
        return self._rng.choice(all_props)

    def _score_candidate(
        self,
        candidate: SearchCandidate,
        csp_triples: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
    ) -> None:
        """Score a candidate using the multi-criteria fitness function."""

        # 1. Literature evidence: how many CSP triples / papers mention this composition+property
        n_csp = sum(
            1 for t in csp_triples
            if t.get("composition", "").lower() in candidate.composition.lower()
            or candidate.composition.lower() in t.get("composition", "").lower()
        )
        n_papers = sum(
            1 for p in papers
            if candidate.composition.lower()
            in (p.get("title", "") + p.get("abstract", "")).lower()
        )
        candidate.literature_evidence = min(1.0, n_csp * 0.15 + n_papers * 0.10)

        # 2. Novelty: fewer papers = more novel
        if n_papers == 0:
            candidate.novelty = 0.8
        elif n_papers <= 2:
            candidate.novelty = 0.6
        elif n_papers <= 5:
            candidate.novelty = 0.4
        else:
            candidate.novelty = 0.2

        # 3. Physical plausibility: check structure compatibility + range
        violations = MaterialPhysicsValidator.validate_structure_compatibility(
            candidate.composition, candidate.structure,
        )
        if violations:
            candidate.physical_plausibility = 0.1
        else:
            candidate.physical_plausibility = 0.8

        # 4. Falsifiability: has numerical prediction?
        if candidate.predicted_range and candidate.predicted_range != "TBD":
            candidate.falsifiability = 0.9
        else:
            candidate.falsifiability = 0.3

        # 5. Synthesizability: known structure = easier to synthesize
        if candidate.structure and candidate.structure != "unknown":
            candidate.synthesizability = 0.7
        else:
            candidate.synthesizability = 0.3

        # 6. Database gap: no CSP triple with this exact comp+struct+prop
        exact_match = any(
            t.get("composition", "") == candidate.composition
            and t.get("structure", "") == candidate.structure
            and t.get("property_name", "") == candidate.property_name
            for t in csp_triples
        )
        candidate.database_gap = 0.1 if exact_match else 0.8

        # Composite score
        candidate.score = (
            self.W_EVIDENCE * candidate.literature_evidence
            + self.W_NOVELTY * candidate.novelty
            + self.W_PHYSICS * candidate.physical_plausibility
            + self.W_FALSIFIABILITY * candidate.falsifiability
            + self.W_SYNTHESIS * candidate.synthesizability
            + self.W_GAP * candidate.database_gap
        )

    def _find_composition(
        self, keywords: List[str], csp_triples: List[Dict[str, Any]],
    ) -> str:
        """Find a composition from keywords or CSP triples."""
        for kw in keywords:
            for t in csp_triples:
                if kw.lower() in t.get("composition", "").lower():
                    return t["composition"]
            # Check if keyword itself looks like a formula
            import re
            if re.match(r'^[A-Z][a-z]?\d*', kw) and len(kw) >= 2:
                return kw
        return ""

    def _find_structure(
        self, keywords: List[str], csp_triples: List[Dict[str, Any]],
    ) -> str:
        """Find a structure from keywords or CSP triples."""
        for kw in keywords:
            for t in csp_triples:
                if kw.lower() == t.get("structure", "").lower():
                    return t["structure"]
            # Check if keyword matches a known structure type
            for s in CSPExtractor.STRUCTURE_TYPES:
                if kw.lower() == s.lower():
                    return s
        return ""

    def _find_property(
        self, keywords: List[str], csp_triples: List[Dict[str, Any]],
    ) -> str:
        """Find a property from keywords or CSP triples."""
        for kw in keywords:
            for t in csp_triples:
                if kw.lower() == t.get("property_name", "").lower():
                    return t["property_name"]
            # Check if keyword matches a known property
            for prop_name in CSPExtractor.PROPERTY_DICT:
                if kw.lower() == prop_name.lower():
                    return prop_name
        return ""

    def _extract_range(self, statement: str) -> str:
        """Extract a predicted range from a hypothesis statement."""
        import re
        # Look for patterns like "3.0–3.5" or "≈2.6" or "85-115"
        match = re.search(r'(\d+\.?\d*)\s*[–\-]\s*(\d+\.?\d*)', statement)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        match = re.search(r'≈\s*(\d+\.?\d*)', statement)
        if match:
            return match.group(1)
        return ""
