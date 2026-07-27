"""GapEvidenceBuilder — Build credible Research Gaps with evidence chains.

Each Gap includes:
  - gap_description: what is not yet studied
  - supporting_papers: papers that hint at the gap (>= 2 required)
  - counter_evidence: papers that partially cover or contradict
  - novelty_score: 0-1, how novel this gap likely is
  - feasibility_score: 0-1, how feasible to verify
  - evidence_chain: traceable chain from papers -> CSP -> gap
  - verification_plan: suggested DFT/experimental path
  - reverse_search_queries: 3-5 synonymous queries for Sciverse reverse search
  - reviewer_assessment: scientific peer-review style judgment
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class GapEvidence:
    """A single Research Gap with full evidence chain."""
    gap_id: str
    gap_description: str
    gap_type: str                           # "csp_missing", "structure_unexplored", "property_unpredicted"
    keywords: List[str] = field(default_factory=list)

    # Evidence
    supporting_papers: List[Dict[str, Any]] = field(default_factory=list)
    counter_evidence: List[Dict[str, Any]] = field(default_factory=list)
    csp_context: List[Dict[str, Any]] = field(default_factory=list)

    # Scores
    novelty_score: float = 0.0              # 0=known, 1=likely novel
    feasibility_score: float = 0.0          # 0=hard to verify, 1=easy
    evidence_strength: float = 0.0          # composite of paper count + CSP coverage

    # Chain
    evidence_chain: List[str] = field(default_factory=list)
    verification_plan: str = ""

    # Reverse search
    reverse_search_queries: List[str] = field(default_factory=list)
    reverse_search_done: bool = False

    # Scientific review
    reviewer_assessment: str = ""
    why_not_known: str = ""

    # Metadata
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id,
            "gap_description": self.gap_description,
            "gap_type": self.gap_type,
            "keywords": self.keywords[:8],
            "supporting_papers": self.supporting_papers,
            "counter_evidence": self.counter_evidence,
            "csp_context": self.csp_context[:5],
            "novelty_score": round(self.novelty_score, 3),
            "feasibility_score": round(self.feasibility_score, 3),
            "evidence_strength": round(self.evidence_strength, 3),
            "evidence_chain": self.evidence_chain,
            "verification_plan": self.verification_plan,
            "reverse_search_queries": self.reverse_search_queries,
            "reverse_search_done": self.reverse_search_done,
            "reviewer_assessment": self.reviewer_assessment,
            "why_not_known": self.why_not_known,
        }


class GapEvidenceBuilder:
    """Build credible Research Gaps from hypotheses, CSP triples, and papers.

    Usage:
        builder = GapEvidenceBuilder()
        gaps = builder.build_gaps(
            hypotheses=hypos,
            csp_triples=all_triples,
            papers=all_papers,
            failed_trials=failures,
        )
    """

    MIN_SUPPORTING_PAPERS = 2

    # Property synonym groups for reverse search query expansion
    _PROPERTY_SYNONYMS: Dict[str, List[str]] = {
        "bandgap": ["band gap", "energy gap", "optical gap", "electronic gap"],
        "conductivity": ["electrical conductivity", "ionic conductivity", "electronic conductivity"],
        "thermal_conductivity": ["thermal conductance", "heat conduction", "phonon transport"],
        "hardness": ["Vickers hardness", "nanoindentation hardness", "mechanical hardness"],
        "youngs_modulus": ["elastic modulus", "Young's modulus", "stiffness"],
        "seebeck": ["Seebeck coefficient", "thermoelectric power", "thermopower"],
        "zt": ["figure of merit", "thermoelectric efficiency", "ZT"],
        "dielectric_constant": ["dielectric permittivity", "relative permittivity", "dielectric constant"],
        "piezoelectric_coeff": ["piezoelectric coefficient", "d33", "piezoelectric response"],
        "curie_temperature": ["Curie temperature", "ferroelectric transition", "Tc"],
        "coercivity": ["coercive field", "coercive force", "magnetic coercivity"],
        "magnetization": ["magnetic moment", "saturation magnetization", "magnetization"],
        "shielding_effectiveness": ["EMI shielding", "electromagnetic shielding", "shielding efficiency"],
        "density": ["mass density", "bulk density", "specific gravity"],
        "formation_energy": ["formation enthalpy", "heat of formation", "formation energy"],
    }

    def __init__(self) -> None:
        self._gaps: List[GapEvidence] = []
        self._seen_keys: Set[str] = set()

    def build_gaps(
        self,
        hypotheses: List[Dict[str, Any]],
        csp_triples: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
        failed_trials: Optional[List[Dict[str, Any]]] = None,
        max_gaps: int = 20,
    ) -> List[GapEvidence]:
        """Build a list of credible Research Gaps.

        Args:
            hypotheses: hypothesis dicts (from host.state.sample_hypotheses)
            csp_triples: CSP triple dicts (from host.get_csp_knowledge)
            papers: paper dicts with paper_id, title, abstract, keywords
            failed_trials: failed trial results (optional, adds failure-based gaps)
            max_gaps: maximum gaps to return

        Returns:
            List of GapEvidence objects, sorted by composite score
        """
        self._gaps = []
        self._seen_keys = set()

        # 1. Build gaps from gap_filling / gap hypotheses
        for h in hypotheses:
            h_type = h.get("hypothesis_type") or h.get("type", "")
            if h_type not in ("gap", "gap_filling"):
                continue

            gap = self._build_gap_from_hypothesis(h, csp_triples, papers)
            if gap is not None:
                self._add_gap(gap)

        # 2. Build gaps from CSP coverage analysis
        csp_gaps = self._build_csp_coverage_gaps(csp_triples, papers)
        for g in csp_gaps:
            self._add_gap(g)

        # 3. Build gaps from failed trials
        if failed_trials:
            trial_gaps = self._build_failure_gaps(failed_trials, csp_triples, papers)
            for g in trial_gaps:
                self._add_gap(g)

        # 4. Deduplicate and score
        self._deduplicate()
        self._score_all()
        self._generate_reverse_search_queries()
        self._generate_reviewer_assessment()

        # 5. Sort and return top-N
        self._gaps.sort(
            key=lambda g: g.novelty_score * 0.5 + g.evidence_strength * 0.3 + g.feasibility_score * 0.2,
            reverse=True,
        )
        return self._gaps[:max_gaps]

    def _build_gap_from_hypothesis(
        self,
        h: Dict[str, Any],
        csp_triples: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
    ) -> Optional[GapEvidence]:
        """Build a Gap from a gap_filling hypothesis."""
        statement = h.get("statement", "")
        if not statement or len(statement) < 10:
            return None

        keywords = h.get("keywords", [])
        gap_id = f"gap_{h.get('hypothesis_id', '')[:12]}"

        # Extract composition and property from keywords for targeted counter-search
        comp = self._extract_composition_from_keywords(keywords, csp_triples)
        prop = self._extract_property_from_keywords(keywords, csp_triples)

        # Find supporting papers from hypothesis source
        supporting: List[Dict[str, Any]] = []
        for role, pid_key, title_key in [
            ("source_A", "paper_a_id", "paper_a_title"),
            ("source_B", "paper_b_id", "paper_b_title"),
        ]:
            pid = h.get(pid_key, "")
            ptitle = h.get(title_key, "")
            if pid:
                supporting.append({
                    "paper_id": pid,
                    "title": ptitle,
                    "role": role,
                    "relevance": "direct_source",
                })

        # Find additional supporting papers by keyword match
        for p in papers:
            if len(supporting) >= 5:
                break
            p_id = p.get("paper_id", p.get("id", ""))
            if any(s["paper_id"] == p_id for s in supporting):
                continue
            p_title = p.get("title", "")
            p_abstract = p.get("abstract", "")
            p_text = (p_title + " " + p_abstract).lower()
            matched_kw = [kw for kw in keywords if kw.lower() in p_text]
            if len(matched_kw) >= 2:
                supporting.append({
                    "paper_id": p_id,
                    "title": p_title[:100],
                    "role": "keyword_match",
                    "relevance": f"matched: {', '.join(matched_kw[:3])}",
                })

        # Need at least 2 supporting papers
        if len(supporting) < self.MIN_SUPPORTING_PAPERS:
            return None

        # Counter-evidence search: multi-strategy
        counter = self._search_counter_evidence(
            keywords, comp, prop, supporting, papers, csp_triples,
        )

        # CSP context
        csp_ctx: List[Dict[str, Any]] = []
        for t in csp_triples:
            t_comp = t.get("composition", "").lower()
            t_prop = t.get("property_name", "").lower()
            for kw in keywords[:3]:
                if kw.lower() in t_comp or kw.lower() in t_prop:
                    csp_ctx.append(t)
                    break
            if len(csp_ctx) >= 5:
                break

        # Determine gap type
        gap_type = "csp_missing"
        if "structure" in statement.lower() and "unknown" not in statement.lower():
            gap_type = "structure_unexplored"
        elif "property" in statement.lower() or "performance" in statement.lower():
            gap_type = "property_unpredicted"

        # Evidence chain
        chain = [
            f"Source papers: {len(supporting)} papers identified",
            f"CSP triples: {len(csp_ctx)} related triples found",
            f"Counter-evidence: {len(counter)} papers with overlapping coverage",
        ]
        if csp_ctx:
            chain.append(
                f"CSP context: {csp_ctx[0].get('composition', '?')}|"
                f"{csp_ctx[0].get('structure', '?')}|"
                f"{csp_ctx[0].get('property_name', '?')}"
            )

        # Verification plan
        if csp_ctx:
            comp_v = csp_ctx[0].get("composition", "target material")
            prop_v = csp_ctx[0].get("property_name", "target property")
            verif = (
                f"DFT calculation: compute {prop_v} of {comp_v} in the predicted structure; "
                f"compare with predicted range. "
                f"If no known experimental data exists, synthesize and measure."
            )
        else:
            verif = "Literature search + DFT calculation to verify the predicted property."

        return GapEvidence(
            gap_id=gap_id,
            gap_description=statement,
            gap_type=gap_type,
            keywords=keywords[:8],
            supporting_papers=supporting,
            counter_evidence=counter,
            csp_context=csp_ctx,
            evidence_chain=chain,
            verification_plan=verif,
        )

    def _search_counter_evidence(
        self,
        keywords: List[str],
        composition: str,
        property_name: str,
        supporting: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
        csp_triples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Multi-strategy counter-evidence search.

        Strategy 1: Exact composition + property match in CSP triples
        Strategy 2: Paper keyword overlap (threshold=2, lowered from 3)
        Strategy 3: Composition match in papers with different property
        """
        counter: List[Dict[str, Any]] = []
        supporting_ids = {s["paper_id"] for s in supporting}

        # Strategy 1: Check if CSP triples already cover this comp+prop
        if composition and property_name:
            for t in csp_triples:
                t_comp = t.get("composition", "").lower()
                t_prop = t.get("property_name", "").lower()
                if (composition.lower() == t_comp and
                    property_name.lower() == t_prop and
                    t.get("property_value") is not None):
                    # This comp+prop is already known — strong counter-evidence
                    src = t.get("source_paper_id", "")
                    if src and src not in supporting_ids:
                        counter.append({
                            "paper_id": src,
                            "title": t.get("source_paper_title", "")[:100],
                            "role": "exact_csp_match",
                            "relevance": f"CSP already has {composition}.{property_name}={t.get('property_value')}",
                        })

        # Strategy 2: Paper keyword overlap (lowered threshold to 2)
        for p in papers:
            if len(counter) >= 8:
                break
            p_id = p.get("paper_id", p.get("id", ""))
            if p_id in supporting_ids:
                continue
            if any(c["paper_id"] == p_id for c in counter):
                continue
            p_title = p.get("title", "")
            p_abstract = p.get("abstract", "")
            p_text = (p_title + " " + p_abstract).lower()
            matched_kw = [kw for kw in keywords[:5] if kw.lower() in p_text]
            if len(matched_kw) >= 2:
                counter.append({
                    "paper_id": p_id,
                    "title": p_title[:100],
                    "role": "keyword_overlap",
                    "relevance": f"overlapping keywords ({len(matched_kw)}): {', '.join(matched_kw[:3])}",
                })

        # Strategy 3: Composition match with any property
        if composition:
            for p in papers:
                if len(counter) >= 8:
                    break
                p_id = p.get("paper_id", p.get("id", ""))
                if p_id in supporting_ids:
                    continue
                if any(c["paper_id"] == p_id for c in counter):
                    continue
                p_title = p.get("title", "")
                p_abstract = p.get("abstract", "")
                p_text = (p_title + " " + p_abstract).lower()
                if composition.lower() in p_text:
                    counter.append({
                        "paper_id": p_id,
                        "title": p_title[:100],
                        "role": "composition_covered",
                        "relevance": f"mentions {composition} (may cover the gap)",
                    })

        return counter

    def _build_csp_coverage_gaps(
        self,
        csp_triples: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
    ) -> List[GapEvidence]:
        """Find gaps by analyzing CSP coverage: compositions with known structure
        but missing property values, or vice versa."""
        gaps: List[GapEvidence] = []

        # Group by composition
        comp_data: Dict[str, List[Dict[str, Any]]] = {}
        for t in csp_triples:
            comp = t.get("composition", "")
            if not comp:
                continue
            comp_data.setdefault(comp, []).append(t)

        for comp, triples in comp_data.items():
            if len(triples) < 1:
                continue

            # Find structures known for this composition
            known_structs = {t.get("structure", "") for t in triples if t.get("structure")}
            # Find properties measured
            known_props = {
                t.get("property_name", ""): t.get("property_value")
                for t in triples
                if t.get("property_name")
            }

            # Gap: composition has structure but no property value
            missing_props = [p for p, v in known_props.items() if v is None]
            if not missing_props and len(known_props) >= 2:
                # Check if there's a property that hasn't been studied at all
                all_known_props = set(known_props.keys())
                # Look for related compositions that have more properties
                for other_comp, other_triples in comp_data.items():
                    if other_comp == comp:
                        continue
                    other_props = {t.get("property_name", "") for t in other_triples}
                    missing_from_comp = other_props - all_known_props
                    if missing_from_comp and len(missing_from_comp) <= 3:
                        # This composition is missing properties that similar materials have
                        supporting = self._find_papers_for_composition(comp, papers)
                        if len(supporting) >= self.MIN_SUPPORTING_PAPERS:
                            gap_id = f"gap_csp_{hashlib.md5(comp.encode()).hexdigest()[:8]}"
                            missing_str = ", ".join(list(missing_from_comp)[:3])
                            gap = GapEvidence(
                                gap_id=gap_id,
                                gap_description=(
                                    f"{comp} has been studied for {', '.join(list(all_known_props)[:3])}, "
                                    f"but its {missing_str} has not been reported, "
                                    f"despite being measured for related {other_comp}."
                                ),
                                gap_type="property_unpredicted",
                                keywords=[comp] + list(missing_from_comp)[:3],
                                supporting_papers=supporting[:5],
                                csp_context=[t for t in triples[:3]],
                                evidence_chain=[
                                    f"Composition: {comp}",
                                    f"Known properties: {', '.join(list(all_known_props)[:5])}",
                                    f"Missing (found in related materials): {missing_str}",
                                    f"Supporting papers: {len(supporting)}",
                                ],
                                verification_plan=(
                                    f"DFT/experimental measurement of {missing_str} for {comp}."
                                ),
                            )
                            # Add counter-evidence for this gap
                            first_missing = list(missing_from_comp)[0] if missing_from_comp else ""
                            gap.counter_evidence = self._search_counter_evidence(
                                [comp] + list(missing_from_comp)[:3],
                                comp, first_missing,
                                gap.supporting_papers, papers, csp_triples,
                            )
                            gaps.append(gap)
                            break

        return gaps

    def _build_failure_gaps(
        self,
        failed_trials: List[Dict[str, Any]],
        csp_triples: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
    ) -> List[GapEvidence]:
        """Build gaps from repeated trial failures."""
        gaps: List[GapEvidence] = []

        # Group failures by keywords
        kw_failures: Dict[str, List[Dict[str, Any]]] = {}
        for f in failed_trials:
            for kw in f.get("keywords", [])[:3]:
                kw_failures.setdefault(kw, []).append(f)

        for kw, fails in kw_failures.items():
            if len(fails) < 3:
                continue

            supporting = self._find_papers_for_keyword(kw, papers)
            if len(supporting) < self.MIN_SUPPORTING_PAPERS:
                continue

            gap_id = f"gap_fail_{hashlib.md5(kw.encode()).hexdigest()[:8]}"
            gap = GapEvidence(
                gap_id=gap_id,
                gap_description=(
                    f"Multiple trials ({len(fails)} attempts) targeting '{kw}' "
                    f"have failed, suggesting an unaddressed technical challenge "
                    f"or knowledge gap in this area."
                ),
                gap_type="trial_failure_cluster",
                keywords=[kw],
                supporting_papers=supporting[:5],
                evidence_chain=[
                    f"Keyword: {kw}",
                    f"Failed trials: {len(fails)}",
                    f"Failure pattern: repeated low scores",
                ],
                verification_plan=(
                    f"Analyze failure causes for '{kw}' trials; "
                    f"consider alternative approaches or identify missing physical constraints."
                ),
            )
            gap.counter_evidence = self._search_counter_evidence(
                [kw], kw, "", gap.supporting_papers, papers, csp_triples,
            )
            gaps.append(gap)

        return gaps

    def _generate_reverse_search_queries(self) -> None:
        """Generate 3-5 synonymous Sciverse search queries for each gap."""
        for g in self._gaps:
            queries: List[str] = []

            # Extract composition and property from keywords
            comp = ""
            prop = ""
            for kw in g.keywords:
                kw_lower = kw.lower()
                # Check if keyword looks like a composition (has uppercase + optional digits)
                if not comp and any(c.isupper() for c in kw) and len(kw) <= 15:
                    comp = kw
                # Check if keyword matches a known property
                if not prop:
                    for prop_name, synonyms in self._PROPERTY_SYNONYMS.items():
                        if kw_lower == prop_name.lower() or kw_lower in [s.lower() for s in synonyms]:
                            prop = prop_name
                            break

            # Generate queries
            if comp and prop:
                synonyms = self._PROPERTY_SYNONYMS.get(prop, [prop])
                queries.append(f"{comp} {prop}")
                for syn in synonyms[:2]:
                    queries.append(f"{comp} {syn}")
                queries.append(f"{comp} {prop} measurement")
                queries.append(f"{comp} {prop} calculation DFT")
            elif comp:
                queries.append(f"{comp} properties")
                queries.append(f"{comp} structure")
                queries.append(f"{comp} characterization")
            elif g.keywords:
                queries.append(" ".join(g.keywords[:3]))
                queries.append(" ".join(g.keywords[:2]) + " study")
                queries.append(" ".join(g.keywords[:2]) + " review")

            g.reverse_search_queries = queries[:5]
            g.reverse_search_done = len(queries) > 0

    def _generate_reviewer_assessment(self) -> None:
        """Generate a scientific peer-review style assessment for each gap."""
        for g in self._gaps:
            n_support = len(g.supporting_papers)
            n_counter = len(g.counter_evidence)
            n_csp = len(g.csp_context)
            has_exact_match = any(
                c.get("role") == "exact_csp_match" for c in g.counter_evidence
            )

            # Why not known
            if has_exact_match:
                g.why_not_known = (
                    "This gap may already be partially addressed: "
                    "an exact CSP match was found in the knowledge base. "
                    "However, the prediction range or structural condition may still be novel."
                )
            elif n_counter == 0:
                g.why_not_known = (
                    "No counter-evidence was found in the current corpus. "
                    "This does not guarantee novelty beyond the corpus — "
                    "a reverse Sciverse search is recommended to confirm."
                )
            elif n_counter <= 2:
                g.why_not_known = (
                    f"{n_counter} paper(s) with overlapping coverage were found, "
                    f"but none directly addresses the specific combination of "
                    f"composition + structure + property identified in this gap. "
                    f"The gap is likely partially novel."
                )
            else:
                g.why_not_known = (
                    f"{n_counter} papers with overlapping coverage were found. "
                    f"While no single paper fully covers this gap, the high overlap "
                    f"suggests it may be close to existing work. "
                    f"A detailed manual review is recommended."
                )

            # Reviewer assessment
            if n_support >= 3 and n_counter == 0 and n_csp >= 1:
                g.reviewer_assessment = (
                    "Strong candidate: well-supported by multiple papers, "
                    "no counter-evidence found, CSP context available. "
                    "Recommend prioritizing for DFT verification."
                )
            elif n_support >= 2 and n_counter <= 1:
                g.reviewer_assessment = (
                    "Moderate candidate: supported by sufficient papers, "
                    "limited counter-evidence. Worth investigating but "
                    "reverse literature search needed before claiming novelty."
                )
            elif n_support >= 2 and n_counter <= 3:
                g.reviewer_assessment = (
                    "Conditional candidate: supported but with notable overlapping work. "
                    "Novelty depends on the specific structural/processing conditions. "
                    "Requires expert judgment to differentiate from existing studies."
                )
            else:
                g.reviewer_assessment = (
                    "Weak candidate: insufficient support or too much overlapping evidence. "
                    "Not recommended for priority verification."
                )

    def _score_all(self) -> None:
        """Compute novelty, feasibility, and evidence_strength scores for all gaps.

        Improved scoring:
        - Novelty: considers counter-evidence type (exact match vs keyword overlap)
        - Evidence strength: weighted by evidence type
        - Feasibility: considers CSP context + verification plan + counter-evidence
        """
        for g in self._gaps:
            n_papers = len(g.supporting_papers)
            n_csp = len(g.csp_context)
            n_counter = len(g.counter_evidence)

            # Count counter-evidence by type
            n_exact = sum(1 for c in g.counter_evidence if c.get("role") == "exact_csp_match")
            n_overlap = sum(1 for c in g.counter_evidence if c.get("role") == "keyword_overlap")
            n_comp = sum(1 for c in g.counter_evidence if c.get("role") == "composition_covered")

            # Evidence strength: supporting papers + CSP coverage
            g.evidence_strength = min(1.0, n_papers * 0.15 + n_csp * 0.10)

            # Novelty scoring: differentiated by counter-evidence type
            if n_exact > 0:
                # Exact CSP match found — likely already known
                g.novelty_score = max(0.05, 0.2 - n_exact * 0.05)
            elif n_counter == 0:
                # No counter-evidence at all
                g.novelty_score = min(0.95, 0.55 + n_papers * 0.08)
            else:
                # Has counter-evidence but no exact match
                # Keyword overlap reduces novelty less than composition coverage
                penalty = n_overlap * 0.08 + n_comp * 0.12
                g.novelty_score = max(0.15, 0.65 - penalty)

            # Feasibility: has CSP context + verification plan + low counter-evidence
            base_feasibility = 0.3
            if g.csp_context:
                base_feasibility += 0.2
            if g.verification_plan:
                base_feasibility += 0.2
            if n_counter <= 1:
                base_feasibility += 0.15
            if n_exact > 0:
                base_feasibility -= 0.1  # already known = less interesting to verify
            g.feasibility_score = min(0.95, max(0.1, base_feasibility))

            # Penalty: too few supporting papers
            if n_papers < self.MIN_SUPPORTING_PAPERS:
                g.novelty_score *= 0.3
                g.evidence_strength *= 0.3

    def _extract_composition_from_keywords(
        self, keywords: List[str], csp_triples: List[Dict[str, Any]],
    ) -> str:
        """Extract the most likely composition from keywords."""
        for kw in keywords:
            for t in csp_triples:
                if kw.lower() in t.get("composition", "").lower():
                    return t["composition"]
            # Check if keyword itself looks like a formula
            import re
            if re.match(r'^[A-Z][a-z]?\d*', kw) and len(kw) >= 2 and len(kw) <= 15:
                return kw
        return ""

    def _extract_property_from_keywords(
        self, keywords: List[str], csp_triples: List[Dict[str, Any]],
    ) -> str:
        """Extract the most likely property name from keywords."""
        for kw in keywords:
            for t in csp_triples:
                if kw.lower() == t.get("property_name", "").lower():
                    return t["property_name"]
            # Check synonym groups
            for prop_name, synonyms in self._PROPERTY_SYNONYMS.items():
                if kw.lower() == prop_name.lower() or kw.lower() in [s.lower() for s in synonyms]:
                    return prop_name
        return ""

    def _find_papers_for_composition(
        self, composition: str, papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find papers mentioning a composition."""
        result = []
        comp_lower = composition.lower()
        for p in papers:
            text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            if comp_lower in text:
                result.append({
                    "paper_id": p.get("paper_id", p.get("id", "")),
                    "title": p.get("title", "")[:100],
                    "role": "composition_match",
                    "relevance": f"mentions {composition}",
                })
        return result

    def _find_papers_for_keyword(
        self, keyword: str, papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find papers mentioning a keyword."""
        result = []
        kw_lower = keyword.lower()
        for p in papers:
            text = (p.get("title", "") + " " + p.get("abstract", "")).lower()
            if kw_lower in text:
                result.append({
                    "paper_id": p.get("paper_id", p.get("id", "")),
                    "title": p.get("title", "")[:100],
                    "role": "keyword_match",
                    "relevance": f"mentions {keyword}",
                })
        return result

    def _add_gap(self, gap: GapEvidence) -> None:
        """Add a gap, skipping duplicates."""
        # Dedup key: first 3 keywords sorted
        key_parts = sorted(gap.keywords[:3])
        key = "|".join(key_parts) + ":" + gap.gap_type
        if key in self._seen_keys:
            # Merge: add supporting papers
            for existing in self._gaps:
                existing_key = "|".join(sorted(existing.keywords[:3])) + ":" + existing.gap_type
                if existing_key == key:
                    for sp in gap.supporting_papers:
                        if not any(s["paper_id"] == sp["paper_id"] for s in existing.supporting_papers):
                            existing.supporting_papers.append(sp)
                    # Merge counter-evidence too
                    for ce in gap.counter_evidence:
                        if not any(c["paper_id"] == ce["paper_id"] for c in existing.counter_evidence):
                            existing.counter_evidence.append(ce)
                    return
            return
        self._seen_keys.add(key)
        self._gaps.append(gap)

    def _deduplicate(self) -> None:
        """Remove near-duplicate gaps."""
        if len(self._gaps) <= 1:
            return

        unique: List[GapEvidence] = []
        seen_descs: List[str] = []

        for g in self._gaps:
            # Simple dedup: check first 50 chars of description
            desc_prefix = g.gap_description[:50].lower()
            is_dup = False
            for seen_desc in seen_descs:
                # Jaccard similarity on words
                w1 = set(desc_prefix.split())
                w2 = set(seen_desc.split())
                if w1 and w2:
                    jaccard = len(w1 & w2) / len(w1 | w2)
                    if jaccard > 0.7:
                        is_dup = True
                        break
            if not is_dup:
                unique.append(g)
                seen_descs.append(desc_prefix)

        self._gaps = unique
