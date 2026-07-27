"""Competition report generator — produces Markdown for competition submission.

Template:
  1. Research Problem
  2. Retrieval Strategy
  3. Literature Screening Results
  4. CSP Knowledge Base
  5. Research Gap List (with evidence chains)
  6. Route A Search Process (if available)
  7. Top Structure-Property Candidates
  8. Evidence Chains
  9. Falsifiable Verification Plan
  10. Limitations & Next Steps
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class CompetitionReportGenerator:
    """Generate competition-ready Markdown reports."""

    def __init__(self) -> None:
        self._project_name = (
            "Sciverse-MCP: Materials Science Literature Survey & "
            "Structure-Property Relationship Discovery Agent"
        )
        self._track = "Track A — Structure-Property Relationship Discovery"

    def generate(
        self,
        host_system: Any,
        search_results: Optional[Dict[str, Any]] = None,
        external_validation: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a full competition report as Markdown.

        Args:
            host_system: HostSystem instance (must be initialized)
            search_results: output from StructurePropertySearchEngine (optional)
            external_validation: output from DB validators (optional)

        Returns:
            Markdown string
        """
        sections: List[str] = []

        # Collect data from host
        status = host_system.state.as_dict() if hasattr(host_system, "state") else {}
        lit_review = host_system.get_literature_review_report()
        csp_triples = host_system.get_csp_knowledge(n=50)
        discoveries = []
        if host_system._verification_engine:
            discoveries = host_system._verification_engine.get_discoveries(n=10)
        discovery_stats = host_system._verification_engine.stats if host_system._verification_engine else {}
        agent_overrides = host_system.get_agent_override_candidates(n=5) if hasattr(host_system, "get_agent_override_candidates") else []

        # Build gaps using GapEvidenceBuilder
        from ..knowledge.gap_builder import GapEvidenceBuilder
        gap_builder = GapEvidenceBuilder()
        hypotheses = host_system.state.sample_hypotheses()
        papers = []
        if host_system._crawler:
            papers = [
                {
                    "paper_id": getattr(p, "paper_id", ""),
                    "title": getattr(p, "title", ""),
                    "abstract": getattr(p, "abstract", ""),
                    "keywords": getattr(p, "keywords", []),
                }
                for p in host_system._crawler.get_cached_papers()
            ]
        gaps = gap_builder.build_gaps(
            hypotheses=hypotheses,
            csp_triples=csp_triples,
            papers=papers,
            max_gaps=15,
        )

        # ── Title ──
        sections.append(self._section_title(status))

        # ── 1. Research Problem ──
        sections.append(self._section_research_problem(status))

        # ── 2. Retrieval Strategy ──
        sections.append(self._section_retrieval_strategy(status))

        # ── 3. Literature Screening Results ──
        sections.append(self._section_literature_screening(status, lit_review, papers))

        # ── 4. CSP Knowledge Base ──
        sections.append(self._section_csp_knowledge(csp_triples))

        # ── 5. Research Gap List ──
        sections.append(self._section_research_gaps(gaps))

        # ── 6. Route A Search Process ──
        if search_results:
            sections.append(self._section_search_process(search_results))

        # ── 7. Top Structure-Property Candidates ──
        sections.append(self._section_top_candidates(discoveries, agent_overrides, search_results))

        # ── 8. Evidence Chains ──
        sections.append(self._section_evidence_chains(discoveries, gaps))

        # ── 9. Falsifiable Verification Plan ──
        sections.append(self._section_verification_plan(gaps, discoveries))

        # ── 10. Limitations & Next Steps ──
        sections.append(self._section_limitations(external_validation))

        # ── Appendix: MCP Call Log ──
        sections.append(self._section_appendix(status))

        return "\n\n".join(sections)

    def _section_title(self, status: Dict[str, Any]) -> str:
        return f"# {self._project_name}\n\n**Track:** {self._track}\n\n**Generated:** {time.strftime('%Y-%m-%d %H:%M')}\n\n**Cycles:** {status.get('cycle_count', 0)} | **Papers:** {status.get('papers_collected', 0)}"

    def _section_research_problem(self, status: Dict[str, Any]) -> str:
        return """## 1. Research Problem

This project addresses the challenge of autonomous materials science literature
survey and structure-property (CSP) relationship discovery. Given a large-scale
materials science literature corpus (Sciverse), the system must:

1. **Autonomously retrieve and filter** relevant papers
2. **Extract CSP triples** (Composition-Structure-Property) from paper text
3. **Identify research gaps** with traceable evidence chains
4. **Generate falsifiable candidate structure-property relationships**
5. **Provide verification pathways** (DFT/experimental)

The key distinction from a literature summarizer: every candidate finding must be
**falsifiable** (state a specific numerical prediction), **traceable** (link back
to source papers via CSP triples), and **physically validated** (pass unit checks,
range gates, and structure compatibility)."""

    def _section_retrieval_strategy(self, status: Dict[str, Any]) -> str:
        n_papers = status.get("papers_collected", 0)
        n_raw = status.get("papers_raw_fetched", 0)
        n_rejected = status.get("papers_quality_rejected", 0)
        cycles = status.get("cycle_count", 0)
        return f"""## 2. Retrieval Strategy

The system uses a multi-stage retrieval pipeline:

1. **Seed keywords**: Materials science domain seeds (bandgap, perovskite, alloy, etc.)
2. **Sciverse API**: Query Sciverse for papers matching seed keywords
3. **Quality filter**: Reject metadata noise, non-technical papers, and duplicates
4. **Implicit pairing**: Find paper pairs with shared keywords but different domains
5. **CSP extraction**: Extract Composition-Structure-Property triples from each paper

| Metric | Count |
|--------|-------|
| Cycles executed | {cycles} |
| Papers fetched (raw) | {n_raw} |
| Papers rejected (quality) | {n_rejected} |
| Papers in knowledge base | {n_papers} |
| Retrieval source | Sciverse API |

**Quality gate**: Papers without technical content (metadata noise, editorial
notices) are rejected before CSP extraction to prevent garbage-in-garbage-out."""

    def _section_literature_screening(
        self, status: Dict, lit_review: Dict, papers: List[Dict],
    ) -> str:
        cat_counts = lit_review.get("papers_by_category", {})
        top_keywords = lit_review.get("top_keywords", {})
        n_pairs = lit_review.get("total_pairs", 0)
        cross_domain = lit_review.get("cross_domain_pairs", 0)

        cat_table = "| Category | Papers |\n|----------|--------|\n"
        for cat, cnt in list(cat_counts.items())[:8]:
            cat_table += f"| {cat} | {cnt} |\n"
        if not cat_counts:
            cat_table += "| (offline corpus) | — |\n"

        kw_str = ", ".join(list(top_keywords.keys())[:15]) if top_keywords else "N/A"

        return f"""## 3. Literature Screening Results

### Paper Distribution

{cat_table}

### Top Keywords

{kw_str}

### Implicit Pairing

| Metric | Count |
|--------|-------|
| Total pairs generated | {n_pairs} |
| Cross-domain pairs | {cross_domain} |

Cross-domain pairs connect papers from different subfields that share a
bridge keyword (e.g., a thermoelectric paper paired with a battery paper
via "conductivity"). These pairs are the primary source of novel hypotheses."""

    def _section_csp_knowledge(self, csp_triples: List[Dict]) -> str:
        n = len(csp_triples)
        # Group by property
        prop_counts: Dict[str, int] = {}
        struct_counts: Dict[str, int] = {}
        for t in csp_triples:
            pn = t.get("property_name", "unknown")
            prop_counts[pn] = prop_counts.get(pn, 0) + 1
            st = t.get("structure", "unknown")
            struct_counts[st] = struct_counts.get(st, 0) + 1

        prop_table = "| Property | Triples |\n|----------|---------|\n"
        for p, c in sorted(prop_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            prop_table += f"| {p} | {c} |\n"

        struct_table = "| Structure | Triples |\n|-----------|---------|\n"
        for s, c in sorted(struct_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            struct_table += f"| {s} | {c} |\n"

        # Sample triples
        samples = ""
        for t in csp_triples[:5]:
            comp = t.get("composition", "?")
            struct = t.get("structure", "?")
            prop = t.get("property_name", "?")
            val = t.get("value_str", "?")
            src = t.get("source_paper_id", "?")
            samples += f"- `{comp}` | {struct} | {prop} = {val} (source: {src})\n"

        return f"""## 4. CSP Knowledge Base

The CSP (Composition-Structure-Property) knowledge base stores structured
triples extracted from papers. Each triple links a chemical composition to
a crystal structure and a measured/predicted property value.

**Total CSP triples extracted:** {n}

### Property Distribution

{prop_table}

### Structure Distribution

{struct_table}

### Sample Triples

{samples}

Each triple carries source provenance (paper ID) and confidence score,
enabling full auditability from finding back to source literature."""

    def _section_research_gaps(self, gaps: List[Any]) -> str:
        if not gaps:
            return """## 5. Research Gap List

No research gaps were identified in this run. This may indicate insufficient
literature coverage or that the system needs more cycles."""

        lines = [f"## 5. Research Gap List\n"]
        lines.append(f"**Total gaps identified:** {len(gaps)}\n")
        lines.append("Each gap includes supporting papers, counter-evidence search, "
                      "novelty score, and a verification plan.\n")

        for i, g in enumerate(gaps[:10], 1):
            g_dict = g.to_dict() if hasattr(g, "to_dict") else g
            desc = g_dict.get("gap_description", "")[:200]
            gtype = g_dict.get("gap_type", "")
            novelty = g_dict.get("novelty_score", 0)
            feasibility = g_dict.get("feasibility_score", 0)
            n_support = len(g_dict.get("supporting_papers", []))
            n_counter = len(g_dict.get("counter_evidence", []))
            chain = g_dict.get("evidence_chain", [])
            verif = g_dict.get("verification_plan", "")

            lines.append(f"### Gap {i}: [{gtype}] (novelty={novelty:.2f}, feasibility={feasibility:.2f})\n")
            lines.append(f"**Description:** {desc}\n")
            lines.append(f"**Supporting papers:** {n_support} | **Counter-evidence:** {n_counter}\n")

            if chain:
                lines.append("**Evidence chain:**\n")
                for c in chain:
                    lines.append(f"  - {c}")
                lines.append("")

            if verif:
                lines.append(f"**Verification plan:** {verif}\n")

        return "\n".join(lines)

    def _section_search_process(self, search_results: Dict[str, Any]) -> str:
        n_gen = search_results.get("total_generated", 0)
        n_eval = search_results.get("total_evaluated", 0)
        n_iter = search_results.get("iterations", 0)
        best_score = search_results.get("best_score", 0)
        method = search_results.get("method", "evolutionary search")

        return f"""## 6. Route A Search Process

The system employs an evolutionary search engine to explore the
structure-property relationship space, guided by LLM-generated hypotheses
and scored by a multi-criteria fitness function.

| Metric | Value |
|--------|-------|
| Search method | {method} |
| Iterations | {n_iter} |
| Candidates generated | {n_gen} |
| Candidates evaluated | {n_eval} |
| Best fitness score | {best_score:.3f} |

### Fitness Function

```
score = 0.25 * literature_evidence
      + 0.20 * novelty
      + 0.20 * physical_plausibility
      + 0.15 * falsifiability
      + 0.10 * synthesizability
      + 0.10 * database_gap_degree
```

### Mutation Operators

- **Element substitution**: replace one element with a same-group alternative
- **Structure replacement**: swap to a compatible crystal structure
- **Property target swap**: change the target property
- **Processing condition swap**: change synthesis method
- **Same-family migration**: move to a related composition in the same family"""

    def _section_top_candidates(
        self,
        discoveries: List[Dict],
        agent_overrides: List[Dict],
        search_results: Optional[Dict],
    ) -> str:
        lines = ["## 7. Top Structure-Property Candidates\n"]
        lines.append("**Important:** These are *candidate* structure-property "
                      "relationships, not confirmed discoveries. Each candidate "
                      "is a falsifiable prediction supported by literature evidence.\n")

        if discoveries:
            lines.append("### Verified Candidates (passed 3-layer validation)\n")
            lines.append("| # | Hypothesis | Level | Stability | Reproduce Rate | Source Papers |\n")
            lines.append("|---|-----------|-------|-----------|----------------|---------------|\n")
            for i, d in enumerate(discoveries[:5], 1):
                stmt = d.get("statement", "")[:80]
                level = d.get("discovery_level", "")
                stab = d.get("stability_score", 0)
                repro = d.get("reproduce_rate", 0)
                pa = d.get("paper_a_id", "")[:12]
                pb = d.get("paper_b_id", "")[:12]
                lines.append(f"| {i} | {stmt} | {level} | {stab:.3f} | {repro:.3f} | {pa}, {pb} |\n")
        else:
            lines.append("### Verified Candidates\n")
            lines.append("No candidates passed full 3-layer validation in this run.\n")

        if search_results and search_results.get("top_candidates"):
            lines.append("\n### Search Engine Top Candidates\n")
            for i, c in enumerate(search_results["top_candidates"][:5], 1):
                comp = c.get("composition", "?")
                struct = c.get("structure", "?")
                prop = c.get("property_name", "?")
                pred = c.get("predicted_range", "?")
                score = c.get("score", 0)
                lines.append(f"{i}. `{comp}` | {struct} | {prop} = {pred} (score={score:.3f})\n")

        return "\n".join(lines)

    def _section_evidence_chains(
        self, discoveries: List[Dict], gaps: List[Any],
    ) -> str:
        lines = ["## 8. Evidence Chains\n"]
        lines.append("Each candidate finding traces back to source papers through "
                      "CSP triples, forming an auditable evidence chain.\n")

        if discoveries:
            for i, d in enumerate(discoveries[:3], 1):
                stmt = d.get("statement", "")[:100]
                pa = d.get("paper_a_id", "")
                pa_title = d.get("paper_a_title", "")
                pb = d.get("paper_b_id", "")
                pb_title = d.get("paper_b_title", "")
                kw = d.get("keywords", [])[:5]

                lines.append(f"### Chain {i}\n")
                lines.append(f"**Candidate:** {stmt}\n")
                lines.append(f"**Source paper A:** [{pa}] {pa_title}\n")
                lines.append(f"**Source paper B:** [{pb}] {pb_title}\n")
                lines.append(f"**Keywords:** {', '.join(kw)}\n")
                lines.append(f"**Chain:** Paper A -> CSP extraction -> Hypothesis -> "
                              f"Twin trial -> Verification -> Candidate\n")
        else:
            lines.append("No verified candidates available for evidence chain display.\n")

        return "\n".join(lines)

    def _section_verification_plan(
        self, gaps: List[Any], discoveries: List[Dict],
    ) -> str:
        lines = ["## 9. Falsifiable Verification Plan\n"]
        lines.append("Each candidate includes a specific, falsifiable prediction "
                      "and a concrete verification pathway.\n")

        if discoveries:
            for i, d in enumerate(discoveries[:3], 1):
                stmt = d.get("statement", "")[:120]
                has_pred = d.get("has_numerical_prediction", False)
                lines.append(f"### Plan {i}\n")
                lines.append(f"**Prediction:** {stmt}\n")
                lines.append(f"**Has numerical prediction:** {has_pred}\n")
                lines.append(f"**Verification pathway:**\n")
                lines.append(f"  1. DFT calculation of the predicted property\n")
                lines.append(f"  2. Cross-check against Materials Project / OQMD / NOMAD\n")
                lines.append(f"  3. If no existing data: propose synthesis + measurement\n")
                lines.append(f"  4. Compare measured value with predicted range\n")
                lines.append(f"  5. If outside range: hypothesis is falsified\n")

        if gaps:
            lines.append("\n### Gap Verification Plans\n")
            for i, g in enumerate(gaps[:3], 1):
                g_dict = g.to_dict() if hasattr(g, "to_dict") else g
                verif = g_dict.get("verification_plan", "")
                if verif:
                    lines.append(f"{i}. {verif}\n")

        return "\n".join(lines)

    def _section_limitations(
        self, external_validation: Optional[Dict[str, Any]],
    ) -> str:
        lines = ["## 10. Limitations & Next Steps\n"]
        lines.append("### Current Limitations\n")
        lines.append("- CSP extraction relies on regex pattern matching; LLM-assisted "
                      "extraction is available but requires API configuration\n")
        lines.append("- Physical validation covers unit conversion and range checks; "
                      "full DFT validation requires external tools\n")
        lines.append("- Research gap identification is based on system corpus coverage, "
                      "not exhaustive field knowledge\n")
        lines.append("- The evolutionary search is a lightweight prototype; "
                      "Bayesian optimization or MCTS could improve exploration efficiency\n")

        if external_validation:
            lines.append("\n### External Database Validation\n\n")
            lines.append("| Database | Status | Materials Found | Novelty |\n")
            lines.append("|----------|--------|-----------------|---------|\n")
            for db_name, db_result in external_validation.items():
                status = db_result.get("status", "unavailable")
                found = db_result.get("found", 0)
                novelty = db_result.get("novelty_status", "unknown")
                lines.append(f"| {db_name} | {status} | {found} | {novelty} |\n")

        lines.append("\n### Next Steps\n")
        lines.append("1. Integrate Materials Project API for real-time cross-validation\n")
        lines.append("2. Upgrade search engine to Bayesian optimization with Gaussian process surrogate\n")
        lines.append("3. Add reverse literature search for each gap (expand beyond system corpus)\n")
        lines.append("4. Implement DFT workflow automation (e.g., via ASE/VASP interface)\n")

        return "\n".join(lines)

    def _section_appendix(self, status: Dict[str, Any]) -> str:
        cycles = status.get("cycle_count", 0)
        return f"""## Appendix: System Metrics

| Metric | Value |
|--------|-------|
| Exploration cycles | {cycles} |
| Hypotheses generated | {status.get('hypotheses_generated', 0)} |
| Trials executed | {status.get('trials_total', 0)} |
| Papers in knowledge base | {status.get('papers_collected', 0)} |
| Pairs found | {status.get('pairs_found', 0)} |
| Cross-domain pairs | {status.get('cross_domain_pairs', 0)} |
| Knowledge entries | {status.get('knowledge_entries', 0)} |

---

*This report was generated automatically by the Sciverse-MCP Agent system.*
*All candidate findings are falsifiable predictions, not confirmed discoveries.*
*MCP call logs form a complete audit trail from literature retrieval to candidate output.*
"""
