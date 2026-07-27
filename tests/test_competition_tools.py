#!/usr/bin/env python3
"""Test P0+P1+P2 competition tools: GapEvidenceBuilder, SearchEngine, Report, DB validators."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sci_host.knowledge.gap_builder import GapEvidenceBuilder, GapEvidence
from sci_host.materials.search_engine import StructurePropertySearchEngine, SearchCandidate
from sci_host.materials.db_validators import ExternalDBValidator
from sci_host.reporting import CompetitionReportGenerator


def test_gap_evidence_builder():
    """Test GapEvidenceBuilder with mock data."""
    print("=" * 60)
    print("TEST 1: GapEvidenceBuilder")
    print("=" * 60)

    hypotheses = [
        {
            "hypothesis_id": "h001",
            "hypothesis_type": "gap_filling",
            "statement": "The bandgap of SnSe in orthorhombic structure has not been systematically studied at high pressure, despite its importance for thermoelectric applications.",
            "keywords": ["SnSe", "bandgap", "orthorhombic", "thermoelectric"],
            "paper_a_id": "paper_001",
            "paper_a_title": "Thermoelectric properties of SnSe",
            "paper_b_id": "paper_002",
            "paper_b_title": "High-pressure phases of layered chalcogenides",
        },
        {
            "hypothesis_id": "h002",
            "hypothesis_type": "gap",
            "statement": "No study has reported the thermal conductivity of BaTiO3 in the cubic phase above the Curie temperature.",
            "keywords": ["BaTiO3", "thermal_conductivity", "cubic"],
            "paper_a_id": "paper_003",
            "paper_a_title": "Ferroelectric transitions in BaTiO3",
            "paper_b_id": "paper_004",
            "paper_b_title": "Thermal properties of perovskite oxides",
        },
    ]

    csp_triples = [
        {"composition": "SnSe", "structure": "orthorhombic", "property_name": "bandgap", "property_value": 0.9, "property_unit": "eV", "source_paper_id": "paper_001"},
        {"composition": "BaTiO3", "structure": "tetragonal", "property_name": "bandgap", "property_value": 3.2, "property_unit": "eV", "source_paper_id": "paper_003"},
        {"composition": "BaTiO3", "structure": "cubic", "property_name": "bandgap", "property_value": 3.4, "property_unit": "eV", "source_paper_id": "paper_003"},
        {"composition": "PbTiO3", "structure": "tetragonal", "property_name": "bandgap", "property_value": 3.0, "property_unit": "eV", "source_paper_id": "paper_005"},
        {"composition": "PbTiO3", "structure": "tetragonal", "property_name": "thermal_conductivity", "property_value": 2.5, "property_unit": "W/mK", "source_paper_id": "paper_005"},
    ]

    papers = [
        {"paper_id": "paper_001", "title": "Thermoelectric properties of SnSe crystals", "abstract": "SnSe shows excellent thermoelectric performance with ZT=2.6. The orthorhombic structure is key."},
        {"paper_id": "paper_002", "title": "High-pressure phases of layered chalcogenides", "abstract": "We study SnSe and SnS under high pressure. Phase transitions observed."},
        {"paper_id": "paper_003", "title": "Ferroelectric transitions in BaTiO3", "abstract": "BaTiO3 undergoes tetragonal to cubic transition at Curie temperature 120C."},
        {"paper_id": "paper_004", "title": "Thermal properties of perovskite oxides", "abstract": "Thermal conductivity of perovskite oxides including BaTiO3 and SrTiO3."},
        {"paper_id": "paper_005", "title": "Properties of PbTiO3", "abstract": "PbTiO3 tetragonal phase bandgap and thermal conductivity measured."},
    ]

    builder = GapEvidenceBuilder()
    gaps = builder.build_gaps(
        hypotheses=hypotheses,
        csp_triples=csp_triples,
        papers=papers,
        max_gaps=10,
    )

    print(f"  Gaps built: {len(gaps)}")
    for i, g in enumerate(gaps):
        print(f"  Gap {i+1}: [{g.gap_type}] novelty={g.novelty_score:.2f} "
              f"evidence={g.evidence_strength:.2f} "
              f"supporting={len(g.supporting_papers)} "
              f"counter={len(g.counter_evidence)}")
        print(f"    Description: {g.gap_description[:80]}...")

    assert len(gaps) >= 1, "Should have at least 1 gap"
    assert all(len(g.supporting_papers) >= 2 for g in gaps), "Each gap needs >=2 supporting papers"
    print("  PASS: GapEvidenceBuilder\n")


def test_search_engine():
    """Test StructurePropertySearchEngine."""
    print("=" * 60)
    print("TEST 2: StructurePropertySearchEngine")
    print("=" * 60)

    hypotheses = [
        {
            "hypothesis_id": "h001",
            "keywords": ["BaTiO3", "perovskite", "bandgap"],
            "statement": "BaTiO3 perovskite bandgap is approximately 3.2 eV",
            "paper_a_id": "p001",
            "paper_b_id": "p002",
        },
        {
            "hypothesis_id": "h002",
            "keywords": ["SnSe", "orthorhombic", "thermoelectric"],
            "statement": "SnSe orthorhombic shows ZT in range 2.0-2.8",
            "paper_a_id": "p003",
            "paper_b_id": "p004",
        },
    ]

    csp_triples = [
        {"composition": "BaTiO3", "structure": "perovskite", "property_name": "bandgap", "property_value": 3.2, "property_unit": "eV", "source_paper_id": "p001"},
        {"composition": "BaTiO3", "structure": "tetragonal", "property_name": "bandgap", "property_value": 3.1, "property_unit": "eV", "source_paper_id": "p001"},
        {"composition": "SnSe", "structure": "orthorhombic", "property_name": "thermoelectric_zt", "property_value": 2.6, "property_unit": "", "source_paper_id": "p003"},
        {"composition": "SrTiO3", "structure": "perovskite", "property_name": "bandgap", "property_value": 3.2, "property_unit": "eV", "source_paper_id": "p005"},
        {"composition": "PbTiO3", "structure": "tetragonal", "property_name": "bandgap", "property_value": 3.0, "property_unit": "eV", "source_paper_id": "p006"},
    ]

    papers = [
        {"title": "BaTiO3 perovskite properties", "abstract": "BaTiO3 is a ferroelectric perovskite with bandgap 3.2 eV."},
        {"title": "SnSe thermoelectric", "abstract": "SnSe shows excellent thermoelectric ZT."},
        {"title": "SrTiO3 cubic perovskite", "abstract": "SrTiO3 bandgap measurement."},
    ]

    engine = StructurePropertySearchEngine()
    result = engine.search(
        seed_hypotheses=hypotheses,
        csp_triples=csp_triples,
        papers=papers,
        iterations=3,
        population_size=15,
        top_k=5,
    )

    print(f"  Total generated: {result['total_generated']}")
    print(f"  Total evaluated: {result['total_evaluated']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Best score: {result['best_score']:.3f}")
    print(f"  Top candidates: {len(result['top_candidates'])}")

    for i, c in enumerate(result["top_candidates"]):
        print(f"  Cand {i+1}: {c['composition']} | {c['structure']} | "
              f"{c['property_name']} = {c['predicted_range']} "
              f"(score={c['score']:.3f}, gen={c['generation']})")
        if c["mutation_history"]:
            print(f"    Mutations: {' -> '.join(c['mutation_history'])}")

    assert result["total_evaluated"] > 0, "Should have evaluated candidates"
    assert len(result["top_candidates"]) > 0, "Should have top candidates"
    print("  PASS: StructurePropertySearchEngine\n")


def test_db_validator():
    """Test ExternalDBValidator (without API keys)."""
    print("=" * 60)
    print("TEST 3: ExternalDBValidator")
    print("=" * 60)

    candidates = [
        {"composition": "BaTiO3", "structure": "perovskite", "property_name": "bandgap"},
        {"composition": "SnSe", "structure": "orthorhombic", "property_name": "thermoelectric_zt"},
        {"composition": "LiFePO4", "structure": "olivine", "property_name": "voltage"},
    ]

    validator = ExternalDBValidator()
    result = validator.validate_candidates(candidates, [])

    print(f"  Candidates validated: {result['candidates_validated']}")
    print(f"  Materials Project: {result['materials_project']['status']}")
    print(f"  OQMD: {result['oqmd']['status']}")
    print(f"  NOMAD: {result['nomad']['status']}")
    print(f"  Novelty status: {result['novelty_status']}")

    # Test manual import
    validator.import_manual_result("materials_project", "BaTiO3", True, 3.2, "eV", "perovskite")
    result2 = validator.validate_candidates(candidates, [])
    print(f"  After manual import - MP found: {result2['materials_project'].get('found_count', 0)}")

    assert "novelty_status" in result, "Should have novelty_status"
    print("  PASS: ExternalDBValidator\n")


def test_competition_report_standalone():
    """Test CompetitionReportGenerator output structure (without full host)."""
    print("=" * 60)
    print("TEST 4: CompetitionReportGenerator (structure check)")
    print("=" * 60)

    gen = CompetitionReportGenerator()

    # Check that all section methods exist
    assert hasattr(gen, "generate"), "Should have generate method"
    assert hasattr(gen, "_section_title"), "Should have _section_title"
    assert hasattr(gen, "_section_research_problem"), "Should have _section_research_problem"
    assert hasattr(gen, "_section_retrieval_strategy"), "Should have _section_retrieval_strategy"
    assert hasattr(gen, "_section_literature_screening"), "Should have _section_literature_screening"
    assert hasattr(gen, "_section_csp_knowledge"), "Should have _section_csp_knowledge"
    assert hasattr(gen, "_section_research_gaps"), "Should have _section_research_gaps"
    assert hasattr(gen, "_section_search_process"), "Should have _section_search_process"
    assert hasattr(gen, "_section_top_candidates"), "Should have _section_top_candidates"
    assert hasattr(gen, "_section_evidence_chains"), "Should have _section_evidence_chains"
    assert hasattr(gen, "_section_verification_plan"), "Should have _section_verification_plan"
    assert hasattr(gen, "_section_limitations"), "Should have _section_limitations"

    # Test individual section generation
    title = gen._section_title({"cycles": 5, "papers_collected": 100})
    assert "Sciverse-MCP" in title, "Title should contain project name"

    problem = gen._section_research_problem({"cycles": 5})
    assert "Research Problem" in problem, "Should contain section header"

    print("  All section methods present")
    print("  PASS: CompetitionReportGenerator structure\n")


def test_imports():
    """Test that all new modules can be imported."""
    print("=" * 60)
    print("TEST 5: Module imports")
    print("=" * 60)

    from sci_host.knowledge.gap_builder import GapEvidenceBuilder, GapEvidence
    from sci_host.materials.search_engine import StructurePropertySearchEngine, SearchCandidate
    from sci_host.materials.db_validators import ExternalDBValidator
    from sci_host.reporting import CompetitionReportGenerator

    print("  All imports successful")
    print("  PASS: Module imports\n")


if __name__ == "__main__":
    test_imports()
    test_gap_evidence_builder()
    test_search_engine()
    test_db_validator()
    test_competition_report_standalone()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
