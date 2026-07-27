#!/usr/bin/env python3
"""End-to-end test: create materials host, run cycles, generate competition report."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sci_host import HostSystem, HostConfig


def test_e2e_competition_report():
    """Create a materials host, run a few cycles, generate report."""
    print("=" * 60)
    print("E2E TEST: Competition Report Generation")
    print("=" * 60)

    # Create materials mode host
    config = HostConfig(
        host_id="e2e-test-001",
        materials_mode=True,
    )
    host = HostSystem(config=config)
    host.start()

    print("  Host created and started")

    # Run a few cycles to populate data
    print("  Running 3 cycles...")
    for i in range(3):
        try:
            result = host.step()
            print(f"  Cycle {i+1}: papers={result.get('papers_collected', 0)}, "
                  f"hypos={result.get('hypotheses_generated', 0)}, "
                  f"trials={result.get('trials_executed', 0)}")
        except Exception as e:
            print(f"  Cycle {i+1} error: {e}")

    # Test gap evidence
    print("\n  Testing gap evidence report...")
    gaps = host.build_gap_evidence_report(n=5)
    print(f"  Gaps found: {len(gaps)}")
    for g in gaps[:2]:
        print(f"    - [{g.get('gap_type', '')}] {g.get('gap_description', '')[:60]}...")
        print(f"      novelty={g.get('novelty_score', 0):.2f} "
              f"supporting={len(g.get('supporting_papers', []))}")

    # Test search engine
    print("\n  Testing structure-property search...")
    search_result = host.run_structure_property_search(
        iterations=2, population_size=10, top_k=3,
    )
    print(f"  Search: {search_result.get('total_evaluated', 0)} evaluated, "
          f"best={search_result.get('best_score', 0):.3f}")
    for c in search_result.get("top_candidates", [])[:3]:
        print(f"    - {c.get('composition', '')} | {c.get('structure', '')} | "
              f"{c.get('property_name', '')} (score={c.get('score', 0):.3f})")

    # Test external DB validation
    print("\n  Testing external DB validation...")
    db_result = host.validate_external_databases()
    print(f"  MP: {db_result.get('materials_project', {}).get('status', 'N/A')}")
    print(f"  OQMD: {db_result.get('oqmd', {}).get('status', 'N/A')}")
    print(f"  NOMAD: {db_result.get('nomad', {}).get('status', 'N/A')}")
    print(f"  Novelty: {db_result.get('novelty_status', 'N/A')}")

    # Generate competition report
    print("\n  Generating competition report...")
    report = host.generate_competition_report(
        include_search=True,
        include_external_validation=False,
    )
    print(f"  Report length: {len(report)} chars")

    # Check report structure
    sections = [
        "## 1. Research Problem",
        "## 2. Retrieval Strategy",
        "## 3. Literature Screening Results",
        "## 4. CSP Knowledge Base",
        "## 5. Research Gap List",
        "## 6. Route A Search Process",
        "## 7. Top Structure-Property Candidates",
        "## 8. Evidence Chains",
        "## 9. Falsifiable Verification Plan",
        "## 10. Limitations & Next Steps",
    ]
    for s in sections:
        assert s in report, f"Missing section: {s}"
        print(f"  ✓ {s}")

    # Save report
    report_file = os.path.join(
        os.path.dirname(__file__), "competition_report_e2e.md",
    )
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\n  Report saved to: {report_file}")

    host.stop()
    print("\n  PASS: E2E competition report generation\n")


if __name__ == "__main__":
    try:
        success = test_e2e_competition_report()
        if success:
            print("=" * 60)
            print("E2E TEST PASSED")
            print("=" * 60)
    except Exception as e:
        print(f"\nE2E TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
