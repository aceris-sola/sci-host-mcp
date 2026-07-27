"""External database validators — pluggable adapters for Materials Project / OQMD / NOMAD.

If API keys are available, queries the live database.
If not, returns "unavailable" and supports manual import / local cache.

Usage:
    validator = ExternalDBValidator()
    result = validator.validate_candidates(candidates, csp_triples)
    # result = {
    #     "materials_project": {"status": "unavailable", ...},
    #     "oqmd": {"status": "unavailable", ...},
    #     "nomad": {"status": "unavailable", ...},
    #     "novelty_status": "missing",
    # }
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class ExternalDBValidator:
    """Cross-validate candidate materials against external databases.

    Supports Materials Project, OQMD, and NOMAD via pluggable adapters.
    Falls back gracefully to "unavailable" when no API key is configured.
    """

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self._cache_dir = cache_dir or os.path.expanduser("~/.sci_host/db_cache")
        os.makedirs(self._cache_dir, exist_ok=True)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

        # API keys (from environment)
        self._mp_api_key = os.environ.get("MP_API_KEY", "")
        self._oqmd_api_key = os.environ.get("OQMD_API_KEY", "")
        self._nomad_token = os.environ.get("NOMAD_TOKEN", "")

    def validate_candidates(
        self,
        candidates: List[Dict[str, Any]],
        csp_triples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate a list of candidate materials against external databases.

        Args:
            candidates: list of candidate dicts with composition, structure, property_name
            csp_triples: CSP triples from system knowledge base (for comparison)

        Returns:
            {
                "materials_project": {"status", "checked", "found", "not_found", ...},
                "oqmd": {...},
                "nomad": {...},
                "novelty_status": "known" | "partially_known" | "missing",
                "candidates_validated": int,
            }
        """
        result: Dict[str, Any] = {
            "candidates_validated": len(candidates),
            "materials_project": self._query_materials_project(candidates),
            "oqmd": self._query_oqmd(candidates),
            "nomad": self._query_nomad(candidates),
        }

        # Determine overall novelty status
        mp_status = result["materials_project"].get("status", "")
        oqmd_status = result["oqmd"].get("status", "")
        nomad_status = result["nomad"].get("status", "")

        mp_found = result["materials_project"].get("found_count", 0)
        oqmd_found = result["oqmd"].get("found_count", 0)
        nomad_found = result["nomad"].get("found_count", 0)

        total_found = mp_found + oqmd_found + nomad_found
        n_cands = len(candidates)

        if n_cands == 0:
            result["novelty_status"] = "unknown"
        elif total_found >= n_cands:
            result["novelty_status"] = "known"
        elif total_found > 0:
            result["novelty_status"] = "partially_known"
        else:
            result["novelty_status"] = "missing"

        return result

    def _query_materials_project(
        self, candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Query Materials Project API for material existence.

        If MP_API_KEY is not set, returns "unavailable".
        """
        if not self._mp_api_key:
            return {
                "status": "unavailable",
                "message": "MP_API_KEY not set. Set environment variable to enable.",
                "found_count": 0,
                "not_found_count": 0,
                "checked": 0,
                "results": [],
            }

        # If API key available, attempt query
        try:
            return self._query_mp_live(candidates)
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "found_count": 0,
                "not_found_count": 0,
                "checked": len(candidates),
                "results": [],
            }

    def _query_mp_live(
        self, candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Query Materials Project live API (requires mp_api package)."""
        try:
            from mp_api.client import MPRester
        except ImportError:
            return {
                "status": "unavailable",
                "message": "mp_api package not installed. Run: pip install mp_api",
                "found_count": 0,
                "not_found_count": 0,
                "checked": 0,
                "results": [],
            }

        results: List[Dict[str, Any]] = []
        found_count = 0
        not_found_count = 0

        with MPRester(self._mp_api_key) as mpr:
            for cand in candidates:
                comp = cand.get("composition", "")
                if not comp:
                    continue

                # Check cache first
                cache_key = f"mp_{comp}"
                if cache_key in self._cache:
                    cached = self._cache[cache_key]
                    results.append(cached)
                    if cached["found"]:
                        found_count += 1
                    else:
                        not_found_count += 1
                    continue

                try:
                    docs = mpr.materials.search(formula=comp, num_sites=(1, 50))
                    if docs and len(docs) > 0:
                        entry = {
                            "composition": comp,
                            "found": True,
                            "database": "materials_project",
                            "n_entries": len(docs),
                            "structures": [
                                getattr(d, "structure", None).__class__.__name__
                                if hasattr(d, "structure") else "N/A"
                                for d in docs[:3]
                            ],
                            "novelty_status": "known",
                        }
                        found_count += 1
                    else:
                        entry = {
                            "composition": comp,
                            "found": False,
                            "database": "materials_project",
                            "n_entries": 0,
                            "novelty_status": "missing",
                        }
                        not_found_count += 1
                    self._cache[cache_key] = entry
                    results.append(entry)
                except Exception as e:
                    entry = {
                        "composition": comp,
                        "found": False,
                        "database": "materials_project",
                        "error": str(e)[:200],
                        "novelty_status": "error",
                    }
                    results.append(entry)

        return {
            "status": "found" if found_count > 0 else "not_found",
            "found_count": found_count,
            "not_found_count": not_found_count,
            "checked": len(candidates),
            "results": results,
        }

    def _query_oqmd(
        self, candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Query OQMD (Open Quantum Materials Database).

        OQMD REST API: http://oqmd.org/restful/materials/formula/{formula}
        """
        results: List[Dict[str, Any]] = []
        found_count = 0
        not_found_count = 0

        for cand in candidates[:20]:  # Limit to avoid rate limits
            comp = cand.get("composition", "")
            if not comp:
                continue

            # Check cache
            cache_key = f"oqmd_{comp}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                results.append(cached)
                if cached["found"]:
                    found_count += 1
                else:
                    not_found_count += 1
                continue

            # Try live query (OQMD REST API doesn't require key)
            try:
                import urllib.request
                import urllib.error

                url = f"http://oqmd.org/restful/materials/formula/{comp}"
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())

                n_results = len(data.get("data", []))
                entry = {
                    "composition": comp,
                    "found": n_results > 0,
                    "database": "oqmd",
                    "n_entries": n_results,
                    "novelty_status": "known" if n_results > 0 else "missing",
                }
                self._cache[cache_key] = entry
                results.append(entry)
                if n_results > 0:
                    found_count += 1
                else:
                    not_found_count += 1
            except Exception as e:
                entry = {
                    "composition": comp,
                    "found": False,
                    "database": "oqmd",
                    "error": str(e)[:200],
                    "novelty_status": "unavailable",
                }
                results.append(entry)

        return {
            "status": "found" if found_count > 0 else "not_found",
            "found_count": found_count,
            "not_found_count": not_found_count,
            "checked": len(results),
            "results": results,
        }

    def _query_nomad(
        self, candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Query NOMAD (Novel Materials Discovery) repository.

        NOMAD API: https://nomad-lab.eu/prod/v1/api/v1/
        """
        results: List[Dict[str, Any]] = []
        found_count = 0
        not_found_count = 0

        for cand in candidates[:20]:
            comp = cand.get("composition", "")
            if not comp:
                continue

            cache_key = f"nomad_{comp}"
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                results.append(cached)
                if cached["found"]:
                    found_count += 1
                else:
                    not_found_count += 1
                continue

            try:
                import urllib.request
                import urllib.error

                # NOMAD API search by formula
                url = (
                    "https://nomad-lab.eu/prod/v1/api/v1/entries/search"
                    "?q=formula&values.formula=" + comp
                )
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                if self._nomad_token:
                    req.add_header("Authorization", f"Bearer {self._nomad_token}")

                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())

                n_results = data.get("pagination", {}).get("total", 0)
                entry = {
                    "composition": comp,
                    "found": n_results > 0,
                    "database": "nomad",
                    "n_entries": n_results,
                    "novelty_status": "known" if n_results > 0 else "missing",
                }
                self._cache[cache_key] = entry
                results.append(entry)
                if n_results > 0:
                    found_count += 1
                else:
                    not_found_count += 1
            except Exception as e:
                entry = {
                    "composition": comp,
                    "found": False,
                    "database": "nomad",
                    "error": str(e)[:200],
                    "novelty_status": "unavailable",
                }
                results.append(entry)

        return {
            "status": "found" if found_count > 0 else "not_found",
            "found_count": found_count,
            "not_found_count": not_found_count,
            "checked": len(results),
            "results": results,
        }

    def import_manual_result(
        self,
        database: str,
        composition: str,
        found: bool,
        property_value: Optional[float] = None,
        property_unit: str = "",
        structure: str = "",
    ) -> None:
        """Manually import a database query result (for offline workflows)."""
        cache_key = f"{database.lower()}_{composition}"
        entry = {
            "composition": composition,
            "found": found,
            "database": database.lower(),
            "novelty_status": "known" if found else "missing",
            "manually_imported": True,
        }
        if property_value is not None:
            entry["property_value"] = property_value
            entry["property_unit"] = property_unit
        if structure:
            entry["structure"] = structure
        self._cache[cache_key] = entry
        self._save_cache()

    def _load_cache(self) -> None:
        cache_file = os.path.join(self._cache_dir, "db_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        cache_file = os.path.join(self._cache_dir, "db_cache.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
