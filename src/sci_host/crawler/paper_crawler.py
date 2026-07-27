"""论文/材料持续采集器.

7×24 小时持续运行，不断采集新的论文和材料。
支持两种模式:
    1. 离线模式: 使用内置多领域语料模拟论文流（不依赖网络）
    2. 在线模式: 通过 arXiv API 采集真实论文（需要网络）

设计灵感来自原项目的 PhysicalEntity.read_sensors() 模式:
    crawler.crawl_batch() → List[Paper]  (类比 read_sensors → Dict)
"""
from __future__ import annotations

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import CrawlerConfig
from ..config import ResearchQualityConfig
from ..research_quality import ResearchQualityGate

_logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """论文/材料数据结构."""
    paper_id: str
    title: str
    abstract: str
    authors: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    year: int = 2025
    source: str = "unknown"
    
    data_source: str = ""
    
    embedding: Optional[Any] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """用于嵌入的文本（标题 + 摘要 + 关键词）."""
        return f"{self.title}. {self.abstract}. {' '.join(self.keywords)}"

    @property
    def primary_category(self) -> str:
        return self.categories[0] if self.categories else "unknown"

    def __hash__(self) -> int:
        return hash(self.paper_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Paper):
            return self.paper_id == other.paper_id
        return False


class PaperSource:
    """论文数据源（离线/在线）.

    支持三种语料模式:
        1. 通用科研语料 (默认): 机器人/ML/digital twin 等跨领域论文
        2. 材料科学语料 (materials_mode=True): 钙钛矿/锂电池/热电/铁电等材料论文
        3. 在线 arXiv: 通过 arXiv API 采集真实论文
    """

    
    OFFLINE_CORPUS: List[Dict[str, Any]] = [
        
        {"title": "Learning Robotic Manipulation via Action Chunking Transformers",
         "abstract": "We propose an action chunking approach using transformer architectures for robotic manipulation. "
                     "The model predicts sequences of actions conditioned on visual observations, enabling smooth "
                     "control with reduced compounding error. Experiments on bimanual manipulation tasks show "
                     "significant improvement over standard behavioral cloning.",
         "categories": ["cs.RO", "cs.AI", "cs.LG"],
         "keywords": ["robotics", "transformer", "manipulation", "behavioral cloning", "action chunking"]},

        {"title": "Sim-to-Real Transfer for Humanoid Robots via Domain Randomization",
         "abstract": "Transferring policies from simulation to real-world humanoid robots remains challenging. "
                     "We introduce a domain randomization framework that bridges the reality gap by augmenting "
                     "physical parameters during training. The approach achieves robust locomotion on hardware "
                     "without fine-tuning.",
         "categories": ["cs.RO", "cs.LG"],
         "keywords": ["sim-to-real", "humanoid", "domain randomization", "locomotion", "transfer"]},

        {"title": "Digital Twin-Driven Predictive Maintenance for Manufacturing Systems",
         "abstract": "Digital twins provide real-time mirrors of physical manufacturing systems. We develop "
                     "a predictive maintenance framework using twin models that forecast equipment degradation "
                     "and schedule interventions proactively. The system integrates sensor data streaming "
                     "with physics-informed neural networks.",
         "categories": ["cs.AI", "eess.SP"],
         "keywords": ["digital twin", "predictive maintenance", "manufacturing", "physics-informed", "sensor"]},

        
        {"title": "Continual Learning with Elastic Weight Consolidation for Non-Stationary Environments",
         "abstract": "Neural networks suffer from catastrophic forgetting when learning sequential tasks. "
                     "We extend elastic weight consolidation (EWC) with Fisher information matrix estimation "
                     "for non-stationary distributions. Our method maintains stable performance across "
                     "task sequences while preserving critical parameters.",
         "categories": ["cs.LG", "cs.AI", "stat.ML"],
         "keywords": ["continual learning", "elastic weight consolidation", "catastrophic forgetting",
                      "Fisher information", "non-stationary"]},

        {"title": "Meta-Learning Fast Adaptation for Few-Shot Robot Learning",
         "abstract": "We apply model-agnostic meta-learning (MAML) to enable rapid adaptation of robot "
                     "policies from few demonstrations. The inner loop adapts to new tasks while the outer "
                     "loop optimizes meta-parameters. Results show 5-shot adaptation on manipulation tasks.",
         "categories": ["cs.LG", "cs.RO"],
         "keywords": ["meta-learning", "MAML", "few-shot", "robot learning", "adaptation"]},

        
        {"title": "Causal Discovery from Time Series via Granger Causality with Confounders",
         "abstract": "Granger causality is widely used for temporal causal discovery but suffers from "
                     "hidden confounders. We propose a deconfounded Granger test using instrumental "
                     "variables. The method recovers true causal graphs in simulation and applies to "
                     "neuroscience data.",
         "categories": ["stat.ML", "cs.AI"],
         "keywords": ["causal discovery", "Granger causality", "confounders", "time series", "instrumental variables"]},

        {"title": "Counterfactual Reasoning for Safe Reinforcement Learning",
         "abstract": "Safe exploration in reinforcement learning requires reasoning about what could go "
                     "wrong. We integrate counterfactual reasoning into RL agents, allowing them to "
                     "evaluate alternative action consequences before execution. The approach reduces "
                     "constraint violations by 70 percent in safety-critical benchmarks.",
         "categories": ["cs.LG", "cs.AI"],
         "keywords": ["counterfactual", "safe reinforcement learning", "constraint violation",
                      "exploration", "safety"]},

        
        {"title": "Building Scientific Knowledge Graphs from Literature with Large Language Models",
         "abstract": "We present a pipeline for constructing domain-specific knowledge graphs from "
                     "scientific publications using large language models for entity extraction and "
                     "relation classification. The graph supports multi-hop reasoning for hypothesis "
                     "generation and literature-based discovery.",
         "categories": ["cs.AI", "cs.CL", "cs.DB"],
         "keywords": ["knowledge graph", "scientific literature", "entity extraction",
                      "relation classification", "multi-hop reasoning", "LLM"]},

        {"title": "Semantic Similarity for Cross-Domain Knowledge Transfer",
         "abstract": "Transferring knowledge across domains requires measuring semantic similarity between "
                     "concepts. We propose a graph-based similarity metric that combines structural proximity "
                     "with embedding distance. The method enables zero-shot transfer between distant domains.",
         "categories": ["cs.AI", "cs.IR"],
         "keywords": ["semantic similarity", "knowledge transfer", "cross-domain",
                      "zero-shot", "graph embedding"]},

        
        {"title": "Evidential Deep Learning for Uncertainty-Aware Medical Diagnosis",
         "abstract": "Medical diagnosis models must quantify uncertainty for safe deployment. We apply "
                     "evidential deep learning with Dirichlet distribution outputs to decompose epistemic "
                     "and aleatoric uncertainty. The approach identifies cases requiring expert review.",
         "categories": ["cs.LG", "q-bio.QM"],
         "keywords": ["evidential deep learning", "uncertainty quantification", "medical diagnosis",
                      "epistemic uncertainty", "aleatoric uncertainty", "Dirichlet"]},

        {"title": "Bayesian Optimization for Sample-Efficient Hyperparameter Tuning",
         "abstract": "Hyperparameter optimization is expensive for large models. We use Gaussian process "
                     "Bayesian optimization with acquisition functions balancing exploration and "
                     "exploitation. The method reduces tuning cost by 5x compared to grid search.",
         "categories": ["cs.LG", "stat.ML"],
         "keywords": ["Bayesian optimization", "Gaussian process", "hyperparameter",
                      "acquisition function", "sample efficiency"]},

        
        {"title": "Diffusion Models for Scenario Generation in Autonomous Driving",
         "abstract": "Generating diverse driving scenarios is critical for testing autonomous vehicles. "
                     "We adapt denoising diffusion probabilistic models to synthesize realistic traffic "
                     "situations conditioned on road layouts. The generated scenarios improve edge case "
                     "coverage in simulation testing.",
         "categories": ["cs.CV", "cs.AI", "cs.RO"],
         "keywords": ["diffusion model", "scenario generation", "autonomous driving",
                      "denoising", "edge case"]},

        {"title": "Physics-Informed Diffusion for Molecule Generation",
         "abstract": "Generating valid molecules requires satisfying physical constraints. We embed "
                     "quantum mechanical constraints into the diffusion process, ensuring generated "
                     "molecules are physically plausible. The approach outperforms unconstrained baselines "
                     "on drug-likeness metrics.",
         "categories": ["physics.chem-ph", "cs.LG"],
         "keywords": ["diffusion model", "molecule generation", "physics-informed",
                      "quantum mechanics", "drug discovery"]},

        
        {"title": "Graph Attention Networks for Multi-Robot Coordination",
         "abstract": "Coordinating multiple robots requires reasoning about inter-robot relationships. "
                     "We model robot teams as dynamic graphs and use graph attention networks to learn "
                     "coordination policies. The approach scales to 20 robots with emergent communication.",
         "categories": ["cs.RO", "cs.LG", "cs.MA"],
         "keywords": ["graph attention network", "multi-robot", "coordination",
                      "emergent communication", "dynamic graph"]},

        {"title": "Graph Neural Networks for Drug-Drug Interaction Prediction",
         "abstract": "Predicting drug-drug interactions is essential for patient safety. We represent "
                     "drug combinations as molecular graphs and train graph neural networks to predict "
                     "interaction types. The model achieves high accuracy on benchmark datasets.",
         "categories": ["q-bio.QM", "cs.LG"],
         "keywords": ["graph neural network", "drug interaction", "molecular graph",
                      "patient safety", "prediction"]},

        
        {"title": "Negative Transfer Detection in Multi-Task Learning",
         "abstract": "Multi-task learning can hurt performance when tasks conflict. We propose a method "
                     "to detect negative transfer by measuring task affinity gradients. When negative "
                     "transfer is detected, the system automatically routes tasks to separate models.",
         "categories": ["cs.LG", "cs.AI"],
         "keywords": ["negative transfer", "multi-task learning", "task affinity",
                      "gradient", "routing"]},

        {"title": "Federated Learning with Byzantine-Robust Aggregation",
         "abstract": "Federated learning is vulnerable to malicious clients. We combine Krum algorithm "
                     "with trimmed mean aggregation to filter Byzantine updates. The method maintains "
                     "model accuracy even when 30 percent of clients are adversarial.",
         "categories": ["cs.LG", "cs.CR"],
         "keywords": ["federated learning", "Byzantine", "Krum", "trimmed mean",
                      "robust aggregation", "adversarial"]},

        
        {"title": "Physics-Informed Neural Networks for Fluid Dynamics Simulation",
         "abstract": "Solving Navier-Stokes equations with traditional methods is computationally expensive. "
                     "We embed the governing equations as soft constraints in neural network loss functions. "
                     "The physics-informed approach achieves accurate solutions with orders of magnitude "
                     "speedup.",
         "categories": ["physics.flu-dyn", "cs.LG", "cs.NA"],
         "keywords": ["physics-informed neural network", "Navier-Stokes", "fluid dynamics",
                      "governing equations", "soft constraint"]},

        {"title": "Quantum-Inspired Optimization for Robot Path Planning",
         "abstract": "Robot path planning in complex environments is NP-hard. We adapt variational quantum "
                     "eigensolver concepts to classical optimization, using parameterized circuits as "
                     "ansatz. The quantum-inspired approach finds near-optimal paths faster than A*.",
         "categories": ["cs.RO", "quant-ph"],
         "keywords": ["quantum optimization", "path planning", "variational quantum eigensolver",
                      "ansatz", "NP-hard"]},

        
        {"title": "Evolutionary Game Theory for Multi-Agent Reinforcement Learning",
         "abstract": "Multi-agent RL often converges to suboptimal equilibria. We apply evolutionary game "
                     "theory to analyze Nash equilibria stability and design reward shaping that guides "
                     "agents toward Pareto-optimal outcomes. The framework generalizes to cooperative "
                     "and competitive settings.",
         "categories": ["cs.GT", "cs.MA", "cs.LG"],
         "keywords": ["evolutionary game theory", "multi-agent", "Nash equilibrium",
                      "Pareto optimal", "reward shaping"]},

        {"title": "Game-Theoretic Resource Allocation in Edge Computing",
         "abstract": "Allocating computing resources at the network edge involves competing users. We "
                     "model resource allocation as a Bayesian game and compute mechanisms that are "
                     "truthful and individually rational. Simulations show improved social welfare.",
         "categories": ["cs.GT", "cs.DC", "cs.NI"],
         "keywords": ["game theory", "resource allocation", "edge computing",
                      "Bayesian game", "mechanism design", "social welfare"]},

        
        {"title": "Blockchain-Based Provenance Tracking for Scientific Data Integrity",
         "abstract": "Ensuring scientific data integrity is critical for reproducibility. We use "
                     "blockchain with Merkle tree anchoring and zero-knowledge proofs to create "
                     "tamper-evident provenance records. The system enables verification without "
                     "revealing sensitive data.",
         "categories": ["cs.CR", "cs.DB"],
         "keywords": ["blockchain", "provenance", "Merkle tree", "zero-knowledge proof",
                      "data integrity", "reproducibility"]},

        
        {"title": "Sleep-Inspired Memory Consolidation for Lifelong Learning Agents",
         "abstract": "Biological sleep consolidates memories through replay. We implement a computational "
                     "analog where agents periodically replay experiences with prioritized sampling, "
                     "strengthening important memories and pruning redundant ones. The method extends "
                     "lifelong learning horizons.",
         "categories": ["cs.AI", "cs.LG", "q-bio.NC"],
         "keywords": ["memory consolidation", "lifelong learning", "replay",
                      "prioritized sampling", "biological inspiration"]},

        {"title": "Multi-Fidelity Bayesian Optimization for Materials Discovery",
         "abstract": "Materials discovery involves expensive high-fidelity and cheap low-fidelity "
                     "evaluations. We use multi-fidelity Gaussian process models to intelligently "
                     "allocate evaluation budget. Active sampling selects the most informative "
                     "experiments at each fidelity level.",
         "categories": ["cond-mat.mtrl-sci", "cs.LG", "stat.ML"],
         "keywords": ["multi-fidelity", "Bayesian optimization", "materials discovery",
                      "Gaussian process", "active sampling"]},

        {"title": "Temporal Consistency Validation for Synthetic Sensor Data",
         "abstract": "Synthetic sensor data must be temporally consistent for valid simulation. We propose "
                     "validation metrics including long-range drift detection, conservation law checking, "
                     "and power spectral density matching. The framework quantifies trustworthiness of "
                     "generated time series.",
         "categories": ["eess.SP", "cs.LG", "stat.ME"],
         "keywords": ["temporal consistency", "synthetic data", "drift detection",
                      "conservation law", "power spectral density", "validation"]},

        {"title": "Cross-Reality Domain Transfer with Maximum Mean Discrepancy",
         "abstract": "Bridging simulation and reality requires domain adaptation. We use maximum mean "
                     "discrepancy (MMD) to measure distribution differences and learn bidirectional "
                     "mappings between simulated and real data spaces. Residual compensation further "
                     "reduces transfer error.",
         "categories": ["cs.LG", "cs.RO", "stat.ML"],
         "keywords": ["domain transfer", "maximum mean discrepancy", "bidirectional mapping",
                      "residual compensation", "simulation to reality"]},

        {"title": "Adaptive Sampling for Efficient Digital Twin Calibration",
         "abstract": "Calibrating digital twins requires selecting informative data points. We design "
                     "an acquisition function balancing uncertainty, novelty, and cost. Greedy budget "
                     "allocation with online updates enables efficient calibration within error targets.",
         "categories": ["cs.AI", "eess.SP", "stat.ML"],
         "keywords": ["adaptive sampling", "digital twin", "calibration",
                      "acquisition function", "uncertainty", "novelty"]},

        {"title": "Trajectory Synthesis with Lagrangian Constraints for Robot Imitation",
         "abstract": "Generating physically feasible trajectories for robot imitation learning is challenging. "
                     "We synthesize trajectories using Bezier curve interpolation and frequency domain "
                     "perturbation, validated by Lagrangian residual computation. The approach generates "
                     "diverse feasible motions from limited demonstrations.",
         "categories": ["cs.RO", "cs.LG"],
         "keywords": ["trajectory synthesis", "Lagrangian constraint", "Bezier curve",
                      "imitation learning", "frequency domain"]},

        {"title": "LLM-Powered Reasoning for Automated Scientific Hypothesis Generation",
         "abstract": "Large language models can generate scientific hypotheses by synthesizing knowledge "
                     "across papers. We develop a chain-of-thought reasoning pipeline that grounds "
                     "language model outputs in structured domain knowledge. The system proposes "
                     "testable hypotheses with confidence scores.",
         "categories": ["cs.CL", "cs.AI"],
         "keywords": ["large language model", "hypothesis generation", "chain-of-thought",
                      "scientific reasoning", "knowledge synthesis"]},

        {"title": "Temporal Compression for Accelerated Physics Simulation",
         "abstract": "Physics simulation is bottlenecked by uniform time stepping. We propose adaptive "
                     "temporal compression that uses state entropy and Jacobian norms to dynamically "
                     "adjust simulation speedup. Critical phases get fine-grained stepping while "
                     "trivial phases are fast-forwarded.",
         "categories": ["cs.LG", "physics.comp-ph"],
         "keywords": ["temporal compression", "physics simulation", "state entropy",
                      "Jacobian", "adaptive stepping"]},

        {"title": "Variational Quantum Circuits for Twin Model Parameter Calibration",
         "abstract": "Calibrating complex digital twin models has many parameters. We use variational "
                     "quantum circuits with parameter-shift gradients to optimize twin parameters. "
                     "The quantum approach explores parameter spaces more efficiently than classical "
                     "gradient descent in certain regimes.",
         "categories": ["quant-ph", "cs.AI"],
         "keywords": ["variational quantum circuit", "parameter calibration", "digital twin",
                      "parameter-shift gradient", "quantum optimization"]},
    ]

    def __init__(
        self,
        config: CrawlerConfig,
        research_seeds: Optional[List[str]] = None,
        materials_mode: bool = False,
    ) -> None:
        self.config = config
        self.research_seeds = research_seeds or []
        self.materials_mode = materials_mode
        self._offline_index = 0
        self._offline_shuffle: List[int] = []
        
        self._arxiv_start = 0
        self._openalex_cursor = "*"
        self._fallback_count = 0  

        
        if materials_mode:
            from ..materials import MATERIAL_CORPUS, MATERIAL_CATEGORIES
            self._materials_corpus = MATERIAL_CORPUS
            if not self.config.categories or self.config.categories == [
                "cs.AI", "cs.LG", "cs.RO", "cs.CL",
                "physics", "q-bio", "stat.ML",
            ]:
                self.config.categories = list(MATERIAL_CATEGORIES)
        else:
            self._materials_corpus = []

    def fetch(self, batch_size: int) -> List[Dict[str, Any]]:
        """获取一批论文数据（原始字典格式）."""
        if self.config.offline_mode:
            return self._fetch_offline(batch_size)
        elif getattr(self.config, "source", "arxiv") == "openalex":
            results = self._fetch_openalex(batch_size)
            self._check_fallback(results)
            return results
        elif getattr(self.config, "source", "arxiv") == "sciverse":
            results = self._fetch_sciverse(batch_size)
            self._check_fallback(results)
            return results
        else:
            results = self._fetch_online(batch_size)
            self._check_fallback(results)
            return results

    def _check_fallback(self, results: List[Dict[str, Any]]) -> None:
        """检测是否回退到离线数据, 并标记论文来源."""
        if not results:
            return
        
        offline_count = sum(
            1 for r in results
            if r.get("paper_id", "").startswith("paper_")
        )
        if offline_count > 0 and not self.config.offline_mode:
            self._fallback_count += 1
            
            for r in results:
                if r.get("paper_id", "").startswith("paper_"):
                    r["data_source"] = "offline_fallback"
                else:
                    r["data_source"] = "online"
        else:
            for r in results:
                r.setdefault("data_source", "online")

    def _fetch_offline(self, batch_size: int) -> List[Dict[str, Any]]:
        """离线模式：从内置语料循环采样."""
        corpus = self._materials_corpus if self.materials_mode else self.OFFLINE_CORPUS
        n = len(corpus)
        if not self._offline_shuffle:
            self._offline_shuffle = list(range(n))
            random.shuffle(self._offline_shuffle)

        results: List[Dict[str, Any]] = []
        for _ in range(batch_size):
            idx = self._offline_shuffle[self._offline_index % n]
            self._offline_index += 1
            paper_data = dict(corpus[idx])
            
            paper_data = self._add_variant(paper_data)
            results.append(paper_data)
        return results

    def _add_variant(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """为离线语料添加变体（模拟持续有新论文流入）."""
        variant = dict(paper_data)
        
        suffix = hashlib.md5(
            f"{paper_data['title']}{time.time()}{random.random()}".encode()
        ).hexdigest()[:8]
        variant["paper_id"] = f"paper_{suffix}"
        variant["year"] = random.choice([2023, 2024, 2025, 2026])
        
        prefixes = ["A Novel", "Towards", "Enhanced", "Adaptive", "Scalable",
                    "Robust", "Efficient", "Unified", "Deep", "Progressive"]
        if not any(paper_data["title"].startswith(p) for p in prefixes):
            variant["title"] = f"{random.choice(prefixes)} {paper_data['title']}"
        return variant

    def _fetch_online(self, batch_size: int) -> List[Dict[str, Any]]:
        """在线模式：通过 arXiv API 采集.

        注意: 需要网络连接和 urllib 库。
        arXiv 官方建议同一 IP 请求间隔 >= 3 秒；若触发 429，使用指数退避重试。
        如果网络不可用，回退到离线模式。
        """
        import urllib.request
        import urllib.parse
        import urllib.error
        import xml.etree.ElementTree as ET

        
        if self.config.categories:
            cat_query = " OR ".join(f"cat:{cat}" for cat in self.config.categories)
        else:
            cat_query = "cat:cond-mat.mtrl-sci"

        
        if self.config.keywords:
            
            kw_query = " OR ".join(f"all:\"{kw}\"" for kw in self.config.keywords[:5])
            search_query = f"({cat_query}) OR ({kw_query})"
        else:
            search_query = cat_query

        params = urllib.parse.urlencode({
            "search_query": search_query,
            "start": self._arxiv_start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        })
        url = f"https://export.arxiv.org/api/query?{params}"

        
        base_delay = 3.0
        max_retries = 5
        last_exception = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "sci_host/1.0 (mailto:research@example.com)"},
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    xml_data = resp.read().decode("utf-8")

                
                self._arxiv_start += batch_size

                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                results: List[Dict[str, Any]] = []
                for entry in root.findall("atom:entry", ns):
                    title = entry.find("atom:title", ns)
                    summary = entry.find("atom:summary", ns)
                    published = entry.find("atom:published", ns)
                    arxiv_id = entry.find("atom:id", ns)

                    title_text = title.text.strip().replace("\n", " ") if title is not None else ""
                    abstract_text = summary.text.strip().replace("\n", " ") if summary is not None else ""
                    year = int(published.text[:4]) if published is not None else 2025
                    raw_id = arxiv_id.text.split("/")[-1] if arxiv_id is not None else ""

                    
                    categories = []
                    for cat in entry.findall("atom:category", ns):
                        term = cat.get("term", "")
                        if term:
                            categories.append(term)

                    
                    authors = []
                    for author in entry.findall("atom:author", ns):
                        name_el = author.find("atom:name", ns)
                        if name_el is not None and name_el.text:
                            authors.append(name_el.text.strip())

                    results.append({
                        "paper_id": f"arxiv_{raw_id}",
                        "title": title_text,
                        "abstract": abstract_text,
                        "categories": categories[:5],
                        "keywords": self._extract_keywords(title_text, abstract_text),
                        "authors": authors[:10],
                        "year": year,
                        "source": "arxiv",
                    })

                return results if results else self._fetch_offline(batch_size)

            except urllib.error.HTTPError as e:
                last_exception = e
                if e.code == 429:
                    delay = base_delay * (2 ** attempt)
                    _logger.warning("[arXiv] 429 Too Many Requests, retry %d/%d after %ds...", attempt + 1, max_retries, delay)
                    time.sleep(delay)
                else:
                    break
            except Exception as e:
                last_exception = e
                break

        
        _logger.warning("[arXiv] 在线采集失败 (%s), 回退到离线语料", last_exception)
        import traceback
        traceback.print_exc()
        return self._fetch_offline(batch_size)

    def _fetch_sciverse(self, batch_size: int) -> List[Dict[str, Any]]:
        """在线模式：通过 SCIVERSE agentic-search API 采集真实论文.

        查询轮转策略：每轮随机组合 2-3 个种子词，避免单词轮转快速耗尽。
        """
        import json as _json
        import urllib.request as _ureq
        import urllib.error as _uerr
        import os as _os
        import random as _rng

        token = _os.environ.get("SCIVERSE_API_TOKEN", "")
        if not token:
            _logger.info("[SCIVERSE] SCIVERSE_API_TOKEN not set, fallback to offline")
            return self._fetch_offline(batch_size)

        
        all_seeds = list(self.config.keywords) + list(self.research_seeds)
        if not all_seeds:
            all_seeds = [
                "materials science", "perovskite", "battery", "thermoelectric",
                "catalyst", "alloy", "semiconductor", "polymer", "composite",
                "crystal", "magnetic", "optical", "mechanical", "thermal",
                "corrosion", "nanostructure", "thin film", "ceramic", "metal",
                "electrochemical", "photocatalysis", "hydrogen", "energy storage",
            ]
        query_idx = getattr(self, "_sciverse_query_idx", 0)
        self._sciverse_query_idx = query_idx + 1
        _rng_obj = _rng.Random(query_idx)

        
        if query_idx == 0 and len(all_seeds) >= 3:
            query = " ".join(all_seeds[:3])
        else:
            n_pick = _rng_obj.randint(2, min(3, len(all_seeds)))
            picked = _rng_obj.sample(all_seeds, n_pick)
            query = " ".join(picked)

        
        if query_idx % 3 == 2:
            modifiers = [
                "recent advances", "review", "performance", "synthesis",
                "properties", "application", "mechanism", "design",
                "optimization", "characterization",
            ]
            query = query + " " + _rng_obj.choice(modifiers)

        query = query[:400]

        payload = _json.dumps({
            "query": query,
            "top_k": min(batch_size, 50),
            "sub_queries": 0,
        }).encode("utf-8")
        url = "https://api.sciverse.space/agentic-search"

        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                req = _ureq.Request(
                    url,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                        "User-Agent": "sci_host/1.0",
                    },
                    method="POST",
                )
                with _ureq.urlopen(req, timeout=45) as resp:
                    data = _json.loads(resp.read().decode("utf-8"))

                results: List[Dict[str, Any]] = []
                seen_titles: set = set()
                for hit in data.get("hits", []):
                    title = (hit.get("title") or "").strip()
                    if not title or title in seen_titles:
                        continue
                    seen_titles.add(title)
                    abstract = (hit.get("abstract") or "").strip()
                    chunk_text = (hit.get("chunk") or "").strip()
                    if not abstract and chunk_text:
                        abstract = chunk_text[:500]
                    doc_id = hit.get("doc_id", "")
                    authors = hit.get("author", []) or []
                    year = hit.get("publication_published_year", 2025) or 2025
                    venue = hit.get("publication_venue_name_unified", "")
                    primary_topic = hit.get("primary_topic", "")
                    score = hit.get("score", 0.0)
                    citation_count = hit.get("citation_count", 0)

                    categories = []
                    if primary_topic:
                        if isinstance(primary_topic, str):
                            if primary_topic.startswith("{"):
                                try:
                                    _pt = _json.loads(primary_topic)
                                    primary_topic = _pt.get("display_name", "") or ""
                                except Exception:
                                    primary_topic = ""
                            if primary_topic:
                                categories.append(primary_topic)
                        elif isinstance(primary_topic, dict):
                            dn = primary_topic.get("display_name", "")
                            if dn:
                                categories.append(dn)

                    keywords = self._extract_keywords(title, abstract)
                    results.append({
                        "paper_id": f"sciverse_{doc_id}",
                        "title": title,
                        "abstract": abstract,
                        "categories": categories[:5],
                        "keywords": keywords,
                        "authors": authors[:10],
                        "year": year,
                        "source": "sciverse",
                        "metadata": {
                            "doc_id": doc_id,
                            "score": score,
                            "venue": venue,
                            "citation_count": citation_count,
                            "chunk_id": hit.get("chunk_id", ""),
                            "primary_topic_domain": hit.get("primary_topic_domain", ""),
                        },
                    })

                _logger.info("[SCIVERSE] query=%s returned %d papers", query[:50], len(results))
                return results if results else self._fetch_offline(batch_size)

            except _uerr.HTTPError as e:
                last_error = e
                if e.code == 429:
                    import time as _time
                    _time.sleep(2 ** attempt)
                elif e.code == 401:
                    _logger.warning("[SCIVERSE] Invalid API token, fallback to offline")
                    return self._fetch_offline(batch_size)
                else:
                    _logger.warning("[SCIVERSE] HTTP %s: %s", e.code, e.reason)
                    break
            except Exception as e:
                last_error = e
                _logger.warning("[SCIVERSE] Request failed: %s", e)
                break

        _logger.warning("[SCIVERSE] fetch failed, fallback to offline")
        return self._fetch_offline(batch_size)

    def _fetch_openalex(self, batch_size: int) -> List[Dict[str, Any]]:
        """在线模式：通过 OpenAlex API 采集真实论文（聚焦材料科学）.

        OpenAlex 用 concepts（概念）标注每篇论文，比 arXiv 分类更适合跨领域发现。
        材料科学概念 id = C192562407 (Materials science)。若用户给了 keywords，作为 search 细化。
        注意: 需要网络连接；abstract 以 inverted index 形式返回，需解码。
        """
        try:
            import urllib.request
            import urllib.parse
            import json

            
            search_text = " ".join(self.config.keywords[:3]) if self.config.keywords else ""
            query_params = {
                "filter": "concepts.id:C192562407,has_abstract:true",
                "per_page": min(batch_size, 50),
                "cursor": self._openalex_cursor,
            }
            if search_text:
                query_params["search"] = search_text

            params = urllib.parse.urlencode(query_params)
            url = f"https://api.openalex.org/works?{params}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "sci_host/1.0 (mailto:research@example.com)"},
            )

            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            
            self._openalex_cursor = data.get("meta", {}).get("next_cursor", "*")

            results: List[Dict[str, Any]] = []
            for w in data.get("results", []):
                title = (w.get("title") or "").strip()
                abstract = self._decode_inverted_index(w.get("abstract_inverted_index"))
                if not title or not abstract:
                    continue
                raw_id = (w.get("id") or "").split("/")[-1]
                concepts = w.get("concepts", []) or []
                cats = [c.get("display_name", "") for c in concepts[:5]
                        if c.get("display_name")]
                kws = [k.get("keyword", "") for k in (w.get("keywords") or [])[:10]
                       if k.get("keyword")]
                if not kws:
                    kws = [c.get("display_name", "") for c in concepts[:10]
                           if c.get("display_name")]
                authors = [(a.get("author") or {}).get("display_name", "")
                           for a in (w.get("authorships") or [])[:10]]
                results.append({
                    "paper_id": f"openalex_{raw_id}",
                    "title": title,
                    "abstract": abstract,
                    "categories": cats,
                    "keywords": kws,
                    "authors": authors,
                    "year": w.get("publication_year") or 2025,
                    "source": "openalex",
                })

            return results if results else self._fetch_offline(batch_size)

        except Exception as e:
            
            import traceback
            traceback.print_exc()
            return self._fetch_offline(batch_size)

    @staticmethod
    def _decode_inverted_index(inv: Optional[Dict[str, List[int]]]) -> str:
        """解码 OpenAlex 的 abstract_inverted_index（词→位置列表）为文本."""
        if not inv:
            return ""
        slots: List[Tuple[int, str]] = []
        for word, positions in inv.items():
            for p in positions:
                slots.append((p, word))
        slots.sort(key=lambda x: x[0])
        return " ".join(w for _, w in slots)

    @staticmethod
    def _extract_keywords(title: str, abstract: str) -> List[str]:
        """简单关键词提取."""
        text = (title + " " + abstract).lower()
        
        stop = {"the", "a", "an", "for", "of", "to", "in", "on", "with", "and",
                "or", "is", "are", "we", "our", "by", "from", "as", "at", "this",
                "that", "these", "those", "it", "its", "be", "been", "was", "were",
                "has", "have", "had", "will", "would", "could", "should", "may",
                "might", "can", "than", "then", "so", "if", "but", "not", "no",
                
                "conference", "ieee", "international", "proceedings",
                "symposium", "journal", "transactions", "workshop",
                "acm", "springer", "elsevier", "wiley", "doi", "http",
                "https", "www", "com", "org", "pdf", "arxiv", "copyright",
                "license", "volume", "issue", "pages", "pp", "vol",
                "author", "authors", "published", "accepted",
                
                "preliminary", "comprehensive", "systematic", "detailed",
                "numerical", "experimental", "theoretical",
                "novel", "enhanced", "adaptive", "towards",
                "study", "investigation", "analysis", "research",
                "approach", "method", "result", "results",
                "show", "showed", "shown", "demonstrate", "propose",
                "present", "paper", "work", "article"}
        words = [w.strip(".,;:!?()[]{}\"'") for w in text.split()]
        keywords = [w for w in words if len(w) > 3 and w not in stop]
        
        seen: set = set()
        result: List[str] = []
        for w in keywords:
            if w not in seen:
                seen.add(w)
                result.append(w)
            if len(result) >= 10:
                break
        return result


class PaperCrawler:
    """论文持续采集器.

    7×24 小时持续运行，不断采集新的论文/材料。
    每次调用 crawl_batch() 返回一批新论文。

    与原项目 PhysicalEntity 的类比:
        PhysicalEntity.read_sensors() → PaperCrawler.crawl_batch()
        传感器数据 → 论文数据
    """

    def __init__(
        self,
        config: CrawlerConfig,
        embedder: Any = None,
        research_seeds: Optional[List[str]] = None,
        materials_mode: bool = False,
        quality_config: Optional[ResearchQualityConfig] = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.materials_mode = materials_mode
        self.source = PaperSource(config, research_seeds, materials_mode=materials_mode)
        self._quality_gate = ResearchQualityGate(quality_config)
        self._cache: Dict[str, Paper] = {}  # paper_id → Paper
        self._seen_ids: set = set()
        self._seen_titles: List[str] = []  
        self.total_crawled: int = 0
        self.quality_rejected_total: int = 0
        self.quality_rejected_last_batch: int = 0

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def fallback_count(self) -> int:
        """在线采集回退离线的次数 (委托给 PaperSource)."""
        return getattr(self.source, "_fallback_count", 0)

    @staticmethod
    def _normalize_title(title: str) -> str:
        """归一化标题用于去重: 转小写、去标点、去前缀."""
        import re as _re
        t = title.lower().strip()
        
        for prefix in ("a novel ", "towards ", "enhanced ", "adaptive ",
                       "scalable ", "robust ", "efficient ", "unified ",
                       "deep ", "progressive "):
            if t.startswith(prefix):
                t = t[len(prefix):]
        
        t = _re.sub(r'[^a-z0-9\s]', '', t)
        t = _re.sub(r'\s+', ' ', t).strip()
        return t

    def _is_duplicate_title(self, norm_title: str) -> bool:
        """检查归一化标题是否与已有论文重复.

        使用精确匹配 + 前缀匹配 (覆盖子标题变体).
        """
        if not norm_title or len(norm_title) < 10:
            return False
        for existing in self._seen_titles:
            
            if norm_title == existing:
                return True
            
            if len(norm_title) > 20 and len(existing) > 20:
                if norm_title in existing or existing in norm_title:
                    return True
        return False

    def crawl_batch(self) -> List[Paper]:
        """采集一批论文，返回 Paper 列表.

        SCIVERSE 模式下，当首批全部去重为 0 时自动换查询重试最多 2 次.
        """
        papers: List[Paper] = []
        self.quality_rejected_last_batch = 0
        _src = getattr(self.source, 'config', None)
        _is_sciverse = _src and getattr(_src, 'source', '') == 'sciverse'
        max_retries = 2 if _is_sciverse else 0

        for attempt in range(max_retries + 1):
            raw_papers = self.source.fetch(self.config.batch_size)
            papers = []

            for raw in raw_papers:
                paper_id = raw.get("paper_id", "")
                if not paper_id or paper_id in self._seen_ids:
                    continue

                
                raw_title = raw.get("title", "")
                norm_title = self._normalize_title(raw_title)
                if norm_title and self._is_duplicate_title(norm_title):
                    continue

                paper = Paper(
                    paper_id=paper_id,
                    title=raw.get("title", ""),
                    abstract=raw.get("abstract", ""),
                    authors=raw.get("authors", []),
                    categories=raw.get("categories", []),
                    keywords=raw.get("keywords", []),
                    year=raw.get("year", 2025),
                    source=raw.get("source", "unknown"),
                    data_source=raw.get("data_source", ""),
                    metadata=raw.get("metadata", {}),
                )

                
                self._seen_ids.add(paper_id)
                if not self._quality_gate.assess_paper(paper).accepted:
                    self.quality_rejected_last_batch += 1
                    self.quality_rejected_total += 1
                    continue

                
                if self.embedder is not None:
                    paper.embedding = self.embedder.embed(paper.text)

                papers.append(paper)
                self._cache[paper_id] = paper
                if norm_title:
                    self._seen_titles.append(norm_title)
                self.total_crawled += 1

            
            if papers or attempt >= max_retries:
                break

        
        if len(self._cache) > self.config.max_cache:
            
            excess = len(self._cache) - self.config.max_cache
            old_ids = list(self._cache.keys())[:excess]
            for oid in old_ids:
                del self._cache[oid]

        return papers

    def quality_stats(self) -> Dict[str, Any]:
        """返回专项质量闸门统计，供 MCP/流式调用观察."""
        return {
            "enabled": self._quality_gate.enabled,
            "rejected_last_batch": self.quality_rejected_last_batch,
            "rejected_total": self.quality_rejected_total,
        }

    def get_cached_papers(self, n: int = 0) -> List[Paper]:
        """获取缓存中的论文."""
        cached = list(self._cache.values())
        return cached[:n] if n > 0 else cached

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """按 ID 获取论文."""
        return self._cache.get(paper_id)

    def clear_cache(self) -> None:
        """清空缓存."""
        self._cache.clear()
