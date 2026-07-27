"""HostSystem — 科学方向探索宿主系统主入口.

以独立科研仿真运行时为实验环境，在仿真中进行科研方向探索.
核心思想 (用户):
    让模型在数字孪生的虚拟环境里不断试错、排列组合.
    复用原项目已实现的算法算子 (CrossDomainTransfer, CausalDiscovery 等).
    利用 EWC 持续学习，越跑越聪明.
    不要 Sleep Cycle / 梦境合成，7×24 连续运行.

系统架构:
    ┌──────────────────────────────────────────────────────┐
    │                    HostSystem                         │
    │                                                       │
    │  ┌──────────────────────────────────────────────────┐ │
    │  │         ResearchTwin (虚拟实验环境)                │ │
    │  │  PE=论文语料  VM=假设评估器  CL=EWC持续学习        │ │
    │  │  注册算子: CrossDomain / Causal / Uncertainty     │ │
    │  │            KnowledgeGraph / LLM / MultiFidelity   │ │
    │  └──────────────────────────────────────────────────┘ │
    │                                                       │
    │  ┌──────────┐  ┌──────────┐  ┌──────────────┐        │
    │  │ Crawler  │→ │  Pairer  │→ │ Hypothesis   │        │
    │  │ 论文采集  │  │ 隐性配对  │  │ Generator    │        │
    │  └──────────┘  └──────────┘  └──────┬───────┘        │
    │                                      │                │
    │  ┌──────────┐  ┌──────────┐  ┌──────▼───────┐        │
    │  │Knowledge │← │  Trial   │← │  Continuous   │        │
    │  │Accumulator│  │  Engine  │  │    Loop       │        │
    │  │  +EWC    │  │ (孪生仿真) │  │               │        │
    │  └──────────┘  └──────────┘  └──────────────┘        │
    └──────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..config import HostConfig
from .event_bus import HostEvent, HostEventBus
from .state import HostSnapshot, HostState

_logger = logging.getLogger(__name__)


class HostSystem:
    """科学方向探索宿主系统.

    7×24 小时连续运行的科学方向发现系统。
    持续采集论文 → 隐性配对 → 生成假设 → 试错验证 → 知识积累。

    使用方式:
        host = HostSystem(HostConfig.quick())
        host.run()

        host = HostSystem(HostConfig.production())
        host.run()  # 无限循环

        host.start()
        result = host.step()
        host.stop()
    """

    VERSION = "SciHost v1.0.0"

    def __init__(self, config: Optional[HostConfig] = None) -> None:
        self.config = config or HostConfig()
        self.host_id = self.config.host_id
        self.version = self.VERSION

        
        self.event_bus = HostEventBus(history_size=500)

        
        self.state = HostState()

        
        self._crawler = None
        self._pairer = None
        self._hypothesis_gen = None
        self._hypothesis_eval = None
        self._trial_engine = None
        self._knowledge_graph = None
        self._knowledge_acc = None
        self._direction_tracker = None
        self._loop = None
        self._verification_engine = None

        
        self._twin: Optional[Any] = None
        self._materials_mode = self.config.materials_mode

        
        # key=paper_id, value=List[CSPTriple]
        self._agent_csp_cache: Dict[str, List[Any]] = {}
        self._agent_mode: bool = False

        
        self._agent_feedback: Dict[str, Any] = {}  
        self._streaming_mode: bool = False
        self._streaming_state: Dict[str, Any] = {}  

        
        self._running = False
        self._initialized = False

        
        self._logs: List[Dict[str, Any]] = []

    

    def _init_components(self) -> None:
        """懒初始化所有核心组件，包括数字孪生虚拟实验环境."""
        from ..crawler.paper_crawler import PaperCrawler
        from ..pairing.implicit_pairer import ImplicitPairer
        from ..pairing.embedding import TextEmbedder
        from ..hypothesis.generator import HypothesisGenerator
        from ..hypothesis.evaluator import HypothesisEvaluator
        from ..trial.engine import TrialEngine
        from ..trial.knowledge_graph import ScienceKnowledgeGraph
        from ..knowledge.accumulator import KnowledgeAccumulator
        from ..knowledge.direction_tracker import DirectionTracker
        from ..loop.continuous_loop import ContinuousLoop
        from ..trial.verification import VerificationEngine

        
        llm_cfg = self.config.llm_extraction
        if llm_cfg.enabled and llm_cfg.api_key:
            try:
                from ..materials.llm_extractor import LLMConfig, set_llm_config
                set_llm_config(LLMConfig(
                    api_base=llm_cfg.api_base,
                    api_key=llm_cfg.api_key,
                    model=llm_cfg.model,
                    max_tokens=llm_cfg.max_tokens,
                    temperature=llm_cfg.temperature,
                    timeout=llm_cfg.timeout,
                    max_retries=llm_cfg.max_retries,
                    enabled=True,
                ))
                self._log("INFO", f"LLM CSP 抽取已启用: model={llm_cfg.model}")
            except Exception as e:
                self._log("WARN", f"LLM 抽取器初始化失败: {e}")

        cfg = self.config

        
        embedder = TextEmbedder(
            max_features=cfg.pairing.max_features,
            ngram_range=cfg.pairing.ngram_range,
        )

        
        self._crawler = PaperCrawler(
            config=cfg.crawler,
            embedder=embedder,
            research_seeds=cfg.research_seeds,
            materials_mode=cfg.materials_mode,
            quality_config=cfg.quality,
        )

        
        self._pairer = ImplicitPairer(
            embedder=embedder,
            config=cfg.pairing,
            quality_config=cfg.quality,
        )

        
        
        _domain_kw = list(cfg.research_seeds) + list(cfg.crawler.keywords)
        self._hypothesis_gen = HypothesisGenerator(
            config=cfg.hypothesis,
            materials_mode=cfg.materials_mode,
            domain_keywords=_domain_kw if _domain_kw else None,
            quality_config=cfg.quality,
        )
        self._hypothesis_eval = HypothesisEvaluator()

        
        self._knowledge_graph = ScienceKnowledgeGraph(
            max_nodes=cfg.knowledge.max_graph_nodes,
        )

        
        self._init_twin()

        
        self._trial_engine = TrialEngine(
            config=cfg.trial,
            twin=self._twin,
            knowledge_graph=self._knowledge_graph,
            materials_mode=self._materials_mode,
        )

        
        self._knowledge_acc = KnowledgeAccumulator(config=cfg.knowledge)

        
        self._direction_tracker = DirectionTracker(
            config=cfg.knowledge,
            quality_config=cfg.quality,
        )

        
        self._verification_engine = VerificationEngine(twin=self._twin)

        
        self._loop = ContinuousLoop(
            host=self,
            config=cfg.loop,
        )

        self._initialized = True
        self._log("components_initialized", {
            "host_id": self.host_id,
            "twin_algorithms": list(self._twin.algorithms.keys()) if self._twin else [],
        })

    def _init_twin(self) -> None:
        """创建独立科研仿真实例并注册评估算子.

        运行时提供的评估算子:
        - cross_domain_transfer: 评估跨域迁移可行性
        - semantic_knowledge_graph: 知识图谱支持度
        - uncertainty_quantifier: 不确定性量化
        - causal_discovery: 因果一致性
        - llm_reasoning: 语义推理
        - multi_fidelity_fusion: 多保真度融合
        在线校准状态保存算子权重和试错摘要。
        """
        from ..research_twin import ResearchTwin, register_research_operator

        
        self._twin = ResearchTwin(
            twin_id=f"scihost-{self.host_id}",
            domain="sci_host",
            ewc_lambda=0.4,
            buffer_size=10000,
        )

        algorithms_to_register = [
            "cross_domain_transfer",
            "semantic_knowledge_graph",
            "uncertainty_quantifier",
            "causal_discovery",
            "llm_reasoning",
            "multi_fidelity_fusion",
            "graph_topology",
            "meta_learning",
        ]
        for algo_name in algorithms_to_register:
            try:
                register_research_operator(self._twin, algo_name)
            except Exception as e:
                self._log("algorithm_register_failed", {
                    "name": algo_name, "error": str(e),
                })

        self._log("twin_initialized", {
            "algorithms": list(self._twin.algorithms.keys()),
        })

    

    def start(self) -> None:
        """启动宿主系统（初始化组件，不进入循环）."""
        if not self._initialized:
            self._init_components()
        self._running = True
        self.state.start_time = time.time()
        self._publish("host.start", {"host_id": self.host_id, "version": self.version})
        self._log("host_started", {})

    def stop(self) -> None:
        """停止宿主系统."""
        self._running = False
        self._publish("host.stop", {"uptime": self.state.uptime_seconds})
        self._log("host_stopped", {})

    @property
    def running(self) -> bool:
        return self._running

    def run(self) -> None:
        """7×24 连续运行（阻塞）.

        内部调用 ContinuousLoop，无限循环直到 stop() 或达到 max_iterations。
        """
        if not self._initialized:
            self._init_components()
        self.start()
        try:
            self._loop.run()
        except KeyboardInterrupt:
            _logger.info("[HostSystem] 收到中断信号，正在停止...")
        finally:
            self.stop()

    

    def step(self) -> Dict[str, Any]:
        """执行一轮完整探索循环.

        一轮循环包含:
            1. 采集论文 → 2. 隐性配对 → 3. 生成假设 →
            4. 孪生仿真试错 → 5. 验证复现 → 6. 知识积累 → 7. 方向更新

        Returns:
            本轮探索的完整结果摘要
        """
        if not self._initialized:
            self._init_components()

        start = time.time()
        result: Dict[str, Any] = {
            "cycle": self.state.cycle_count,
            "timestamp": start,
        }

        
        papers = self._crawl_step()
        result["papers_crawled"] = len(papers)
        result["papers_quality_rejected"] = self.state.papers_quality_rejected
        result["offline_fallback_crawls"] = self.state.offline_fallback_crawls

        
        pairs = self._pair_step(papers)
        result["pairs_found"] = len(pairs)

        
        hypotheses = self._hypothesize_step(pairs)
        result["hypotheses_generated"] = len(hypotheses)
        result["sample_hypotheses"] = self.state.sample_hypotheses()

        
        trial_results = self._trial_step(hypotheses)
        result["trials_passed"] = sum(1 for r in trial_results if r.passed)
        result["trials_failed"] = sum(1 for r in trial_results if not r.passed)

        
        verification_result = self._verification_step(
            hypotheses, trial_results,
        )
        result["verification"] = verification_result

        
        knowledge_result = self._knowledge_step(trial_results, papers)
        result["knowledge_updated"] = knowledge_result

        
        direction_result = self._direction_step(trial_results, verification_result)
        result["directions_updated"] = direction_result

        
        self.state.cycle_count += 1
        elapsed = time.time() - start
        result["elapsed_ms"] = round(elapsed * 1000, 1)

        self._publish("loop.cycle", result)
        return result

    

    def get_papers_for_csp(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取待 CSP 抽取的论文列表 (供 MCP agent 读取).

        Agent (如 Claude) 读取这些论文的标题+摘要,
        自行抽取 CSP 三元组后通过 submit_agent_csp() 提交.

        Args:
            n: 最多返回论文数

        Returns:
            论文列表, 每条含 paper_id, title, abstract
        """
        if not self._initialized:
            self._init_components()

        papers_out: List[Dict[str, Any]] = []
        seen_ids: set = set()

        
        if self._pairer is not None and hasattr(self._pairer, "_paper_pool"):
            for pid, p in self._pairer._paper_pool.items():
                if pid in seen_ids:
                    continue
                if pid in self._agent_csp_cache:
                    continue  
                seen_ids.add(pid)
                papers_out.append({
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "abstract": p.abstract or "",
                    "keywords": p.keywords[:5] if p.keywords else [],
                    "categories": p.categories[:3] if p.categories else [],
                })
                if len(papers_out) >= n:
                    break

        
        if len(papers_out) < n and self._crawler is not None:
            cache = getattr(self._crawler, "_cache", [])
            cache_items = cache.values() if isinstance(cache, dict) else cache
            for p in cache_items:
                if not hasattr(p, "paper_id"):
                    continue
                if p.paper_id in seen_ids:
                    continue
                if p.paper_id in self._agent_csp_cache:
                    continue
                seen_ids.add(p.paper_id)
                papers_out.append({
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "abstract": p.abstract or "",
                    "keywords": p.keywords[:5] if p.keywords else [],
                    "categories": p.categories[:3] if p.categories else [],
                })
                if len(papers_out) >= n:
                    break

        return papers_out

    def submit_agent_csp(self, csp_triples: List[Dict[str, Any]]) -> int:
        """接收 MCP agent 提交的 CSP 三元组.

        Agent (如 Claude) 读取论文摘要后自行抽取三元组,
        通过此方法提交到系统知识库.

        Args:
            csp_triples: 三元组列表, 每条含:
                - paper_id: 论文 ID (必填)
                - composition: 化学组分 (如 "BaTiO3")
                - structure: 晶体结构 (如 "perovskite")
                - property_name: 性能名 (如 "bandgap")
                - property_value: 性能值 (数值或 null)
                - property_unit: 单位 (如 "eV")

        Returns:
            成功存入的三元组数
        """
        if not self._initialized:
            self._init_components()

        from ..materials import CSPTriple

        count = 0
        for item in csp_triples:
            paper_id = item.get("paper_id", "")
            comp = str(item.get("composition", "")).strip()
            struct = str(item.get("structure", "unknown")).strip()
            prop_name = str(item.get("property_name", "")).strip()

            if not comp or not prop_name:
                continue

            
            prop_value = item.get("property_value")
            if prop_value is not None:
                try:
                    prop_value = float(prop_value)
                except (ValueError, TypeError):
                    prop_value = None

            prop_unit = str(item.get("property_unit", "")).strip()

            triple = CSPTriple(
                composition=comp,
                structure=struct,
                property_name=prop_name,
                property_value=prop_value,
                property_unit=prop_unit,
                source_paper_id=paper_id,
                source_paper_title=item.get("paper_title", ""),
                source_section="agent_extracted",
                confidence=0.95,  
            )

            if paper_id not in self._agent_csp_cache:
                self._agent_csp_cache[paper_id] = []
            self._agent_csp_cache[paper_id].append(triple)
            count += 1

        
        if self._hypothesis_gen is not None and hasattr(self._hypothesis_gen, "_csp_knowledge"):
            for paper_id, triples in self._agent_csp_cache.items():
                for triple in triples:
                    if triple.key not in self._hypothesis_gen._csp_knowledge:
                        self._hypothesis_gen._csp_knowledge[triple.key] = triple

        self._agent_mode = True
        self._log("INFO", f"Agent CSP: 接收 {count} 条三元组, 覆盖 {len(self._agent_csp_cache)} 篇论文")
        return count

    def step_agent(self) -> Dict[str, Any]:
        """Agent 模式探索循环.

        与 step() 的区别:
        - 假设生成阶段优先使用 agent 提交的 CSP 三元组
        - 不调用内部 LLM API (agent 即 LLM)
        - 如果某论文没有 agent CSP, 降级为正则抽取

        Returns:
            本轮探索结果
        """
        if not self._initialized:
            self._init_components()

        
        self._agent_mode = True

        start = time.time()
        result: Dict[str, Any] = {
            "cycle": self.state.cycle_count,
            "timestamp": start,
            "mode": "agent",
        }

        
        papers = self._crawl_step()
        result["papers_crawled"] = len(papers)
        result["papers_quality_rejected"] = self.state.papers_quality_rejected
        result["offline_fallback_crawls"] = self.state.offline_fallback_crawls

        
        pairs = self._pair_step(papers)
        result["pairs_found"] = len(pairs)

        
        hypotheses = self._hypothesize_step_agent(pairs)
        result["hypotheses_generated"] = len(hypotheses)
        result["sample_hypotheses"] = self.state.sample_hypotheses()

        
        trial_results = self._trial_step(hypotheses)
        result["trials_passed"] = sum(1 for r in trial_results if r.passed)
        result["trials_failed"] = sum(1 for r in trial_results if not r.passed)

        
        verification_result = self._verification_step(
            hypotheses, trial_results,
        )
        result["verification"] = verification_result

        
        knowledge_result = self._knowledge_step(trial_results, papers)
        result["knowledge_updated"] = knowledge_result

        
        direction_result = self._direction_step(trial_results, verification_result)
        result["directions_updated"] = direction_result

        self.state.cycle_count += 1
        elapsed = time.time() - start
        result["elapsed_ms"] = round(elapsed * 1000, 1)
        result["agent_csp_count"] = sum(len(v) for v in self._agent_csp_cache.values())

        self._publish("loop.cycle", result)
        return result

    def _hypothesize_step_agent(self, pairs: List[Any]) -> List[Any]:
        """Agent 模式假设生成: 优先使用 agent CSP 三元组."""
        try:
            
            if self._agent_csp_cache and self._hypothesis_gen is not None:
                
                from ..materials import CSPExtractor
                original_extract = CSPExtractor.extract

                @classmethod
                def _agent_extract(cls, text: str, paper_id: str = "", paper_title: str = "") -> List[Any]:
                    """Agent CSP 优先抽取."""
                    
                    if paper_id and paper_id in self._agent_csp_cache:
                        return self._agent_csp_cache[paper_id]
                    
                    return original_extract.__func__(cls, text, paper_id=paper_id, paper_title=paper_title)

                
                CSPExtractor.extract = _agent_extract
                try:
                    hypotheses = self._hypothesis_gen.generate(pairs)
                finally:
                    
                    CSPExtractor.extract = original_extract
            else:
                hypotheses = self._hypothesis_gen.generate(pairs)

            self.state.hypotheses_generated += len(hypotheses)
            self.state.hypotheses_active += len(hypotheses)
            for h in hypotheses:
                self._publish("hypothesis.generated", {
                    "id": h.hypothesis_id,
                    "type": h.hypothesis_type,
                    "statement": h.statement[:100],
                    "method": h.metadata.get("generation_method", "template"),
                })
                
                self.state.add_recent_hypothesis({
                    "hypothesis_id": h.hypothesis_id,
                    "statement": h.statement[:200],
                    "type": h.hypothesis_type,
                    "hypothesis_type": h.hypothesis_type,
                    "method": h.metadata.get("generation_method", "template"),
                    "confidence": h.confidence,
                    "novelty": h.novelty,
                    "keywords": list(h.keywords[:10]),
                    "paper_a_id": h.paper_a_id,
                    "paper_a_title": h.paper_a_title,
                    "paper_b_id": h.paper_b_id,
                    "paper_b_title": h.paper_b_title,
                })
            return hypotheses
        except Exception as e:
            self._handle_error("hypothesize_agent", str(e))
            return []

    

    def stream_crawl(self) -> Dict[str, Any]:
        """流式阶段 1: 采集论文, 返回给 Claude 评估.

        Claude 可以:
        - 查看论文标题/摘要
        - 标记哪些论文值得关注
        - 提供关键词调整建议
        """
        if not self._initialized:
            self._init_components()
        self._streaming_mode = True

        papers = self._crawl_step()
        papers_data = [
            {
                "paper_id": p.paper_id,
                "title": p.title,
                "abstract": (p.abstract or "")[:200],
                "keywords": p.keywords[:5] if p.keywords else [],
                "categories": p.categories[:3] if p.categories else [],
                "data_source": getattr(p, "data_source", "unknown"),
            }
            for p in papers
        ]

        self._streaming_state["papers"] = papers
        self._streaming_state["papers_data"] = papers_data

        
        fallback_count = self._crawler.fallback_count if self._crawler else 0

        
        _online = sum(1 for p in papers if getattr(p, "data_source", "") == "online")
        _offline = sum(1 for p in papers if getattr(p, "data_source", "") in ("offline_fallback", "offline"))
        if _offline > 0 and _online == 0:
            _data_source = "offline_fallback"
        elif _offline > 0:
            _data_source = "mixed"
        else:
            _data_source = "online"

        
        _agent_required = False
        _agent_reason = ""
        if fallback_count > 0:
            _agent_required = True
            _agent_reason = f"在线采集回退离线 {fallback_count} 次, 建议检查 API 配置"
        elif len(papers) == 0:
            _agent_required = True
            _agent_reason = "本轮未采集到任何论文, 建议调整关键词或数据源"

        return {
            "stage": "crawl",
            "papers_crawled": len(papers),
            "papers": papers_data,
            "data_source": _data_source,
            "fallback_count": fallback_count,
            "online_papers": _online,
            "offline_papers": _offline,
            "quality": self._crawler.quality_stats() if self._crawler else {},
            "fallback_warning": f"检测到 {fallback_count} 次在线采集回退离线, 请检查 API token/网络" if fallback_count > 0 else "",
            "agent_action_required": _agent_required,
            "agent_action_reason": _agent_reason,
            "next": "pair (调用 sci_stream_pair) 或 feedback (调用 sci_stream_feedback)",
        }

    def stream_pair(self) -> Dict[str, Any]:
        """流式阶段 2: 隐性配对, 返回配对结果给 Claude 评估.

        Claude 可以:
        - 查看哪些跨领域配对最有前景
        - 标记值得深挖的配对
        - 提供配对阈值调整建议
        """
        if not self._initialized:
            self._init_components()

        papers = self._streaming_state.get("papers", [])
        if not papers:
            
            
            return {
                "stage": "pair",
                "error": "no_papers_in_stream",
                "pairs_found": 0,
                "cross_domain": 0,
                "pairs": [],
                "agent_action_required": True,
                "agent_action_reason": "流式配对阶段无论文可用, 请先调用 sci_stream_crawl 采集论文",
                "next": "crawl (调用 sci_stream_crawl 先采集论文)",
            }

        
        
        focus_paper_ids = self._streaming_state.get("focus_paper_ids", set())
        if focus_paper_ids:
            focus_papers = [p for p in papers if p.paper_id in focus_paper_ids]
            other_papers = [p for p in papers if p.paper_id not in focus_paper_ids]
            papers = focus_papers + other_papers
            self._log("INFO", f"Stream pair: 优先配对 {len(focus_papers)} 篇关注论文")

        pairs = self._pair_step(papers) if papers else []
        pairs_data = [
            {
                "pair_id": getattr(p, "pair_id", ""),
                "paper_a_title": getattr(p, "paper_a_title", ""),
                "paper_b_title": getattr(p, "paper_b_title", ""),
                "similarity": round(getattr(p, "similarity", 0), 3),
                "cross_domain": getattr(p, "cross_domain", False),
                "bridge_keywords": getattr(p, "bridge_keywords", [])[:5],
            }
            for p in pairs
        ]

        self._streaming_state["pairs"] = pairs
        self._streaming_state["pairs_data"] = pairs_data

        
        _cross_count = sum(1 for p in pairs if getattr(p, "cross_domain", False))
        _agent_required = _cross_count > 0 or len(pairs) == 0
        _agent_reason = ""
        if _cross_count > 0:
            _agent_reason = f"发现 {_cross_count} 个跨领域配对, 建议评估桥接关键词的技术相关性"
        elif len(pairs) == 0:
            _agent_reason = "未产生任何配对, 建议调整相似度阈值或关键词"

        return {
            "stage": "pair",
            "pairs_found": len(pairs),
            "cross_domain": _cross_count,
            "pairs": pairs_data[:15],
            "agent_action_required": _agent_required,
            "agent_action_reason": _agent_reason,
            "next": "hypothesize (调用 sci_stream_hypothesize) 或 feedback",
        }

    def stream_hypothesize(self) -> Dict[str, Any]:
        """流式阶段 3: 生成假设, 返回给 Claude 评估.

        Claude 可以:
        - 查看假设陈述
        - 判断假设是否有科学意义
        - 标记值得试错的假设
        - 提供假设改进建议
        """
        if not self._initialized:
            self._init_components()

        pairs = self._streaming_state.get("pairs", [])
        if not pairs:
            return {"stage": "hypothesize", "error": "no pairs, call stream_pair first"}

        
        hypotheses = self._hypothesize_step_agent(pairs) if self._agent_csp_cache else self._hypothesize_step(pairs)

        
        
        focus_pair_ids = self._streaming_state.get("focus_pair_ids", set())
        if focus_pair_ids and hypotheses:
            focused = [h for h in hypotheses if h.source_pair in focus_pair_ids]
            others = [h for h in hypotheses if h.source_pair not in focus_pair_ids]
            if focused:
                hypotheses = focused + others
                self._log("INFO", f"Stream hypothesize: 优先处理 {len(focused)} 个关注配对的假设")

        hypos_data = [
            {
                "hypothesis_id": h.hypothesis_id,
                "statement": h.statement[:200],
                "type": h.hypothesis_type,
                "keywords": h.keywords[:5],
                "paper_a_title": h.paper_a_title,
                "paper_b_title": h.paper_b_title,
                "has_prediction": h.metadata.get("has_numerical_prediction", False),
            }
            for h in hypotheses
        ]

        self._streaming_state["hypotheses"] = hypotheses
        self._streaming_state["hypos_data"] = hypos_data

        
        _agent_required = len(hypotheses) == 0
        _agent_reason = ""
        if len(hypotheses) == 0:
            _agent_reason = "未生成任何假设 (可能因可证伪性闸门过滤), 建议检查配对质量"
        
        elif all(not h.metadata.get("has_numerical_prediction", False) for h in hypotheses):
            _agent_required = True
            _agent_reason = "所有假设均无数值预测, 可证伪性较低, 建议人工审查"

        return {
            "stage": "hypothesize",
            "hypotheses_generated": len(hypotheses),
            "hypotheses": hypos_data,
            "agent_action_required": _agent_required,
            "agent_action_reason": _agent_reason,
            "next": "trial (调用 sci_stream_trial) 或 feedback",
        }

    def stream_trial(self) -> Dict[str, Any]:
        """流式阶段 4: 孪生仿真试错, 返回每个假设的算子预测给 Claude.

        Claude 可以:
        - 查看 6 个算法算子 + LLM 流式算子的预测
        - 判断哪些假设通过得有道理, 哪些是碰巧
        - 对通过的假设标记 "值得复现验证"
        - 对失败的假设分析原因
        """
        if not self._initialized:
            self._init_components()

        hypotheses = self._streaming_state.get("hypotheses", [])
        if not hypotheses:
            return {"stage": "trial", "error": "no hypotheses, call stream_hypothesize first"}

        
        if self._twin and self._twin.pe and self._crawler:
            all_papers = self._crawler.get_cached_papers()
            self._twin.pe.update_corpus_stats(all_papers, [])

        
        
        focus_hypo_ids = self._streaming_state.get("focus_hypo_ids", set())
        if focus_hypo_ids and hypotheses:
            focused = [h for h in hypotheses if h.hypothesis_id in focus_hypo_ids]
            others = [h for h in hypotheses if h.hypothesis_id not in focus_hypo_ids]
            if focused:
                hypotheses = focused + others
                self._log("INFO", f"Stream trial: 优先试错 {len(focused)} 个关注假设")
                self._streaming_state["hypotheses"] = hypotheses

        trial_results = self._trial_step(hypotheses)

        
        feedback = self._agent_feedback.get("trial", {})
        if feedback:
            for r in trial_results:
                
                if r.hypothesis_id in feedback.get("force_pass", []):
                    r.passed = True
                    r.metadata["agent_override"] = True
                if r.hypothesis_id in feedback.get("force_fail", []):
                    r.passed = False

        trials_data = [
            {
                "hypothesis_id": r.hypothesis_id,
                "statement": r.statement[:150],
                "passed": r.passed,
                "score": r.score,
                "consistency": r.consistency,
                "algo_predictions": r.algo_predictions,
                "failure_reason": r.failure_reason[:100] if r.failure_reason else "",
                "llm_reasoning": r.metadata.get("llm_reasoning", ""),
                "credibility": r.credibility,
                "credibility_note": r.credibility_note,
            }
            for r in trial_results
        ]

        self._streaming_state["trial_results"] = trial_results
        self._streaming_state["trials_data"] = trials_data

        
        _passed = sum(1 for r in trial_results if r.passed)
        _failed = len(trial_results) - _passed
        _agent_required = False
        _agent_reason = ""
        
        if _passed > 0 and _failed == 0:
            _agent_required = True
            _agent_reason = "所有假设均通过, 可能阈值过松, 建议人工复核"
        
        elif _passed > 0 and _failed > 0 and abs(_passed - _failed) <= 2:
            _agent_required = True
            _agent_reason = f"通过 {_passed} vs 失败 {_failed}, 分歧较大, 建议评估边界案例"
        
        elif _passed == 0 and _failed > 0:
            _agent_required = True
            _agent_reason = "所有假设均失败, 建议检查配对和假设生成质量"

        return {
            "stage": "trial",
            "trials_total": len(trial_results),
            "trials_passed": _passed,
            "trials_failed": _failed,
            "trials": trials_data,
            "agent_action_required": _agent_required,
            "agent_action_reason": _agent_reason,
            "next": "verify (调用 sci_stream_verify) 或 feedback",
        }

    def stream_verify(self) -> Dict[str, Any]:
        """流式阶段 5: 验证复现, 返回认证发现给 Claude.

        Claude 可以:
        - 查看哪些假设被认证为科学发现
        - 评估发现级别 (Gold/Silver/Bronze) 是否合理
        - 提供最终研究建议
        """
        if not self._initialized:
            self._init_components()

        hypotheses = self._streaming_state.get("hypotheses", [])
        trial_results = self._streaming_state.get("trial_results", [])
        if not trial_results:
            return {"stage": "verify", "error": "no trial results, call stream_trial first"}

        
        
        
        
        trial_feedback = self._agent_feedback.get("trial", {})
        if trial_feedback:
            force_pass_ids = set(trial_feedback.get("force_pass", []))
            force_fail_ids = set(trial_feedback.get("force_fail", []))
            applied_count = 0
            for r in trial_results:
                original_passed = r.passed
                if r.hypothesis_id in force_pass_ids:
                    r.passed = True
                    r.metadata["agent_override"] = True
                    if not original_passed:
                        r.failure_reason = ""
                        applied_count += 1
                if r.hypothesis_id in force_fail_ids:
                    r.passed = False
                    if original_passed:
                        r.failure_reason = "Agent 强制失败: " + trial_feedback.get("reasoning", "")[:100]
                        applied_count += 1
            if applied_count:
                self._log("INFO", f"Stream verify: 重新应用 trial feedback, 修改了 {applied_count} 个结果")
                
                self._streaming_state["trial_results"] = trial_results

        
        _force_pass_ids = set(trial_feedback.get("force_pass", [])) if trial_feedback else set()

        verification_result = self._verification_step(hypotheses, trial_results, force_pass_ids=_force_pass_ids)
        knowledge_result = self._knowledge_step(trial_results, self._streaming_state.get("papers", []))
        direction_result = self._direction_step(trial_results, verification_result)

        
        
        verify_feedback = self._agent_feedback.get("verify", {})
        direction_hint = verify_feedback.get("direction_hint", "")
        if direction_hint:
            
            if hasattr(self._direction_tracker, "add_manual_direction"):
                self._direction_tracker.add_manual_direction(direction_hint)
            self._log("INFO", f"Stream verify: 方向建议已注入: {direction_hint[:80]}")
            direction_result["agent_direction_hint"] = direction_hint

        self.state.cycle_count += 1

        
        csp_triples = self.get_csp_knowledge(n=10)

        self._streaming_state = {}  

        return {
            "stage": "verify",
            "verification": verification_result,
            "knowledge_updated": knowledge_result,
            "directions": direction_result,
            "csp_triples": csp_triples,
            "cycle_complete": True,
            "next": "重新开始: sci_stream_crawl",
        }

    def stream_feedback(self, stage: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """接收 Claude 在任意阶段的反馈, 影响后续流程.

        Args:
            stage: 当前阶段名 ("crawl" / "pair" / "hypothesize" / "trial" / "verify")
            feedback: 反馈内容, 可包含:
                - focus_paper_ids: 值得关注的论文 ID (crawl 阶段)
                - focus_pair_ids: 值得深挖的配对 ID (pair 阶段)
                - force_pass: 强制通过的假设 ID (trial 阶段)
                - force_fail: 强制失败的假设 ID (trial 阶段)
                - reasoning: Claude 的推理文字
                - direction_hint: 研究方向建议

        Returns:
            确认信息
        """
        self._agent_feedback[stage] = feedback
        reasoning = feedback.get("reasoning", "")

        
        if stage == "crawl":
            
            focus_ids = feedback.get("focus_paper_ids", [])
            if focus_ids:
                self._streaming_state["focus_paper_ids"] = set(focus_ids)

        elif stage == "pair":
            
            focus_pair_ids = feedback.get("focus_pair_ids", [])
            if focus_pair_ids:
                self._streaming_state["focus_pair_ids"] = set(focus_pair_ids)

        elif stage == "hypothesize":
            
            focus_hypo_ids = feedback.get("focus_hypothesis_ids", [])
            if focus_hypo_ids:
                self._streaming_state["focus_hypo_ids"] = set(focus_hypo_ids)

        elif stage == "trial":
            
            
            pass

        self._log("INFO", f"Stream feedback [{stage}]: {reasoning[:80] if reasoning else '(no reasoning)'}")

        return {
            "status": "accepted",
            "stage": stage,
            "feedback_applied": True,
            "message": f"反馈已记录, 将影响后续 {stage} 阶段",
        }

    

    def _crawl_step(self) -> List[Any]:
        """采集论文.

        统计三个层次的论文计数:
          - papers_raw_fetched: 原始获取数 (含被质量闸门拒绝的)
          - papers_quality_rejected: 被质量闸门拒绝的
          - papers_collected: 质量闸门后的有效论文数
        同时跟踪在线回退离线次数.
        """
        try:
            
            prev_fallback = self.state.offline_fallback_crawls
            papers = self._crawler.crawl_batch()
            cur_fallback = self._crawler.fallback_count if self._crawler else 0
            fallback_delta = cur_fallback - prev_fallback

            
            q_stats = self._crawler.quality_stats() if self._crawler else {}
            rejected_this_batch = q_stats.get("rejected_last_batch", 0)

            
            self.state.papers_collected += len(papers)
            self.state.papers_raw_fetched += len(papers) + rejected_this_batch
            self.state.papers_quality_rejected += rejected_this_batch
            self.state.papers_in_cache = self._crawler.cache_size if self._crawler else 0
            self.state.offline_fallback_crawls = cur_fallback

            if papers:
                self._publish("crawler.batch", {
                    "count": len(papers),
                    "rejected": rejected_this_batch,
                    "fallback_delta": fallback_delta,
                })
            return papers
        except Exception as e:
            self._handle_error("crawl", str(e))
            return []

    def _pair_step(self, papers: List[Any]) -> List[Any]:
        """隐性配对."""
        try:
            if not papers:
                return []
            
            self._pairer.add_papers(papers)
            pairs = self._pairer.find_pairs()
            self.state.pairs_found += len(pairs)
            cross = sum(1 for p in pairs if getattr(p, "cross_domain", False))
            self.state.cross_domain_pairs += cross
            if pairs:
                self._publish("pairing.done", {
                    "count": len(pairs), "cross_domain": cross,
                })
            return pairs
        except Exception as e:
            self._handle_error("pair", str(e))
            return []

    def _hypothesize_step(self, pairs: List[Any]) -> List[Any]:
        """生成假设."""
        try:
            hypotheses = self._hypothesis_gen.generate(pairs)
            self.state.hypotheses_generated += len(hypotheses)
            self.state.hypotheses_active += len(hypotheses)
            for h in hypotheses:
                self._publish("hypothesis.generated", {
                    "id": h.hypothesis_id,
                    "type": h.hypothesis_type,
                    "statement": h.statement[:100],
                    "method": h.metadata.get("generation_method", "template"),
                })
                
                self.state.add_recent_hypothesis({
                    "hypothesis_id": h.hypothesis_id,
                    "statement": h.statement[:200],
                    "type": h.hypothesis_type,
                    "hypothesis_type": h.hypothesis_type,
                    "method": h.metadata.get("generation_method", "template"),
                    "confidence": h.confidence,
                    "novelty": h.novelty,
                    "keywords": list(h.keywords[:10]),
                    "paper_a_id": h.paper_a_id,
                    "paper_a_title": h.paper_a_title,
                    "paper_b_id": h.paper_b_id,
                    "paper_b_title": h.paper_b_title,
                })
            
            llm_stats = getattr(self._hypothesis_gen, "llm_stats", None)
            if llm_stats and llm_stats.get("llm_available"):
                self._log("INFO", (
                    f"LLM 假设生成: 成功={llm_stats['llm_success']}, "
                    f"降级={llm_stats['llm_fallback']}"
                ))
            return hypotheses
        except Exception as e:
            self._handle_error("hypothesize", str(e))
            return []

    def _trial_step(self, hypotheses: List[Any]) -> List[Any]:
        """孪生仿真试错验证.

        每个假设被放进独立科研仿真环境:
        - 算法算子做预测 (cross_domain, causal, uncertainty 等)
        - 检查多算法预测的自洽性
        - EWC 持续学习校准算子权重
        """
        try:
            
            if self._twin and self._twin.pe and self._crawler:
                
                all_papers = self._crawler.get_cached_papers()
                self._twin.pe.update_corpus_stats(all_papers, [])

            results = self._trial_engine.run_trials(hypotheses)
            for r in results:
                self.state.trials_total += 1
                if r.passed:
                    self.state.trials_passed += 1
                    self._publish("trial.passed", {
                        "hypothesis_id": r.hypothesis_id,
                        "score": round(r.score, 3),
                        "consistency": round(r.consistency, 3),
                        "algo_predictions": r.algo_predictions,
                    })
                else:
                    self.state.trials_failed += 1
                    self._publish("trial.failed", {
                        "hypothesis_id": r.hypothesis_id,
                        "score": round(r.score, 3),
                        "consistency": round(r.consistency, 3),
                        "reason": r.failure_reason,
                    })
            self.state.hypotheses_active = max(0, self.state.hypotheses_active - len(results))
            return results
        except Exception as e:
            self._handle_error("trial", str(e))
            return []

    def _knowledge_step(self, trial_results: List[Any], papers: List[Any]) -> Dict[str, Any]:
        """知识积累 (区分初筛通过 vs 认证发现)."""
        try:
            added = 0
            for r in trial_results:
                if r.passed:
                    self._knowledge_acc.add_validated(r)
                    self._knowledge_graph.add_hypothesis(r)
                    added += 1
                else:
                    self._knowledge_acc.add_failed(r)

            
            for p in papers:
                self._knowledge_graph.add_paper(p)

            self.state.knowledge_entries = self._knowledge_acc.entry_count
            self._publish("knowledge.updated", {"added": added})
            return {"added": added, "total": self.state.knowledge_entries}
        except Exception as e:
            self._handle_error("knowledge", str(e))
            return {"error": str(e)}

    def _verification_step(
        self,
        hypotheses: List[Any],
        trial_results: List[Any],
        force_pass_ids: Optional[set] = None,
    ) -> Dict[str, Any]:
        """验证复现: 对初筛通过的假设进行多轮复现验证.

        流程:
        1. 筛出试错通过的假设
        2. 每个假设重跑 5 次 (参数微扰) + 交叉验证
        3. 稳定性达标的被认证为"科学发现"
        4. force_pass_ids 中的假设跳过验证, 直接标为发现
        """
        try:
            if force_pass_ids is None:
                force_pass_ids = set()

            
            passed_hypos: Dict[str, Any] = {}
            for r in trial_results:
                if r.passed or r.hypothesis_id in force_pass_ids:
                    
                    for h in hypotheses:
                        if h.hypothesis_id == r.hypothesis_id:
                            passed_hypos[h.hypothesis_id] = (h, r)
                            break

            if not passed_hypos:
                return {
                    "verified": 0,
                    "discoveries": 0,
                    "discoveries_detail": [],
                    "agent_overrides": 0,
                    "agent_override_detail": [],
                }

            discoveries_count = 0
            verified_count = 0
            discoveries_detail: List[Dict[str, Any]] = []
            agent_override_count = 0
            agent_override_detail: List[Dict[str, Any]] = []

            for hypo_id, (hypo, original_result) in passed_hypos.items():
                
                if hypo_id in force_pass_ids:
                    verified_count += 1
                    agent_override_count += 1
                    
                    self._verification_engine.record_agent_override(
                        hypo,
                        original_result,
                        self._agent_feedback.get("trial", {}).get("reasoning", ""),
                    )
                    agent_override_detail.append({
                        "hypothesis_id": hypo_id,
                        "statement": hypo.statement[:120],
                        "stability": 0.0,
                        "reproduce_rate": 0.0,
                        "level": "Agent override candidate (未经复现验证)",
                        "passed_runs": "agent_override (not counted as formal discovery)",
                    })
                    self._publish("discovery.agent_override", {
                        "hypothesis_id": hypo_id,
                        "level": "Agent override candidate",
                        "note": "未经复现验证, 不计入正式候选发现",
                    })
                    continue

                report = self._verification_engine.verify(hypo, original_result)
                verified_count += 1

                if report.is_discovery:
                    discoveries_count += 1
                    discoveries_detail.append({
                        "hypothesis_id": hypo_id,
                        "statement": hypo.statement[:120],
                        "stability": report.stability_score,
                        "reproduce_rate": report.reproduce_rate,
                        "level": report.discovery_level,
                        "passed_runs": f"{report.passed_runs}/{report.total_runs}",
                    })
                    self._publish("discovery.certified", {
                        "hypothesis_id": hypo_id,
                        "stability": report.stability_score,
                        "level": report.discovery_level,
                    })
                else:
                    self._publish("verification.failed", {
                        "hypothesis_id": hypo_id,
                        "stability": report.stability_score,
                        "reproduce_rate": report.reproduce_rate,
                    })

            return {
                "verified": verified_count,
                "discoveries": discoveries_count,
                "discoveries_detail": discoveries_detail,
                "agent_overrides": agent_override_count,
                "agent_override_detail": agent_override_detail,
            }
        except Exception as e:
            self._handle_error("verification", str(e))
            return {"error": str(e)}

    def _direction_step(
        self,
        trial_results: List[Any],
        verification_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """方向更新 — 仅认证发现才大幅提升置信度."""
        try:
            
            certified_ids: set = set()
            if verification_result and "discoveries_detail" in verification_result:
                for d in verification_result["discoveries_detail"]:
                    certified_ids.add(d.get("hypothesis_id", ""))

            for r in trial_results:
                if r.passed:
                    if r.hypothesis_id in certified_ids:
                        
                        self._direction_tracker.update_from_trial(r)
                    else:
                        
                        self._direction_tracker.update_from_trial(r)
                else:
                    self._direction_tracker.record_failure(r)

            
            eliminated = self._direction_tracker.decay_and_eliminate()

            self.state.directions_tracked = self._direction_tracker.direction_count
            self.state.directions_promising = self._direction_tracker.promising_count

            if eliminated:
                for d in eliminated:
                    self._publish("direction.eliminated", {"direction": d})

            return {
                "tracked": self.state.directions_tracked,
                "promising": self.state.directions_promising,
                "eliminated": len(eliminated),
            }
        except Exception as e:
            self._handle_error("direction", str(e))
            return {"error": str(e)}

    

    def get_top_directions(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最有前景的研究方向."""
        if self._direction_tracker is None:
            return []
        return self._direction_tracker.top_directions(n)

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """获取知识库摘要."""
        if self._knowledge_acc is None:
            return {}
        return self._knowledge_acc.summary()

    def get_discoveries(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取已认证的候选发现 (仅正式候选发现, 不含 Agent override).

        本系统的"发现"是内部算法验证的候选发现,
        不是 DFT/实验/真实仿真验证的正式科学发现.
        """
        if self._verification_engine is None:
            return []
        return self._verification_engine.get_discoveries(n)

    def get_discovery_stats(self) -> Dict[str, Any]:
        """获取候选发现统计仪表盘.

        明确区分正式候选发现与 Agent override candidates.
        """
        if self._verification_engine is None:
            return {}
        return self._verification_engine.stats

    def get_agent_override_candidates(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取 Agent override candidates (未经复现验证).

        这些是 Agent (Claude/LLM) 通过 force_pass 标记的假设,
        没有经过三层验证, 仅作为人机协作的候选.
        """
        if self._verification_engine is None:
            return []
        return self._verification_engine.get_agent_override_candidates(n)

    def get_verification_detail(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """获取某个假设的完整验证报告 (含每次重跑详情)."""
        if self._verification_engine is None:
            return None
        return self._verification_engine.get_verification_detail(hypothesis_id)

    def get_recent_verifications(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的验证报告."""
        if self._verification_engine is None:
            return []
        return self._verification_engine.get_recent_verifications(n)

    def get_twin_status(self) -> Dict[str, Any]:
        """获取数字孪生虚拟实验环境状态."""
        if self._twin is None:
            return {"initialized": False}
        twin = self._twin
        return {
            "initialized": True,
            "twin_id": twin.twin_id,
            "domain": twin.domain,
            "version": twin.version,
            "algorithms": list(twin.algorithms.keys()),
            "pe_health": twin.pe.health() if twin.pe else 0.0,
            "vm_health": twin.vm.health() if twin.vm else 0.0,
            "dd_samples": twin.dd.total_samples if twin.dd else 0,
            "cl_mode": str(twin.cl.mode) if twin.cl else "N/A",
            "cl_sleep_cycles": twin.cl.sleep_cycles if twin.cl else 0,
            "cl_skill_count": twin.cl.skill_count if twin.cl else 0,
            "ewc_info": self._trial_engine.ewc_info if self._trial_engine else {},
        }

    def get_ewc_status(self) -> Dict[str, Any]:
        """获取 EWC 持续学习状态."""
        if self._trial_engine is None:
            return {}
        return self._trial_engine.ewc_info

    def get_recent_pairs(self, n: int = 10) -> List[Dict[str, Any]]:
        """获取最近的隐性配对."""
        if self._pairer is None:
            return []
        return self._pairer.recent_pairs(n)

    def get_csp_knowledge(
        self, composition: str = "", property_name: str = "", n: int = 20,
    ) -> List[Dict[str, Any]]:
        """获取 CSP (组分-结构-性能) 知识库.

        赛题: "组分、结构、工艺与性能之间的关联"
        每条记录包含完整溯源链 (source_paper_id).

        Args:
            composition: 按化学组分过滤 (模糊匹配), 空=不过滤
            property_name: 按性能名过滤, 空=不过滤
            n: 最多返回条数
        """
        if self._hypothesis_gen is None or not hasattr(self._hypothesis_gen, "_csp_knowledge"):
            return []
        results: List[Dict[str, Any]] = []
        for key, triple in self._hypothesis_gen._csp_knowledge.items():
            if composition and composition.lower() not in triple.composition.lower():
                continue
            if property_name and property_name.lower() not in triple.property_name.lower():
                continue
            
            if triple.structure == "unknown" and (
                triple.property_name == "general" or triple.property_value is None
            ):
                continue
            if triple.property_name == "general" and triple.property_value is None:
                continue
            results.append(triple.to_dict())
            if len(results) >= n:
                break
        return results

    def get_discovery_report(self, hypothesis_id: str = "", n: int = 5) -> List[Dict[str, Any]]:
        """生成结构化候选发现报告 (含文献溯源链).

        赛题: "文献溯源的完整性与可信度"
        每份报告包含:
        - 假设陈述 + 类型
        - 预测值 (可证伪性)
        - 来源论文 A/B (标题 + ID)
        - CSP 三元组 (组分-结构-性能)
        - 验证详情 (稳定性 + 复现率 + 每轮详情)
        - 溯源链: 从假设 → 配对 → 论文 → CSP 三元组

        注意: 报告中的"发现"是内部算法验证的候选发现,
        不是 DFT/实验/真实仿真验证的正式科学发现.

        Args:
            hypothesis_id: 指定假设 ID, 空=取 top-N 发现
            n: 最多返回报告数
        """
        if self._verification_engine is None:
            return []

        
        if hypothesis_id:
            discoveries = [self._verification_engine.get_verification_detail(hypothesis_id)]
            discoveries = [d for d in discoveries if d]
        else:
            discoveries = self._verification_engine.get_discoveries(n)

        reports: List[Dict[str, Any]] = []
        for disc in discoveries:
            hid = disc.get("hypothesis_id", "")
            report: Dict[str, Any] = {
                "hypothesis_id": hid,
                "statement": disc.get("statement", ""),
                "hypothesis_type": disc.get("hypothesis_type", ""),
                "discovery_level": disc.get("discovery_level", ""),
                "stability_score": disc.get("stability_score", 0.0),
                "reproduce_rate": disc.get("reproduce_rate", 0.0),
                "keywords": disc.get("keywords", []),
                "has_numerical_prediction": disc.get("metadata", {}).get(
                    "has_numerical_prediction", False,
                ),
            }

            
            detail = self._verification_engine.get_verification_detail(hid)
            if detail:
                report["verification"] = {
                    "total_runs": detail.get("total_runs", 0),
                    "passed_runs": detail.get("passed_runs", 0),
                    "cross_validation_passed": detail.get("cross_validation_passed", False),
                    "cross_validation_detail": detail.get("cross_validation_detail", {}),
                    "runs": detail.get("runs", []),
                }

            
            report["traceability"] = {
                "paper_a_id": disc.get("metadata", {}).get("paper_a_id", ""),
                "paper_a_title": disc.get("metadata", {}).get("paper_a_title", ""),
                "paper_b_id": disc.get("metadata", {}).get("paper_b_id", ""),
                "paper_b_title": disc.get("metadata", {}).get("paper_b_title", ""),
                "source_pair": disc.get("metadata", {}).get("source_pair", ""),
            }

            
            csp_triples = []
            if hasattr(self._hypothesis_gen, "_csp_knowledge"):
                for kw in disc.get("keywords", []):
                    for key, triple in self._hypothesis_gen._csp_knowledge.items():
                        if (kw.lower() in triple.composition.lower() or
                            kw.lower() in triple.structure.lower() or
                            kw.lower() in triple.property_name.lower()):
                            td = triple.to_dict()
                            if td not in csp_triples:
                                csp_triples.append(td)
                        if len(csp_triples) >= 10:
                            break
                    if len(csp_triples) >= 10:
                        break
            report["csp_triples"] = csp_triples

            reports.append(report)

        return reports

    def get_research_gap_report(self, n: int = 10) -> List[Dict[str, Any]]:
        """生成 Research Gap 识别报告.

        赛题基础任务: 识别文献中的研究空白 (Research Gap).
        每份报告包含:
        - gap_id: 空白标识
        - gap_description: 空白描述 (什么还没做)
        - evidence_papers: 支持该空白的论文 (标题 + ID + 摘要片段)
        - counter_evidence: 反证论文 (如有)
        - novelty_assessment: 新颖性判断 (是否已被其他论文覆盖)
        - suggested_hypotheses: 建议的假设方向
        - csp_context: 相关 CSP 知识 (如有)

        Args:
            n: 最多返回报告数
        """
        if not self._initialized:
            self._init_components()

        gaps: List[Dict[str, Any]] = []

        
        recent_hypos = self.state.sample_hypotheses()
        for h in recent_hypos:
            h_type = h.get("hypothesis_type") or h.get("type", "")
            if h_type not in ("gap", "gap_filling"):
                continue
            gap_entry: Dict[str, Any] = {
                "gap_id": f"gap_{h.get('hypothesis_id', '')[:12]}",
                "gap_description": h.get("statement", ""),
                "gap_type": h_type,
                "keywords": h.get("keywords", [])[:5],
                "evidence_papers": [],
                "counter_evidence": [],
                "novelty_assessment": "待评估",
                "suggested_hypotheses": [],
                "csp_context": [],
            }

            
            if h.get("paper_a_id"):
                gap_entry["evidence_papers"].append({
                    "paper_id": h.get("paper_a_id", ""),
                    "title": h.get("paper_a_title", ""),
                    "role": "source_A",
                })
            if h.get("paper_b_id"):
                gap_entry["evidence_papers"].append({
                    "paper_id": h.get("paper_b_id", ""),
                    "title": h.get("paper_b_title", ""),
                    "role": "source_B",
                })

            
            if self._knowledge_acc is not None:
                kws = h.get("keywords", [])
                search_kw = kws[0] if kws else ""
                existing = self._knowledge_acc.search_by_keyword(search_kw) if search_kw else []
                if existing:
                    gap_entry["novelty_assessment"] = (
                        f"知识库中已有 {len(existing)} 条相关记录, "
                        f"需确认该空白是否已被覆盖"
                    )
                else:
                    gap_entry["novelty_assessment"] = "知识库中无相关记录, 新颖性较高"

            
            if hasattr(self._hypothesis_gen, "_csp_knowledge"):
                for kw in h.get("keywords", [])[:3]:
                    for key, triple in self._hypothesis_gen._csp_knowledge.items():
                        if (kw.lower() in triple.composition.lower() or
                            kw.lower() in triple.structure.lower() or
                            kw.lower() in triple.property_name.lower()):
                            td = triple.to_dict()
                            if td not in gap_entry["csp_context"]:
                                gap_entry["csp_context"].append(td)
                            if len(gap_entry["csp_context"]) >= 5:
                                break
                    if len(gap_entry["csp_context"]) >= 5:
                        break

            gaps.append(gap_entry)
            if len(gaps) >= n:
                break

        
        if self._direction_tracker is not None and len(gaps) < n:
            for d in self._direction_tracker.top_directions(20):
                if d.get("failure_count", 0) >= 3 and d.get("support_count", 0) == 0:
                    gap_entry = {
                        "gap_id": f"gap_dir_{d.get('direction_id', '')[:8]}",
                        "gap_description": (
                            f"方向 \"{d.get('label', '')}\" 在多次试错中均失败 "
                            f"({d.get('failure_count', 0)} 次), "
                            f"可能存在尚未解决的技术空白"
                        ),
                        "gap_type": "trial_failure_cluster",
                        "keywords": d.get("keywords", [])[:5],
                        "evidence_papers": [],
                        "counter_evidence": [],
                        "novelty_assessment": "多次试错失败, 需深入分析失败原因",
                        "suggested_hypotheses": [d.get("statement", "")],
                        "csp_context": [],
                    }
                    gaps.append(gap_entry)
                    if len(gaps) >= n:
                        break

        return gaps

    def get_literature_review_report(self) -> Dict[str, Any]:
        """生成文献调研综述报告.

        赛题基础任务: 文献调研与综述.
        报告包含:
        - total_papers: 采集论文总数
        - papers_by_category: 按领域分类统计
        - top_keywords: 高频关键词
        - cross_domain_pairs: 跨领域配对统计
        - key_findings: 关键候选发现
        - research_gaps: 识别到的研究空白
        - csp_summary: CSP 知识库摘要
        """
        if not self._initialized:
            self._init_components()

        report: Dict[str, Any] = {
            "total_papers": self.state.papers_collected,
            "papers_raw_fetched": self.state.papers_raw_fetched,
            "papers_quality_rejected": self.state.papers_quality_rejected,
            "cycles": self.state.cycle_count,
        }

        
        if self._crawler is not None:
            papers = self._crawler.get_cached_papers()
            cat_counts: Dict[str, int] = {}
            kw_counts: Dict[str, int] = {}
            for p in papers:
                for cat in getattr(p, "categories", []):
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                for kw in getattr(p, "keywords", []):
                    kw_lower = kw.lower()
                    kw_counts[kw_lower] = kw_counts.get(kw_lower, 0) + 1
            report["papers_by_category"] = dict(
                sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            )
            report["top_keywords"] = dict(
                sorted(kw_counts.items(), key=lambda x: x[1], reverse=True)[:20]
            )

        
        if self._pairer is not None:
            recent_pairs = self._pairer.recent_pairs(100)
            cross_count = sum(1 for p in recent_pairs if p.get("cross_domain", False))
            report["cross_domain_pairs"] = {
                "total": len(recent_pairs),
                "cross_domain": cross_count,
                "same_domain": len(recent_pairs) - cross_count,
            }

        
        report["key_findings"] = self.get_discoveries(5)

        
        report["research_gaps"] = self.get_research_gap_report(5)

        
        if hasattr(self._hypothesis_gen, "_csp_knowledge"):
            csp_all = list(self._hypothesis_gen._csp_knowledge.values())
            report["csp_summary"] = {
                "total_triples": len(csp_all),
                "compositions": len(set(t.composition for t in csp_all)),
                "structures": len(set(t.structure for t in csp_all)),
                "properties": len(set(t.property_name for t in csp_all)),
            }

        
        report["discovery_stats"] = self.get_discovery_stats()

        return report

    def build_gap_evidence_report(self, n: int = 15) -> List[Dict[str, Any]]:
        """Build credible Research Gaps with evidence chains.

        Uses GapEvidenceBuilder to produce gaps with:
        - supporting_papers (>= 2 required)
        - counter_evidence (reverse search)
        - novelty_score, feasibility_score
        - evidence_chain, verification_plan
        """
        if not self._initialized:
            self._init_components()

        from ..knowledge.gap_builder import GapEvidenceBuilder

        hypotheses = self.state.sample_hypotheses()
        csp_triples = self.get_csp_knowledge(n=100)

        papers: List[Dict[str, Any]] = []
        if self._crawler is not None:
            for p in self._crawler.get_cached_papers():
                papers.append({
                    "paper_id": getattr(p, "paper_id", ""),
                    "title": getattr(p, "title", ""),
                    "abstract": getattr(p, "abstract", ""),
                    "keywords": getattr(p, "keywords", []),
                })

        failed_trials: List[Dict[str, Any]] = []
        if self._knowledge_acc is not None:
            for f in self._knowledge_acc.get_recent_failures(20):
                failed_trials.append({
                    "keywords": getattr(f, "keywords", []),
                    "statement": getattr(f, "statement", ""),
                })

        builder = GapEvidenceBuilder()
        gaps = builder.build_gaps(
            hypotheses=hypotheses,
            csp_triples=csp_triples,
            papers=papers,
            failed_trials=failed_trials,
            max_gaps=n,
        )
        return [g.to_dict() for g in gaps]

    def run_structure_property_search(
        self,
        iterations: int = 5,
        population_size: int = 20,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run evolutionary search for structure-property relationships.

        Route A: search/optimization algorithm + LLM deep fusion.
        Seeds population from hypotheses + CSP triples, then evolves.
        """
        if not self._initialized:
            self._init_components()

        from ..materials.search_engine import StructurePropertySearchEngine

        hypotheses = self.state.sample_hypotheses()
        csp_triples = self.get_csp_knowledge(n=100)

        papers: List[Dict[str, Any]] = []
        if self._crawler is not None:
            for p in self._crawler.get_cached_papers():
                papers.append({
                    "title": getattr(p, "title", ""),
                    "abstract": getattr(p, "abstract", ""),
                })

        engine = StructurePropertySearchEngine()
        result = engine.search(
            seed_hypotheses=hypotheses,
            csp_triples=csp_triples,
            papers=papers,
            iterations=iterations,
            population_size=population_size,
            top_k=top_k,
        )
        return result

    def validate_external_databases(
        self, candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Cross-validate candidates against Materials Project / OQMD / NOMAD.

        If no API keys are configured, returns "unavailable" for each DB.
        Supports manual import via import_manual_result().
        """
        from ..materials.db_validators import ExternalDBValidator

        validator = ExternalDBValidator()

        if candidates is None:
            # Use top discoveries as candidates
            candidates = self.get_discoveries(10)

        csp_triples = self.get_csp_knowledge(n=50)
        return validator.validate_candidates(candidates, csp_triples)

    def generate_competition_report(
        self,
        include_search: bool = True,
        include_external_validation: bool = False,
    ) -> str:
        """Generate a competition-ready Markdown report.

        Template:
        1. Research Problem
        2. Retrieval Strategy
        3. Literature Screening Results
        4. CSP Knowledge Base
        5. Research Gap List (with evidence chains)
        6. Route A Search Process
        7. Top Structure-Property Candidates
        8. Evidence Chains
        9. Falsifiable Verification Plan
        10. Limitations & Next Steps
        """
        if not self._initialized:
            self._init_components()

        from ..reporting import CompetitionReportGenerator

        search_results = None
        if include_search:
            try:
                search_results = self.run_structure_property_search(
                    iterations=3, population_size=15, top_k=5,
                )
            except Exception as e:
                _logger.warning("Search engine failed: %s", e)

        external_validation = None
        if include_external_validation:
            try:
                external_validation = self.validate_external_databases()
            except Exception as e:
                _logger.warning("External validation failed: %s", e)

        generator = CompetitionReportGenerator()
        return generator.generate(
            host_system=self,
            search_results=search_results,
            external_validation=external_validation,
        )

    def snapshot(self) -> HostSnapshot:
        """生成系统快照."""
        return HostSnapshot(
            state=self.state.as_dict(),
            top_directions=self.get_top_directions(5),
            recent_events=[
                {"topic": e.topic, "payload": e.payload}
                for e in self.event_bus.history()[-20:]
            ],
        )

    def status(self) -> str:
        """打印系统状态."""
        return self.state.summary()

    

    def _publish(self, topic: str, payload: Dict[str, Any]) -> None:
        self.event_bus.publish(HostEvent(
            topic=topic, payload=payload, source=self.host_id,
        ))

    def _log(self, event: str, data: Dict[str, Any]) -> None:
        self._logs.append({"time": time.time(), "event": event, "data": data})
        if len(self._logs) > 2000:
            self._logs = self._logs[-2000:]

    def _handle_error(self, stage: str, error: str) -> None:
        self.state.errors += 1
        self.state.last_error = f"[{stage}] {error}"
        self._publish("loop.error", {"stage": stage, "error": error})
        self._log("error", {"stage": stage, "error": error})

    def logs(self) -> List[Dict[str, Any]]:
        return list(self._logs)

    def __repr__(self) -> str:
        return (
            f"<HostSystem id={self.host_id} version={self.version} "
            f"cycles={self.state.cycle_count} running={self._running}>"
        )
