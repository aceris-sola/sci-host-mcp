"""宿主系统配置."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CrawlerConfig:
    """论文采集器配置."""
    
    crawl_interval: float = 60.0
    
    batch_size: int = 20
    
    categories: List[str] = field(default_factory=lambda: [
        "cs.AI", "cs.LG", "cs.RO", "cs.CL",
        "physics", "q-bio", "stat.ML",
    ])
    
    max_cache: int = 5000
    
    offline_mode: bool = True
    
    offline_corpus_path: Optional[str] = None
    
    keywords: List[str] = field(default_factory=list)
    
    source: str = "arxiv"


@dataclass
class ResearchQualityConfig:
    """科研信号质量闸门配置.

    默认关闭以保持通用/离线语料的兼容性。专项检索可开启后，要求论文
    同时具备目标方向命中、技术机制和可观测证据，减少元数据噪声流入后续阶段。
    """
    enabled: bool = False
    focus_terms: List[str] = field(default_factory=list)
    min_score: float = 0.45
    min_mechanism_hits: int = 1
    min_evidence_hits: int = 1
    require_focus: bool = True
    min_abstract_chars: int = 80
    min_falsifiability_score: float = 0.45
    require_technical_pair_bridge: bool = True


@dataclass
class PairingConfig:
    """隐性配对配置."""
    
    embed_dim: int = 128
    
    similarity_threshold: float = 0.10
    
    max_pairs_per_round: int = 50
    
    cross_domain: bool = True
    
    cross_domain_threshold: float = 0.05
    
    max_features: int = 2000
    
    ngram_range: tuple = (1, 2)


@dataclass
class HypothesisConfig:
    """假设生成配置."""
    
    max_per_round: int = 10
    
    min_pair_similarity: float = 0.05
    
    type_weights: Dict[str, float] = field(default_factory=lambda: {
        "analogy": 0.3,       
        "contradiction": 0.2,  
        "gap": 0.3,           
        "combination": 0.2,    
    })


@dataclass
class TrialConfig:
    """试错引擎配置."""
    
    max_trials_per_round: int = 20
    
    strategies: List[str] = field(default_factory=lambda: [
        "consistency_check",    
        "literature_cross_ref", 
        "novelty_score",        
        "feasibility_score",    
    ])
    
    pass_threshold: float = 0.5
    
    max_retries: int = 3


@dataclass
class KnowledgeConfig:
    """知识积累配置."""
    
    max_directions: int = 100
    
    confidence_decay: float = 0.995
    
    confidence_boost: float = 0.1
    
    elimination_threshold: float = 0.05
    
    max_graph_nodes: int = 10000


@dataclass
class LoopConfig:
    """连续循环配置."""
    
    loop_interval: float = 30.0
    
    max_iterations: int = 0
    
    sample_per_round: int = 30
    
    auto_save: bool = True
    
    save_path: Optional[str] = None
    
    log_level: str = "INFO"


@dataclass
class LLMExtractionConfig:
    """LLM 摘要级 CSP 抽取配置."""
    
    enabled: bool = False
    
    api_base: str = "https://api.stepfun.com/step_plan/v1"
    # API Key
    api_key: str = ""
    
    model: str = "step-router-v1"
    
    max_tokens: int = 2000
    
    temperature: float = 0.1
    
    timeout: int = 120
    
    max_retries: int = 2


@dataclass
class HostConfig:
    """宿主系统总配置."""
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    hypothesis: HypothesisConfig = field(default_factory=HypothesisConfig)
    trial: TrialConfig = field(default_factory=TrialConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    quality: ResearchQualityConfig = field(default_factory=ResearchQualityConfig)
    
    llm_extraction: LLMExtractionConfig = field(default_factory=LLMExtractionConfig)
    
    host_id: str = "sci-host-001"
    
    research_seeds: List[str] = field(default_factory=lambda: [
        "robotics", "machine learning", "digital twin",
        "continual learning", "transfer learning",
    ])
    
    materials_mode: bool = False

    @classmethod
    def quick(cls, offline: bool = True, loop_interval: float = 5.0) -> "HostConfig":
        """快速配置：适合演示和测试."""
        cfg = cls()
        cfg.crawler.offline_mode = offline
        cfg.loop.loop_interval = loop_interval
        cfg.loop.max_iterations = 5  
        cfg.crawler.batch_size = 20
        cfg.pairing.max_pairs_per_round = 20
        cfg.hypothesis.max_per_round = 8
        cfg.trial.max_trials_per_round = 15
        return cfg

    @classmethod
    def production(cls) -> "HostConfig":
        """生产配置：7×24 连续运行."""
        cfg = cls()
        cfg.crawler.offline_mode = False
        cfg.loop.max_iterations = 0  
        cfg.loop.loop_interval = 60.0
        cfg.crawler.batch_size = 50
        cfg.pairing.max_pairs_per_round = 100
        cfg.hypothesis.max_per_round = 20
        cfg.trial.max_trials_per_round = 40
        return cfg

    @classmethod
    def materials(cls, offline: bool = True, loop_interval: float = 5.0) -> "HostConfig":
        """材料科学配置 (赛题方向三).

        面向「材料科学文献驱动的科学发现智能体」赛题:
        - 使用材料科学内置语料 (钙钛矿/锂电池/热电/铁电/合金等)
        - 开启 CSP 抽取和材料科学假设模板
        - arXiv 分类聚焦 cond-mat.mtrl-sci
        - 研究种子聚焦材料科学关键词
        """
        from .materials import MATERIAL_SEEDS, MATERIAL_CATEGORIES
        cfg = cls()
        cfg.materials_mode = True
        cfg.crawler.offline_mode = offline
        cfg.crawler.categories = list(MATERIAL_CATEGORIES)
        cfg.crawler.keywords = list(MATERIAL_SEEDS)
        cfg.research_seeds = list(MATERIAL_SEEDS)
        cfg.loop.loop_interval = loop_interval
        cfg.loop.max_iterations = 5
        cfg.crawler.batch_size = 20
        cfg.pairing.max_pairs_per_round = 20
        cfg.hypothesis.max_per_round = 8
        cfg.trial.max_trials_per_round = 15
        return cfg
    @classmethod
    def sciverse(cls, loop_interval: float = 5.0) -> "HostConfig":
        """SCIVERSE 数据源配置.

        使用 SCIVERSE agentic-search API 采集 4.65 亿学术元数据.
        需要环境变量 SCIVERSE_API_TOKEN.
        """
        from .materials import MATERIAL_SEEDS, MATERIAL_CATEGORIES
        cfg = cls()
        cfg.materials_mode = True
        cfg.crawler.offline_mode = False
        cfg.crawler.source = "sciverse"
        cfg.crawler.categories = list(MATERIAL_CATEGORIES)
        cfg.crawler.keywords = list(MATERIAL_SEEDS)
        cfg.research_seeds = list(MATERIAL_SEEDS)
        cfg.loop.loop_interval = loop_interval
        cfg.loop.max_iterations = 5
        cfg.crawler.batch_size = 20
        cfg.pairing.max_pairs_per_round = 20
        cfg.hypothesis.max_per_round = 8
        cfg.trial.max_trials_per_round = 15
        return cfg

    def with_llm(
        self,
        api_key: str,
        api_base: str = "https://api.stepfun.com/step_plan/v1",
        model: str = "step-router-v1",
    ) -> "HostConfig":
        """启用 LLM 摘要级 CSP 抽取.

        Args:
            api_key: LLM API Key
            api_base: API 地址
            model: 模型名

        Returns:
            self (链式调用)
        """
        self.llm_extraction.enabled = True
        self.llm_extraction.api_key = api_key
        self.llm_extraction.api_base = api_base
        self.llm_extraction.model = model
        return self
