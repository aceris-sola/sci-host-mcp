"""科学方向探索宿主系统 (Scientific Direction Discovery Host System).

面向材料科学文献调研与可证伪假设生成的持续运行系统。

核心特性:
    1. 隐性配对 — 持续采集论文/材料，发现跨领域隐性关联
    2. 试错与重复 — 假设生成 → 验证试错 → 知识积累，循环往复
    3. 7×24 小时运行 — 无梦境/睡眠机制，连续不间断探索

主入口:
    from sci_host import HostSystem, HostConfig
    host = HostSystem(HostConfig())
    host.run()  # 7×24 连续运行

MCP Server (供 LLM 调用):
    python -m sci_host.mcp_server
    python -m sci_host.mcp_server --http 8080
"""
from __future__ import annotations

from .config import HostConfig
from .core.host_system import HostSystem
from .core.event_bus import HostEvent, HostEventBus
from .core.state import HostState, HostSnapshot
from .crawler.paper_crawler import PaperCrawler, Paper, PaperSource
from .pairing.implicit_pairer import ImplicitPairer, PaperPair
from .pairing.embedding import TextEmbedder
from .hypothesis.generator import HypothesisGenerator, Hypothesis
from .hypothesis.evaluator import HypothesisEvaluator
from .trial.engine import TrialEngine, TrialResult
from .trial.knowledge_graph import ScienceKnowledgeGraph
from .loop.continuous_loop import ContinuousLoop
from .knowledge.accumulator import KnowledgeAccumulator
from .knowledge.direction_tracker import DirectionTracker, ResearchDirection

__version__ = "1.0.0"

__all__ = [
    "HostSystem",
    "HostConfig",
    "HostEvent",
    "HostEventBus",
    "HostState",
    "HostSnapshot",
    "PaperCrawler",
    "Paper",
    "PaperSource",
    "ImplicitPairer",
    "PaperPair",
    "TextEmbedder",
    "HypothesisGenerator",
    "Hypothesis",
    "HypothesisEvaluator",
    "TrialEngine",
    "TrialResult",
    "ScienceKnowledgeGraph",
    "ContinuousLoop",
    "KnowledgeAccumulator",
    "DirectionTracker",
    "ResearchDirection",
    "__version__",
]
