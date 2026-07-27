"""科学方向探索宿主系统 MCP Server — 通过 MCP 协议暴露科学探索能力给外部 LLM。

使用方式:
    python -m sci_host.mcp_server                      # stdio 模式 (本地)
    python -m sci_host.mcp_server --http 8080           # HTTP 模式 (远程)
    python sci_host/mcp_server.py                       # 直接运行

MCP 客户端配置 (Claude Desktop / Cursor / VS Code / 任意支持 MCP 的软件):
    {
      "mcpServers": {
        "sci-host": {
          "command": "python",
          "args": ["/path/to/sci_host/mcp_server.py"]
        }
      }
    }

暴露的 MCP 工具 (LLM 可调用):
    1. sci_create_host            — 创建科学探索宿主实例 (内嵌科研仿真运行时)
    2. sci_create_materials_host  — 一键创建材料科学发现宿主 (赛题方向三)
    3. sci_step                   — 执行一轮完整探索循环 (孪生仿真试错)
    4. sci_run_cycles             — 连续运行 N 轮探索
    5. sci_get_status             — 获取系统运行状态
    6. sci_get_directions         — 获取最有前景的研究方向
    7. sci_get_knowledge          — 获取知识库摘要
    8. sci_search_knowledge       — 按关键词搜索已验证知识
    9. sci_get_top_validated      — 获取评分最高的已验证假设
    10. sci_get_failures          — 获取最近的试错失败 (学习教训)
    11. sci_get_pairs             — 获取最近的隐性配对
    12. sci_get_snapshot          — 获取完整系统快照
    13. sci_get_graph             — 获取科学知识图谱摘要
    14. sci_add_paper             — 手动添加论文供分析
    15. sci_get_twin_status       — 获取数字孪生虚拟环境状态 (算法算子/EWC)
    16. sci_get_discoveries       — 获取已认证的科学发现 (按稳定性排序)
    17. sci_get_discovery_stats   — 获取发现统计仪表盘 (发现率/金/银/铜级)
    18. sci_get_verification      — 获取某个假设的完整验证报告 (含每次重跑详情)
    19. sci_get_csp_knowledge     — 查询 CSP(组分-结构-性能) 知识库 (材料科学专属)
    20. sci_get_discovery_report  — 获取含文献溯源链的发现报告 (材料科学专属)
    21. sci_get_info              — 查询系统信息
    22. sci_list_hosts            — 列出所有活跃宿主实例
    23. sci_remove_host           — 移除宿主实例
    ── Agent CSP 模式 (LLM agent 自主抽取, 不需 API key) ──
    24. sci_get_papers_for_csp    — 获取待抽取 CSP 的论文列表 (agent 读取摘要)
    25. sci_submit_csp             — agent 提交抽取的 CSP 三元组到知识库
    26. sci_step_agent             — 用 agent CSP 三元组运行探索循环
    ── 流式推理模式 (Claude 在孪生仿真每个阶段实时推理) ──
    27. sci_stream_crawl           — 阶段1: 采集论文 → Claude 评估
    28. sci_stream_pair            — 阶段2: 隐性配对 → Claude 评估
    29. sci_stream_hypothesize     — 阶段3: 生成假设 → Claude 评估
    30. sci_stream_trial           — 阶段4: 孪生仿真 → Claude 实时看算子预测
    31. sci_stream_verify          — 阶段5: 验证复现 → Claude 看认证发现
    32. sci_stream_feedback        — Claude 在任意阶段提交推理/覆盖判断
    ── 比赛工具 (P0+P1+P2) ──
    33. sci_get_gap_evidence       — 获取带证据链的 Research Gap 报告 (>=2 篇支持文献)
    34. sci_run_structure_property_search — 路线 A: 进化搜索构效关系候选
    35. sci_generate_competition_report — 生成参赛 Markdown 报告 (10 节固定模板)
    36. sci_validate_external_databases — 外部数据库交叉验证 (MP/OQMD/NOMAD)
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, List

_logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "FastMCP 未安装，请运行: pip install mcp[cli]"
    )


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sci_host import HostSystem, HostConfig, __version__
from sci_host.config import (
    CrawlerConfig, PairingConfig, HypothesisConfig,
    TrialConfig, KnowledgeConfig, LoopConfig,
)



class HostManager:
    """线程安全的宿主实例管理器。"""

    def __init__(self) -> None:
        self._hosts: Dict[str, HostSystem] = {}
        self._lock = threading.Lock()

    def put(self, host_id: str, host: HostSystem) -> None:
        with self._lock:
            self._hosts[host_id] = host

    def get(self, host_id: str) -> HostSystem:
        with self._lock:
            if host_id not in self._hosts:
                raise ValueError(
                    f"宿主实例 '{host_id}' 不存在。可用实例: {list(self._hosts.keys())}"
                )
            return self._hosts[host_id]

    def list_ids(self) -> List[str]:
        with self._lock:
            return list(self._hosts.keys())

    def remove(self, host_id: str) -> bool:
        with self._lock:
            host = self._hosts.pop(host_id, None)
            if host is not None:
                try:
                    host.stop()
                except Exception:
                    pass
                return True
            return False



_manager = HostManager()


mcp = FastMCP("sci-host")




class CreateHostInput(BaseModel):
    """创建宿主实例参数。"""
    host_id: str = Field(
        default="sci-host-001",
        description="宿主实例唯一 ID，用于后续引用",
    )
    offline: bool = Field(
        default=True,
        description="是否使用离线模式（内置跨领域语料，不访问网络）。False 则通过 arXiv API 采集真实论文",
    )
    batch_size: int = Field(
        default=20, ge=1, le=200,
        description="每轮采集论文数量",
    )
    max_pairs: int = Field(
        default=50, ge=1, le=500,
        description="每轮最大隐性配对数",
    )
    max_hypotheses: int = Field(
        default=10, ge=1, le=100,
        description="每轮最大假设生成数",
    )
    max_trials: int = Field(
        default=20, ge=1, le=200,
        description="每轮最大试错验证数",
    )
    pass_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="试错通过阈值 (0-1)，越高越严格",
    )
    similarity_threshold: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="同领域配对相似度阈值，越低越敏感",
    )
    cross_domain_threshold: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="跨领域配对相似度阈值，低于同领域阈值以鼓励发现远距离关联",
    )
    research_seeds: List[str] = Field(
        default_factory=lambda: ["robotics", "machine learning", "digital twin",
                                  "continual learning", "transfer learning"],
        description="研究兴趣种子，初始关注方向",
    )
    categories: List[str] = Field(
        default_factory=list,
        description="arXiv 分类列表，限定采集领域，如 ['cond-mat.mtrl-sci','physics.app-ph','cs.LG']；为空则使用全局默认分类",
    )
    source: str = Field(
        default="arxiv",
        description="在线数据源: 'arxiv' / 'openalex' / 'sciverse' (SCIVERSE 4.65 亿学术元数据语义检索, 需 SCIVERSE_API_TOKEN)",
    )
    materials_mode: bool = Field(
        default=False,
        description="是否启用材料科学模式 (赛题方向三)。启用后: 使用材料科学语料 (钙钛矿/锂电池/热电/铁电/合金等)、CSP知识抽取、材料科学假设模板、cond-mat.mtrl-sci 分类",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="在线检索关键词，用于聚焦特定子领域。OpenAlex 模式作为 search 细化 (如工程材料: ['fatigue','composite','steel','concrete','tensile'])，arXiv 模式作为关键词过滤；为空则使用全局/材料默认",
    )
    quality_gate: bool = Field(
        default=False,
        description="是否启用科研信号质量闸门。启用后过滤元数据噪声，并要求论文有技术机制和可观测证据；专项检索建议开启",
    )
    quality_focus_terms: List[str] = Field(
        default_factory=list,
        description="质量闸门使用的目标术语；为空时复用 keywords 或 research_seeds",
    )
    quality_min_score: float = Field(
        default=0.45, ge=0.0, le=1.0,
        description="论文质量最低分数，越高越严格",
    )
    llm_api_key: str = Field(
        default="",
        description="LLM API Key，用于摘要级 CSP 抽取。提供后自动启用 LLM 抽取 (比正则准确率高)，不提供则用正则降级",
    )
    llm_api_base: str = Field(
        default="https://api.stepfun.com/step_plan/v1",
        description="LLM API 地址",
    )
    llm_model: str = Field(
        default="step-router-v1",
        description="LLM 模型名",
    )


class HostIdInput(BaseModel):
    """宿主实例 ID 参数。"""
    host_id: str = Field(description="宿主实例 ID")


class RunCyclesInput(BaseModel):
    """连续运行参数。"""
    host_id: str = Field(description="宿主实例 ID")
    cycles: int = Field(
        default=5, ge=1, le=1000,
        description="运行的探索循环次数",
    )
    interval: float = Field(
        default=0.0, ge=0.0, le=3600.0,
        description="每轮之间的间隔（秒），0 表示不等待",
    )


class GetDirectionsInput(BaseModel):
    """获取研究方向参数。"""
    host_id: str = Field(description="宿主实例 ID")
    top_n: int = Field(
        default=10, ge=1, le=100,
        description="返回的前 N 个方向",
    )


class SearchKnowledgeInput(BaseModel):
    """知识搜索参数。"""
    host_id: str = Field(description="宿主实例 ID")
    keyword: str = Field(description="搜索关键词，如 'learning'、'quantum'、'digital twin'")


class GetPairsInput(BaseModel):
    """获取配对参数。"""
    host_id: str = Field(description="宿主实例 ID")
    n: int = Field(
        default=10, ge=1, le=100,
        description="返回的最近配对数量",
    )


class GetTopValidatedInput(BaseModel):
    """获取已验证知识参数。"""
    host_id: str = Field(description="宿主实例 ID")
    n: int = Field(
        default=10, ge=1, le=100,
        description="返回的前 N 条已验证知识",
    )


class GetFailuresInput(BaseModel):
    """获取失败知识参数。"""
    host_id: str = Field(description="宿主实例 ID")
    n: int = Field(
        default=10, ge=1, le=100,
        description="返回的最近失败数量",
    )


class AddPaperInput(BaseModel):
    """手动添加论文参数。"""
    host_id: str = Field(description="宿主实例 ID")
    title: str = Field(description="论文标题")
    abstract: str = Field(description="论文摘要")
    keywords: List[str] = Field(
        default_factory=list,
        description="论文关键词列表",
    )
    categories: List[str] = Field(
        default_factory=list,
        description="论文分类列表，如 ['cs.AI', 'cs.LG']",
    )
    authors: List[str] = Field(
        default_factory=list,
        description="作者列表",
    )
    year: int = Field(default=2025, description="发表年份")




@mcp.tool(
    description=(
        "创建一个科学方向探索宿主实例。宿主会持续采集论文、发现隐性配对、"
        "生成假设、试错验证、积累知识、追踪研究方向。"
        "创建后可用 sci_step 执行单轮探索，或用 sci_run_cycles 连续运行。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_create_host(input: CreateHostInput) -> Dict[str, Any]:
    """创建科学探索宿主实例。

    宿主系统会 7×24 持续运行，不断:
      1. 采集论文/材料
      2. 发现跨领域隐性配对
      3. 生成可验证的科学假设
      4. 通过多策略试错验证假设
      5. 积累知识并追踪有前景的研究方向
    """
    config = HostConfig()
    config.host_id = input.host_id
    config.crawler.offline_mode = input.offline
    config.crawler.batch_size = input.batch_size
    config.pairing.max_pairs_per_round = input.max_pairs
    config.pairing.similarity_threshold = input.similarity_threshold
    config.pairing.cross_domain_threshold = input.cross_domain_threshold
    config.hypothesis.max_per_round = input.max_hypotheses
    config.trial.max_trials_per_round = input.max_trials
    config.trial.pass_threshold = input.pass_threshold
    config.research_seeds = input.research_seeds
    if input.categories:
        config.crawler.categories = input.categories
    config.crawler.source = input.source
    config.crawler.keywords = input.keywords
    config.materials_mode = input.materials_mode

    
    if input.llm_api_key:
        config.llm_extraction.enabled = True
        config.llm_extraction.api_key = input.llm_api_key
        config.llm_extraction.api_base = input.llm_api_base
        config.llm_extraction.model = input.llm_model

    
    if input.materials_mode:
        from sci_host.materials import MATERIAL_SEEDS, MATERIAL_CATEGORIES
        if not input.categories:
            config.crawler.categories = list(MATERIAL_CATEGORIES)
        if not input.research_seeds or input.research_seeds == ["robotics", "machine learning", "digital twin", "continual learning", "transfer learning"]:
            config.research_seeds = list(MATERIAL_SEEDS)
        if not input.keywords:
            config.crawler.keywords = list(MATERIAL_SEEDS)

    
    if input.quality_gate:
        config.quality.enabled = True
        config.quality.focus_terms = list(
            input.quality_focus_terms or input.keywords or config.research_seeds
        )
        config.quality.min_score = input.quality_min_score

    host = HostSystem(config)
    host.start()
    _manager.put(input.host_id, host)

    return {
        "status": "success",
        "host_id": input.host_id,
        "version": host.version,
        "offline_mode": input.offline,
        "config": {
            "batch_size": input.batch_size,
            "max_pairs": input.max_pairs,
            "max_hypotheses": input.max_hypotheses,
            "max_trials": input.max_trials,
            "pass_threshold": input.pass_threshold,
            "similarity_threshold": input.similarity_threshold,
            "cross_domain_threshold": input.cross_domain_threshold,
            "research_seeds": input.research_seeds,
            "quality": {
                "enabled": config.quality.enabled,
                "focus_terms": config.quality.focus_terms,
                "min_score": config.quality.min_score,
            },
        },
        "message": f"宿主 '{input.host_id}' 创建成功，可调用 sci_step 开始探索",
        "materials_mode": input.materials_mode,
    }


@mcp.tool(
    description=(
        "一键创建材料科学文献驱动的科学发现宿主 (赛题方向三)。"
        "自动启用: 材料科学语料 (钙钛矿/锂电池/热电/铁电/合金/超导/ML4Materials)、"
        "CSP(组分-结构-性能)知识抽取、材料科学假设模板 (构效关系/组分迁移/隐藏关联)、"
        "cond-mat.mtrl-sci 分类。"
        "创建后调用 sci_step 开始探索，调用 sci_get_discovery_report 获取含溯源链的发现报告。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_create_materials_host(input: CreateHostInput) -> Dict[str, Any]:
    """一键创建材料科学发现宿主.

    面向赛题「材料科学文献驱动的科学发现智能体」:
    - 内置 22 篇材料科学论文语料 (perovskite/LiFePO4/Bi2Te3/BaTiO3/HEA等)
    - CSP 抽取: 从论文中提取 组分-结构-性能 三元组
    - 可证伪假设: 生成含数值预测的构效关系/隐藏关联/空白填补假设
    - 文献溯源: 每个发现可追溯到源论文
    """
    input.materials_mode = True
    return sci_create_host(input)

@mcp.tool(
    description=(
        "一键创建 SCIVERSE 数据源材料科学发现宿主。"
        "使用 SCIVERSE agentic-search API 采集 4.65 亿学术元数据，"
        "自动启用材料科学模式 (CSP 抽取 + 构效关系假设 + 文献溯源)。"
        "需要环境变量 SCIVERSE_API_TOKEN。"
        "创建后调用 sci_step 开始探索。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_create_sciverse_host(input: CreateHostInput) -> Dict[str, Any]:
    input.materials_mode = True
    input.source = "sciverse"
    input.offline = False
    if not input.keywords:
        from sci_host.materials import MATERIAL_SEEDS
        input.keywords = list(MATERIAL_SEEDS)
    if not input.categories:
        from sci_host.materials import MATERIAL_CATEGORIES
        input.categories = list(MATERIAL_CATEGORIES)
    if not input.research_seeds or input.research_seeds == ["robotics", "machine learning", "digital twin", "continual learning", "transfer learning"]:
        from sci_host.materials import MATERIAL_SEEDS
        input.research_seeds = list(MATERIAL_SEEDS)
    return sci_create_host(input)



@mcp.tool(
    description=(
        "执行一轮完整的科学探索循环: 采集论文 → 隐性配对 → 生成假设 → "
        "试错验证 → 知识积累 → 方向更新。"
        "每轮约产生 10-50 篇论文、若干隐性配对和假设。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_step(input: HostIdInput) -> Dict[str, Any]:
    """执行单轮探索循环。

    一轮循环包含:
      1. 采集论文 (Crawler)
      2. 隐性配对 (Pairer) — 发现跨领域关联
      3. 生成假设 (Hypothesis Generator)
      4. 试错验证 (Trial Engine) — 多策略验证
      5. 知识积累 (Knowledge Accumulator)
      6. 方向更新 (Direction Tracker)
    """
    host = _manager.get(input.host_id)
    result = host.step()

    
    csp_triples = host.get_csp_knowledge(n=5)

    
    recent_verifications = host.get_recent_verifications(3)

    
    discovery_stats = host.get_discovery_stats()

    return {
        "host_id": input.host_id,
        "cycle": result["cycle"],
        "papers_crawled": result["papers_crawled"],
        "pairs_found": result["pairs_found"],
        "hypotheses_generated": result["hypotheses_generated"],
        "trials_passed": result["trials_passed"],
        "trials_failed": result["trials_failed"],
        "elapsed_ms": result["elapsed_ms"],
        "directions": result.get("directions_updated", {}),
        
        "csp_triples": csp_triples,
        
        "recent_verifications": recent_verifications,
        
        "discovery_stats": discovery_stats,
    }


@mcp.tool(
    description=(
        "连续运行 N 轮科学探索循环。适合快速积累知识和发现研究方向。"
        "每轮之间可选等待 interval 秒。运行后会返回累积统计。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_run_cycles(input: RunCyclesInput) -> Dict[str, Any]:
    """连续运行多轮探索循环。"""
    import time

    host = _manager.get(input.host_id)
    results_per_cycle: List[Dict[str, Any]] = []

    for i in range(input.cycles):
        result = host.step()
        results_per_cycle.append({
            "cycle": result["cycle"],
            "papers": result["papers_crawled"],
            "pairs": result["pairs_found"],
            "hypotheses": result["hypotheses_generated"],
            "passed": result["trials_passed"],
            "failed": result["trials_failed"],
            "ms": result["elapsed_ms"],
            "fallback": result.get("offline_fallback_crawls", 0),
            "rejected": result.get("papers_quality_rejected", 0),
            "sample_hypotheses": result.get("sample_hypotheses", []),
        })
        if input.interval > 0 and i < input.cycles - 1:
            time.sleep(input.interval)

    
    directions = host.get_top_directions(5)
    knowledge = host.get_knowledge_summary()

    return {
        "host_id": input.host_id,
        "cycles_run": input.cycles,
        "per_cycle_summary": results_per_cycle,
        "final_status": host.state.as_dict(),
        "top_directions": directions,
        "knowledge_summary": {
            "validated": knowledge.get("validated", 0),
            "failed": knowledge.get("failed", 0),
            "avg_score": knowledge.get("avg_score", 0),
            "avg_novelty": knowledge.get("avg_novelty", 0),
            "top_keywords": knowledge.get("top_keywords", []),
        },
        "sample_hypotheses": host.state.sample_hypotheses(),
    }


@mcp.tool(
    description=(
        "获取宿主系统的完整运行状态，包括运行时间、循环次数、"
        "论文采集数、配对数、假设数、试错统计、知识积累、方向追踪等。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_status(input: HostIdInput) -> Dict[str, Any]:
    """查询宿主系统运行状态。"""
    host = _manager.get(input.host_id)
    return {
        "host_id": input.host_id,
        "status": host.state.as_dict(),
        "readable": host.status(),
    }


@mcp.tool(
    description=(
        "获取最有前景的研究方向。每个方向包含标签、关键词、置信度、"
        "支持证据数、假设类型等。按置信度排序。"
        "这是系统的核心输出——代表系统发现的可能值得深入研究的科学方向。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_directions(input: GetDirectionsInput) -> Dict[str, Any]:
    """获取最有前景的研究方向。"""
    host = _manager.get(input.host_id)
    directions = host.get_top_directions(input.top_n)
    return {
        "host_id": input.host_id,
        "total_directions": host.state.directions_tracked,
        "promising_directions": host.state.directions_promising,
        "top_directions": directions,
    }


@mcp.tool(
    description=(
        "获取知识库摘要，包括已验证知识数、失败知识数、假设类型分布、"
        "平均评分、平均新颖性、高频关键词等。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_knowledge(input: HostIdInput) -> Dict[str, Any]:
    """获取知识库摘要。"""
    host = _manager.get(input.host_id)
    summary = host.get_knowledge_summary()

    
    learnings = {}
    if host._knowledge_acc is not None:
        learnings = host._knowledge_acc.get_learnings()

    return {
        "host_id": input.host_id,
        "knowledge_summary": summary,
        "learnings_from_failures": learnings,
    }


@mcp.tool(
    description=(
        "按关键词搜索已验证的知识条目。返回包含该关键词的所有已验证假设。"
        "用于查找特定主题的科学发现。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_search_knowledge(input: SearchKnowledgeInput) -> Dict[str, Any]:
    """按关键词搜索已验证知识。"""
    host = _manager.get(input.host_id)
    if host._knowledge_acc is None:
        return {"host_id": input.host_id, "results": [], "message": "知识库尚未初始化"}

    entries = host._knowledge_acc.search_by_keyword(input.keyword)
    results = [
        {
            "entry_id": e.entry_id,
            "statement": e.statement,
            "type": e.hypothesis_type,
            "score": e.score,
            "novelty": e.novelty,
            "keywords": e.keywords,
            "source_papers": e.source_papers,
        }
        for e in entries
    ]
    return {
        "host_id": input.host_id,
        "keyword": input.keyword,
        "found": len(results),
        "results": results,
    }


@mcp.tool(
    description=(
        "获取评分最高的已验证假设（科学发现）。按综合评分排序。"
        "每条包含假设陈述、类型、评分、新颖性、关键词和来源论文。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_top_validated(input: GetTopValidatedInput) -> Dict[str, Any]:
    """获取评分最高的已验证知识。"""
    host = _manager.get(input.host_id)
    if host._knowledge_acc is None:
        return {"host_id": input.host_id, "results": []}

    entries = host._knowledge_acc.get_top_validated(input.n)
    results = [
        {
            "entry_id": e.entry_id,
            "statement": e.statement,
            "type": e.hypothesis_type,
            "score": round(e.score, 3),
            "novelty": round(e.novelty, 3),
            "keywords": e.keywords,
            "source_papers": e.source_papers,
            "retry_count": e.retry_count,
        }
        for e in entries
    ]
    return {
        "host_id": input.host_id,
        "total_validated": host._knowledge_acc.validated_count,
        "top_entries": results,
    }


@mcp.tool(
    description=(
        "获取最近的试错失败记录。系统从失败中学习，"
        "失败模式分析可帮助理解哪些假设类型或配对策略效果不佳。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_failures(input: GetFailuresInput) -> Dict[str, Any]:
    """获取最近的失败知识（学习教训）。"""
    host = _manager.get(input.host_id)
    if host._knowledge_acc is None:
        return {"host_id": input.host_id, "results": []}

    entries = host._knowledge_acc.get_recent_failures(input.n)
    results = [
        {
            "entry_id": e.entry_id,
            "statement": e.statement,
            "type": e.hypothesis_type,
            "score": round(e.score, 3),
            "keywords": e.keywords,
            "retry_count": e.retry_count,
        }
        for e in entries
    ]
    return {
        "host_id": input.host_id,
        "total_failures": host._knowledge_acc.failed_count,
        "recent_failures": results,
    }


@mcp.tool(
    description=(
        "获取最近的隐性配对。每个配对包含两篇论文的标题、相似度、"
        "配对类型（跨领域桥接/方法迁移/概念互补/直接相似）、桥接关键词和配对理由。"
        "这是系统发现跨领域关联的核心输出。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_pairs(input: GetPairsInput) -> Dict[str, Any]:
    """获取最近的隐性配对。"""
    host = _manager.get(input.host_id)
    pairs = host.get_recent_pairs(input.n)
    return {
        "host_id": input.host_id,
        "total_pairs_found": host.state.pairs_found,
        "cross_domain_pairs": host.state.cross_domain_pairs,
        "recent_pairs": pairs,
    }


@mcp.tool(
    description=(
        "获取系统完整快照，包含状态数据、Top 研究方向和最近事件历史。"
        "适合做完整的状态检查或持久化。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_snapshot(input: HostIdInput) -> Dict[str, Any]:
    """获取系统完整快照。"""
    host = _manager.get(input.host_id)
    snapshot = host.snapshot()
    return {
        "host_id": input.host_id,
        **snapshot.as_dict(),
    }


@mcp.tool(
    description=(
        "获取科学知识图谱的摘要统计，包括节点数、边数、"
        "节点类型分布（论文/假设/概念）、关系类型分布、概念聚类数等。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_graph(input: HostIdInput) -> Dict[str, Any]:
    """获取科学知识图谱摘要。"""
    host = _manager.get(input.host_id)
    if host._knowledge_graph is None:
        return {"host_id": input.host_id, "message": "知识图谱尚未初始化"}

    summary = host._knowledge_graph.summary()
    clusters = host._knowledge_graph.get_concept_clusters()

    
    sorted_clusters = sorted(clusters.items(), key=lambda x: -len(x[1]))[:10]
    top_clusters = [
        {
            "concept": host._knowledge_graph._nodes[cid].label if cid in host._knowledge_graph._nodes else cid,
            "connected_papers": len(papers),
        }
        for cid, papers in sorted_clusters
    ]

    return {
        "host_id": input.host_id,
        "graph_summary": summary,
        "top_concept_clusters": top_clusters,
    }


@mcp.tool(
    description=(
        "手动添加一篇论文到宿主系统进行分析。添加后论文会进入配对池，"
        "参与下一轮隐性配对和假设生成。适合分析特定论文或私有文献。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_add_paper(input: AddPaperInput) -> Dict[str, Any]:
    """手动添加论文供分析。"""
    host = _manager.get(input.host_id)

    import hashlib
    paper_id = f"manual_{hashlib.md5(input.title.encode()).hexdigest()[:8]}"

    from sci_host.crawler.paper_crawler import Paper
    paper = Paper(
        paper_id=paper_id,
        title=input.title,
        abstract=input.abstract,
        authors=input.authors,
        categories=input.categories,
        keywords=input.keywords,
        year=input.year,
        source="manual",
    )

    
    if host._pairer is not None and host._pairer.embedder is not None:
        paper.embedding = host._pairer.embedder.embed(paper.text)

    
    if host._pairer is not None:
        host._pairer.add_papers([paper])

    
    if host._knowledge_graph is not None:
        host._knowledge_graph.add_paper(paper)

    return {
        "status": "success",
        "host_id": input.host_id,
        "paper_id": paper_id,
        "title": input.title,
        "message": f"论文已添加，将在下一轮探索中参与配对分析",
    }


@mcp.tool(
    description=(
        "查询系统信息：版本号、活跃宿主实例列表。"
        "也可用于健康检查。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_info() -> Dict[str, Any]:
    """查询系统信息。"""
    return {
        "system": "Scientific Direction Discovery Host System",
        "version": __version__,
        "description": "7×24 持续运行的科学方向探索系统 | 数字孪生虚拟环境 | 算法算子试错 | EWC持续学习",
        "active_hosts": _manager.list_ids(),
        "capabilities": [
            "论文持续采集 (离线语料 + arXiv API + OpenAlex API + SCIVERSE 4.65亿学术元数据)",
            "跨领域隐性配对 (TF-IDF + SVD 嵌入)",
            "科学假设生成 (类比/矛盾/空白/组合)",
            "孪生仿真试错 (算法算子验证 + 自洽性检查)",
            "EWC 持续学习 (算子权重校准 + 防遗忘)",
            "知识积累与方向追踪",
            "科学知识图谱构建",
            "材料科学模式: CSP 抽取 + 构效关系假设 + 文献溯源 (赛题方向三)",
            "LLM 摘要级 CSP 抽取 (Step Fun / OpenAI 兼容 API, 正则降级)",
            "Agent CSP 模式: MCP agent (Claude) 自主抽取, 无需 API key",
        ],
    }


@mcp.tool(
    description="列出所有活跃的宿主实例 ID。",
    annotations={"readOnlyHint": True},
)
def sci_list_hosts() -> Dict[str, Any]:
    """列出所有活跃宿主实例。"""
    ids = _manager.list_ids()
    hosts_info = []
    for hid in ids:
        try:
            host = _manager.get(hid)
            hosts_info.append({
                "host_id": hid,
                "cycles": host.state.cycle_count,
                "papers_collected": host.state.papers_collected,
                "directions_tracked": host.state.directions_tracked,
                "running": host.running,
            })
        except Exception:
            pass
    return {
        "total_hosts": len(ids),
        "hosts": hosts_info,
    }


@mcp.tool(
    description=(
        "获取数字孪生虚拟实验环境的详细状态，包括:"
        "已注册的算法算子 (CrossDomainTransfer/CausalDiscovery/UncertaintyQuantifier等)、"
        "各维度健康度 (PE论文语料/VM假设评估器/CL持续学习)、"
        "EWC 持续学习状态 (算子权重/校准误差/技能数)、"
        "孪生交互记录数等。"
        "这是理解系统如何在虚拟环境中进行试错的核心接口。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_twin_status(input: HostIdInput) -> Dict[str, Any]:
    """获取数字孪生虚拟实验环境状态。"""
    host = _manager.get(input.host_id)
    return {
        "host_id": input.host_id,
        **host.get_twin_status(),
    }


@mcp.tool(
    description=(
        "获取正式候选发现 (Gold/Silver/Bronze)，按稳定性分数排序。"
        "每个候选发现包含假设陈述、类型、关键词、稳定性分数、复现率、发现级别。"
        "只有通过多轮复现验证 + 交叉验证的假设才会被认证为候选发现。"
        "Agent override candidates 不包含在内，请用 sci_get_agent_overrides 查看。"
        "注意: 本系统的验证是内部算法验证，不是 DFT/实验/真实仿真验证。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_discoveries(input: GetTopValidatedInput) -> Dict[str, Any]:
    """获取正式候选发现 (不含 Agent override)."""
    host = _manager.get(input.host_id)
    discoveries = host.get_discoveries(input.n)
    stats = host.get_discovery_stats()
    return {
        "host_id": input.host_id,
        "total_candidate_discoveries": stats.get("formal_candidate_discoveries", 0),
        "agent_override_candidates": stats.get("agent_override_candidates", 0),
        "discoveries": discoveries,
    }


@mcp.tool(
    description=(
        "获取候选发现统计仪表盘: 总验证数、正式候选发现数、Agent override 数、"
        "Gold/Silver/Bronze 候选分布、平均稳定性、平均复现率。"
        "用于评估系统的科研发现效率和质量。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_discovery_stats(input: HostIdInput) -> Dict[str, Any]:
    """获取候选发现统计仪表盘."""
    host = _manager.get(input.host_id)
    return {
        "host_id": input.host_id,
        **host.get_discovery_stats(),
    }


class GetVerificationInput(BaseModel):
    """获取验证报告参数."""
    host_id: str = Field(description="宿主实例 ID")
    hypothesis_id: str = Field(description="假设 ID")


@mcp.tool(
    description=(
        "获取某个假设的完整验证报告，包含: "
        "是否被认证为发现、发现级别、稳定性分数、复现率、"
        "每次重跑的详细结果 (参数扰动方式/通过状态/各算子评分/违规原因)、"
        "交叉验证详情。用于深入理解某个假设为什么被接受或拒绝。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_verification(input: GetVerificationInput) -> Dict[str, Any]:
    """获取某个假设的完整验证报告."""
    host = _manager.get(input.host_id)
    detail = host.get_verification_detail(input.hypothesis_id)
    if detail is None:
        return {
            "host_id": input.host_id,
            "hypothesis_id": input.hypothesis_id,
            "found": False,
            "message": "未找到该假设的验证记录",
        }
    return {
        "host_id": input.host_id,
        "found": True,
        **detail,
    }


@mcp.tool(
    description="移除一个宿主实例，释放资源。",
    annotations={"readOnlyHint": False, "destructiveHint": True},
)
def sci_remove_host(input: HostIdInput) -> Dict[str, Any]:
    """移除宿主实例。"""
    removed = _manager.remove(input.host_id)
    return {
        "host_id": input.host_id,
        "removed": removed,
        "message": f"宿主 '{input.host_id}' 已移除" if removed else f"宿主 '{input.host_id}' 不存在",
    }




class GetCSPKnowledgeInput(BaseModel):
    """查询 CSP 知识库参数."""
    host_id: str = Field(description="宿主实例 ID")
    composition: str = Field(
        default="",
        description="按化学组分过滤 (模糊匹配)，如 'BaTiO3'、'LiFePO4'。空=不过滤",
    )
    property_name: str = Field(
        default="",
        description="按性能名过滤，如 'bandgap'、'conductivity'、'tc'。空=不过滤",
    )
    n: int = Field(
        default=20, ge=1, le=200,
        description="最多返回条数",
    )


@mcp.tool(
    description=(
        "查询 CSP (组分-结构-性能) 知识库。"
        "每条记录包含: 化学组分 (如 BaTiO3)、晶体结构 (如 perovskite)、"
        "性能名+值 (如 bandgap=3.2 eV)、来源论文 ID (溯源)、置信度。"
        "可按组分或性能名过滤。这是材料科学构效关系知识的核心存储。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_csp_knowledge(input: GetCSPKnowledgeInput) -> Dict[str, Any]:
    """查询 CSP 三元组知识库."""
    host = _manager.get(input.host_id)
    triples = host.get_csp_knowledge(
        composition=input.composition,
        property_name=input.property_name,
        n=input.n,
    )
    return {
        "host_id": input.host_id,
        "total": len(triples),
        "filter": {
            "composition": input.composition or "(all)",
            "property_name": input.property_name or "(all)",
        },
        "csp_triples": triples,
    }


class GetDiscoveryReportInput(BaseModel):
    """获取发现报告参数."""
    host_id: str = Field(description="宿主实例 ID")
    hypothesis_id: str = Field(
        default="",
        description="指定假设 ID 获取单份报告。空=获取 top-N 发现报告",
    )
    n: int = Field(
        default=5, ge=1, le=50,
        description="最多返回报告数 (当 hypothesis_id 为空时有效)",
    )


@mcp.tool(
    description=(
        "获取结构化科学发现报告 (含完整文献溯源链)。"
        "每份报告包含: 假设陈述+类型、发现级别 (Gold/Silver/Bronze)、稳定性+复现率、"
        "来源论文 A/B (标题+ID)、CSP 三元组 (组分-结构-性能)、"
        "验证详情 (每轮重跑结果+交叉验证)、是否含数值预测 (可证伪性)。"
        "赛题: '文献溯源的完整性与可信度' — 每个发现可追溯到源论文和 CSP 三元组。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_discovery_report(input: GetDiscoveryReportInput) -> Dict[str, Any]:
    """获取含溯源链的发现报告."""
    host = _manager.get(input.host_id)
    reports = host.get_discovery_report(
        hypothesis_id=input.hypothesis_id,
        n=input.n,
    )
    return {
        "host_id": input.host_id,
        "total_reports": len(reports),
        "reports": reports,
    }


class GetResearchGapReportInput(BaseModel):
    """Research Gap 报告参数."""
    host_id: str = Field(description="宿主实例 ID")
    n: int = Field(
        default=10, ge=1, le=50,
        description="最多返回报告数",
    )


@mcp.tool(
    description=(
        "【基础任务】获取 Research Gap 识别报告。"
        "每份报告包含: 空白描述、证据论文、反证、新颖性判断、CSP 上下文。"
        "赛题基础任务要求识别文献中的研究空白。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_research_gap_report(input: GetResearchGapReportInput) -> Dict[str, Any]:
    """获取 Research Gap 识别报告."""
    host = _manager.get(input.host_id)
    gaps = host.get_research_gap_report(n=input.n)
    return {
        "host_id": input.host_id,
        "total_gaps": len(gaps),
        "gaps": gaps,
    }


class LiteratureReviewInput(BaseModel):
    """文献调研综述报告参数."""
    host_id: str = Field(description="宿主实例 ID")


@mcp.tool(
    description=(
        "【基础任务】获取文献调研综述报告。"
        "包含: 采集论文统计、领域分布、高频关键词、跨领域配对、"
        "关键候选发现、研究空白、CSP 知识库摘要。"
        "赛题基础任务要求文献调研与综述。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_literature_review(input: LiteratureReviewInput) -> Dict[str, Any]:
    """获取文献调研综述报告."""
    host = _manager.get(input.host_id)
    report = host.get_literature_review_report()
    return {
        "host_id": input.host_id,
        "report": report,
    }


class GetAgentOverrideInput(BaseModel):
    """Agent override candidates 参数."""
    host_id: str = Field(description="宿主实例 ID")
    n: int = Field(
        default=10, ge=1, le=50,
        description="最多返回条数",
    )


@mcp.tool(
    description=(
        "获取 Agent override candidates (未经复现验证)。"
        "这些是 Agent (Claude/LLM) 通过 force_pass 标记的假设, "
        "没有经过三层验证, 仅作为人机协作的候选。"
        "与正式候选发现 (sci_get_discoveries) 明确区分。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_agent_overrides(input: GetAgentOverrideInput) -> Dict[str, Any]:
    """获取 Agent override candidates."""
    host = _manager.get(input.host_id)
    candidates = host.get_agent_override_candidates(n=input.n)
    return {
        "host_id": input.host_id,
        "total_candidates": len(candidates),
        "candidates": candidates,
    }


# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════

class GetPapersForCSPInput(BaseModel):
    """获取待 CSP 抽取论文列表参数."""
    host_id: str = Field(description="宿主实例 ID")
    n: int = Field(
        default=10, ge=1, le=50,
        description="最多返回论文数",
    )


@mcp.tool(
    description=(
        "【Agent CSP 模式】获取待抽取 CSP 三元组的论文列表。"
        "返回论文的 paper_id、title、abstract。"
        "你 (LLM agent) 读取这些摘要后, 自行抽取组分-结构-性能三元组, "
        "然后调用 sci_submit_csp 提交结果。"
        "这样用你自身的推理能力做 CSP 抽取, 不需要单独配 API key。"
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_papers_for_csp(input: GetPapersForCSPInput) -> Dict[str, Any]:
    """获取待 CSP 抽取的论文列表.

    工作流程:
        1. 调用此工具获取论文列表 (含标题+摘要)
        2. 你 (agent) 读取摘要, 抽取 CSP 三元组
        3. 调用 sci_submit_csp 提交三元组
        4. 调用 sci_step_agent 运行探索循环
    """
    host = _manager.get(input.host_id)
    papers = host.get_papers_for_csp(n=input.n)
    return {
        "host_id": input.host_id,
        "total": len(papers),
        "papers": papers,
        "instructions": (
            "请读取每篇论文的 title + abstract, 抽取 CSP 三元组:\n"
            "  - composition: 化学组分 (如 BaTiO3)\n"
            "  - structure: 晶体结构 (如 perovskite)\n"
            "  - property_name: 性能名 (如 bandgap, conductivity, piezoelectric_coeff)\n"
            "  - property_value: 数值 (如 3.2), 无值则 null\n"
            "  - property_unit: 单位 (如 eV, S/cm, pC/N)\n"
            "然后调用 sci_submit_csp 提交, 格式: [{paper_id, composition, structure, property_name, property_value, property_unit}]"
        ),
    }


class SubmitCSPInput(BaseModel):
    """提交 CSP 三元组参数."""
    host_id: str = Field(description="宿主实例 ID")
    csp_triples: List[Dict[str, Any]] = Field(
        description=(
            "CSP 三元组列表, 每条含: "
            "paper_id (论文ID), composition (化学组分如BaTiO3), "
            "structure (晶体结构如perovskite), "
            "property_name (性能名如bandgap), "
            "property_value (数值, 无则null), "
            "property_unit (单位如eV)"
        ),
    )


@mcp.tool(
    description=(
        "【Agent CSP 模式】提交你 (LLM agent) 抽取的 CSP 三元组到系统知识库。"
        "三元组会注入假设生成器, 后续 sci_step_agent 将使用这些三元组生成构效关系假设。"
        "这是 agent 自主推理的核心环节——你的抽取质量直接决定发现质量。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_submit_csp(input: SubmitCSPInput) -> Dict[str, Any]:
    """提交 agent 抽取的 CSP 三元组."""
    host = _manager.get(input.host_id)
    count = host.submit_agent_csp(input.csp_triples)
    return {
        "status": "success",
        "host_id": input.host_id,
        "submitted": len(input.csp_triples),
        "accepted": count,
        "message": f"已接收 {count} 条 CSP 三元组, 可调用 sci_step_agent 运行探索循环",
    }


@mcp.tool(
    description=(
        "【Agent CSP 模式】使用 agent 提交的 CSP 三元组运行一轮探索循环。"
        "与 sci_step 的区别: 假设生成阶段优先使用你 (agent) 提交的三元组, "
        "不调用内部 LLM API。未覆盖的论文降级为正则抽取。"
        "流程: 采集论文 → 配对 → 用 agent CSP 生成假设 → 试错验证 → 知识积累。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_step_agent(input: HostIdInput) -> Dict[str, Any]:
    """Agent 模式探索循环."""
    host = _manager.get(input.host_id)
    result = host.step_agent()

    csp_triples = host.get_csp_knowledge(n=5)
    recent_verifications = host.get_recent_verifications(3)
    discovery_stats = host.get_discovery_stats()

    return {
        "host_id": input.host_id,
        "mode": "agent",
        "cycle": result["cycle"],
        "papers_crawled": result["papers_crawled"],
        "pairs_found": result["pairs_found"],
        "hypotheses_generated": result["hypotheses_generated"],
        "trials_passed": result["trials_passed"],
        "trials_failed": result["trials_failed"],
        "elapsed_ms": result["elapsed_ms"],
        "agent_csp_count": result.get("agent_csp_count", 0),
        "directions": result.get("directions_updated", {}),
        "csp_triples": csp_triples,
        "recent_verifications": recent_verifications,
        "discovery_stats": discovery_stats,
    }


# ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════

class StreamFeedbackInput(BaseModel):
    """流式反馈参数."""
    host_id: str = Field(description="宿主实例 ID")
    stage: str = Field(
        description="当前阶段: crawl / pair / hypothesize / trial / verify",
    )
    reasoning: str = Field(
        default="",
        description="你的推理过程 (为什么这样判断)",
    )
    focus_paper_ids: List[str] = Field(
        default_factory=list,
        description="[crawl阶段] 值得关注的论文 ID 列表",
    )
    focus_pair_ids: List[str] = Field(
        default_factory=list,
        description="[pair阶段] 值得深挖的配对 ID 列表",
    )
    focus_hypothesis_ids: List[str] = Field(
        default_factory=list,
        description="[hypothesize阶段] 值得试错的假设 ID 列表",
    )
    force_pass: List[str] = Field(
        default_factory=list,
        description="[trial阶段] 强制通过的假设 ID (你认为孪生误判了)",
    )
    force_fail: List[str] = Field(
        default_factory=list,
        description="[trial阶段] 强制失败的假设 ID (你认为孪生放水了)",
    )
    direction_hint: str = Field(
        default="",
        description="[verify阶段] 研究方向建议",
    )


@mcp.tool(
    description=(
        "【流式推理 阶段1】采集论文, 返回标题+摘要给你 (Claude) 评估。"
        "你读取论文后, 可以调用 sci_stream_feedback 提供关注论文标记, "
        "然后调用 sci_stream_pair 进入下一阶段。"
        "这是流式模式——你在孪生仿真的每个阶段都参与推理, 不是等结果出来再看。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_stream_crawl(input: HostIdInput) -> Dict[str, Any]:
    """流式阶段 1: 采集论文."""
    host = _manager.get(input.host_id)
    return host.stream_crawl()


@mcp.tool(
    description=(
        "【流式推理 阶段2】隐性配对, 返回跨领域配对结果给你评估。"
        "你可以查看哪些配对最有前景, 调用 sci_stream_feedback 标记值得深挖的配对, "
        "然后调用 sci_stream_hypothesize 生成假设。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_stream_pair(input: HostIdInput) -> Dict[str, Any]:
    """流式阶段 2: 隐性配对."""
    host = _manager.get(input.host_id)
    return host.stream_pair()


@mcp.tool(
    description=(
        "【流式推理 阶段3】生成假设, 返回假设陈述给你评估。"
        "你判断哪些假设有科学意义, 调用 sci_stream_feedback 标记值得试错的假设, "
        "然后调用 sci_stream_trial 送入孪生仿真。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_stream_hypothesize(input: HostIdInput) -> Dict[str, Any]:
    """流式阶段 3: 生成假设."""
    host = _manager.get(input.host_id)
    return host.stream_hypothesize()


@mcp.tool(
    description=(
        "【流式推理 阶段4】孪生仿真试错, 返回 6 个算法算子 + LLM 流式算子的预测给你。"
        "你可以看到每个假设的 cross_domain / causal / uncertainty / knowledge_graph / llm_reasoning 等算子分数。"
        "如果你认为孪生误判了, 调用 sci_stream_feedback 用 force_pass / force_fail 覆盖。"
        "然后调用 sci_stream_verify 做复现验证。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_stream_trial(input: HostIdInput) -> Dict[str, Any]:
    """流式阶段 4: 孪生仿真试错."""
    host = _manager.get(input.host_id)
    return host.stream_trial()


@mcp.tool(
    description=(
        "【流式推理 阶段5】验证复现, 返回认证发现 (Gold/Silver/Bronze) 给你。"
        "你可以查看最终发现报告, 提供研究方向建议。"
        "调用后一轮探索循环完成, 可重新从 sci_stream_crawl 开始。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_stream_verify(input: HostIdInput) -> Dict[str, Any]:
    """流式阶段 5: 验证复现."""
    host = _manager.get(input.host_id)
    return host.stream_verify()


@mcp.tool(
    description=(
        "【流式推理 反馈】在任意阶段提交你的推理和判断, 影响后续流程。"
        "例如: 在 trial 阶段用 force_pass 覆盖孪生的失败判定 (你认为孪生太保守), "
        "或在 crawl 阶段标记值得关注的论文。"
        "你的推理 (reasoning) 会被记录到系统日志, 作为人类/agent 知识的溯源。"
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False},
)
def sci_stream_feedback(input: StreamFeedbackInput) -> Dict[str, Any]:
    """流式反馈: Claude 在任意阶段提交推理和判断."""
    host = _manager.get(input.host_id)
    feedback = {
        "reasoning": input.reasoning,
        "focus_paper_ids": input.focus_paper_ids,
        "focus_pair_ids": input.focus_pair_ids,
        "focus_hypothesis_ids": input.focus_hypothesis_ids,
        "force_pass": input.force_pass,
        "force_fail": input.force_fail,
        "direction_hint": input.direction_hint,
    }
    return host.stream_feedback(input.stage, feedback)


# ════════════════════════════════════════════════════════════
#  Competition Tools (P0+P1+P2)
#  Gap evidence, search engine, competition report, DB validation
# ════════════════════════════════════════════════════════════

class GetGapEvidenceInput(BaseModel):
    """Gap evidence report parameters."""
    host_id: str = Field(description="Host instance ID")
    n: int = Field(default=15, ge=1, le=50, description="Max gaps to return")


@mcp.tool(
    description=(
        "Get Research Gap report with full evidence chains. "
        "Each gap includes: description, supporting papers (>=2), "
        "counter-evidence, novelty_score, feasibility_score, "
        "evidence_chain, and verification_plan. "
        "This is more rigorous than sci_get_research_gap_report."
    ),
    annotations={"readOnlyHint": True},
)
def sci_get_gap_evidence(input: GetGapEvidenceInput) -> Dict[str, Any]:
    """Get Research Gap report with evidence chains."""
    host = _manager.get(input.host_id)
    gaps = host.build_gap_evidence_report(n=input.n)
    return {
        "host_id": input.host_id,
        "total_gaps": len(gaps),
        "gaps": gaps,
    }


class RunSearchInput(BaseModel):
    """Structure-property search parameters."""
    host_id: str = Field(description="Host instance ID")
    iterations: int = Field(default=5, ge=1, le=20, description="Evolution iterations")
    population_size: int = Field(default=20, ge=5, le=100, description="Population size")
    top_k: int = Field(default=5, ge=1, le=20, description="Top candidates to return")


@mcp.tool(
    description=(
        "Run evolutionary search for structure-property relationships (Route A). "
        "Seeds population from LLM-generated hypotheses + CSP triples, "
        "then evolves via element substitution, structure replacement, "
        "and property target swap. "
        "Returns top-N candidates with fitness scores."
    ),
    annotations={"readOnlyHint": True},
)
def sci_run_structure_property_search(input: RunSearchInput) -> Dict[str, Any]:
    """Run evolutionary search for structure-property candidates."""
    host = _manager.get(input.host_id)
    result = host.run_structure_property_search(
        iterations=input.iterations,
        population_size=input.population_size,
        top_k=input.top_k,
    )
    return {
        "host_id": input.host_id,
        **result,
    }


class GenerateReportInput(BaseModel):
    """Competition report parameters."""
    host_id: str = Field(description="Host instance ID")
    include_search: bool = Field(default=True, description="Include Route A search results")
    include_external_validation: bool = Field(
        default=False,
        description="Include external DB validation (requires API keys)",
    )


@mcp.tool(
    description=(
        "Generate a competition-ready Markdown report. "
        "Template: 1.Research Problem 2.Retrieval Strategy "
        "3.Literature Screening 4.CSP Knowledge Base "
        "5.Research Gaps 6.Route A Search 7.Top Candidates "
        "8.Evidence Chains 9.Verification Plan 10.Limitations. "
        "Output is Markdown, ready for competition submission."
    ),
    annotations={"readOnlyHint": True},
)
def sci_generate_competition_report(input: GenerateReportInput) -> Dict[str, Any]:
    """Generate competition report as Markdown."""
    host = _manager.get(input.host_id)
    report_md = host.generate_competition_report(
        include_search=input.include_search,
        include_external_validation=input.include_external_validation,
    )
    return {
        "host_id": input.host_id,
        "report_format": "markdown",
        "report": report_md,
        "report_length": len(report_md),
    }


class ValidateDBInput(BaseModel):
    """External database validation parameters."""
    host_id: str = Field(description="Host instance ID")
    candidates: List[Dict[str, Any]] = Field(
        default=[],
        description="Candidates to validate (empty=use system discoveries)",
    )


@mcp.tool(
    description=(
        "Cross-validate candidates against external databases: "
        "Materials Project, OQMD, NOMAD. "
        "Checks if material exists, structure is known, "
        "property has been recorded. "
        "If no API keys configured, returns 'unavailable'. "
        "Set MP_API_KEY env var to enable Materials Project."
    ),
    annotations={"readOnlyHint": True},
)
def sci_validate_external_databases(input: ValidateDBInput) -> Dict[str, Any]:
    """Validate candidates against external databases."""
    host = _manager.get(input.host_id)
    candidates = input.candidates if input.candidates else None
    result = host.validate_external_databases(candidates=candidates)
    return {
        "host_id": input.host_id,
        **result,
    }



def main() -> None:
    """MCP 服务器入口。"""
    import argparse
    import logging as _logging

    
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        description="科学方向探索宿主系统 MCP Server",
    )
    parser.add_argument(
        "--http", type=int, default=None,
        help="HTTP 模式端口 (默认 stdio 模式)",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="HTTP 模式绑定的主机地址 (默认 0.0.0.0)",
    )
    args = parser.parse_args()

    if args.http:
        mcp.settings.host = args.host
        mcp.settings.port = args.http
        _logger.info("[SciHost MCP] HTTP 模式启动，%s:%s", args.host, args.http)
    else:
        _logger.info("[SciHost MCP] stdio 模式启动")

    mcp.run()


if __name__ == "__main__":
    main()
