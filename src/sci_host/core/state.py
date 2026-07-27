"""宿主系统状态管理.

跟踪系统全局状态，包括:
- 已采集论文数 (原始 + 质量闸门后)
- 已发现配对数
- 已生成假设数
- 试错统计
- 知识库规模
- 运行时间
- 在线回退离线次数
- 假设采样 (最近几条, 供状态报告展示)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List


@dataclass
class HostState:
    """宿主系统运行状态（可变，实时更新）."""
    
    start_time: float = field(default_factory=time.time)
    
    cycle_count: int = 0
    
    papers_collected: int = 0           
    papers_raw_fetched: int = 0         
    papers_quality_rejected: int = 0    
    papers_in_cache: int = 0
    
    offline_fallback_crawls: int = 0    
    
    pairs_found: int = 0
    cross_domain_pairs: int = 0
    
    hypotheses_generated: int = 0
    hypotheses_active: int = 0
    
    trials_total: int = 0
    trials_passed: int = 0
    trials_failed: int = 0
    
    knowledge_entries: int = 0
    directions_tracked: int = 0
    directions_promising: int = 0
    
    errors: int = 0
    last_error: str = ""
    
    recent_hypotheses: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=5),
    )

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def uptime_str(self) -> str:
        s = int(self.uptime_seconds)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f"{h}h{m}m{s}s"

    @property
    def trial_pass_rate(self) -> float:
        if self.trials_total == 0:
            return 0.0
        return self.trials_passed / self.trials_total

    def add_recent_hypothesis(self, hypo: Dict[str, Any]) -> None:
        """添加一条假设到最近采样缓冲."""
        self.recent_hypotheses.append(hypo)

    def sample_hypotheses(self) -> List[Dict[str, Any]]:
        """返回最近假设采样列表 (从旧到新)."""
        return list(self.recent_hypotheses)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "uptime": self.uptime_str,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "cycle_count": self.cycle_count,
            "papers_collected": self.papers_collected,
            "papers_raw_fetched": self.papers_raw_fetched,
            "papers_quality_rejected": self.papers_quality_rejected,
            "papers_in_cache": self.papers_in_cache,
            "offline_fallback_crawls": self.offline_fallback_crawls,
            "pairs_found": self.pairs_found,
            "cross_domain_pairs": self.cross_domain_pairs,
            "hypotheses_generated": self.hypotheses_generated,
            "hypotheses_active": self.hypotheses_active,
            "trials_total": self.trials_total,
            "trials_passed": self.trials_passed,
            "trials_failed": self.trials_failed,
            "trial_pass_rate": round(self.trial_pass_rate, 3),
            "knowledge_entries": self.knowledge_entries,
            "directions_tracked": self.directions_tracked,
            "directions_promising": self.directions_promising,
            "errors": self.errors,
            "sample_hypotheses": self.sample_hypotheses(),
        }

    def summary(self) -> str:
        """生成可读的状态摘要."""
        d = self.as_dict()
        lines = [
            f"═══ 宿主系统状态 ═══",
            f"  运行时间: {d['uptime']}",
            f"  探索循环: {d['cycle_count']} 轮",
            f"  论文采集: {d['papers_collected']} 篇 (原始 {d['papers_raw_fetched']}, "
            f"质量拒绝 {d['papers_quality_rejected']}, 缓存 {d['papers_in_cache']})",
            f"  在线回退: {d['offline_fallback_crawls']} 次",
            f"  隐性配对: {d['pairs_found']} 对 (跨领域 {d['cross_domain_pairs']})",
            f"  假设生成: {d['hypotheses_generated']} 个 (活跃 {d['hypotheses_active']})",
            f"  试错验证: {d['trials_total']} 次 (通过 {d['trials_passed']}, "
            f"通过率 {d['trial_pass_rate']:.1%})",
            f"  知识积累: {d['knowledge_entries']} 条",
            f"  方向追踪: {d['directions_tracked']} 个 (有前景 {d['directions_promising']})",
            f"  错误: {d['errors']}",
        ]
        
        samples = d.get("sample_hypotheses", [])
        if samples:
            lines.append(f"  最近假设采样:")
            for i, s in enumerate(samples[:3], 1):
                stmt = s.get("statement", "")[:80]
                lines.append(f"    {i}. [{s.get('type', '?')}] {stmt}...")
        return "\n".join(lines)


@dataclass
class HostSnapshot:
    """宿主系统快照（不可变，用于持久化）."""
    timestamp: float = field(default_factory=time.time)
    state: Dict[str, Any] = field(default_factory=dict)
    top_directions: List[Dict[str, Any]] = field(default_factory=list)
    recent_events: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "state": self.state,
            "top_directions": self.top_directions,
            "recent_events": self.recent_events,
        }
