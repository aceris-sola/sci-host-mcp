"""科学知识图谱 — 构建论文-假设-概念的关系网络.

复用原项目 SemanticKnowledgeGraph 的设计模式（邻接表 + BFS 路径 + 相似查询），
但节点从"数字孪生实例"改为"论文/假设/概念"。

节点类型:
    - paper: 论文节点
    - hypothesis: 假设节点
    - concept: 概念节点（从关键词提取）

边类型:
    - cites: 引用关系
    - relates_to: 相关关系
    - validates: 验证关系（假设被试错验证）
    - contradicts: 矛盾关系
    - derives_from: 假设来源关系
    - bridges: 桥接关系（概念桥接两篇论文）
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class ScienceNode:
    """知识图谱节点."""
    node_id: str
    node_type: str          # paper / hypothesis / concept
    label: str              
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ScienceEdge:
    """知识图谱边."""
    source: str
    target: str
    relation: str           # cites / relates_to / validates / contradicts / derives_from / bridges
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)



RELATIONS: Set[str] = {
    "cites", "relates_to", "validates", "contradicts",
    "derives_from", "bridges", "extends",
}


class ScienceKnowledgeGraph:
    """科学知识图谱.

    随试错持续积累，构建论文-假设-概念的关系网络。
    支持路径查询、社区发现、桥接分析。

    与原项目 SemanticKnowledgeGraph 的对比:
        原项目: TwinNode (twin_id, domain, capabilities) + Jaccard 相似度
        本系统: ScienceNode (paper/hypothesis/concept) + 多关系类型
    """

    def __init__(self, max_nodes: int = 10000) -> None:
        self.max_nodes = max_nodes
        self._nodes: Dict[str, ScienceNode] = {}
        self._adj: Dict[str, List[ScienceEdge]] = {}
        self._adj_reverse: Dict[str, List[ScienceEdge]] = {}  

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(edges) for edges in self._adj.values())

    def add_paper(self, paper: Any) -> None:
        """添加论文节点."""
        pid = getattr(paper, "paper_id", "")
        if not pid:
            return
        if pid not in self._nodes:
            self._add_node(ScienceNode(
                node_id=pid,
                node_type="paper",
                label=getattr(paper, "title", pid)[:80],
                properties={
                    "categories": getattr(paper, "categories", []),
                    "keywords": getattr(paper, "keywords", []),
                    "year": getattr(paper, "year", 2025),
                    "source": getattr(paper, "source", "unknown"),
                },
            ))
            
            keywords = getattr(paper, "keywords", [])
            for kw in keywords[:5]:
                concept_id = f"concept_{kw.lower().replace(' ', '_')}"
                if concept_id not in self._nodes:
                    self._add_node(ScienceNode(
                        node_id=concept_id,
                        node_type="concept",
                        label=kw,
                        properties={"keyword": kw},
                    ))
                self._add_edge(ScienceEdge(
                    source=pid, target=concept_id,
                    relation="relates_to",
                    weight=1.0,
                ))

        self._enforce_limit()

    def add_hypothesis(self, trial_result: Any) -> None:
        """添加验证通过的假设节点."""
        hid = getattr(trial_result, "hypothesis_id", "")
        if not hid:
            return
        if hid not in self._nodes:
            self._add_node(ScienceNode(
                node_id=hid,
                node_type="hypothesis",
                label=getattr(trial_result, "statement", hid)[:80],
                properties={
                    "type": getattr(trial_result, "hypothesis_type", ""),
                    "score": getattr(trial_result, "score", 0.0),
                    "novelty": getattr(trial_result, "novelty", 0.0),
                    "source_pair": getattr(trial_result, "source_pair", ""),
                },
            ))

            
            paper_a_id = getattr(trial_result, "paper_a_id", "")
            paper_b_id = getattr(trial_result, "paper_b_id", "")
            if paper_a_id and paper_a_id in self._nodes:
                self._add_edge(ScienceEdge(
                    source=hid, target=paper_a_id,
                    relation="derives_from", weight=0.8,
                ))
            if paper_b_id and paper_b_id in self._nodes:
                self._add_edge(ScienceEdge(
                    source=hid, target=paper_b_id,
                    relation="derives_from", weight=0.8,
                ))

            
            keywords = getattr(trial_result, "keywords", [])
            for kw in keywords[:3]:
                concept_id = f"concept_{kw.lower().replace(' ', '_')}"
                if concept_id in self._nodes:
                    self._add_edge(ScienceEdge(
                        source=hid, target=concept_id,
                        relation="validates", weight=0.6,
                    ))

        self._enforce_limit()

    def find_bridge_concepts(self, paper_a_id: str, paper_b_id: str) -> List[str]:
        """找到两篇论文之间的桥接概念（共同邻居）."""
        if paper_a_id not in self._nodes or paper_b_id not in self._nodes:
            return []
        neighbors_a = set(self._get_neighbors(paper_a_id))
        neighbors_b = set(self._get_neighbors(paper_b_id))
        common = neighbors_a & neighbors_b
        
        bridges = [n for n in common if self._nodes.get(n, None) and
                   self._nodes[n].node_type == "concept"]
        return bridges

    def find_path(self, node_a: str, node_b: str, max_depth: int = 5) -> List[str]:
        """BFS 最短路径."""
        if node_a not in self._nodes or node_b not in self._nodes:
            return []
        if node_a == node_b:
            return [node_a]
        visited: Set[str] = {node_a}
        queue: deque = deque([(node_a, [node_a])])
        while queue:
            cur, path = queue.popleft()
            if len(path) > max_depth:
                continue
            for nb in self._get_neighbors(cur):
                if nb in visited:
                    continue
                new_path = path + [nb]
                if nb == node_b:
                    return new_path
                visited.add(nb)
                queue.append((nb, new_path))
        return []

    def get_concept_clusters(self) -> Dict[str, List[str]]:
        """获取概念聚类（每个概念连接的论文列表）."""
        clusters: Dict[str, List[str]] = {}
        for node_id, node in self._nodes.items():
            if node.node_type != "concept":
                continue
            papers = [
                e.source for e in self._adj_reverse.get(node_id, [])
                if e.relation == "relates_to"
                and self._nodes.get(e.source, None)
                and self._nodes[e.source].node_type == "paper"
            ]
            if papers:
                clusters[node_id] = papers
        return clusters

    def get_hypotheses_for_concept(self, concept_id: str) -> List[str]:
        """获取与某概念相关的已验证假设."""
        result: List[str] = []
        for edge in self._adj_reverse.get(concept_id, []):
            if edge.relation == "validates":
                result.append(edge.source)
        return result

    def summary(self) -> Dict[str, Any]:
        """图谱摘要."""
        type_counts: Dict[str, int] = {}
        for node in self._nodes.values():
            type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1

        relation_counts: Dict[str, int] = {}
        for edges in self._adj.values():
            for e in edges:
                relation_counts[e.relation] = relation_counts.get(e.relation, 0) + 1

        return {
            "total_nodes": self.node_count,
            "total_edges": self.edge_count,
            "node_types": type_counts,
            "relation_types": relation_counts,
            "concept_clusters": len(self.get_concept_clusters()),
        }

    

    def _add_node(self, node: ScienceNode) -> None:
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node
            self._adj[node.node_id] = []
            self._adj_reverse[node.node_id] = []

    def _add_edge(self, edge: ScienceEdge) -> None:
        if edge.source not in self._nodes:
            self._add_node(ScienceNode(edge.source, "unknown", edge.source))
        if edge.target not in self._nodes:
            self._add_node(ScienceNode(edge.target, "unknown", edge.target))
        self._adj[edge.source].append(edge)
        self._adj_reverse[edge.target].append(edge)

    def _get_neighbors(self, node_id: str) -> List[str]:
        """获取邻居（双向）."""
        result: Set[str] = set()
        for e in self._adj.get(node_id, []):
            result.add(e.target)
        for e in self._adj_reverse.get(node_id, []):
            result.add(e.source)
        return list(result)

    def _enforce_limit(self) -> None:
        """执行节点数上限."""
        if len(self._nodes) <= self.max_nodes:
            return
        
        papers = sorted(
            [(n.created_at, nid) for nid, n in self._nodes.items()
             if n.node_type == "paper"],
        )
        excess = len(self._nodes) - self.max_nodes
        for _, nid in papers[:excess]:
            self._remove_node(nid)

    def _remove_node(self, node_id: str) -> None:
        """删除节点及其关联边."""
        self._nodes.pop(node_id, None)
        self._adj.pop(node_id, None)
        self._adj_reverse.pop(node_id, None)
        
        for nid in list(self._adj.keys()):
            self._adj[nid] = [e for e in self._adj[nid] if e.target != node_id]
        for nid in list(self._adj_reverse.keys()):
            self._adj_reverse[nid] = [e for e in self._adj_reverse[nid] if e.source != node_id]
