"""宿主系统事件总线.

提供科研宿主内部的 EventBus 设计模式，
为科学探索宿主系统提供松耦合通信机制。
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class HostEvent:
    """宿主事件."""
    topic: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""

    def __post_init__(self) -> None:
        if not self.topic:
            raise ValueError("事件主题不能为空")


EventHandler = Callable[[HostEvent], None]


class HostEventBus:
    """发布-订阅事件总线.

    支持:
    - 主题订阅（通配符后缀匹配，如 "crawler.*" 匹配所有 crawler 开头的事件）
    - 同步分发
    - 事件历史记录
    """

    
    TOPICS = {
        "host.start": "宿主系统启动",
        "host.stop": "宿主系统停止",
        "crawler.batch": "采集器完成一批采集",
        "crawler.error": "采集器出错",
        "pairing.done": "配对引擎完成一轮配对",
        "pairing.found": "发现新的隐性配对",
        "hypothesis.generated": "生成新假设",
        "trial.started": "试错开始",
        "trial.passed": "试错通过",
        "trial.failed": "试错失败",
        "trial.retried": "试错重试",
        "knowledge.updated": "知识库更新",
        "direction.discovered": "发现新研究方向",
        "direction.eliminated": "方向被淘汰",
        "loop.cycle": "一轮探索循环完成",
        "loop.error": "循环出错",
    }

    def __init__(self, history_size: int = 500) -> None:
        self._subscribers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._history: List[HostEvent] = []
        self._history_size = history_size

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """订阅主题。支持前缀通配，例如 "crawler.*"."""
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        if topic in self._subscribers:
            try:
                self._subscribers[topic].remove(handler)
            except ValueError:
                pass

    def publish(self, event: HostEvent) -> None:
        """发布事件，同步通知所有匹配的订阅者."""
        if self._history_size > 0:
            self._history.append(event)
            if len(self._history) > self._history_size:
                self._history = self._history[-self._history_size:]

        for topic, handlers in self._subscribers.items():
            if self._matches(topic, event.topic):
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception:
                        pass  

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic == prefix or topic.startswith(prefix + ".")
        return pattern == topic

    def history(self) -> List[HostEvent]:
        return list(self._history)

    def history_by_topic(self, topic_prefix: str) -> List[HostEvent]:
        """按主题前缀过滤历史事件."""
        return [e for e in self._history if e.topic.startswith(topic_prefix)]

    def clear(self) -> None:
        self._subscribers.clear()
        self._history.clear()
