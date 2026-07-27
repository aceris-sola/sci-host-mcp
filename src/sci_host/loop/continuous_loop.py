"""连续循环 — 7×24 小时不间断探索.

替代原项目的 Sleep Cycle 日夜循环机制。
原项目: 白天(Awake) → 夜晚(Sleep) → 白天 → ... (有梦境/睡眠)
本系统: 采集 → 配对 → 假设 → 试错 → 积累 → 采集 → ... (无睡眠)

关键区别:
    - 无 Sleep/Awake 模式切换，始终"清醒"
    - 无梦境合成，用假设生成替代
    - 无 EWC 防遗忘，用方向置信度衰减替代
    - 持续运行，7×24 小时不间断
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

from ..config import LoopConfig

_logger = logging.getLogger(__name__)


class ContinuousLoop:
    """7×24 连续探索循环.

    核心循环:
        while running:
            1. 采集论文 (Crawler)
            2. 隐性配对 (Pairer)
            3. 生成假设 (Hypothesis Generator)
            4. 试错验证 (Trial Engine)
            5. 知识积累 (Knowledge Accumulator)
            6. 方向更新 (Direction Tracker)
            7. 等待 loop_interval 秒
            8. 回到 1

    与原项目 Sleep Cycle 的对比:
        原项目:
            awake_step() → sleep_step() → awake_step() → ...
            (白天感知执行, 夜晚梦境学习)

        本系统:
            step() → step() → step() → ... (无限循环)
            (每步都是完整的探索循环, 无需睡眠)
    """

    def __init__(self, host: Any, config: LoopConfig) -> None:
        self.host = host
        self.config = config
        self._iteration: int = 0
        self._running: bool = False
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._error_streak: int = 0
        self._max_error_streak: int = 10

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加每轮循环结束后的回调."""
        self._callbacks.append(callback)

    def run(self) -> None:
        """启动 7×24 连续循环（阻塞）.

        循环直到:
            - host.stop() 被调用
            - 达到 max_iterations（0 = 无限）
            - 连续错误超过阈值
            - KeyboardInterrupt
        """
        self._running = True
        self._iteration = 0

        _logger.info("[ContinuousLoop] 启动 7×24 连续探索循环")
        _logger.info("  循环间隔: %ds", self.config.loop_interval)
        _logger.info("  最大迭代: %s", '无限' if self.config.max_iterations == 0 else self.config.max_iterations)
        _logger.info("  按 Ctrl+C 停止")

        try:
            while self._running:
                if (self.config.max_iterations > 0 and
                        self._iteration >= self.config.max_iterations):
                    _logger.info("[ContinuousLoop] 达到最大迭代次数 %d，停止", self.config.max_iterations)
                    break

                try:
                    result = self.host.step()
                    self._iteration += 1
                    self._error_streak = 0

                    
                    self._print_cycle_summary(result)

                    
                    for cb in self._callbacks:
                        try:
                            cb(result)
                        except Exception:
                            pass

                    
                    if self.config.auto_save and self.config.save_path:
                        if self._iteration % 5 == 0:  
                            self._save_state()

                except Exception as e:
                    self._error_streak += 1
                    self.host._handle_error("loop", str(e))
                    if self._error_streak >= self._max_error_streak:
                        _logger.error("[ContinuousLoop] 连续错误 %d 次，停止", self._error_streak)
                        break

                
                if self._running:
                    time.sleep(self.config.loop_interval)

        except KeyboardInterrupt:
            _logger.info("[ContinuousLoop] 收到中断信号")
        finally:
            self._running = False
            if self.config.auto_save and self.config.save_path:
                self._save_state()
            _logger.info("[ContinuousLoop] 已停止 (共 %d 轮)", self._iteration)

    def stop(self) -> None:
        """停止循环."""
        self._running = False

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def running(self) -> bool:
        return self._running

    def _print_cycle_summary(self, result: Dict[str, Any]) -> None:
        """打印每轮摘要."""
        cycle = result.get("cycle", 0)
        papers = result.get("papers_crawled", 0)
        pairs = result.get("pairs_found", 0)
        hypos = result.get("hypotheses_generated", 0)
        passed = result.get("trials_passed", 0)
        failed = result.get("trials_failed", 0)
        elapsed = result.get("elapsed_ms", 0)

        _logger.info("  [Cycle %d] 论文:%d 配对:%d 假设:%d 通过:%d 失败:%d (%.0fms)",
                      cycle, papers, pairs, hypos, passed, failed, elapsed)

        
        directions = self.host.get_top_directions(3)
        if directions and cycle > 0 and cycle % 3 == 0:
            _logger.info("  🎯 当前最有前景的方向:")
            for i, d in enumerate(directions, 1):
                _logger.info("     %d. %s (置信度=%.3f)", i, d['label'][:60], d['confidence'])

    def _save_state(self) -> None:
        """保存系统状态."""
        if not self.config.save_path:
            return
        try:
            os.makedirs(self.config.save_path, exist_ok=True)
            snapshot = self.host.snapshot()
            filepath = os.path.join(
                self.config.save_path,
                f"snapshot_{int(time.time())}.json",
            )
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot.as_dict(), f, ensure_ascii=False, indent=2)
        except Exception:
            pass
