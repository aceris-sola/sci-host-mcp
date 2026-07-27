"""LLM 流式推理器 — 在孪生仿真过程中实时评估每个假设.

与 llm_extractor.py (前置 CSP 抽取) 不同,
本模块在 twin.awake_step() 之后、判定通过/失败之前介入:
    1. 孪生 6 个算子给出预测
    2. LLM 看到假设 + 算子预测 → 给出推理评分 + 推理理由
    3. LLM 评分作为第 7 个算子 ("llm_streaming") 加入加权平均
    4. LLM 推理理由存入试错结果, 供后续溯源

这样 LLM 不是旁观者, 而是孪生推理链的一环.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)




def _get_llm_config():
    """获取已配置的 LLM 配置."""
    try:
        from .llm_extractor import get_llm_extractor
        ext = get_llm_extractor()
        if ext and ext.is_available:
            return ext.config
    except Exception:
        pass
    return None




_STREAMING_PROMPT = """You are a materials science reasoning expert embedded inside a digital twin simulation.

A hypothesis has been evaluated by 6 algorithm operators. Your job is to review their predictions and give your own reasoning-based score.

Hypothesis: {hypothesis}

Algorithm operator predictions (0-1, higher=more feasible):
{algo_predictions}

CSP knowledge context:
{csp_context}

Task:
1. Briefly reason about whether this hypothesis is scientifically sound (1-2 sentences).
2. Give a score from 0.0 to 1.0 (your confidence in the hypothesis).

Output format (JSON only, no other text):
{{"reasoning": "your brief reasoning", "score": 0.75}}"""




class LLMStreamingReasoner:
    """LLM 流式推理器 — 在孪生仿真中实时评估假设.

    用法:
        reasoner = LLMStreamingReasoner()
        if reasoner.is_available:
            result = reasoner.evaluate(hypothesis, algo_predictions, csp_triples)
            # result = {"score": 0.75, "reasoning": "...", "operator": "llm_streaming"}
    """

    def __init__(self):
        self._call_count = 0
        self._fail_count = 0
        self._cache: Dict[str, Dict[str, Any]] = {}

    @property
    def is_available(self) -> bool:
        cfg = _get_llm_config()
        return cfg is not None and cfg.enabled and bool(cfg.api_key)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_calls": self._call_count,
            "failures": self._fail_count,
        }

    def evaluate(
        self,
        hypothesis: Any,
        algo_predictions: Dict[str, float],
        csp_triples: List[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """在孪生仿真后实时评估假设.

        Args:
            hypothesis: 假设对象 (有 statement, hypothesis_type 等属性)
            algo_predictions: 孪生 6 个算子的预测 {algo_name: score}
            csp_triples: 相关 CSP 三元组 (可选上下文)

        Returns:
            {"score": float, "reasoning": str, "operator": "llm_streaming"}
            或 None (如果 LLM 不可用)
        """
        if not self.is_available:
            return None

        statement = getattr(hypothesis, "statement", str(hypothesis))
        cache_key = statement[:150]

        if cache_key in self._cache:
            return self._cache[cache_key]

        self._call_count += 1

        try:
            cfg = _get_llm_config()

            
            algo_str = "\n".join(
                f"  - {k}: {v:.3f}" for k, v in algo_predictions.items()
            )

            csp_str = ""
            if csp_triples:
                csp_str = "\n".join(
                    f"  - {t.composition} | {t.structure} | {t.property_name} = {t.value_str}"
                    for t in csp_triples[:5]
                )
            elif not csp_triples:
                csp_str = "  (no CSP context available)"

            prompt = _STREAMING_PROMPT.format(
                hypothesis=statement[:300],
                algo_predictions=algo_str,
                csp_context=csp_str,
            )

            
            response = self._call_llm(cfg, prompt)
            if not response:
                self._fail_count += 1
                return None

            
            result = self._parse_response(response)
            if result:
                self._cache[cache_key] = result
                return result

            self._fail_count += 1
            return None

        except Exception as e:
            _logger.warning("[LLM-Stream] evaluate failed: %s", e)
            self._fail_count += 1
            return None

    def _call_llm(self, cfg, prompt: str) -> str:
        """调用 LLM API."""
        import httpx

        payload = json.dumps({
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": "You are a materials science reasoning expert. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.2,
        })

        try:
            r = httpx.post(
                f"{cfg.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg.api_key}",
                    "Content-Type": "application/json",
                },
                content=payload,
                timeout=cfg.timeout,
            )
            if r.status_code != 200:
                return ""
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            _logger.warning("[LLM-Stream] API call failed: %s", e)
            return ""

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON."""
        
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                score = max(0.0, min(1.0, score))
                return {
                    "score": score,
                    "reasoning": str(data.get("reasoning", ""))[:200],
                    "operator": "llm_streaming",
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                score = max(0.0, min(1.0, float(data["score"])))
                return {
                    "score": score,
                    "reasoning": str(data.get("reasoning", ""))[:200],
                    "operator": "llm_streaming",
                }
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        
        score_match = re.search(r'"score"\s*:\s*([\d.]+)', response)
        if score_match:
            try:
                score = max(0.0, min(1.0, float(score_match.group(1))))
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]+)"', response)
                reasoning = reasoning_match.group(1)[:200] if reasoning_match else ""
                return {
                    "score": score,
                    "reasoning": reasoning,
                    "operator": "llm_streaming",
                }
            except (ValueError, TypeError):
                pass

        return None




_global_reasoner: Optional[LLMStreamingReasoner] = None


def get_streaming_reasoner() -> LLMStreamingReasoner:
    """获取全局流式推理器实例."""
    global _global_reasoner
    if _global_reasoner is None:
        _global_reasoner = LLMStreamingReasoner()
    return _global_reasoner
