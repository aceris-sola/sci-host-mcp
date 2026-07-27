"""LLM 驱动的假设生成器 — 因果推理式科学假设生成.

与模板填空式生成 (generator.py TEMPLATES) 不同,
本模块让 LLM 作为"科学推理者"直接生成假设:

    输入: 论文配对 (标题+摘要+关键词) + CSP 三元组 + 配对类型
    ↓
    LLM 因果推理:
      1. 识别两篇论文之间的因果机制差异
      2. 提出可证伪的预测 (含具体变量和方向)
      3. 给出推理链 (为什么 A → B)
    ↓
    输出: 结构化假设 (statement + rationale + predicted_value + causal_chain)

调用链:
    HypothesisGenerator.generate(pairs)
        → 尝试 LLM 生成 (如果 LLM 可用)
        → LLM 失败则降级为模板生成 (原有逻辑)

LLM 返回 JSON:
    {
        "statement": "在 cable-driven 机器人中, 增大 cable 预紧力从 X 到 Y,
                       预期末端刚度提升 ≈15%, 但关节耦合误差增加 ≤2°",
        "rationale": "Paper A 证明 cable 预紧力与刚度的线性关系在 ≥500N 成立;
                      Paper B 的欠驱动模型显示耦合误差对刚度敏感...",
        "hypothesis_type": "structure_property",
        "predicted_variable": "end_effector_stiffness",
        "predicted_direction": "increase",
        "predicted_magnitude": 0.15,
        "causal_chain": ["cable_pretension ↑", "joint_stiffness ↑", "coupling_error ↑"],
        "falsification_criterion": "如果末端刚度提升 <5% 或耦合误差 >5°, 假设被证伪",
        "confidence": 0.7
    }
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)


from . import CSPTriple




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




_SYSTEM_PROMPT = """You are a scientific hypothesis generation expert embedded in a digital twin research system.

Your task: given a pair of research papers and their extracted knowledge, generate ONE falsifiable, causally-grounded scientific hypothesis.

CRITICAL REQUIREMENTS:
1. The hypothesis must be FALSIFIABLE — state a specific prediction that could be proven wrong by experiment or simulation.
2. The hypothesis must have a CAUSAL CHAIN — explain WHY the predicted effect should occur, linking mechanism → intermediate variable → outcome.
3. The hypothesis must be SPECIFIC — use actual method names, material names, or variable names from the papers, NOT generic terms like "the approach" or "the method".
4. The hypothesis must be NOVEL — it should not be trivially stated in either paper; it should be a NEW inference from combining their insights.
5. If the papers are from different domains, the hypothesis should bridge them via a shared underlying mechanism.
6. The hypothesis should include QUANTITATIVE METRICS whenever possible — compare alternatives on cost, torque/force output, efficiency, lifetime/durability, weight, or other measurable engineering metrics.

FORBIDDEN:
- Do NOT generate "transfer X from domain A to domain B" statements without specifying what will change and why.
- Do NOT use vague terms: "improve performance", "enhance efficiency", "achieve better results".
- Do NOT generate hypotheses about topics unrelated to the papers' actual content.
- Do NOT produce hypotheses without a comparison baseline (e.g., "compared to harmonic drive" or "vs. brushless motor").

OUTPUT FORMAT: Output ONLY a JSON object (no markdown, no explanation). The JSON must have these fields:
{
    "statement": "A concise, falsifiable hypothesis statement (1-3 sentences). Include specific variables, predicted direction, approximate magnitude, and a comparison baseline.",
    "rationale": "Why this hypothesis makes sense given the two papers (2-4 sentences). Reference specific findings from each paper.",
    "hypothesis_type": "one of: structure_property, hidden_link, gap_filling, analogy, contradiction, combination",
    "predicted_variable": "the main variable being predicted (e.g., 'torque_density', 'cost_per_unit', 'tracking_error', 'fatigue_life')",
    "predicted_direction": "increase or decrease",
    "predicted_magnitude": 0.15,
    "comparison_baseline": "What is being compared against (e.g., 'harmonic drive', 'brushless motor', 'PZT ceramic')",
    "causal_chain": ["step 1: mechanism", "step 2: intermediate effect", "step 3: outcome"],
    "falsification_criterion": "What specific observation would disprove this hypothesis",
    "confidence": 0.7
}"""

_USER_PROMPT_TEMPLATE = """Generate a falsifiable scientific hypothesis from this paper pair:

## Paper A
Title: {title_a}
Abstract: {abstract_a}
Keywords: {keywords_a}

## Paper B
Title: {title_b}
Abstract: {abstract_b}
Keywords: {keywords_b}

## Pair Info
- Pair type: {pair_type}
- Bridge keywords: {bridge_keywords}
- Cross-domain: {cross_domain}

## Extracted CSP Knowledge (Composition-Structure-Property)
{csp_context}

## Domain Context
{domain_context}

Generate ONE hypothesis that combines insights from both papers. The hypothesis should:
1. Be testable in a digital twin simulation.
2. Include specific quantitative metrics (e.g., torque density N·m/kg, cost $/unit, efficiency %, lifetime cycles).
3. Compare against a concrete baseline (e.g., "compared to harmonic drive" or "vs. PZT ceramic").
4. State what would falsify the prediction."""




class LLMHypothesisGenerator:
    """LLM 驱动的科学假设生成器.

    用法:
        gen = LLMHypothesisGenerator()
        if gen.is_available:
            result = gen.generate_hypothesis(pair, csp_a, csp_b, domain_context)
            # result = {"statement": ..., "rationale": ..., "causal_chain": ...}
    """

    def __init__(self):
        self._call_count = 0
        self._fail_count = 0
        self._cache: Dict[str, Dict[str, Any]] = {}

    @property
    def is_available(self) -> bool:
        """LLM 是否可用."""
        cfg = _get_llm_config()
        return cfg is not None and cfg.enabled and bool(cfg.api_key)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_calls": self._call_count,
            "failures": self._fail_count,
            "cache_hits": sum(1 for _ in self._cache),
        }

    def generate_hypothesis(
        self,
        pair: Any,
        csp_a: Optional[List[CSPTriple]] = None,
        csp_b: Optional[List[CSPTriple]] = None,
        domain_context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """用 LLM 从论文配对 + CSP 知识生成假设.

        Args:
            pair: PaperPair 对象 (有 paper_a_title, paper_b_title, 等属性)
            csp_a: Paper A 的 CSP 三元组 (可选)
            csp_b: Paper B 的 CSP 三元组 (可选)
            domain_context: 领域上下文描述 (研究种子等)

        Returns:
            结构化假设 dict, 或 None (如果 LLM 不可用/失败)
            {
                "statement": str,
                "rationale": str,
                "hypothesis_type": str,
                "predicted_variable": str,
                "predicted_direction": str,
                "predicted_magnitude": float | None,
                "causal_chain": List[str],
                "falsification_criterion": str,
                "confidence": float,
            }
        """
        if not self.is_available:
            return None

        
        cache_key = f"{pair.pair_id}:{len(csp_a or [])}:{len(csp_b or [])}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._call_count += 1

        try:
            cfg = _get_llm_config()

            
            prompt = self._build_prompt(pair, csp_a, csp_b, domain_context)

            
            response = self._call_llm(cfg, prompt)
            if not response:
                self._fail_count += 1
                return None

            
            result = self._parse_response(response)
            if result:
                
                result = self._post_validate(result, pair)
                if result:
                    self._cache[cache_key] = result
                    return result

            self._fail_count += 1
            return None

        except Exception as e:
            _logger.warning("[LLM-HypoGen] generate failed: %s", e)
            self._fail_count += 1
            return None

    def generate_batch(
        self,
        pairs: List[Any],
        csp_map: Optional[Dict[str, Dict[str, List[CSPTriple]]]] = None,
        domain_context: str = "",
        max_count: int = 10,
    ) -> List[Dict[str, Any]]:
        """批量生成假设.

        Args:
            pairs: PaperPair 列表
            csp_map: {pair_id: {"a": [CSPTriple...], "b": [CSPTriple...]}}
            domain_context: 领域上下文
            max_count: 最多生成多少个

        Returns:
            假设 dict 列表
        """
        results: List[Dict[str, Any]] = []
        csp_map = csp_map or {}

        for pair in pairs:
            if len(results) >= max_count:
                break

            csp_info = csp_map.get(pair.pair_id, {})
            csp_a = csp_info.get("a", [])
            csp_b = csp_info.get("b", [])

            result = self.generate_hypothesis(pair, csp_a, csp_b, domain_context)
            if result:
                results.append(result)

        return results

    

    def _build_prompt(
        self,
        pair: Any,
        csp_a: Optional[List[CSPTriple]],
        csp_b: Optional[List[CSPTriple]],
        domain_context: str,
    ) -> str:
        """构造用户提示."""

        
        csp_lines: List[str] = []
        if csp_a:
            csp_lines.append("Paper A CSP triples:")
            for t in csp_a[:5]:
                csp_lines.append(
                    f"  - {t.composition} | {t.structure} | "
                    f"{t.property_name} = {t.value_str}"
                )
        if csp_b:
            csp_lines.append("Paper B CSP triples:")
            for t in csp_b[:5]:
                csp_lines.append(
                    f"  - {t.composition} | {t.structure} | "
                    f"{t.property_name} = {t.value_str}"
                )
        csp_context = "\n".join(csp_lines) if csp_lines else "(no CSP triples extracted)"

        
        abstract_a = (getattr(pair, "paper_a_abstract", "") or "")[:500]
        abstract_b = (getattr(pair, "paper_b_abstract", "") or "")[:500]

        
        kws_a = getattr(pair, "paper_a_keywords", []) or []
        kws_b = getattr(pair, "paper_b_keywords", []) or []

        return _USER_PROMPT_TEMPLATE.format(
            title_a=getattr(pair, "paper_a_title", ""),
            abstract_a=abstract_a or "(no abstract available)",
            keywords_a=", ".join(kws_a[:8]) if kws_a else "(none)",
            title_b=getattr(pair, "paper_b_title", ""),
            abstract_b=abstract_b or "(no abstract available)",
            keywords_b=", ".join(kws_b[:8]) if kws_b else "(none)",
            pair_type=getattr(pair, "pair_type", "unknown"),
            bridge_keywords=", ".join(getattr(pair, "bridge_keywords", []) or []),
            cross_domain=str(getattr(pair, "cross_domain", False)),
            csp_context=csp_context,
            domain_context=domain_context or "(general scientific research)",
        )

    def _call_llm(self, cfg, prompt: str) -> str:
        """调用 LLM API."""
        try:
            import httpx
        except ImportError:
            return self._call_llm_urllib(cfg, prompt)

        payload = json.dumps({
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1000,
            "temperature": 0.4,  
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
                _logger.warning("[LLM-HypoGen] HTTP %d: %s", r.status_code, r.text[:200])
                return ""
            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            _logger.warning("[LLM-HypoGen] httpx failed: %s, trying urllib", e)
            return self._call_llm_urllib(cfg, prompt)

    def _call_llm_urllib(self, cfg, prompt: str) -> str:
        """用 urllib 调用 LLM (fallback)."""
        import urllib.request as ureq

        payload = json.dumps({
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1000,
            "temperature": 0.4,
        }).encode("utf-8")

        req = ureq.Request(
            f"{cfg.api_base}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with ureq.urlopen(req, timeout=cfg.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON 为结构化假设.

        鲁棒解析: 支持 markdown 代码块、裸 JSON、截断 JSON.
        """
        data = None

        
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            pass

        
        if data is None:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

        
        if data is None:
            first_brace = response.find('{')
            last_brace = response.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = response[first_brace:last_brace + 1]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    
                    cleaned = re.sub(r',\s*}', '}', json_str)
                    cleaned = re.sub(r',\s*]', ']', cleaned)
                    try:
                        data = json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass

        if not isinstance(data, dict):
            return None

        
        statement = str(data.get("statement", "")).strip()
        if not statement or len(statement) < 15:
            return None

        result: Dict[str, Any] = {
            "statement": statement,
            "rationale": str(data.get("rationale", ""))[:500],
            "hypothesis_type": str(data.get("hypothesis_type", "unknown")).strip(),
            "predicted_variable": str(data.get("predicted_variable", "")).strip(),
            "predicted_direction": str(data.get("predicted_direction", "")).strip(),
            "comparison_baseline": str(data.get("comparison_baseline", "")).strip(),
            "causal_chain": data.get("causal_chain", []) if isinstance(data.get("causal_chain"), list) else [],
            "falsification_criterion": str(data.get("falsification_criterion", "")).strip(),
            "confidence": 0.5,
        }

        
        mag_raw = data.get("predicted_magnitude")
        if mag_raw is not None:
            try:
                result["predicted_magnitude"] = float(mag_raw)
            except (ValueError, TypeError):
                result["predicted_magnitude"] = None
        else:
            result["predicted_magnitude"] = None

        
        conf_raw = data.get("confidence")
        if conf_raw is not None:
            try:
                result["confidence"] = max(0.0, min(1.0, float(conf_raw)))
            except (ValueError, TypeError):
                pass

        return result

    def _post_validate(
        self, result: Dict[str, Any], pair: Any,
    ) -> Optional[Dict[str, Any]]:
        """后处理验证: 检查 LLM 生成的假设质量.

        Returns:
            通过验证的 result, 或 None (如果质量不达标)
        """
        statement = result.get("statement", "")

        
        _vague_patterns = [
            "transfer x from",
            "improve performance",
            "enhance efficiency",
            "achieve better results",
            "the proposed method",
            "the approach",
        ]
        stmt_lower = statement.lower()
        for pattern in _vague_patterns:
            if pattern in stmt_lower:
                _logger.debug("[LLM-HypoGen] rejected vague statement: %s", pattern)
                return None

        
        if len(statement) < 30:
            return None

        
        causal_chain = result.get("causal_chain", [])
        hypo_type = result.get("hypothesis_type", "")
        if not causal_chain and hypo_type not in ("gap_filling", "unknown"):
            
            
            result["confidence"] *= 0.6

        
        _valid_types = {
            "structure_property", "hidden_link", "gap_filling",
            "analogy", "contradiction", "combination",
        }
        if result["hypothesis_type"] not in _valid_types:
            
            if "矛盾" in statement or "contradict" in stmt_lower:
                result["hypothesis_type"] = "contradiction"
            elif "组合" in statement or "combin" in stmt_lower:
                result["hypothesis_type"] = "combination"
            elif "迁移" in statement or "transfer" in stmt_lower or "analogy" in stmt_lower:
                result["hypothesis_type"] = "analogy"
            elif "空白" in statement or "gap" in stmt_lower or "未" in statement:
                result["hypothesis_type"] = "gap_filling"
            elif "关联" in statement or "link" in stmt_lower or "hidden" in stmt_lower:
                result["hypothesis_type"] = "hidden_link"
            else:
                result["hypothesis_type"] = "hidden_link"  

        return result




_global_generator: Optional[LLMHypothesisGenerator] = None


def get_hypothesis_generator() -> LLMHypothesisGenerator:
    """获取全局 LLM 假设生成器实例."""
    global _global_generator
    if _global_generator is None:
        _global_generator = LLMHypothesisGenerator()
    return _global_generator
