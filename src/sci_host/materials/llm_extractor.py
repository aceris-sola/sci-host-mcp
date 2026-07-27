"""LLM 驱动的 CSP 三元组抽取器 — 摘要级精确抽取.

用大语言模型从论文标题+摘要中抽取组分-结构-性能三元组,
比纯正则方法在真实摘要上准确率高一个数量级.

调用链:
    CSPExtractor.extract(text)
        → 先尝试 LLM 抽取 (如果配置了 LLM)
        → LLM 失败则降级为正则抽取 (原有逻辑)

LLM 返回 JSON 格式:
    [
        {
            "composition": "BaTiO3",
            "structure": "perovskite",
            "property_name": "piezoelectric_coeff",
            "property_value": 190,
            "property_unit": "pC/N"
        },
        ...
    ]
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)


from . import CSPTriple




class LLMConfig:
    """LLM 抽取器配置."""

    def __init__(
        self,
        api_base: str = "https://api.stepfun.com/step_plan/v1",
        api_key: str = "",
        model: str = "step-router-v1",
        max_tokens: int = 2000,
        temperature: float = 0.1,
        timeout: int = 120,
        max_retries: int = 2,
        enabled: bool = False,
    ):
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.enabled = enabled

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量创建配置."""
        return cls(
            api_base=os.environ.get("LLM_API_BASE", "https://api.stepfun.com/step_plan/v1"),
            api_key=os.environ.get("LLM_API_KEY", ""),
            model=os.environ.get("LLM_MODEL", "step-router-v1"),
            enabled=bool(os.environ.get("LLM_API_KEY", "")),
        )




_SYSTEM_PROMPT = """You are a materials science information extraction expert.

Your task: extract Composition-Structure-Property (CSP) triples from a paper title and abstract.

IMPORTANT: Output ONLY a JSON array. No reasoning, no explanations, no markdown formatting. Just the raw JSON array.

Rules:
1. Composition: the chemical formula of the main material studied (e.g., "BaTiO3", "CsPbI3", "La0.7Sr0.3MnO3", "NiTi", "Mg-Al alloy"). Use standard chemical notation. If no specific formula is mentioned, use "unknown".
2. Structure: the crystal structure or material form (e.g., "perovskite", "spinel", "layered", "amorphous", "composite", "nanotube", "thin film", "bulk"). If not mentioned, use "unknown".
3. Property name: the material property being measured or discussed. Use these standard names when applicable:
   - bandgap, conductivity, resistivity, mobility, dielectric_constant, dielectric_loss
   - piezoelectric_coeff, pyroelectric_coeff, coupling_factor, tc (Curie temp)
   - magnetization, coercivity, susceptibility
   - seebeck, zt, power_factor, thermal_conductivity
   - specific_capacity, voltage, energy_density, coulombic_efficiency
   - overpotential, tafel_slope, faradaic_efficiency
   - youngs_modulus, hardness, yield_strength, tensile_strength, flexural_strength
   - corrosion_rate, surface_area, particle_size, density
   - frequency, melting_point, thermal_expansion
   If the property doesn't match any standard name, use a descriptive lowercase name with underscores.
4. Property value: the numerical value if explicitly stated. null if not given.
5. Property unit: the unit (e.g., "eV", "S/cm", "pC/N", "GPa", "MPa", "K", "nm"). Empty string if not applicable.

Output format: a JSON array of objects. Only output the JSON, no other text.
If no CSP triples can be extracted, return an empty array [].

Example input: "BaTiO3 perovskite ceramic shows excellent piezoelectric properties with d33=190 pC/N."
Example output:
[
  {"composition": "BaTiO3", "structure": "perovskite", "property_name": "piezoelectric_coeff", "property_value": 190, "property_unit": "pC/N"}
]

Example input: "We study the thermoelectric performance of Bi2Te3 nanowires."
Example output:
[
  {"composition": "Bi2Te3", "structure": "nanotube", "property_name": "zt", "property_value": null, "property_unit": ""}
]"""




class LLMCSPExtractor:
    """LLM 驱动的 CSP 三元组抽取器.

    用法:
        extractor = LLMCSPExtractor(config)
        triples = extractor.extract("BaTiO3 perovskite d33=190", paper_id="p1", paper_title="Test")
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self._cache: Dict[str, List[CSPTriple]] = {}
        self._call_count = 0
        self._fail_count = 0

    @property
    def is_available(self) -> bool:
        """LLM 是否可用."""
        return self.config.enabled and bool(self.config.api_key)

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_calls": self._call_count,
            "failures": self._fail_count,
            "cache_hits": sum(1 for _ in self._cache),
        }

    def extract(
        self,
        text: str,
        paper_id: str = "",
        paper_title: str = "",
    ) -> List[CSPTriple]:
        """用 LLM 从文本中抽取 CSP 三元组.

        带重试逻辑: LLM 可能超时或返回不可解析的文本.

        Args:
            text: 论文标题+摘要
            paper_id: 论文 ID (溯源)
            paper_title: 论文标题 (溯源)

        Returns:
            CSPTriple 列表, 失败时返回空列表
        """
        if not self.is_available:
            return []

        
        cache_key = text[:200]
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._call_count += 1

        for attempt in range(self.config.max_retries + 1):
            try:
                response_text = self._call_llm(text)
                if not response_text:
                    if attempt < self.config.max_retries:
                        continue
                    self._fail_count += 1
                    return []

                triples = self._parse_response(response_text, paper_id, paper_title)

                
                if triples:
                    self._cache[cache_key] = triples
                    return triples

                
                if attempt < self.config.max_retries:
                    continue

                
                self._cache[cache_key] = []
                return []

            except Exception as e:
                _logger.warning("[LLM-CSP] attempt %d failed: %s", attempt + 1, e)
                if attempt >= self.config.max_retries:
                    self._fail_count += 1
                    return []

        self._fail_count += 1
        return []

    def _call_llm(self, text: str) -> str:
        """调用 LLM API, 返回文本响应."""
        try:
            import httpx
        except ImportError:
            import urllib.request as _ureq
            import urllib.error as _uerr
            return self._call_llm_urllib(text, _ureq, _uerr)

        payload = json.dumps({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract CSP triples from this paper:\n\n{text[:3000]}"},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        })

        try:
            r = httpx.post(
                f"{self.config.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                content=payload,
                timeout=self.config.timeout,
            )
            if r.status_code != 200:
                _logger.warning("[LLM-CSP] HTTP %d: %s", r.status_code, r.text[:200])
                return ""

            data = r.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception as e:
            _logger.warning("[LLM-CSP] httpx failed: %s, trying urllib", e)
            return self._call_llm_urllib(text)

    def _call_llm_urllib(self, text: str, _ureq=None, _uerr=None) -> str:
        """用 urllib 调用 LLM (fallback)."""
        if _ureq is None:
            import urllib.request as _ureq
            import urllib.error as _uerr

        payload = json.dumps({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract CSP triples from this paper:\n\n{text[:3000]}"},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }).encode("utf-8")

        req = _ureq.Request(
            f"{self.config.api_base}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with _ureq.urlopen(req, timeout=self.config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def _parse_response(
        self,
        response_text: str,
        paper_id: str,
        paper_title: str,
    ) -> List[CSPTriple]:
        """解析 LLM 返回的 JSON 为 CSPTriple 列表.

        step-router-v1 可能输出 advisor 咨询文本 + markdown 代码块,
        需要鲁棒地提取 JSON 数组.
        """
        data = None

        
        try:
            data = json.loads(response_text)
        except json.JSONDecodeError:
            pass

        
        if data is None:
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

        
        if data is None:
            
            first_bracket = response_text.find('[')
            last_bracket = response_text.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                json_str = response_text[first_bracket:last_bracket + 1]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    
                    cleaned = re.sub(r',\s*]', ']', json_str)
                    cleaned = re.sub(r',\s*}', '}', cleaned)
                    try:
                        data = json.loads(cleaned)
                    except json.JSONDecodeError:
                        pass

        
        if data is None:
            objects = []
            
            for m in re.finditer(r'\{[^{}]*"composition"[^{}]*\}', response_text, re.DOTALL):
                try:
                    obj = json.loads(m.group(0))
                    objects.append(obj)
                except json.JSONDecodeError:
                    pass
            data = objects if objects else []

        if not isinstance(data, list):
            data = []

        if not isinstance(data, list):
            return []

        triples: List[CSPTriple] = []
        for item in data:
            if not isinstance(item, dict):
                continue

            comp = str(item.get("composition", "")).strip()
            struct = str(item.get("structure", "")).strip()
            prop_name = str(item.get("property_name", "")).strip()

            
            if not comp or comp == "unknown":
                continue
            if not prop_name or prop_name == "general":
                continue

            
            prop_value_raw = item.get("property_value")
            prop_value = None
            if prop_value_raw is not None:
                try:
                    if isinstance(prop_value_raw, (int, float)):
                        prop_value = float(prop_value_raw)
                    else:
                        
                        val_str = str(prop_value_raw).strip()
                        num_match = re.search(r'-?[\d.]+(?:e[+-]?\d+)?', val_str, re.IGNORECASE)
                        if num_match:
                            prop_value = float(num_match.group(0))
                except (ValueError, TypeError):
                    prop_value = None

            prop_unit = str(item.get("property_unit", "")).strip()

            triple = CSPTriple(
                composition=comp,
                structure=struct if struct else "unknown",
                property_name=prop_name,
                property_value=prop_value,
                property_unit=prop_unit,
                source_paper_id=paper_id,
                source_paper_title=paper_title,
                source_section="abstract",
                confidence=0.9,  
            )
            triples.append(triple)

        return triples




_global_extractor: Optional[LLMCSPExtractor] = None


def get_llm_extractor() -> Optional[LLMCSPExtractor]:
    """获取全局 LLM 抽取器实例 (如果可用)."""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = LLMCSPExtractor()
    return _global_extractor if _global_extractor.is_available else None


def set_llm_config(config: LLMConfig) -> None:
    """设置全局 LLM 配置."""
    global _global_extractor
    _global_extractor = LLMCSPExtractor(config)
