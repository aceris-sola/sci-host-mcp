"""材料科学领域定制层 — 实体模型、知识抽取、假设模板、文献源.

本模块将通用科研探索系统适配为材料科学文献驱动的科学发现智能体.

核心概念:
    CSP 三元组: Composition(组分) - Structure(结构) - Property(性能)
    这是材料科学的知识基石，所有发现都围绕 CSP 关系展开.

赛题要求:
    "基于大规模材料科学文献库，构建能够自主阅读、推理并产出
     可验证科学发现的智能体系统"
    "产出具有可证伪性的科学发现——包括材料_性质关联、
     隐藏知识连接、新材料设计假设等"

本模块提供:
    - MaterialEntity: 材料实体 (组分/结构/工艺/性能 + 溯源)
    - CSPTriple: 组分-结构-性能三元组
    - CSPExtractor: 从论文中抽取 CSP 三元组
    - MaterialHypothesisTemplates: 材料科学假设模板
    - MATERIAL_CORPUS: 内置材料科学论文语料
    - MATERIAL_CATEGORIES: 材料科学 arXiv 分类
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


MATERIAL_CATEGORIES: List[str] = [
    "cond-mat.mtrl-sci",
    "cond-mat.mes-hall",
    "cond-mat.supr-con",
    "physics.chem-ph",
    "physics.comp-ph",
    "cs.LG",
]


MATERIAL_SEEDS: List[str] = [
    "materials science", "band gap", "perovskite", "crystal structure",
    "density functional theory", "DFT", "alloy", "composite",
    "semiconductor", "catalyst", "battery", "photovoltaic",
    "thermoelectric", "ferroelectric", "piezoelectric",
    "conductivity", "magnetism", "superconductivity",
    "phase diagram", "synthesis", "calcination", "sintering",
    "thin film", "nanoparticle", "crystal growth",
    "machine learning for materials", "materials discovery",
    "high-throughput", "property prediction",
]


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

@dataclass
class MaterialEntity:
    """材料实体 — 从论文中抽取的结构化材料知识.

    赛题: "组分、结构、工艺与性能之间的关联"
    本实体将这四要素结构化，附带溯源信息构成可审计证据链.
    """
    composition: str
    structure: str = ""
    processing: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)  # {"bandgap": 3.2, "conductivity": 1.5e-4}
    source_paper_id: str = ""
    source_paper_title: str = ""
    source_section: str = ""            # "abstract", "table_2", "fig_3_caption"
    source_doi: str = ""
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def formula_clean(self) -> str:
        """清理后的化学式."""
        return re.sub(r'\s+', '', self.composition)

    @property
    def has_property(self) -> bool:
        return len(self.properties) > 0

    @property
    def property_summary(self) -> str:
        """性能摘要字符串."""
        if not self.properties:
            return "unknown"
        parts = []
        for k, v in self.properties.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4g}")
            else:
                parts.append(f"{k}={v}")
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composition": self.composition,
            "structure": self.structure,
            "processing": self.processing,
            "properties": self.properties,
            "source_paper_id": self.source_paper_id,
            "source_paper_title": self.source_paper_title[:80],
            "source_section": self.source_section,
            "confidence": self.confidence,
        }


@dataclass
class CSPTriple:
    """组分-结构-性能三元组 — 材料科学知识的基本单元.

    赛题: "构效关系发现"
    CSP 三元组是构效关系的最小可验证单元:
        Composition + Structure → Property (value)

    可证伪性: 预测的 Property 值可被实验/计算直接验证.
    """
    composition: str
    structure: str
    property_name: str          # "bandgap", "conductivity", "tc"
    property_value: Optional[float] = None
    property_unit: str = ""     # "eV", "S/cm", "K"
    source_paper_id: str = ""
    source_paper_title: str = ""
    source_section: str = ""
    confidence: float = 0.5

    @property
    def key(self) -> str:
        """唯一键: 组分+结构+性能名."""
        return f"{self.composition}|{self.structure}|{self.property_name}"

    @property
    def is_prediction(self) -> bool:
        """是否为预测 (无已知值)."""
        return self.property_value is None

    @property
    def value_str(self) -> str:
        if self.property_value is None:
            return "?"
        if self.property_unit:
            return f"{self.property_value:.4g} {self.property_unit}"
        return f"{self.property_value:.4g}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composition": self.composition,
            "structure": self.structure,
            "property_name": self.property_name,
            "property_value": self.property_value,
            "property_unit": self.property_unit,
            "value_str": self.value_str,
            "source_paper_id": self.source_paper_id,
            "source_paper_title": self.source_paper_title,
            "source_section": self.source_section,
            "confidence": self.confidence,
            "is_prediction": self.is_prediction,
        }


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

class CSPExtractor:
    """从论文文本中抽取 CSP 三元组.

    赛题: "自主检索筛选、知识抽取"
    本抽取器使用规则 + 模式匹配从论文标题/摘要中提取:
        - 化学组分 (化学式模式)
        - 晶体结构 (已知结构类型词典)
        - 性能名+值 (性能词典 + 数值模式)

    虽然不如 LLM 抽取精确，但在离线/无 LLM 时保证基本可用.
    生产环境可替换为 LLM 抽取.
    """

    STRUCTURE_TYPES: List[str] = [
        "perovskite", "double perovskite", "layered perovskite",
        "antiperovskite", "post-perovskite",
        "Ruddlesden-Popper", "Dion-Jacobson", "Aurivillius",
        "brownmillerite",
        "cubic", "tetragonal", "orthorhombic", "hexagonal",
        "monoclinic", "trigonal", "rhombohedral", "triclinic",
        "rocksalt", "fluorite", "rutile", "wurtzite", "zincblende",
        "diamond", "graphite", "spinel", "olivine", "ilmenite",
        "corundum", "anatase", "brookite", "cristobalite",
        "amorphous", "layered", "quasicrystal",
        "half-Heusler", "full-Heusler", "Heusler",
        "L12", "L1_2", "B2", "D03", "D0_3", "A15",
        "C14", "C15", "C36", "Laves", "CsCl",
        "skutterudite", "clathrate", "chalcopyrite", "stannite",
        "kesterite", "tetrahedrite",
        "NiAs", "CdI2", "MoS2", "delafossite",
        "kagome", "honeycomb",
        "garnet", "scheelite", "pyrochlore", "zircon",
        "monazite", "xenotime", "apatite", "baddeleyite",
        "weberite", "fergusonite", "marcasite", "pyrite",
        "arsenopyrite", "covellite", "chalcocite",
        "spinel", "inverse spinel", "normal spinel",
        "hollandite", "tunnel", "framework", "NASICON",
        "garnet", "perovskite",
        "A15", "cuprate", "iron-based", "iron pnictide",
        "kagome", "pyrochlore", "frustrated",
        "body-centered", "face-centered", "simple cubic",
        "body-centered tetragonal", "BCT",
        "tridymite", "coesite", "stishovite", "moganite",
        "calcite", "aragonite", "dolomite", "magnesite",
        "talc", "serpentine", "kaolinite", "montmorillonite",
        "albite", "orthoclase", "anorthite", "quartz",
        "mullite", "cordierite", "spodumene", "wollastonite",
        "enstatite", "diopside", "forsterite", "fayalite",
        "argyrodite", "thio-LISICON", "anti-perovskite",
        "garnet-type", "perovskite-type",
        "MOF", "ZIF", "zeolite", "MFI", "FAU",
        "graphene", "nanotube", "fullerene", "nanoribbon",
        "β-sheet", "α-helix", "collagen", "hydroxyapatite",
        "topological insulator", "Weyl semimetal", "Dirac semimetal",
        "topological", "nodal-line", "flat-band",
        "high-entropy alloy", "HEA", "complex concentrated alloy",
        "multiprincipal element", "CCA",
    ]

    PROPERTY_DICT: Dict[str, Tuple[List[str], str]] = {
        "bandgap": (["band gap", "bandgap", "energy gap", "Eg", "optical gap"], "eV"),
        "conductivity": (["conductivity", "electrical conductivity", "ionic conductivity", "σ"], "S/cm"),
        "resistivity": (["resistivity", "electrical resistivity"], "Ω·cm"),
        "mobility": (["mobility", "carrier mobility", "electron mobility", "hole mobility"], "cm²/Vs"),
        "carrier_concentration": (["carrier concentration", "carrier density", "doping concentration"], "cm⁻³"),
        "work_function": (["work function", "workfunction"], "eV"),
        "hall_coefficient": (["Hall coefficient", "Hall constant"], "cm³/C"),

        "tc": (["Curie temperature", "Curie temp", "Tc", "ferroelectric transition"], "K"),
        "dielectric_constant": (["dielectric constant", "relative permittivity", "εr"], ""),
        "dielectric_loss": (["dielectric loss", "loss tangent", "tan δ", "tan delta"], ""),
        "piezoelectric_coeff": (["piezoelectric coefficient", "d33", "d31", "d15", "piezoelectric constant"], "pC/N"),
        "pyroelectric_coeff": (["pyroelectric coefficient", "pyroelectric constant"], "μC/m²K"),
        "coupling_factor": (["electromechanical coupling", "coupling coefficient", "kp", "coupling factor"], ""),
        "spontaneous_polarization": (["spontaneous polarization", "Ps"], "μC/cm²"),
        "remanent_polarization": (["remanent polarization", "remanent polarisation", "Pr", "residual polarization"], "μC/cm²"),

        "tn": (["Néel temperature", "Néel temp", "TN", "antiferromagnetic transition"], "K"),
        "magnetization": (["magnetization", "saturation magnetization", "Ms"], "emu/g"),
        "coercivity": (["coercivity", "coercive field", "Hc"], "Oe"),
        "susceptibility": (["magnetic susceptibility", "susceptibility", "χ"], ""),
        "permeability": (["permeability", "magnetic permeability", "μr"], ""),
        "anisotropy_field": (["anisotropy field", "anisotropy constant", "Ha"], "Oe"),

        "seebeck": (["Seebeck coefficient", "thermopower", "thermoelectric power", "Seebeck"], "μV/K"),
        "zt": (["figure of merit", "ZT", "thermoelectric figure"], ""),
        "power_factor": (["power factor", "PF"], "μW/mK²"),

        "superconducting_tc": (["superconducting transition", "superconducting critical temperature", "superconducting Tc"], "K"),
        "critical_field": (["critical field", "Hc1", "Hc2", "upper critical field", "lower critical field"], "T"),
        "critical_current": (["critical current density", "Jc", "critical current"], "A/cm²"),
        "penetration_depth": (["penetration depth", "London penetration depth"], "nm"),
        "coherence_length": (["coherence length"], "nm"),

        "specific_capacity": (["specific capacity", "gravimetric capacity"], "mAh/g"),
        "voltage": (["voltage", "operating voltage", "working voltage", "open circuit voltage", "OCV"], "V"),
        "energy_density": (["energy density"], "Wh/kg"),
        "power_density": (["power density"], "W/kg"),
        "coulombic_efficiency": (["coulombic efficiency", "coulomb efficiency"], "%"),
        "cycle_life": (["cycle life", "cycling stability", "cycle stability"], ""),

        "overpotential": (["overpotential", "over-potential", "η"], "V"),
        "tafel_slope": (["Tafel slope", "Tafel"], "mV/dec"),
        "tof": (["turnover frequency", "TOF", "turnover rate"], "s⁻¹"),
        "exchange_current": (["exchange current density", "j0"], "mA/cm²"),
        "faradaic_efficiency": (["faradaic efficiency", "faraday efficiency", "FE"], "%"),
        "onset_potential": (["onset potential", "onset overpotential"], "V"),

        "refractive_index": (["refractive index", "index of refraction"], ""),
        "absorption_coeff": (["absorption coefficient", "absorption"], "cm⁻¹"),
        "plqy": (["photoluminescence quantum yield", "PLQY", "quantum yield", "QY"], "%"),
        "emission_wavelength": (["emission wavelength", "emission peak", "emission", "PL peak", "photoluminescence peak"], "nm"),
        "absorption_edge": (["absorption edge", "absorption threshold"], "nm"),

        "youngs_modulus": (["Young's modulus", "elastic modulus", "Young modulus"], "GPa"),
        "hardness": (["hardness", "Vickers hardness", "Hv", "nanoindentation hardness"], "GPa"),
        "yield_strength": (["yield strength", "yield stress"], "MPa"),
        "tensile_strength": (["tensile strength", "ultimate tensile strength", "UTS"], "MPa"),
        "fracture_toughness": (["fracture toughness", "KIC", "K_Ic"], "MPa·m"),
        "elongation": (["elongation", "elongation at break", "strain at break"], "%"),
        "bulk_modulus": (["bulk modulus"], "GPa"),
        "shear_modulus": (["shear modulus"], "GPa"),
        "poisson_ratio": (["Poisson ratio", "Poisson's ratio", "ν"], ""),
        "flexural_strength": (["flexural strength", "bend strength", "bending strength"], "MPa"),

        "thermal_conductivity": (["thermal conductivity", "κ", "kappa"], "W/mK"),
        "specific_heat": (["specific heat", "Cp", "heat capacity", "specific heat capacity"], "J/molK"),
        "thermal_expansion": (["thermal expansion", "coefficient of thermal expansion", "CTE", "thermal expansion coefficient"], "ppm/K"),
        "thermal_diffusivity": (["thermal diffusivity", "thermal diffusion"], "mm²/s"),
        "debye_temp": (["Debye temperature", "θD"], "K"),
        "melting_point": (["melting point", "melting temperature", "Tm"], "K"),
        "glass_transition": (["glass transition", "glass transition temperature", "Tg"], "K"),

        "formation_energy": (["formation energy", "formation enthalpy", "ΔHf"], "eV"),
        "cohesive_energy": (["cohesive energy", "Ec"], "eV"),
        "vacancy_energy": (["vacancy formation energy", "vacancy energy"], "eV"),
        "migration_energy": (["migration energy", "activation energy for migration"], "eV"),
        "activation_energy": (["activation energy", "Ea", "activation barrier"], "eV"),
        "decomposition_temp": (["decomposition temperature", "decomposition", "Td"], "K"),

        "corrosion_rate": (["corrosion rate", "corrosion"], "mm/year"),
        "corrosion_potential": (["corrosion potential", "Ecorr"], "V"),

        "surface_area": (["surface area", "specific surface area", "BET surface area", "BET"], "m²/g"),
        "particle_size": (["particle size", "grain size", "crystallite size", "nanoparticle size"], "nm"),
        "pore_volume": (["pore volume", "total pore volume"], "cm³/g"),
        "pore_size": (["pore size", "average pore diameter", "pore diameter"], "nm"),
        "film_thickness": (["film thickness", "layer thickness", "coating thickness"], "nm"),

        "neutron_cross": (["neutron cross section", "neutron absorption cross", "absorption cross section"], "barn"),

        "frequency": (["resonance frequency", "natural frequency", "operating frequency"], "Hz"),

        # ═══════════════════════════════════════════════════════════════
        # ═══════════════════════════════════════════════════════════════

        "molecular_weight": (["molecular weight", "Mw", "weight-average molecular weight", "Mn"], "g/mol"),
        "crystallinity": (["crystallinity", "degree of crystallinity", "crystalline fraction"], "%"),
        "crosslink_density": (["crosslink density", "cross-link density", "crosslinking density"], "mol/m³"),
        "melt_flow_index": (["melt flow index", "melt index", "MFI"], "g/10min"),
        "melt_temp": (["melting point", "melting temperature", "Tm", "melt temperature"], "K"),
        "degradation_temp": (["degradation temperature", "thermal degradation", "decomposition temperature"], "K"),

        "fiber_volume_fraction": (["fiber volume fraction", "fibre volume fraction", "volume fraction"], "%"),
        "interface_strength": (["interfacial shear strength", "interface shear strength", "interface strength"], "MPa"),
        "interlaminar_strength": (["interlaminar shear strength", "ILSS", "interlaminar strength"], "MPa"),

        "biocompatibility": (["biocompatibility", "cell viability", "cytocompatibility"], "%"),
        "degradation_rate": (["degradation rate", "biodegradation rate", "bioresorption rate"], ""),
        "cell_adhesion": (["cell adhesion", "cell attachment", "cell proliferation"], "%"),

        "shape_recovery": (["shape recovery", "recovery ratio", "shape recovery ratio", "recovery rate"], "%"),
        "phase_transition_temp": (["martensitic transformation", "phase transition temperature", "transformation temperature", "austenite finish", "Af"], "K"),
        "superelasticity": (["superelastic", "superelasticity", "pseudoelastic"], ""),
        "hysteresis": (["hysteresis", "hysteresis loop", "hysteresis width"], ""),

        "pce": (["power conversion efficiency", "PCE", "photoconversion efficiency"], "%"),
        "jsc": (["short-circuit current", "short circuit current", "Jsc", "JSC"], "mA/cm²"),
        "voc": (["open-circuit voltage", "open circuit voltage", "Voc", "VOC"], "V"),
        "fill_factor": (["fill factor", "FF"], ""),
        "ipce": (["incident photon-to-current efficiency", "IPCE", "external quantum efficiency", "EQE"], "%"),

        "gas_response": (["gas response", "sensor response", "sensitivity", "gas sensitivity"], ""),
        "response_time": (["response time", "response and recovery"], "s"),
        "recovery_time": (["recovery time"], "s"),
        "detection_limit": (["detection limit", "limit of detection", "LOD"], "ppm"),

        "friction_coeff": (["friction coefficient", "coefficient of friction", "COF", "friction"], ""),
        "wear_rate": (["wear rate", "wear resistance", "wear"], "mg/N·m"),

        "diffusion_coeff": (["diffusion coefficient", "diffusivity", "diffusion constant"], "cm²/s"),

        "adsorption_capacity": (["adsorption capacity", "adsorption amount", "uptake capacity", "adsorbed amount"], "mmol/g"),

        "hydrogen_capacity": (["hydrogen storage capacity", "hydrogen capacity", "H2 storage", "gravimetric hydrogen capacity"], "wt%"),

        "magnetoresistance": (["magnetoresistance", "MR ratio", "magnetoresistance ratio"], "%"),
        "spin_polarization": (["spin polarization", "spin polarisation"], "%"),
        "tunneling_resistance": (["tunneling magnetoresistance", "TMR"], "%"),

        "magnetoelectric_coeff": (["magnetoelectric coefficient", "magnetoelectric coupling", "ME coefficient"], "mV/cm·Oe"),

        "color_temperature": (["color temperature", "correlated color temperature", "CCT"], "K"),
        "color_rendering": (["color rendering index", "CRI", "color rendering"], ""),
        "luminous_efficiency": (["luminous efficiency", "luminous efficacy", "luminance"], "lm/W"),
        "luminance": (["luminance", "brightness"], "cd/m²"),

        "electrochemical_window": (["electrochemical window", "electrochemical stability window", "ESW"], "V"),
        "ion_transference": (["ion transference number", "transference number", "t+", "transport number"], ""),

        "density": (["density", "bulk density", "mass density"], "g/cm³"),
        "relative_density": (["relative density", "theoretical density fraction", "sintered density"], "%"),
        "porosity": (["porosity", "pore fraction", "void fraction"], "%"),

        "damping_factor": (["damping factor", "damping capacity", "loss factor", "tan delta"], ""),
        "absorption_coeff_acoustic": (["sound absorption coefficient", "acoustic absorption", "noise reduction coefficient", "NRC"], ""),

        "shielding_effectiveness": (["shielding effectiveness", "EMI shielding", "SE", "shielding"], "dB"),

        "lattice_parameter": (["lattice parameter", "lattice constant", "lattice parameter a", "lattice constant a"], "Å"),
        "cell_volume": (["unit cell volume", "cell volume", "lattice volume"], "ų"),

        "compressive_strength": (["compressive strength", "compression strength", "crushing strength"], "MPa"),

        "electrocaloric": (["electrocaloric temperature", "electrocaloric change", "electrocaloric effect", "ΔT"], "K"),

        "breakdown_strength": (["dielectric breakdown", "breakdown strength", "breakdown voltage", "dielectric strength"], "kV/mm"),

        "contact_angle": (["contact angle", "wetting angle", "water contact angle"], "°"),

        "fatigue_strength": (["fatigue strength", "fatigue limit", "endurance limit", "fatigue life"], "MPa"),
        "fatigue_life_cycles": (["fatigue life", "cycles to failure", "Nf"], ""),
    }

    _VALID_ELEMENTS = frozenset({
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og',
    })

    # ₀₁₂₃₄₅₆₇₈₉ → 0123456789
    _UNICODE_SUBSCRIPT_MAP = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')

    _DIGITLESS_WHITELIST = frozenset({
        "BN", "SiC", "FeCo", "TiC", "FeNi", "CoNi", "NiCu", "AgCu",
        "GaN", "GaAs", "GaP", "InP", "InAs", "AlN", "ZnO", "ZnS",
        "ZnSe", "ZnTe", "CdS", "CdSe", "CdTe", "PbS", "PbSe", "PbTe",
        "SnTe", "GeTe", "SbTe", "TiN", "ZrN", "HfN", "VN", "CrN",
        "TiB", "ZrB", "FeSi", "CoSi", "NiSi", "MnSi", "CrSi",
        "FeO", "CoO", "NiO", "CuO", "MnO", "MgO", "CaO", "SrO",
        "BaO", "LiF", "NaF", "NaCl", "KCl", "CsCl", "LiCl", "AgCl",
        "CuCl", "CuBr", "CuI", "AgBr", "AgI", "FeSn", "NiSn",
        "CuSn", "FeAl", "CoAl", "NiAl", "TiAl", "NiGe", "FeGe",
        "CoGe", "FeSb", "CoSb", "NiSb", "CuSb", "FeBi", "NiBi",
        "LaB", "CeB", "EuB", "YB", "SmB", "MgB", "CaB", "BaB",
        "UP", "ThP", "US", "ThS", "UC", "ThC", "USi", "ThSi",
        "FeP", "CoP", "NiP", "FeSe", "CoSe", "NiSe", "FeTe",
        "CoTe", "NiTe", "LaSi", "CeSi", "LaS", "CeS", "LaSe",
        "CeSe", "LaTe", "CeTe", "LaN", "CeN", "ScN", "YN",
        "CrB", "MnB", "MoB", "WB", "LaSn", "CeSn", "LaPb",
        "CePb", "ZrO", "HfO", "VO", "RuO", "RhO", "PdO",
        "SnSe", "SnS", "SnO",
        # GeS, GeSe, GeO
        "GeS", "GeSe", "GeO",
        # PbO
        "PbO",
        # BiSe, BiS, BiO
        "BiSe", "BiS", "BiO",
        # SbS, SbSe, SbO
        "SbS", "SbSe", "SbO",
        # AsSe, AsS, AsO
        "AsSe", "AsS", "AsO",
        # MoS, MoSe, MoO, MoTe
        "MoS", "MoSe", "MoO", "MoTe",
        # WS, WSe, WO, WTe
        "WS", "WSe", "WO", "WTe",
        # NbSe, NbS, TaS, TaSe
        "NbSe", "NbS", "TaS", "TaSe",
        # CrSe, CrS, CrTe, MnSe, MnS, MnTe
        "CrSe", "CrS", "CrTe", "MnSe", "MnS", "MnTe",
        "VSe", "VS", "VTe",
        "TiS", "TiSe",
        # ZrS, ZrSe, HfS, HfSe
        "ZrS", "ZrSe", "HfS", "HfSe",
        # ReS, ReSe
        "ReS", "ReSe",
        # PtS, PtSe, PtO, PdS, PdSe
        "PtS", "PtSe", "PtO", "PdS", "PdSe",
        # IrO, IrS, RuS, RhS
        "IrO", "IrS", "RuS", "RhS",
        # CuS, CuSe, CuTe, AgS, AgSe, AgTe
        "CuS", "CuSe", "CuTe", "AgS", "AgSe", "AgTe",
        # AuS, AuSe
        "AuS", "AuSe",
        "NiS", "CoS", "FeS",
        "SiGe",
        # Binary alloy permutations (CuNi, NiCr, TiAl, etc.)
        "CuNi", "CuCo", "CuFe", "CuMn", "CuCr", "CuZn", "CuAl",
        "NiCr", "NiCo", "NiFe", "NiMn", "NiAl", "NiTi",
        "CoCr", "CoFe", "CoMn",
        "FeCr", "FeMn", "FeAl",
        "TiAl", "TiV", "TiCr",
        "AlSi", "AlMg", "AlZn",
        "MgAl", "MgZn",
        "ZnAl",
        # InSe, InS, GaS, GaSe, GaTe
        "InSe", "InS", "GaS", "GaSe", "GaTe",
        "AlS", "AlSe", "AlAs",
        "BP", "BAs", "BSb",
        "CP", "CAs",
    })

    _FORMULA_PATTERN = re.compile(
        r'\b((?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?){2,})\b'
    )

    _ELEMENT_SPLIT_PATTERN = re.compile(r'([A-Z][a-z]?)(\d+(?:\.\d+)?)?')

    _VALUE_PATTERN = re.compile(
        r'(?<![a-zA-Z0-9])(-?[\d.]+(?:e[+-]?\d+)?)\s*'
        r'(eV/atom|eV|meV|keV|'
        r'Ω·cm|Ωcm|ohm·cm|'
        r'S/cm|'
        r'cm²/Vs|cm²/s|cm³/g|cm⁻³|cm³/C|cm⁻¹|'
        r'μC/m²K|μC/cm²|μV/K|μW/mK²|'
        r'Wh/kg|W/mK|W/m·K|W/kg|lm/W|'
        r'J/molK|'
        r'mAh/g|mA/cm²|A/cm²|'
        r'mg/N·m|mV/dec|mV/cm·Oe|mmol/g|'
        r'MPa·m|MPa|GPa|kPa|kV/mm|'
        r'mm/year|mm²/s|m²/g|mol/m³|'
        r'ppm/K|ppm|'
        r'g/cm³|g/mol|g/10min|'
        r'pC/N|'
        r'wt%|'
        r'cd/m²|'
        r'barn|nm|μm|s⁻¹|Hz|dB|'
        r'Å|°|'
        r'K|Oe|T|V|%)?',
        re.IGNORECASE,
    )

    _FALSE_FORMULAS = {
        "The", "We", "In", "An", "As", "At", "By", "Be", "Do", "Go",
        "He", "If", "Is", "It", "Me", "My", "No", "Of", "On", "Or",
        "So", "To", "Up", "Us", "We", "Am", "An", "Ba", "Ca", "Ga",
        "La", "Mo", "Nb", "Nd", "Ni", "Os", "Po", "Ra", "Rb", "Ru",
        "Sc", "Se", "Si", "Sn", "Sr", "Ta", "Te", "Ti", "Tl", "V",
        "W", "Xe", "Y", "Yb", "Zn", "Zr",
        "For", "But", "Not", "All", "Any", "Can", "Had", "Her",
        "His", "How", "Its", "May", "New", "Now", "Old", "One",
        "Our", "Out", "She", "The", "Their", "There", "These",
        "They", "This", "Those", "Two", "Use", "Was", "Way", "Were",
        "What", "When", "Where", "Which", "Who", "Why", "Will",
        "With", "Would", "You", "Your",
        "This", "That", "From", "Into", "Over", "Such", "Than",
        "Them", "Then", "These", "Those", "Very", "Also",
        "GPa", "MPa", "kPa", "eV", "meV", "keV", "GHz", "THz", "MHz",
        "nm", "cm", "mm", "pm", "fm", "K", "J", "mol", "atom",
        "DFT", "LED", "XRD", "PLD", "MAE", "ORR", "HER", "OER",
        "MOCVD", "FCC", "BCC", "HCP", "HEA", "NCM", "LLZO", "KNN",
        "YBCO", "LSMO", "PVD", "CVD", "SEM", "TEM", "AFM",
        "XPS", "EDS", "EELS", "ICP", "PL", "EL", "EQE",
        "UV", "IR", "NMR", "EPR", "ESR", "RAM",
        "ZT", "NCM811", "LCMO", "PZT", "BST", "BFO", "BTO",
        "LSAT", "STO", "LAO", "SZO", "PFO", "TO",
        "HEA", "MEA", "CIGS", "CZTS", "IGZO", "PVC",
        "MIT", "CDW", "AFM", "FM", "PM",
        "TCMs", "HECs", "UHTCs", "TMDCs", "TMDs", "MOFs",
        "AuNPs", "AgNPs", "CuNPs", "PtNPs",
        "COVID", "MERS", "SARS", "HIV", "DNA", "RNA",
        "LEDs", "OLEDs", "QLEDs", "FETs", "MEMS", "NEMS",
        "GO", "rGO", "CNTs", "SWCNTs", "MWCNTs",
        "PCE", "Voc", "Jsc",
        "ALD", "FZ", "CZ",
        "RBDs", "ACE2", "ML", "AI", "PV", "TE", "TMDC",
        "PANI", "PPy", "ETP", "LSCs", "CoV", "CO",
        "MWCNT", "SWCNT", "ITO", "FTO", "AZO",
        "PBS", "PCB", "PMMA", "PDMS", "PET", "PI",
        "PEO", "PP", "PE", "PS", "PVA", "PVC",
        "PU", "PA", "PC", "POM", "PPO", "PPS",
        "PTFE", "PVDF", "PTFE", "PHB", "PLA",
        "OPV", "OLED", "QLED", "LED",
        "DSC", "DSSC", "SC", "EC",
        "EM", "MW", "NW", "NP", "NS",
        "DFT", "B3LYP", "GGA", "LDA", "PBE",
        "FCC", "BCC", "HCP",
        "RdRp", "HDOCK", "NPs", "VIS", "NIR", "MIR",
        "PEDOT", "LIMNO", "FPPCE", "MX", "CS", "EMI",
        "UV", "IR", "RAM", "GHz", "THz", "MHz",
        "IGMSE", "SIMATS", "SPD", "MP", "ABO3",
        "GPTFF", "SVP", "TZVP", "TMR", "TMD", "TMDs",
        "PIMC", "GPUs", "HDNNP", "XAS", "XRPD", "MLIPs",
        "E4B", "RHEELS", "SXPS", "HAXPES", "CBO", "CFP",
        "III", "IV", "VI", "II", "VII", "VIII",
        "GNN", "CNN", "RNN", "LSTM", "GAN", "VAE",
        "BERT", "GPT", "LLM", "MLP", "SVM", "KNN",
        "PCA", "UMAP", "tSNE", "NLP", "CSP",
        "HEMs", "SPDs", "MLIP", "XRPDs",
        "DFT", "B3LYP", "GGA", "LDA", "PBE",
        "FCC", "BCC", "HCP",
        "HEA", "MEA", "CIGS", "CZTS", "IGZO",
        "MIT", "CDW", "AFM", "FM", "PM",
        "SEM", "TEM", "AFM", "XPS", "EDS", "EELS",
        "PL", "EL", "EQE", "UV", "IR", "NMR",
        "EPR", "ESR",
        "ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU",
        "VWX", "XYZ",
        "QM9", "G0W0",
        "CoV",
        "No",
        "PVC",
        "PCB",
        "PI",
        "PC",
        "PU",
        "PP",
        "PS",
        "CO",
    }

    _ERROR_KEYWORDS = frozenset({
        "mae", "mean absolute error", "rmse", "root mean square",
        "mse", "mean squared error", "std", "standard deviation",
        "uncertainty", "error", "deviation", "bias",
        "prediction error", "training error", "validation error",
        "confidence interval", "standard error",
    })

    @classmethod
    def _split_sentences(cls, text: str) -> List[str]:
        """将文本按句子切分.

        按句号、分号、换行切分，保留每个句子的完整上下文.
        这是 P0-3 修复的基础: 避免 "SnSe ZT=2.6 ... Bi2Te3 ZT=1.1" 跨句污染.

        关键: 必须保护数字中的小数点 (如 3.15, 2.6, 0.18),
        否则 'ZT=2.6' 会被切成 'ZT=2' 和 '6', 导致数值损坏.
        """
        protected = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)

        raw = re.split(r'[;.\n]+', protected)

        sentences = [
            s.strip().replace('<DOT>', '.')
            for s in raw
            if s.strip() and len(s.strip()) > 5
        ]
        return sentences

    @classmethod
    def extract(cls, text: str, paper_id: str = "", paper_title: str = "") -> List[CSPTriple]:
        """从文本中抽取 CSP 三元组.

        优先使用 LLM 抽取 (如果配置了 LLM API), 失败则降级为正则抽取.

        Args:
            text: 论文标题+摘要文本
            paper_id: 论文 ID (溯源)
            paper_title: 论文标题 (溯源)

        Returns:
            CSPTriple 列表
        """
        try:
            from .llm_extractor import get_llm_extractor
            llm_ext = get_llm_extractor()
            if llm_ext is not None:
                llm_triples = llm_ext.extract(text, paper_id=paper_id, paper_title=paper_title)
                if llm_triples:
                    return llm_triples
        except Exception:
            pass

        triples: List[CSPTriple] = []
        seen_keys: set = set()

        sentences = cls._split_sentences(text)

        if not sentences:
            sentences = [text]

        context_comps: List[str] = []

        for sent in sentences:
            sent_lower = sent.lower()

            comps = cls._extract_compositions(sent)
            props: List[Tuple[str, Optional[float], str]] = []
            structs: List[str] = []

            if comps:
                context_comps = comps
                structs = cls._extract_structures(sent_lower)
                props = cls._extract_properties(sent_lower, sent)
            else:
                props = cls._extract_properties(sent_lower, sent)
                if context_comps and props:
                    comps = context_comps
                    structs = cls._extract_structures(sent_lower)
                else:
                    continue
            

            for comp in comps[:2]:
                struct = structs[0] if structs else "unknown"

                if props:
                    for prop_name, prop_value, prop_unit in props:
                        triple = CSPTriple(
                            composition=comp,
                            structure=struct,
                            property_name=prop_name,
                            property_value=prop_value,
                            property_unit=prop_unit,
                            source_paper_id=paper_id,
                            source_paper_title=paper_title,
                            source_section="abstract",
                            confidence=0.7 if prop_value is not None else 0.4,
                        )
                        if triple.key not in seen_keys:
                            seen_keys.add(triple.key)
                            triples.append(triple)
                else:
                    if struct == "unknown":
                        continue
                    triple = CSPTriple(
                        composition=comp,
                        structure=struct,
                        property_name="general",
                        property_value=None,
                        source_paper_id=paper_id,
                        source_paper_title=paper_title,
                        source_section="abstract",
                        confidence=0.3,
                    )
                    if triple.key not in seen_keys:
                        seen_keys.add(triple.key)
                        triples.append(triple)

        triples = [
            t for t in triples
            if not (
                (t.structure == "unknown" and t.property_name == "general")
                or (t.structure == "unknown" and t.property_value is None)
                or (t.property_name == "general" and t.property_value is None)
            )
        ]

        return triples

    @classmethod
    def _parse_formula_elements(cls, formula: str) -> List[str]:
        """解析化学式，返回元素符号列表.

        如 'CsPbI3' → ['Cs', 'Pb', 'I']
        如 'La0.7Sr0.3MnO3' → ['La', 'Sr', 'Mn', 'O']
        如 'GPa' → ['G', 'Pa'] (Pa 不是有效元素)
        如 'DFT' → ['D', 'F', 'T'] (全部非有效元素)
        """
        return [
            elem for elem, _sub in cls._ELEMENT_SPLIT_PATTERN.findall(formula)
            if elem
        ]

    @classmethod
    def _validate_formula_elements(cls, formula: str) -> bool:
        """检查化学式中的所有元素符号是否都是有效的周期表元素.

        这是从根源上拦截非化学式缩写的核心方法:
        - 'DFT' → D 不是元素 → 拒绝
        - 'GPa' → G 不是元素 → 拒绝
        - 'MLP' → M 不是元素 → 拒绝
        - 'CsPbI3' → Cs, Pb, I 全部有效 → 通过
        - 'BaTiO3' → Ba, Ti, O 全部有效 → 通过
        """
        elements = cls._parse_formula_elements(formula)
        if len(elements) < 2:
            return False
        return all(e in cls._VALID_ELEMENTS for e in elements)

    _PROPERTY_CONTEXT_KEYWORDS: List[str] = [
        "bandgap", "band gap", "conductivity", "resistivity", "mobility",
        "dielectric", "piezoelectric", "ferroelectric", "curie",
        "magnetization", "coercivity", "seebeck", "thermoelectric",
        "zt", "figure of merit", "thermal conductivity", "thermal expansion",
        "hardness", "modulus", "yield strength", "tensile strength",
        "fracture", "density", "heat capacity", "debye",
        "superconducting", "superconductor", "critical temperature",
        "formation energy", "cohesive energy",
        "crystal", "structure", "phase", "alloy", "ceramic", "polymer",
        "composite", "semiconductor", "perovskite", "spinel",
        "thin film", "single crystal", "polycrystalline", "nanoparticle",
        "synthesis", "sintering", "annealing", "doping",
        "XRD", "XPS", "SEM", "TEM", "measurement",
        "shielding", "shielding effectiveness", "EMI shielding",
        "foam", "sponge", "aerogel",
        "absorption", "reflection", "transmission",
    ]

    @classmethod
    def _has_property_context(cls, text: str) -> bool:
        """检查文本中是否包含材料性能关键词 (用于无数字化学式的上下文校验).

        当 SnSe 这样的无下标化学式通过元素校验但不在白名单时,
        如果同句中有性能关键词 (如 ZT, bandgap, conductivity),
        则判定为材料科学上下文, 允许通过.
        """
        text_lower = text.lower()
        for kw in cls._PROPERTY_CONTEXT_KEYWORDS:
            if kw.lower() in text_lower:
                return True
        return False

    @classmethod
    def _extract_compositions(cls, text: str) -> List[str]:
        """从文本中抽取化学式.

        预处理流程:
        1. Unicode 下标归一化 (₀₁₂₃₄₅₆₇₈₉ → 0123456789)
           — 修复 CsPbI₃ 被截断为 CsPbI 的根因
        2. LaTeX 下标变体归一化
           — $_{3}$, $_3$, _{3}, \textsubscript{3}, <sub>3</sub> → 3
        3. 元素周期表校验
           — 从根源上拦截 DFT, GPa, MLP 等非化学式缩写
        4. 无数字化学式上下文校验 (P0-3 修复)
           — SnSe 等无下标二元化合物, 通过元素校验 + 同句有性能关键词 → 允许
        """
        text = text.translate(cls._UNICODE_SUBSCRIPT_MAP)
        text = re.sub(r'\$_\{([^}]+)\}\$', r'\1', text)           # $_{3}$ → 3
        text = re.sub(r'\$_([^$]+)\$', r'\1', text)               # $_3$ → 3
        text = re.sub(r'_\{([^}]+)\}', r'\1', text)                # _{3} → 3
        text = re.sub(r'\\textsubscript\{([^}]+)\}', r'\1', text)  # \textsubscript{3} → 3
        text = re.sub(r'<sub>([^<]+)</sub>', r'\1', text)          # <sub>3</sub> → 3
        text = re.sub(r'\{([^}]+)\}', r'\1', text)                # {3} → 3

        matches = cls._FORMULA_PATTERN.findall(text)
        result: List[str] = []
        seen: set = set()

        for m in matches:
            m_clean = m.strip()
            if len(m_clean) < 2 or len(m_clean) > 20:
                continue
            if m_clean in cls._FALSE_FORMULAS:
                continue
            if not cls._validate_formula_elements(m_clean):
                continue
            if not any(c.isdigit() for c in m_clean) and m_clean not in cls._DIGITLESS_WHITELIST:
                if not cls._has_property_context(text):
                    continue
            if m_clean.isalpha() and len(m_clean) > 4:
                consecutive_lower = 0
                is_english = False
                for c in m_clean:
                    if c.islower():
                        consecutive_lower += 1
                        if consecutive_lower >= 3:
                            is_english = True
                            break
                    else:
                        consecutive_lower = 0
                if is_english:
                    continue

            if m_clean not in seen:
                seen.add(m_clean)
                result.append(m_clean)

        return result

    @classmethod
    def _extract_structures(cls, text_lower: str) -> List[str]:
        """从文本中抽取晶体结构类型.

        修复1: 大小写不敏感 — STRUCTURE_TYPES 含混合大小写条目 (如 half-Heusler, NASICON),
               需用 struct.lower() 在 text_lower 中匹配.
        修复2: 长名称优先 — 避免 'perovskite' 子串匹配到 'antiperovskite',
               按长度降序检查.
        """
        result: List[str] = []
        for struct in sorted(cls.STRUCTURE_TYPES, key=len, reverse=True):
            if struct.lower() in text_lower:
                result.append(struct)
        return result

    @classmethod
    def _extract_properties(cls, text_lower: str, original_text: str) -> List[Tuple[str, Optional[float], str]]:
        """从文本中抽取性能名和值.

        Returns:
            [(property_name, value_or_None, unit), ...]
        """
        results: List[Tuple[str, Optional[float], str]] = []
        seen_props: set = set()

        for prop_name, (aliases, default_unit) in cls.PROPERTY_DICT.items():
            if prop_name in seen_props:
                continue

            for alias in aliases:
                alias_lower = alias.lower()
                if re.search(r'\b' + re.escape(alias_lower) + r'\b', text_lower):
                    value, unit = cls._extract_value_near(text_lower, alias_lower, default_unit)
                    if value is not None and unit:
                        if not cls._is_unit_compatible(unit, default_unit):
                            value = None
                            unit = default_unit
                    results.append((prop_name, value, unit or default_unit))
                    seen_props.add(prop_name)
                    break

        return results

    @staticmethod
    def _is_unit_compatible(extracted_unit: str, expected_unit: str) -> bool:
        """检查提取到的单位是否与预期单位兼容.

        全领域扩展: 覆盖 16 大领域的单位组.
        """
        eu = extracted_unit.lower().strip()
        exp = expected_unit.lower().strip()
        if not exp:
            return True
        if not eu:
            return True
        unit_groups = [
            {"ev", "mev", "kev", "ev/atom"},
            {"s/cm"},
            {"ω·cm", "ωcm", "ohm·cm"},
            {"cm²/vs"},
            {"cm⁻³"},
            {"cm³/c"},
            {"k"},
            {"oe"},
            {"t"},
            {"pc/n"},
            {"μc/m²k"},
            {"μc/cm²"},
            {"μv/k"},
            {"μw/mk²"},
            {"gpa"},
            {"mpa"},
            {"mpa·m"},
            {"kpa"},
            {"kv/mm"},
            {"w/mk", "w/m·k"},
            {"j/molk"},
            {"ppm/k"},
            {"mm²/s"},
            {"cm²/s"},
            {"mah/g"},
            {"wh/kg"},
            {"w/kg"},
            {"v"},
            {"a/cm²"},
            {"ma/cm²"},
            {"mv/dec"},
            {"cm⁻¹"},
            {"nm"},
            {"μm"},
            {"m²/g"},
            {"cm³/g"},
            {"mm/year"},
            {"barn"},
            {"hz"},
            {"s⁻¹"},
            {"%", "wt%"},
            {"g/cm³"},
            {"g/mol"},
            {"g/10min"},
            {"mol/m³"},
            {"lm/w"},
            {"cd/m²"},
            {"db"},
            {"å"},
            {"°"},
            {"mmol/g"},
            {"mg/n·m"},
            {"mv/cm·oe"},
            {"ppm"},
        ]
        for group in unit_groups:
            if eu in group and exp in group:
                return True
            if eu in group and exp not in group:
                return False
            if exp in group and eu not in group:
                return False
        return True

    @classmethod
    def _extract_value_near(
        cls, text_lower: str, prop_alias: str, default_unit: str,
    ) -> Tuple[Optional[float], str]:
        """在性能名附近查找数值.

        P0-1 修复: 从性能别名之后开始搜索 (idx + len(alias)),
                   避免别名自身含数字 (如 d33, Hc) 被误匹配.
        P0-2 修复: 跳过 MAE/error/uncertainty 前缀的数值,
                   这些是误差值不是性能值.
        """
        idx = text_lower.find(prop_alias)
        if idx < 0:
            return None, default_unit

        start = idx + len(prop_alias)
        search_region = text_lower[start:start + 150]

        for value_match in cls._VALUE_PATTERN.finditer(search_region):
            match_start = value_match.start()
            val_str = value_match.group(1)
            unit = value_match.group(2) or default_unit

            before_region = search_region[max(0, match_start - 30):match_start]
            is_error_value = False
            for ekw in cls._ERROR_KEYWORDS:
                if ekw in before_region:
                    is_error_value = True
                    break

            if is_error_value:
                continue

            try:
                value = float(val_str)
                return value, unit
            except ValueError:
                continue

        return None, default_unit


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

class MaterialHypothesisTemplates:
    """材料科学假设模板 — 可证伪的构效关系预测.

    赛题: "产出具有可证伪性的科学发现——包括
           材料_性质关联、隐藏知识连接、新材料设计假设"

    四类假设:
        1. structure_property: 结构-性能关联预测
           "X 材料在 Y 结构下的 P 性能预测为 Z"
        2. composition_transfer: 组分替换迁移
           "将 A 中元素 a 替换为 b，P 性能可能从 V1 变为 V2"
        3. process_optimization: 工艺优化预测
           "工艺 G 在材料 M 上的 P 最优区间预测为 [V1, V2]"
        4. hidden_link: 隐藏知识连接
           "论文 A 的 P1≈V1 与论文 B 的 P2≈V2 可能通过 C 关联"
    """

    TEMPLATES: Dict[str, List[str]] = {
        "structure_property": [
            "具有{structure_a}结构的{composition_a}，其{property}预测约为{predicted_value}{unit}；"
            "参考{composition_b}在相同结构下的已知值{known_value}{unit}，"
            "两者差异可能源于组分差异",

            "{composition_a}在{structure_a}结构下的{property}预测值为{predicted_value}{unit}，"
            "该预测基于{composition_b}({structure_b})的已知{property}={known_value}{unit}和结构类比推理",

            "基于{composition_a}与{composition_b}的结构相似性({structure_a})，"
            "预测{composition_a}的{property}≈{predicted_value}{unit}"
            "（{composition_b}的{property}={known_value}{unit}）",
        ],

        "composition_transfer": [
            "将{composition_a}中的{element_a}替换为{element_b}，"
            "在{structure}结构下{property}可能从{known_value}{unit}变为约{predicted_value}{unit}",

            "基于{composition_a}→{composition_b}的组分替换策略，"
            "预测{property}的变化趋势: {known_value}→{predicted_value}{unit}"
            "（结构保持{structure}）",
        ],

        "process_optimization": [
            "{process}工艺在{composition}({structure})上的{property}最优区间"
            "预测为{predicted_value}{unit}附近，"
            "当前文献仅报道了{known_value}{unit}",

            "基于{composition_a}的{process}工艺数据，"
            "预测同类结构{composition_b}的{property}最优工艺窗口"
            "在{predicted_value}{unit}附近",
        ],

        "hidden_link": [
            "《{title_a}》报道的{composition_a}({structure_a})的{property_a}≈{known_value}{unit_a}"
            "与《{title_b}》的{composition_b}({structure_b})的{property_b}≈{value_b}{unit_b}"
            "可能通过'{bridge}'关联，统一模型预测{composition_a}的{property_b}≈{predicted_value}{unit_b}",

            "发现隐藏关联: {composition_a}的{property_a}与{composition_b}的{property_b}"
            "通过'{bridge}'桥接，预测{composition_b}的{property_a}≈{predicted_value}{unit_a}",
        ],

        "gap_filling": [
            "本系统当前语料中未见{composition_a}在{structure_a}结构下的{property}报道，"
            "基于同类结构{composition_b}的{property}={known_value}{unit}，"
            "类比预测{composition_a}的{property}≈{predicted_value}{unit}",
            "数据空白: {composition_a}和{composition_b}均被研究，"
            "但{composition_a}在{structure_b}结构下的{property}在本系统语料中未见，"
            "类比预测≈{predicted_value}{unit}",
        ],
    }

    TYPE_LABELS: Dict[str, str] = {
        "structure_property": "构效关系预测",
        "composition_transfer": "组分迁移预测",
        "process_optimization": "工艺优化预测",
        "hidden_link": "隐藏知识连接",
        "gap_filling": "文献空白填补",
    }

    @classmethod
    def get_template(cls, hypo_type: str, rng: Any = None) -> Optional[str]:
        """获取随机模板."""
        templates = cls.TEMPLATES.get(hypo_type, [])
        if not templates:
            return None
        if rng:
            return rng.choice(templates)
        import random
        return random.choice(templates)

    @classmethod
    def all_types(cls) -> List[str]:
        return list(cls.TEMPLATES.keys())


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

MATERIAL_CORPUS: List[Dict[str, Any]] = [
    {
        "title": "Defect Passivation in Perovskite Solar Cells via Lewis Base Treatment",
        "abstract": "Organic-inorganic perovskite CH3NH3PbI3 solar cells achieve power conversion "
                    "efficiency of 24.8% after Lewis base surface passivation. The bandgap of "
                    "CH3NH3PbI3 is 1.55 eV with tetragonal perovskite structure. Defect density "
                    "reduced by 70% measured by deep-level transient spectroscopy.",
        "categories": ["cond-mat.mtrl-sci", "physics.chem-ph"],
        "keywords": ["perovskite", "solar cell", "defect passivation", "bandgap",
                     "CH3NH3PbI3", "Lewis base", "photovoltaic"],
        "year": 2024,
    },
    {
        "title": "Mixed-Cation Perovskite FAxMA1-xPbI3 for Stable Solar Cells",
        "abstract": "Formamidinium-methylammonium mixed-cation perovskite FA0.85MA0.15PbI3 shows "
                    "improved stability with bandgap of 1.53 eV. The perovskite structure retains "
                    "at 85°C for 1000 hours. Power conversion efficiency reaches 23.2%. "
                    "Formation energy calculated by DFT is -2.8 eV/atom.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["perovskite", "mixed cation", "stability", "bandgap",
                     "FA0.85MA0.15PbI3", "DFT", "formation energy"],
        "year": 2024,
    },
    {
        "title": "CsPbBr3 Perovskite Quantum Dots for Light-Emitting Diodes",
        "abstract": "Cesium lead bromide CsPbBr3 quantum dots exhibit photoluminescence quantum "
                    "yield of 90%. The cubic perovskite structure shows bandgap of 2.3 eV. "
                    "LED external quantum efficiency reaches 12.5%. Synthesized by hot-injection "
                    "method at 170°C.",
        "categories": ["cond-mat.mtrl-sci", "physics.chem-ph"],
        "keywords": ["perovskite", "quantum dots", "LED", "CsPbBr3",
                     "bandgap", "photoluminescence", "cubic"],
        "year": 2023,
    },

    {
        "title": "Olivine LiFePO4 Cathode with Enhanced Rate Capability",
        "abstract": "Olivine-structured LiFePO4 cathode material achieves discharge capacity of "
                    "165 mAh/g at 1C rate. Electronic conductivity improved to 1.2e-2 S/cm "
                    "after carbon coating. The olivine structure shows thermal stability up to "
                    "300°C. DFT calculation gives formation energy of -2.45 eV/atom.",
        "categories": ["cond-mat.mtrl-sci", "physics.chem-ph"],
        "keywords": ["LiFePO4", "olivine", "cathode", "battery", "conductivity",
                     "formation energy", "DFT", "lithium-ion"],
        "year": 2024,
    },
    {
        "title": "LiNi0.8Co0.1Mn0.1O2 Layered Cathode for High-Energy Batteries",
        "abstract": "Layered LiNi0.8Co0.1Mn0.1O2 (NCM811) cathode delivers capacity of 200 mAh/g. "
                    "The hexagonal layered structure shows cation mixing of 2.1%. Thermal "
                    "conductivity measured at 1.5 W/mK. Electronic conductivity is 5.0e-3 S/cm. "
                    "Calcined at 750°C for 12 hours.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["NCM811", "layered", "cathode", "battery", "conductivity",
                     "thermal conductivity", "calcination", "LiNi0.8Co0.1Mn0.1O2"],
        "year": 2024,
    },
    {
        "title": "Garnet Li7La3Zr2O12 Solid Electrolyte for All-Solid-State Batteries",
        "abstract": "Garnet-structured Li7La3Zr2O12 (LLZO) solid electrolyte achieves ionic "
                    "conductivity of 5.7e-4 S/cm at room temperature. The cubic garnet structure "
                    "is stable up to 900°C. Ta-doped Li6.4La3Zr1.4Ta0.6O12 shows improved "
                    "conductivity of 8.0e-4 S/cm. Sintered at 1100°C for 6 hours.",
        "categories": ["cond-mat.mtrl-sci", "physics.chem-ph"],
        "keywords": ["LLZO", "garnet", "solid electrolyte", "ionic conductivity",
                     "sintering", "Li7La3Zr2O12", "cubic"],
        "year": 2023,
    },

    {
        "title": "Bi2Te3 Thermoelectric Material with ZT=1.2 at Room Temperature",
        "abstract": "Rhombohedral Bi2Te3 achieves thermoelectric figure of merit ZT=1.2 at 300K. "
                    "Seebeck coefficient measured at -210 μV/K. Electrical conductivity is "
                    "1.1e5 S/cm. Thermal conductivity reduced to 1.5 W/mK by nanostructuring. "
                    "Rhombohedral structure confirmed by XRD.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["Bi2Te3", "thermoelectric", "ZT", "Seebeck",
                     "conductivity", "thermal conductivity", "rhombohedral"],
        "year": 2024,
    },
    {
        "title": "SnSe Thermoelectric with Record ZT=2.8 at 923K",
        "abstract": "Layered SnSe single crystal achieves ZT=2.8 along b-axis at 923K. "
                    "Seebeck coefficient reaches 510 μV/K. Thermal conductivity drops to "
                    "0.37 W/mK at high temperature. The orthorhombic structure undergoes "
                    "phase transition at 750K. Hall mobility measured at 15 cm²/Vs.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["SnSe", "thermoelectric", "ZT", "Seebeck",
                     "thermal conductivity", "orthorhombic", "mobility", "layered"],
        "year": 2023,
    },

    {
        "title": "BaTiO3 Ferroelectric with Enhanced Piezoelectric Coefficient",
        "abstract": "Tetragonal BaTiO3 ceramic shows piezoelectric coefficient d33=190 pC/N. "
                    "Dielectric constant reaches 1500 at Curie temperature of 403K. "
                    "Sintered at 1300°C for 2 hours. Domain engineering further enhances d33 "
                    "to 350 pC/N. Young's modulus measured at 145 GPa.",
        "categories": ["cond-mat.mtrl-sci", "physics.chem-ph"],
        "keywords": ["BaTiO3", "ferroelectric", "piezoelectric", "perovskite",
                     "dielectric", "Curie temperature", "sintering", "tetragonal"],
        "year": 2024,
    },
    {
        "title": "Lead-Free KNN Piezoelectric Ceramics near Morphotropic Phase Boundary",
        "abstract": "(K0.5Na0.5)NbO3 (KNN) ceramics near morphotropic phase boundary show "
                    "d33=300 pC/N. Orthorhombic to tetragonal transition at 200°C. "
                    "Dielectric constant of 1200 measured at 1kHz. Sintered at 1080°C. "
                    "Hardness measured at 4.8 GPa.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["KNN", "piezoelectric", "lead-free", "perovskite",
                     "morphotropic", "dielectric", "hardness", "orthorhombic"],
        "year": 2023,
    },

    {
        "title": "Pt3Ni Alloy Catalyst for Oxygen Reduction Reaction",
        "abstract": "Pt3Ni alloy catalyst shows 5.1x enhancement in oxygen reduction reaction "
                    "activity vs commercial Pt/C. Face-centered cubic structure confirmed by "
                    "XRD. Formation energy calculated by DFT: -0.85 eV/atom. The Pt-skin "
                    "surface has d-band center of -2.75 eV. Synthesized by sol-gel method.",
        "categories": ["cond-mat.mtrl-sci", "physics.chem-ph"],
        "keywords": ["Pt3Ni", "catalyst", "alloy", "ORR", "formation energy",
                     "DFT", "cubic", "sol-gel"],
        "year": 2024,
    },
    {
        "title": "MoS2 Edge Sites for Hydrogen Evolution Reaction Catalysis",
        "abstract": "Hexagonal MoS2 nanosheets show hydrogen evolution reaction overpotential of "
                    "180 mV at 10 mA/cm². The hexagonal layered structure has bandgap of "
                    "1.8 eV (monolayer). Formation energy: -1.2 eV/atom. Conductivity improved "
                    "by S-vacancy engineering to 2.5e-3 S/cm.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["MoS2", "catalyst", "HER", "bandgap", "formation energy",
                     "conductivity", "hexagonal", "layered"],
        "year": 2023,
    },

    {
        "title": "GaN Wide Bandgap Semiconductor for Power Electronics",
        "abstract": "Wurtzite GaN shows bandgap of 3.4 eV. Electron mobility of 2000 cm²/Vs "
                    "measured at room temperature. Thermal conductivity of 130 W/mK. "
                    "Breakdown voltage of 3.3 MV/cm. Grown by MOCVD at 1050°C on sapphire "
                    "substrate. Young's modulus: 295 GPa.",
        "categories": ["cond-mat.mtrl-sci", "physics.comp-ph"],
        "keywords": ["GaN", "semiconductor", "bandgap", "mobility",
                     "thermal conductivity", "wurtzite", "MOCVD"],
        "year": 2024,
    },
    {
        "title": "SiC Power Device with Figure of Merit 10x Superior to Si",
        "abstract": "4H-SiC hexagonal polytype has bandgap of 3.26 eV. Thermal conductivity "
                    "of 370 W/mK, three times that of silicon. Critical electric field of "
                    "2.8 MV/cm. Electron mobility: 950 cm²/Vs. Sublimation growth at 2300°C. "
                    "Hardness: 28 GPa.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["SiC", "semiconductor", "bandgap", "thermal conductivity",
                     "mobility", "hexagonal", "hardness"],
        "year": 2023,
    },

    {
        "title": "Nd2Fe14B Permanent Magnet with Energy Product 52 MGOe",
        "abstract": "Tetragonal Nd2Fe14B permanent magnet achieves maximum energy product of "
                    "52 MGOe. Saturation magnetization: 1.6 T. Coercivity: 12 kOe. "
                    "Curie temperature: 585K. Sintered at 1080°C. Domain wall energy: "
                    "1.7 mJ/m². The tetragonal structure is confirmed by neutron diffraction.",
        "categories": ["cond-mat.mtrl-sci", "cond-mat.mes-hall"],
        "keywords": ["Nd2Fe14B", "permanent magnet", "magnetization", "coercivity",
                     "Curie temperature", "tetragonal", "sintering"],
        "year": 2024,
    },
    {
        "title": "La0.7Sr0.3MnO3 Colossal Magnetoresistance Manganite",
        "abstract": "Rhombohedral La0.7Sr0.3MnO3 shows colossal magnetoresistance of -60% at "
                    "7T. Curie temperature: 360K. Saturation magnetization: 3.7 μB/Mn. "
                    "The perovskite structure has tolerance factor of 0.91. Conductivity: "
                    "250 S/cm. Epitaxial thin film grown by PLD at 750°C.",
        "categories": ["cond-mat.mtrl-sci", "cond-mat.supr-con"],
        "keywords": ["LSMO", "magnetoresistance", "manganite", "perovskite",
                     "Curie temperature", "magnetization", "conductivity", "rhombohedral"],
        "year": 2023,
    },

    {
        "title": "CoCrFeMnNi High-Entropy Alloy with Excellent Cryogenic Toughness",
        "abstract": "Face-centered cubic CoCrFeMnNi Cantor alloy shows tensile strength of "
                    "1.3 GPa at 77K. Fracture toughness exceeds 200 MPa√m. Young's modulus: "
                    "202 GPa. Formation energy: -0.15 eV/atom. Single-phase FCC structure "
                    "stable from 4K to melting point. Arc-melted and homogenized at 1100°C.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["high-entropy alloy", "CoCrFeMnNi", "Cantor alloy", "FCC",
                     "toughness", "Young's modulus", "formation energy"],
        "year": 2024,
    },
    {
        "title": "AlCoCrFeNi High-Entropy Alloy with BCC Structure and High Hardness",
        "abstract": "Body-centered cubic AlCoCrFeNi HEA achieves Vickers hardness of 5.2 GPa. "
                    "Yield strength: 1.8 GPa. Young's modulus: 175 GPa. The BCC structure "
                    "shows spinodal decomposition at 600°C. Thermal conductivity: 15 W/mK. "
                    "Cohesive energy: -4.2 eV/atom.",
        "categories": ["cond-mat.mtrl-sci"],
        "keywords": ["high-entropy alloy", "AlCoCrFeNi", "BCC", "hardness",
                     "yield strength", "thermal conductivity", "cohesive energy"],
        "year": 2023,
    },

    {
        "title": "Machine Learning Prediction of Bandgap for Inorganic Crystals",
        "abstract": "Graph neural network predicts bandgap of inorganic crystals with MAE of "
                    "0.18 eV on 24000 DFT-calculated compounds. The model uses composition "
                    "and crystal structure features. Perovskite BaTiO3 predicted bandgap: "
                    "3.15 eV (DFT: 3.20 eV). Olivine LiFePO4 predicted: 0.35 eV (DFT: 0.32 eV).",
        "categories": ["cond-mat.mtrl-sci", "cs.LG"],
        "keywords": ["machine learning", "bandgap", "prediction", "graph neural network",
                     "DFT", "perovskite", "BaTiO3", "LiFePO4"],
        "year": 2024,
    },
    {
        "title": "High-Throughput Screening of Thermoelectric Materials via DFT Database",
        "abstract": "High-throughput DFT screening of 48000 compounds identifies 23 promising "
                    "thermoelectric candidates. SnSe predicted ZT=2.6 (experimental: 2.8). "
                    "Bi2Te3 predicted ZT=1.1 (experimental: 1.2). Seebeck coefficient "
                    "prediction MAE: 18 μV/K. Formation energy filter: <-0.5 eV/atom.",
        "categories": ["cond-mat.mtrl-sci", "cs.LG"],
        "keywords": ["high-throughput", "thermoelectric", "DFT", "screening",
                     "ZT", "Seebeck", "formation energy", "SnSe", "Bi2Te3"],
        "year": 2024,
    },

    {
        "title": "Hydride Superconductor LaH10 with Tc=250K under High Pressure",
        "abstract": "Cubic LaH10 hydride superconductor achieves critical temperature Tc=250K "
                    "at 170 GPa. The cubic structure is confirmed by XRD. Formation energy: "
                    "-0.45 eV/atom at 170 GPa. Debye temperature: 1850K. Magnetization "
                    "measurement confirms Meissner effect below Tc.",
        "categories": ["cond-mat.supr-con", "cond-mat.mtrl-sci"],
        "keywords": ["LaH10", "superconductor", "critical temperature", "high pressure",
                     "formation energy", "Debye temperature", "cubic"],
        "year": 2024,
    },
    {
        "title": "Cuprate Superconductor YBCO with Tc=92K and Application in Coated Conductors",
        "abstract": "YBa2Cu3O7 (YBCO) cuprate superconductor shows Tc=92K. The orthorhombic "
                    "perovskite structure has critical current density of 3 MA/cm² at 77K. "
                    "Coherent length: 1.5 nm. London penetration depth: 200 nm. Thin film "
                    "deposited by PLD at 800°C. Thermal conductivity: 10 W/mK.",
        "categories": ["cond-mat.supr-con", "cond-mat.mtrl-sci"],
        "keywords": ["YBCO", "superconductor", "YBa2Cu3O7", "critical temperature",
                     "perovskite", "orthorhombic", "thermal conductivity"],
        "year": 2023,
    },
]


# ═══════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════

class MaterialPhysicsValidator:
    """材料物理常识验证器.

    解决三类问题:
        1. 晶体化学可行性: 化合物不能存在于单元素结构 (如金刚石)
        2. 已知规律识别: 层状结构→低热导率等通用规律不算新发现
        3. 相态一致性: YBCO正交相=超导, 四方相=非超导
        4. 数值量级校验: 磁化强度单位混淆 (T vs emu/g)
    """

    STRUCTURE_STOICHIOMETRY_RULES: Dict[str, Dict[str, Any]] = {
        "diamond":      {"min_elements": 1, "max_elements": 1, "desc": "金刚石结构仅适用于单元素 (C/Si/Ge)"},
        "graphite":     {"min_elements": 1, "max_elements": 1, "desc": "石墨结构仅适用于单元素 (C)"},
        "rocksalt":     {"min_elements": 2, "max_elements": 2, "desc": "岩盐结构为1:1二元化合物"},
        "zincblende":   {"min_elements": 2, "max_elements": 2, "desc": "闪锌矿结构为1:1二元化合物"},
        "wurtzite":     {"min_elements": 2, "max_elements": 2, "desc": "纤锌矿结构为1:1二元化合物"},
        "rutile":       {"min_elements": 2, "max_elements": 2, "desc": "金红石结构为1:2二元化合物"},
        "fluorite":     {"min_elements": 2, "max_elements": 2, "desc": "萤石结构为1:2二元化合物"},
        "perovskite":   {"min_elements": 2, "max_elements": 4, "desc": "钙钛矿ABO3结构"},
        "spinel":       {"min_elements": 2, "max_elements": 3, "desc": "尖晶石AB2O4结构"},
        "garnet":       {"min_elements": 3, "max_elements": 4, "desc": "石榴石A3B2C3O12结构"},
        "ilmenite":     {"min_elements": 3, "max_elements": 3, "desc": "钛铁矿ABO3结构"},
        "scheelite":    {"min_elements": 2, "max_elements": 3, "desc": "白钨矿ABO4结构"},
        "pyrochlore":   {"min_elements": 3, "max_elements": 3, "desc": "烧绿石A2B2O7结构"},
        "hexagonal":    None,
        "cubic":        None,
        "tetragonal":   None,
        "orthorhombic": None,
        "monoclinic":   None,
        "trigonal":     None,
        "amorphous":    None,
        "layered":      None,
    }

    KNOWN_PRINCIPLES: Dict[tuple, str] = {
        ("layered", "thermal_conductivity"): "层状结构普遍具有低晶格热导率 (层间弱相互作用+强非谐性), 属热电领域共识",
        ("layered", "conductivity"): "层状结构的电导率具有显著的层内/层间各向异性, 属已知规律",
        ("perovskite", "dielectric_constant"): "钙钛矿结构普遍具有高介电常数, 属铁电领域共识",
        ("perovskite", "tc"): "钙钛矿铁电体的居里温度与容忍因子相关, 属已知规律",
        ("amorphous", "thermal_conductivity"): "非晶态材料热导率普遍低于晶态, 属声子散射共识",
        ("hexagonal", "bandgap"): "六方结构半导体的带隙与层厚相关, 属二维材料共识",
    }

    PHASE_RULES: Dict[str, Dict[str, Any]] = {
        "YBa2Cu3O7": {
            "orthorhombic": {"tc": True, "superconducting": True, "desc": "正交相=富氧超导相"},
            "tetragonal":   {"tc": False, "superconducting": False, "desc": "四方相=缺氧非超导相"},
            "perovskite":   {"note": "YBCO是层状缺氧钙钛矿衍生结构, 非标准钙钛矿"},
        },
        "YBCO": {
            "orthorhombic": {"tc": True, "superconducting": True, "desc": "正交相=富氧超导相"},
            "tetragonal":   {"tc": False, "superconducting": False, "desc": "四方相=缺氧非超导相"},
        },
    }

    PROPERTY_RANGES: Dict[str, tuple] = {
        "bandgap":             (0.0, 12.0, "eV"),
        "conductivity":        (1e-10, 1e8, "S/cm"),
        "resistivity":         (1e-12, 1e12, "Ω·cm"),
        "mobility":            (0.0, 1e6, "cm²/Vs"),
        "work_function":       (1.0, 8.0, "eV"),
        "tc":                  (0.0, 2000.0, "K"),
        "tn":                  (0.0, 2000.0, "K"),
        "magnetization":       (0.0, 300.0, "emu/g"),
        "coercivity":          (0.0, 1e6, "Oe"),
        "dielectric_constant": (1.0, 1e5, ""),
        "dielectric_loss":     (0.0, 10.0, ""),
        "piezoelectric_coeff": (0.0, 2000.0, "pC/N"),
        "spontaneous_polarization": (0.0, 200.0, "μC/cm²"),
        "remanent_polarization": (0.0, 150.0, "μC/cm²"),
        "seebeck":             (-1000.0, 1000.0, "μV/K"),
        "zt":                  (0.0, 10.0, ""),
        "power_factor":        (0.0, 1e5, "μW/mK²"),
        "superconducting_tc":  (0.0, 300.0, "K"),
        "critical_field":      (0.0, 200.0, "T"),
        "youngs_modulus":      (0.001, 1500.0, "GPa"),
        "bulk_modulus":        (0.001, 2000.0, "GPa"),
        "shear_modulus":       (0.001, 1000.0, "GPa"),
        "hardness":            (0.001, 120.0, "GPa"),
        "yield_strength":      (1.0, 5000.0, "MPa"),
        "tensile_strength":    (1.0, 8000.0, "MPa"),
        "compressive_strength": (0.1, 5000.0, "MPa"),
        "flexural_strength":   (1.0, 3000.0, "MPa"),
        "fracture_toughness":  (0.1, 200.0, "MPa·m"),
        "fatigue_strength":    (1.0, 2000.0, "MPa"),
        "elongation":          (0.0, 1000.0, "%"),
        "poisson_ratio":       (-1.0, 0.5, ""),
        "density":             (0.001, 25.0, "g/cm³"),
        "thermal_conductivity": (0.0, 5000.0, "W/mK"),
        "specific_heat":       (0.0, 2000.0, "J/molK"),
        "thermal_expansion":   (-50.0, 500.0, "ppm/K"),
        "debye_temp":          (0.0, 2000.0, "K"),
        "melting_point":       (0.0, 5000.0, "K"),
        "glass_transition":    (0.0, 1500.0, "K"),
        "formation_energy":    (-10.0, 5.0, "eV/atom"),
        "cohesive_energy":     (-20.0, 0.0, "eV/atom"),
        "activation_energy":   (0.0, 10.0, "eV"),
        "corrosion_rate":      (0.0, 100.0, "mm/year"),
        "specific_capacity":   (0.0, 5000.0, "mAh/g"),
        "voltage":             (0.0, 5.0, "V"),
        "energy_density":      (0.0, 10000.0, "Wh/kg"),
        "power_density":       (0.0, 1e6, "W/kg"),
        "coulombic_efficiency": (0.0, 100.0, "%"),
        "overpotential":       (-3.0, 3.0, "V"),
        "tafel_slope":         (0.0, 1000.0, "mV/dec"),
        "faradaic_efficiency": (0.0, 100.0, "%"),
        "refractive_index":    (1.0, 5.0, ""),
        "plqy":                (0.0, 100.0, "%"),
        "emission_wavelength": (200.0, 5000.0, "nm"),
        "pce":                 (0.0, 50.0, "%"),
        "jsc":                 (0.0, 100.0, "mA/cm²"),
        "voc":                 (0.0, 5.0, "V"),
        "breakdown_strength":  (0.0, 2000.0, "kV/mm"),
        "contact_angle":       (0.0, 180.0, "°"),
        "friction_coeff":      (0.0, 2.0, ""),
        "hydrogen_capacity":   (0.0, 25.0, "wt%"),
    }

    UNIT_CONVERSION: Dict[str, float] = {
        "GPa→MPa": 1000.0,
        "MPa→GPa": 0.001,
        "kPa→MPa": 0.001,
        "Pa→MPa": 1e-6,
        "MPa→kPa": 1000.0,
        "GPa→kPa": 1e6,
        "kPa→GPa": 1e-6,
        "Pa→GPa": 1e-9,
        "g/cm³→kg/m³": 1000.0,
        "kg/m³→g/cm³": 0.001,
        "eV→meV": 1000.0,
        "meV→eV": 0.001,
        "keV→eV": 1000.0,
        "eV→keV": 0.001,
        "W/cmK→W/mK": 100.0,
        "W/mK→W/cmK": 0.01,
        "nm→Å": 10.0,
        "Å→nm": 0.1,
        "MPa·m→Pa·m": 1e6,
    }

    UNIT_COMPAT_GROUPS: Dict[str, List[str]] = {
        "pressure_stress": ["GPa", "MPa", "kPa", "Pa", "GPa·m"],
        "density": ["g/cm³", "kg/m³"],
        "temperature": ["K", "°C", "C"],
        "energy": ["eV", "meV", "keV", "J"],
        "thermal_cond": ["W/mK", "W/cmK", "W/m·K"],
        "electric_field": ["kV/mm", "V/m", "V/cm"],
        "frequency": ["Hz", "kHz", "MHz", "GHz", "THz"],
        "length": ["nm", "μm", "mm", "cm", "m", "Å"],
        "angle": ["°", "deg", "rad"],
        "dimensionless": ["", "%", "wt%"],
    }

    @classmethod
    def count_elements(cls, composition: str) -> int:
        """从化学式中估计元素种类数.

        化学式中每个元素符号以大写字母开头:
        如 BaTiO3 → B, T, O → 3 种元素
        如 Si → S → 1 种元素
        如 Nd2Fe14B → N, F, B → 3 种元素
        如 LiFePO4 → L, F, P → 3 种元素 (O 被过滤因为可能被误计)
        注意: 这只是启发式估计, 不完全精确
        """
        count = 0
        for c in composition:
            if c.isupper():
                count += 1
        return max(1, count)

    @classmethod
    def validate_structure_compatibility(
        cls, composition: str, structure: str,
    ) -> List[str]:
        """验证化学式与晶体结构的物理可行性.

        Returns:
            违规列表, 空列表=可行
        """
        violations: List[str] = []

        if not composition or not structure:
            return violations

        if structure == "unknown":
            return violations

        rule = cls.STRUCTURE_STOICHIOMETRY_RULES.get(structure)
        if rule is None:
            return violations

        n_elements = cls.count_elements(composition)

        if n_elements < rule["min_elements"]:
            violations.append(
                f"晶体化学违反: {composition}({n_elements}种元素)不能形成{structure}结构 "
                f"— {rule['desc']}"
            )
        elif n_elements > rule.get("max_elements", 999):
            violations.append(
                f"晶体化学违反: {composition}({n_elements}种元素)元素数超出{structure}结构上限 "
                f"— {rule['desc']}"
            )

        return violations

    @classmethod
    def check_known_principle(
        cls, structure: str, property_name: str,
    ) -> Optional[str]:
        """检查是否为已知结构-性能规律.

        Returns:
            已知规律描述 (如果不是新发现), None=可能是新发现
        """
        key = (structure, property_name)
        if key in cls.KNOWN_PRINCIPLES:
            return cls.KNOWN_PRINCIPLES[key]

        if structure == "layered" and property_name in (
            "thermal_conductivity", "conductivity", "seebeck", "zt"
        ):
            return cls.KNOWN_PRINCIPLES.get(
                ("layered", property_name),
                f"层状结构的{property_name}具有已知各向异性特征, 属材料物理共识"
            )

        return None

    @classmethod
    def check_phase_consistency(
        cls, composition: str, structure: str, property_name: str,
    ) -> List[str]:
        """检查相态-性能一致性.

        例如: YBCO 的 Tc=92K 属于正交相, 不属于四方相.
        """
        violations: List[str] = []

        phase_rule = cls.PHASE_RULES.get(composition)
        if phase_rule is None:
            clean_comp = composition.strip()
            phase_rule = cls.PHASE_RULES.get(clean_comp)

        if phase_rule is None:
            return violations

        phase_info = phase_rule.get(structure)
        if phase_info is None:
            for _, info in phase_rule.items():
                if isinstance(info, dict) and "note" in info:
                    violations.append(f"结构归类不严谨: {info['note']}")
            return violations

        if property_name in phase_info:
            if phase_info[property_name] is False:
                desc = phase_info.get("desc", "")
                violations.append(
                    f"相态不一致: {composition}的{structure}相({desc})不具备{property_name}属性"
                )

        return violations


    @classmethod
    def _find_unit_group(cls, unit: str) -> Optional[str]:
        """返回单位所属的兼容组名 (None=未知单位)."""
        if not unit:
            return "dimensionless"
        for group, units in cls.UNIT_COMPAT_GROUPS.items():
            if unit in units:
                return group
        return None

    @classmethod
    def convert_unit(
        cls, value: float, from_unit: str, to_unit: str,
    ) -> Tuple[bool, float, str]:
        """将 value 从 from_unit 换算到 to_unit.

        Returns:
            (success, converted_value, message)
            success=False 时 converted_value=0, message 说明原因.
            如果单位不兼容 (不在同一组), success=False.
        """
        if not from_unit and not to_unit:
            return (True, value, "")
        if from_unit == to_unit:
            return (True, value, "")

        key = f"{from_unit}→{to_unit}"
        if key in cls.UNIT_CONVERSION:
            return (True, value * cls.UNIT_CONVERSION[key], "")

        rev_key = f"{to_unit}→{from_unit}"
        if rev_key in cls.UNIT_CONVERSION:
            factor = cls.UNIT_CONVERSION[rev_key]
            if factor != 0:
                return (True, value / factor, "")

        if from_unit == "K" and to_unit in ("°C", "C"):
            return (True, value - 273.15, "")
        if from_unit in ("°C", "C") and to_unit == "K":
            return (True, value + 273.15, "")

        g_from = cls._find_unit_group(from_unit)
        g_to = cls._find_unit_group(to_unit)
        if g_from is not None and g_to is not None and g_from != g_to:
            return (False, 0.0,
                    f"单位不兼容: {from_unit}({g_from}) 与 {to_unit}({g_to}) 无法换算")
        if g_from is None or g_to is None:
            return (False, 0.0,
                    f"未知单位: from={from_unit!r} to={to_unit!r}")

        return (False, 0.0,
                f"单位同组但无换算因子: {from_unit}→{to_unit}")

    @classmethod
    def check_property_range(
        cls,
        property_name: str,
        value: float,
        unit: str = "",
    ) -> Dict[str, Any]:
        """检查数值是否在该性能的物理合理范围内 (P0-2).

        会先将 value 从 unit 换算到 PROPERTY_RANGES 中的参考单位,
        然后比较. 单位不兼容 → 拒绝 (不重置单位).

        Returns:
            {
                "in_range": bool,
                "violations": List[str],
                "converted_value": float | None,
                "reference_unit": str,
                "range": (min, max) | None,
            }
        """
        result: Dict[str, Any] = {
            "in_range": True,
            "violations": [],
            "converted_value": None,
            "reference_unit": "",
            "range": None,
        }

        rng = cls.PROPERTY_RANGES.get(property_name)
        if rng is None:
            return result

        lo, hi, ref_unit = rng
        result["reference_unit"] = ref_unit
        result["range"] = (lo, hi)

        if unit and ref_unit and unit != ref_unit:
            ok, conv, msg = cls.convert_unit(value, unit, ref_unit)
            if not ok:
                result["in_range"] = False
                result["violations"].append(
                    f"单位校验失败: {property_name}={value} {unit}, {msg}"
                )
                return result
            check_val = conv
            result["converted_value"] = conv
        else:
            check_val = value
            result["converted_value"] = value

        if check_val < lo or check_val > hi:
            result["in_range"] = False
            disp_val = f"{check_val:.4g}" if isinstance(check_val, (int, float)) else str(check_val)
            result["violations"].append(
                f"物理范围违反: {property_name}={disp_val} {ref_unit} "
                f"超出合理范围 [{lo}, {hi}] {ref_unit}"
            )

        return result


    ROBOT_MATERIAL_CONSTRAINTS: Dict[str, Dict[str, Any]] = {
        "lightweight_joint": {
            "desc": "轻量化机器人关节材料联合约束",
            "rules": {
                "density":         ("max", 8.0, "g/cm³"),
                "yield_strength":  ("min", 200.0, "MPa"),
                "hardness":        ("min", 1.0, "GPa"),
                "tensile_strength": ("min", 300.0, "MPa"),
            },
        },
        "wear_resistant_surface": {
            "desc": "耐磨表面涂层联合约束",
            "rules": {
                "hardness":        ("min", 10.0, "GPa"),
                "friction_coeff":  ("max", 0.5, ""),
            },
        },
    }

    @classmethod
    def check_joint_csp_consistency(
        cls,
        composition: str,
        structure: str,
        property_name: str,
        value: Optional[float] = None,
        unit: str = "",
    ) -> List[str]:
        """组分-结构-性能三者联合校验 (P0-4).

        检查逻辑:
        1. 化学式与结构兼容 (已有 validate_structure_compatibility)
        2. 结构与性能的已知匹配 (如 perovskite→dielectric, HEA→mechanical)
        3. 数值范围 (如有 value/unit, 调用 check_property_range)
        4. 跨性能物理一致性 (如有多个性能值, 检查彼此关系)

        Returns:
            违规列表 (空=通过)
        """
        violations: List[str] = []

        v1 = cls.validate_structure_compatibility(composition, structure)
        violations.extend(v1)

        if value is not None:
            r = cls.check_property_range(property_name, value, unit)
            violations.extend(r["violations"])

        _STRUCT_PROP_HINTS: Dict[str, List[str]] = {
            "perovskite": ["dielectric_constant", "tc", "piezoelectric_coeff",
                           "magnetization", "conductivity", "bandgap"],
            "HEA": ["yield_strength", "hardness", "tensile_strength",
                    "youngs_modulus", "fracture_toughness"],
            "high-entropy alloy": ["yield_strength", "hardness", "tensile_strength"],
            "diamond": ["hardness", "youngs_modulus", "thermal_conductivity"],
            "graphite": ["conductivity", "thermal_conductivity", "friction_coeff"],
            "amorphous": ["magnetization", "youngs_modulus"],
        }

        return violations

    @classmethod
    def check_robot_material_constraints(
        cls,
        property_values: Dict[str, Tuple[float, str]],
        profile: str = "lightweight_joint",
    ) -> Dict[str, Any]:
        """检查机器人材料联合约束 (P0-4).

        Args:
            property_values: {"density": (7.8, "g/cm³"), "hardness": (2.0, "GPa"), ...}
            profile: 约束档案名

        Returns:
            {
                "satisfied": bool,
                "violations": List[str],
                "warnings": List[str],
            }
        """
        result = {"satisfied": True, "violations": [], "warnings": []}
        prof = cls.ROBOT_MATERIAL_CONSTRAINTS.get(profile)
        if prof is None:
            result["warnings"].append(f"未知机器人材料约束档案: {profile}")
            return result

        for prop, (op, threshold, ref_unit) in prof["rules"].items():
            if prop not in property_values:
                result["warnings"].append(
                    f"缺少{prop}数据, 无法评估联合约束'{profile}'"
                )
                continue
            val, unit = property_values[prop]
            ok, conv, msg = cls.convert_unit(val, unit, ref_unit)
            if not ok:
                result["violations"].append(
                    f"联合约束校验失败: {prop} 单位不兼容 ({msg})"
                )
                result["satisfied"] = False
                continue
            if op == "max" and conv > threshold:
                result["violations"].append(
                    f"机器人材料约束: {prop}={conv:.4g} {ref_unit} > {threshold} {ref_unit} "
                    f"(需低密度/低摩擦)"
                )
                result["satisfied"] = False
            elif op == "min" and conv < threshold:
                result["violations"].append(
                    f"机器人材料约束: {prop}={conv:.4g} {ref_unit} < {threshold} {ref_unit} "
                    f"(需高强度/高硬度)"
                )
                result["satisfied"] = False

        return result

    @classmethod
    def validate_hypothesis(
        cls,
        composition_a: str,
        composition_b: str,
        structure_a: str,
        structure_b: str,
        property_name: str,
        hypo_type: str,
        predicted_value: Optional[float] = None,
        predicted_unit: str = "",
        known_value: Optional[float] = None,
        known_unit: str = "",
    ) -> Dict[str, Any]:
        """全面验证假设的物理合理性 (P0-2 升级).

        新增参数:
            predicted_value: 预测值 (如有)
            predicted_unit: 预测值单位
            known_value: 已知参考值 (如有)
            known_unit: 已知值单位

        Returns:
            {
                "violations": List[str],
                "warnings": List[str],
                "is_known_principle": bool,
                "known_principle_desc": str,
                "severity": str,           # "pass" / "warning" / "reject"
                "range_check": dict,
                "unit_compatible": bool,
            }
        """
        violations: List[str] = []
        warnings: List[str] = []
        is_known = False
        known_desc = ""
        range_check: Dict[str, Any] = {"in_range": True, "violations": []}
        unit_compatible = True

        for comp, struct, label in [
            (composition_a, structure_a, "A"),
            (composition_b, structure_b, "B"),
        ]:
            v = cls.validate_structure_compatibility(comp, struct)
            violations.extend(v)

        for comp, struct, label in [
            (composition_a, structure_a, "A"),
            (composition_b, structure_b, "B"),
        ]:
            v = cls.check_phase_consistency(comp, struct, property_name)
            violations.extend(v)

        for struct in [structure_a, structure_b]:
            desc = cls.check_known_principle(struct, property_name)
            if desc:
                is_known = True
                known_desc = desc
                warnings.append(f"已知规律: {desc}")
                break

        if predicted_value is not None:
            range_check = cls.check_property_range(
                property_name, predicted_value, predicted_unit,
            )
            if not range_check["in_range"]:
                violations.extend(range_check["violations"])
                for vmsg in range_check["violations"]:
                    if "单位" in vmsg or "不兼容" in vmsg:
                        unit_compatible = False

        if predicted_value is not None:
            for comp, struct in [
                (composition_a, structure_a),
                (composition_b, structure_b),
            ]:
                v = cls.check_joint_csp_consistency(
                    comp, struct, property_name,
                    value=predicted_value, unit=predicted_unit,
                )
                violations.extend(v)

        if violations:
            severity = "reject"
        elif is_known:
            severity = "warning"
        else:
            severity = "pass"

        return {
            "violations": violations,
            "warnings": warnings,
            "is_known_principle": is_known,
            "known_principle_desc": known_desc,
            "severity": severity,
            "range_check": range_check,
            "unit_compatible": unit_compatible,
        }
