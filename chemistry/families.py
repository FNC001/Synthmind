from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence


SCHEMA_VERSION = "target_cation_family_v1"
PERIODIC_GROUP_BUCKETS: Sequence[str] = tuple(
    [f"G{i:02d}" for i in range(1, 19)] + ["LN", "AN", "UNK"]
)
ROUTING_LEVELS: Sequence[str] = ("metal", "metalloid", "nonmetal_fallback")


def _pymatgen_types():
    try:
        from pymatgen.core import Composition, Element
    except ImportError as exc:  # pragma: no cover - exercised on training host
        raise RuntimeError(
            "pymatgen is required for target-cation family assignment; "
            "install the project's models optional dependencies"
        ) from exc
    return Composition, Element


def element_group_bucket(element: Any) -> str:
    """Map a pymatgen Element to the versioned routing group bucket."""
    if bool(element.is_lanthanoid):
        return "LN"
    if bool(element.is_actinoid):
        return "AN"
    group = getattr(element, "group", None)
    if group is None:
        return "UNK"
    return f"G{int(group):02d}"


@dataclass(frozen=True)
class FamilyAssignment:
    input_formula: str
    canonical_formula: str
    target_elements: List[str]
    target_cation_elements: List[str]
    target_anion_elements: List[str]
    family_signature_primary: str
    family_id_primary: str
    family_routing_level: str
    family_parse_status: str
    family_schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _stable_family_id(signature: str) -> str:
    readable = signature.replace("+", "_")
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    return f"catfam_v1__{readable}__{digest}"


def assign_cation_family(formula: Any) -> FamilyAssignment:
    """
    Assign the primary target-cation family.

    Metals determine routing. If a composition contains no metals, metalloids
    are used as framework elements; all elements are used only as a final
    fallback. Stoichiometry and anion identity deliberately do not affect the
    primary family signature.
    """
    Composition, Element = _pymatgen_types()
    input_formula = "" if formula is None else str(formula).strip()
    if not input_formula:
        raise ValueError("formula is empty")

    composition = Composition(input_formula).remove_charges()
    amounts = composition.get_el_amt_dict()
    if not amounts:
        raise ValueError(f"formula has no elements: {input_formula!r}")

    elements = sorted((Element(symbol) for symbol in amounts), key=lambda e: e.Z)
    selected = [element for element in elements if bool(element.is_metal)]
    routing_level = "metal"
    if not selected:
        selected = [element for element in elements if bool(element.is_metalloid)]
        routing_level = "metalloid"
    if not selected:
        selected = list(elements)
        routing_level = "nonmetal_fallback"

    selected_symbols = {element.symbol for element in selected}
    group_buckets = sorted({element_group_bucket(element) for element in selected})
    signature = "+".join(group_buckets) if group_buckets else "UNK"

    return FamilyAssignment(
        input_formula=input_formula,
        canonical_formula=composition.reduced_formula,
        target_elements=[element.symbol for element in elements],
        target_cation_elements=[element.symbol for element in selected],
        target_anion_elements=[
            element.symbol for element in elements if element.symbol not in selected_symbols
        ],
        family_signature_primary=signature,
        family_id_primary=_stable_family_id(signature),
        family_routing_level=routing_level,
        family_parse_status="ok",
    )


def family_feature_names(prefix: str = "feat_catfam__") -> List[str]:
    return [
        *[f"{prefix}{bucket}" for bucket in PERIODIC_GROUP_BUCKETS],
        *[f"{prefix}level__{level}" for level in ROUTING_LEVELS],
    ]


def family_feature_vector(
    assignment: FamilyAssignment,
    prefix: str = "feat_catfam__",
) -> List[float]:
    active_groups = set(assignment.family_signature_primary.split("+"))
    return [
        *[1.0 if bucket in active_groups else 0.0 for bucket in PERIODIC_GROUP_BUCKETS],
        *[
            1.0 if assignment.family_routing_level == level else 0.0
            for level in ROUTING_LEVELS
        ],
    ]


def assign_many(formulas: Iterable[Any]) -> List[FamilyAssignment]:
    return [assign_cation_family(formula) for formula in formulas]
