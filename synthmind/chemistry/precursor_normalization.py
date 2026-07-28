from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CANONICALIZATION_VERSION = "precursor_composition_v1"

_SUBSCRIPT_TRANSLATION = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_HYDRATE_PERIOD = re.compile(r"\.(?=\d+(?:H2O|NH3)$)")
_LEADING_MULTIPLIER = re.compile(r"^(\d+(?:\.\d+)?)(?=[A-Z(])(.*)$")
_VARIABLE_STOICHIOMETRY = re.compile(r"(?:^|[)·.])(?:m|n|x|y|z)(?=[A-Z(])")
_UNSAFE_SHORTHAND = re.compile(
    r"(?:NPs?|NWs?|OAc|Cit|UiO|PVP|PEG|PVA|CTAB|SDS|[A-Z][a-z]?I{2,3})$"
)
_SAFE_ORGANIC_LIGAND = re.compile(
    r"(?:CH3COO|CH3CO2|C2H3O2|OCOCH3|C2O4|CO2CO2|C5H7O2|CH3COCHCOCH3)"
)


def _pymatgen_composition():
    try:
        from pymatgen.core import Composition
    except ImportError as exc:  # pragma: no cover - exercised on training host
        raise RuntimeError(
            "pymatgen is required for precursor composition normalization; "
            "install the project's models optional dependencies"
        ) from exc
    return Composition


@dataclass(frozen=True)
class PrecursorNormalization:
    raw_name: str
    normalized_text: str
    canonical_key: str
    canonical_formula: str
    status: str
    error: str = ""
    version: str = CANONICALIZATION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_precursor_text(value: Any) -> str:
    """Normalize typography without changing chemical stoichiometry."""
    text = "" if value is None else unicodedata.normalize("NFKC", str(value)).strip()
    text = text.translate(_SUBSCRIPT_TRANSLATION)
    text = text.replace("∙", "·").replace("•", "·").replace("⋅", "·")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"_\((\d+(?:\.\d+)?)\)", r"\1", text)
    # An ASCII dot is used as a hydrate separator in many source labels. Do
    # not rewrite it when a proper middle dot already exists: in strings such
    # as ``Al(OH)3·0.949H2O`` the ASCII dot is a decimal point.
    if "·" not in text:
        text = _HYDRATE_PERIOD.sub("·", text)
    return text


def _gcd_many(values: Sequence[int]) -> int:
    return reduce(gcd, values) if values else 1


def _integer_composition(amounts: Mapping[str, float]) -> Dict[str, int]:
    fractions = {
        str(symbol): Fraction(float(amount)).limit_denominator(1000)
        for symbol, amount in amounts.items()
        if math.isfinite(float(amount)) and float(amount) > 0
    }
    if not fractions:
        raise ValueError("composition has no positive finite element amounts")
    common_denominator = math.lcm(*(value.denominator for value in fractions.values()))
    integers = {
        symbol: int(value.numerator * (common_denominator // value.denominator))
        for symbol, value in fractions.items()
    }
    divisor = _gcd_many(list(integers.values()))
    return {symbol: value // divisor for symbol, value in integers.items()}


def _parse_fragment(fragment: str):
    Composition = _pymatgen_composition()
    multiplier = 1.0
    body = fragment
    match = _LEADING_MULTIPLIER.match(fragment)
    if match:
        multiplier = float(match.group(1))
        body = match.group(2)
    if not body:
        raise ValueError("empty formula fragment")
    return Composition(body).remove_charges() * multiplier


def _parse_composition(text: str):
    Composition = _pymatgen_composition()
    if not text:
        raise ValueError("precursor name is empty")
    if text.startswith("·") or text.endswith("·"):
        raise ValueError("formula starts or ends with an adduct separator")
    if "-" in text:
        raise ValueError("hyphenated adduct or textual name is not parsed conservatively")
    if _VARIABLE_STOICHIOMETRY.search(text):
        raise ValueError("variable stoichiometry is not composition-canonicalized")
    if _UNSAFE_SHORTHAND.search(text):
        raise ValueError("recognized shorthand or morphology suffix is not a molecular formula")
    total = Composition({})
    for fragment in text.split("·"):
        total += _parse_fragment(fragment)
    if not total:
        raise ValueError("composition is empty")
    return total.remove_charges()


def _fallback_key(text: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z()+.·_-]+", "", text).casefold()
    return "text::" + normalized


def normalize_precursor(value: Any) -> PrecursorNormalization:
    """Return a deterministic composition key, with conservative text fallback.

    Formula order, redundant parentheses, and hydrate-dot typography are treated
    as aliases. Different elemental stoichiometries remain distinct. Ambiguous
    shorthand, named additives, and variable-stoichiometry expressions are kept
    as text labels so they cannot be merged accidentally.
    """
    raw_name = "" if value is None else str(value).strip()
    text = normalize_precursor_text(raw_name)
    try:
        composition = _parse_composition(text)
        integer_amounts = _integer_composition(composition.get_el_amt_dict())
        elements = sorted(composition.elements, key=lambda element: int(element.Z))
        element_symbols = {element.symbol for element in elements}
        contains_metal = any(bool(element.is_metal) for element in elements)
        if re.match(r"^([A-Z][a-z]?)\1", text):
            # Strings such as ``FeFe2O3`` can encode a concatenated mixture,
            # not a reduced FeO formula. Preserve the source distinction.
            return PrecursorNormalization(
                raw_name=raw_name,
                normalized_text=text,
                canonical_key="duplicated_formula_text::" + text.casefold(),
                canonical_formula=str(composition.formula),
                status="duplicated_formula_text",
            )
        if len(elements) == 1:
            # Elemental allotropes and morphology annotations can matter as
            # reagents (C, C60, S8, metal nanoparticles). Do not collapse them
            # merely because the reduced composition is the same element.
            return PrecursorNormalization(
                raw_name=raw_name,
                normalized_text=text,
                canonical_key="elemental_text::" + text.casefold(),
                canonical_formula=str(composition.formula),
                status="elemental_text",
            )
        if {"C", "H"} <= element_symbols and not contains_metal:
            # Elemental composition alone cannot distinguish molecular
            # isomers (for example ethanol and dimethyl ether). Keep these
            # labels text-distinct unless a future curated synonym table says
            # otherwise. Metal salts with organic anions remain eligible for
            # formula-order and hydrate normalization.
            return PrecursorNormalization(
                raw_name=raw_name,
                normalized_text=text,
                canonical_key="molecular_text::" + text.casefold(),
                canonical_formula=str(composition.formula),
                status="molecular_text",
            )
        if {"C", "H"} <= element_symbols and contains_metal and not _SAFE_ORGANIC_LIGAND.search(text):
            # A metal does not remove molecular-isomer ambiguity: normal- and
            # iso-propoxides have the same elemental composition. Only common
            # ligand spellings covered above are merged automatically.
            return PrecursorNormalization(
                raw_name=raw_name,
                normalized_text=text,
                canonical_key="coordination_text::" + text.casefold(),
                canonical_formula=str(composition.formula),
                status="coordination_text",
            )
        key = "composition::" + "|".join(
            f"{element.symbol}:{integer_amounts[element.symbol]}" for element in elements
        )
        canonical_formula = composition.reduced_formula
        return PrecursorNormalization(
            raw_name=raw_name,
            normalized_text=text,
            canonical_key=key,
            canonical_formula=str(canonical_formula),
            status="composition",
        )
    except Exception as exc:
        return PrecursorNormalization(
            raw_name=raw_name,
            normalized_text=text,
            canonical_key=_fallback_key(text),
            canonical_formula="",
            status="text_fallback",
            error=f"{type(exc).__name__}: {exc}",
        )


def normalize_many(values: Iterable[Any]) -> List[PrecursorNormalization]:
    return [normalize_precursor(value) for value in values]
