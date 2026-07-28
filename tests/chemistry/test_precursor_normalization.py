from __future__ import annotations

import unittest

try:
    import pymatgen  # noqa: F401
except ImportError:  # pragma: no cover - local minimal environment
    pymatgen = None

from synthmind.chemistry.precursor_normalization import normalize_precursor


@unittest.skipIf(pymatgen is None, "pymatgen is not installed")
class PrecursorNormalizationTests(unittest.TestCase):
    def test_inorganic_formula_order_aliases_share_a_key(self) -> None:
        values = ["Al2(SO4)3", "(SO4)3Al2"]
        self.assertEqual(len({normalize_precursor(value).canonical_key for value in values}), 1)

    def test_molecular_isomers_are_not_merged_by_elemental_composition(self) -> None:
        ethanol = normalize_precursor("C2H6O")
        dimethyl_ether = normalize_precursor("CH3OCH3")
        self.assertNotEqual(ethanol.canonical_key, dimethyl_ether.canonical_key)

    def test_hydrate_dot_aliases_share_a_key(self) -> None:
        values = ["(CH3COO)2Mn·4H2O", "(CH3COO)2Mn.4H2O", "Mn(CH3COO)2·4H2O"]
        records = [normalize_precursor(value) for value in values]
        self.assertEqual({record.status for record in records}, {"composition"})
        self.assertEqual(len({record.canonical_key for record in records}), 1)

    def test_hydrate_stoichiometry_remains_distinct(self) -> None:
        hydrated = normalize_precursor("CuSO4·5H2O")
        anhydrous = normalize_precursor("CuSO4")
        self.assertNotEqual(hydrated.canonical_key, anhydrous.canonical_key)

    def test_reordered_acetate_aliases_share_a_key(self) -> None:
        a = normalize_precursor("Mn(CH3COO)2·4H2O")
        b = normalize_precursor("(CH3COO)2Mn.4H2O")
        self.assertEqual(a.canonical_key, b.canonical_key)

    def test_metal_alkoxide_isomers_are_not_composition_merged(self) -> None:
        normal_propoxide = normalize_precursor("Ti(OCH2CH2CH3)4")
        isopropoxide = normalize_precursor("Ti(OCH(CH3)2)4")
        self.assertNotEqual(normal_propoxide.canonical_key, isopropoxide.canonical_key)

    def test_ambiguous_acronym_uses_text_fallback(self) -> None:
        record = normalize_precursor("PVP")
        self.assertEqual(record.status, "text_fallback")

    def test_fractional_hydrate_coefficient_keeps_decimal_point(self) -> None:
        record = normalize_precursor("Al(OH)3·0.949H2O")
        self.assertEqual(record.status, "composition")

    def test_concatenated_formula_is_not_reduced_into_another_precursor(self) -> None:
        self.assertNotEqual(
            normalize_precursor("FeFe2O3").canonical_key,
            normalize_precursor("FeO").canonical_key,
        )


if __name__ == "__main__":
    unittest.main()
