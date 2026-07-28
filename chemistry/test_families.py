from __future__ import annotations

import unittest

try:
    import pymatgen  # noqa: F401
except ImportError:  # pragma: no cover - local minimal environment
    pymatgen = None

from synthmind.chemistry.families import assign_cation_family, family_feature_vector


@unittest.skipIf(pymatgen is None, "pymatgen is not installed")
class CationFamilyTests(unittest.TestCase):
    def test_requested_alkali_examples_are_one_family(self) -> None:
        formulas = ["LiCl", "NaCl", "NaBr", "Li2O", "Na2S"]
        assignments = [assign_cation_family(formula) for formula in formulas]
        self.assertEqual({item.family_signature_primary for item in assignments}, {"G01"})
        self.assertEqual(len({item.family_id_primary for item in assignments}), 1)

    def test_multication_family_ignores_anion_and_stoichiometry(self) -> None:
        item = assign_cation_family("SrTiO3")
        self.assertEqual(item.family_signature_primary, "G02+G04")
        self.assertEqual(item.target_cation_elements, ["Ti", "Sr"])
        self.assertEqual(item.target_anion_elements, ["O"])

    def test_family_vector_is_stable(self) -> None:
        a = family_feature_vector(assign_cation_family("LiCl"))
        b = family_feature_vector(assign_cation_family("Na2S"))
        self.assertEqual(a, b)
        self.assertEqual(sum(a), 2.0)  # one group plus the metal routing-level flag

    def test_metalloid_fallback(self) -> None:
        item = assign_cation_family("SiO2")
        self.assertEqual(item.family_signature_primary, "G14")
        self.assertEqual(item.family_routing_level, "metalloid")


if __name__ == "__main__":
    unittest.main()
