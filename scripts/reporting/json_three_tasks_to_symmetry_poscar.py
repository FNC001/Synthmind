#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    from pymatgen.core import Lattice, Structure
except ModuleNotFoundError as exc:
    raise SystemExit(
        "pymatgen is required for symmetry-expanded POSCAR generation. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc


def formula_from_results(data: dict) -> str:
    for item in data["results"]:
        if item["task"] in {"method", "precursor"}:
            m = re.search(r'"([^"]+)"', item.get("prompt", ""))
            if m:
                return m.group(1)
    raise ValueError("formula not found in task prompts")


def parse_structure_description(desc: str):
    sg_text, cell_part, rest = desc.split("|", 2)
    a, b, c, alpha, beta, gamma = [float(x) for x in cell_part.split(",")]
    species = []
    coords = []
    for el, coord_text in re.findall(r"\(([A-Z][a-z]?)-[^[]+\[([^\]]+)\]\)", rest):
        coord = [float(x) % 1.0 for x in coord_text.split()]
        if len(coord) == 3:
            species.append(el)
            coords.append(coord)
    return int(sg_text.strip()), Lattice.from_parameters(a, b, c, alpha, beta, gamma), species, coords


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: json_three_tasks_to_symmetry_poscar.py OUT_DIR INPUT_JSON [INPUT_JSON ...]")
    out_dir = Path(sys.argv[1])
    for raw in sys.argv[2:]:
        src = Path(raw)
        data = json.loads(src.read_text(encoding="utf-8"))
        formula = formula_from_results(data)
        sg, lattice, species, coords = parse_structure_description(data["structure_description"])
        structure = Structure.from_spacegroup(sg, lattice, species, coords, coords_are_cartesian=False, tol=0.01)
        structure = structure.get_sorted_structure()
        dst = out_dir / src.stem.replace("_three_tasks", "") / "POSCAR"
        dst.parent.mkdir(parents=True, exist_ok=True)
        structure.to(filename=str(dst), fmt="poscar")
        lines = dst.read_text(encoding="utf-8").splitlines()
        if lines:
            lines[0] = f"{formula} symmetry-expanded from three_tasks pyxtal description; generated={structure.composition.formula}"
            dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{src.name} -> {dst} target={formula} generated={structure.composition.formula} sites={len(structure)}")


if __name__ == "__main__":
    main()
