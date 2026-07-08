#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_formula(formula: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for el, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        out.append((el, int(num) if num else 1))
    return out


def lattice_vectors(a: float, b: float, c: float, alpha: float, beta: float, gamma: float):
    ar, br, gr = map(math.radians, (alpha, beta, gamma))
    ax, ay, az = a, 0.0, 0.0
    bx, by, bz = b * math.cos(gr), b * math.sin(gr), 0.0
    cx = c * math.cos(br)
    cy = c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / math.sin(gr)
    cz2 = c * c - cx * cx - cy * cy
    cz = math.sqrt(max(cz2, 0.0))
    return [(ax, ay, az), (bx, by, bz), (cx, cy, cz)]


def formula_from_results(data: dict) -> str:
    for item in data["results"]:
        if item["task"] in {"method", "precursor"}:
            m = re.search(r'"([^"]+)"', item.get("prompt", ""))
            if m:
                return m.group(1)
    raise ValueError("formula not found in task prompts")


def parse_structure_description(desc: str):
    first, cell_part, rest = desc.split("|", 2)
    cell = [float(x) for x in cell_part.split(",")]
    sites = []
    pattern = re.compile(r"\(([A-Z][a-z]?)-[^[]+\[([^\]]+)\]\)")
    for el, coord_text in pattern.findall(rest):
        coords = [float(x) for x in coord_text.split()]
        if len(coords) == 3:
            sites.append((el, [x % 1.0 for x in coords]))
    return first.strip(), cell, sites


def choose_coords(sites: list[tuple[str, list[float]]], formula_parts: list[tuple[str, int]]):
    by_el: dict[str, list[list[float]]] = defaultdict(list)
    for el, coord in sites:
        by_el[el].append(coord)

    coords_by_el: dict[str, list[list[float]]] = {}
    for el, count in formula_parts:
        available = by_el.get(el, [])
        if not available:
            raise ValueError(f"no coordinates found for element {el}")
        chosen = []
        idx = 0
        while len(chosen) < count:
            base = available[idx % len(available)]
            # Tiny deterministic offset avoids exact overlap when the reduced formula
            # needs more sites than the unique pyxtal representatives provide.
            cycle = idx // len(available)
            if cycle:
                off = 0.017 * cycle
                coord = [(base[0] + off) % 1, (base[1] + 0.011 * cycle) % 1, (base[2] + 0.013 * cycle) % 1]
            else:
                coord = base
            chosen.append(coord)
            idx += 1
        coords_by_el[el] = chosen
    return coords_by_el


def write_poscar(path: Path, formula: str, cell: list[float], formula_parts, coords_by_el) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors = lattice_vectors(*cell)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{formula} generated from three_tasks pyxtal description\n")
        f.write("1.0\n")
        for vec in vectors:
            f.write("  " + "  ".join(f"{x:.10f}" for x in vec) + "\n")
        f.write("  " + "  ".join(el for el, _ in formula_parts) + "\n")
        f.write("  " + "  ".join(str(count) for _, count in formula_parts) + "\n")
        f.write("Direct\n")
        for el, _ in formula_parts:
            for coord in coords_by_el[el]:
                f.write("  " + "  ".join(f"{x:.10f}" for x in coord) + "\n")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: json_three_tasks_to_poscar.py OUT_DIR INPUT_JSON [INPUT_JSON ...]")
    out_dir = Path(sys.argv[1])
    for raw in sys.argv[2:]:
        src = Path(raw)
        data = json.loads(src.read_text(encoding="utf-8"))
        formula = formula_from_results(data)
        _, cell, sites = parse_structure_description(data["structure_description"])
        formula_parts = parse_formula(formula)
        coords_by_el = choose_coords(sites, formula_parts)
        dst = out_dir / src.stem.replace("_three_tasks", "") / "POSCAR"
        write_poscar(dst, formula, cell, formula_parts, coords_by_el)
        print(f"{src.name} -> {dst} ({formula})")


if __name__ == "__main__":
    main()
