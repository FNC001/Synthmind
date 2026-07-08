#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_poscar_elements(poscar: Path) -> list[str]:
    lines = poscar.read_text(errors="ignore").splitlines()
    if len(lines) < 6:
        return []

    # VASP5 format usually has element symbols on line 6, counts on line 7.
    raw = lines[5].strip().split()
    elems = []
    for x in raw:
        if re.fullmatch(r"[A-Z][a-z]?", x):
            elems.append(x)

    return elems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--poscar", required=True)
    ap.add_argument("--ignore_elements", default="H,O")
    args = ap.parse_args()

    poscar = Path(args.poscar)
    ignore = {x.strip() for x in args.ignore_elements.split(",") if x.strip()}

    elems = parse_poscar_elements(poscar)
    active = [e for e in elems if e not in ignore]

    print(f"poscar={poscar}")
    print(f"elements={elems}")
    print(f"ignore_elements={sorted(ignore)}")
    print(f"active_target_elements={active}")

    if not elems:
        raise SystemExit(2)

    if not active:
        raise SystemExit(10)

    raise SystemExit(0)


if __name__ == "__main__":
    main()
