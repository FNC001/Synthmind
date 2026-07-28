#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(?:°\s*)?(C|F|K|celsius|fahrenheit|kelvin|℃|℉)\b", re.I)
TIME_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|min|mins|minute|minutes|s|sec|secs|second|seconds|d|day|days)\b",
    re.I,
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def temp_to_c(value: float, unit: str) -> float:
    u = unit.lower().replace(" ", "")
    if u in {"k", "kelvin"}:
        return value - 273.15
    if u in {"f", "°f", "℉", "fahrenheit"}:
        return (value - 32.0) * 5.0 / 9.0
    return value


def time_to_h(value: float, unit: str) -> float:
    u = unit.lower()
    if u in {"min", "mins", "minute", "minutes"}:
        return value / 60.0
    if u in {"s", "sec", "secs", "second", "seconds"}:
        return value / 3600.0
    if u in {"d", "day", "days"}:
        return value * 24.0
    return value


def regex_conditions(text: str) -> Tuple[List[float], List[float], Counter, Counter]:
    temps: List[float] = []
    times: List[float] = []
    temp_units: Counter = Counter()
    time_units: Counter = Counter()
    for m in TEMP_RE.finditer(text or ""):
        v = safe_float(m.group(1))
        if v is None:
            continue
        unit = m.group(2)
        temp_units[unit.lower()] += 1
        c = temp_to_c(v, unit)
        if 0 <= c <= 2000:
            temps.append(round(c, 4))
    for m in TIME_RE.finditer(text or ""):
        v = safe_float(m.group(1))
        if v is None:
            continue
        unit = m.group(2)
        time_units[unit.lower()] += 1
        h = time_to_h(v, unit)
        if 0 <= h <= 500:
            times.append(round(h, 4))
    return temps, times, temp_units, time_units


def row_text(row: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in ["synthesis_text", "reaction_string"]:
        if row.get(key):
            parts.append(str(row[key]))
    raw = row.get("raw_synthesis_record")
    if isinstance(raw, dict):
        for key in ["method_raw", "method_compact", "synthesis_text"]:
            if raw.get(key):
                parts.append(str(raw[key]))
        for op in raw.get("operations") or []:
            if isinstance(op, dict):
                for key in ["text", "operation_raw", "operation"]:
                    if op.get(key):
                        parts.append(str(op[key]))
    for step in row.get("steps") or []:
        if isinstance(step, dict) and step.get("text"):
            parts.append(str(step["text"]))
    return " ".join(parts)


def choose_temp(existing: Optional[float], regex_vals: List[float]) -> Tuple[Optional[float], bool, str]:
    vals = sorted(set(regex_vals))
    candidate = max(vals) if vals else None
    if existing is None:
        return candidate, candidate is not None, "filled_from_text" if candidate is not None else "missing"
    if candidate is not None and abs(existing - candidate) > 1e-6:
        # Keep operation/fallback extraction as canonical; report mismatch only.
        return existing, False, "text_mismatch_kept_existing"
    return existing, False, "ok"


def choose_time(existing: Optional[float], regex_vals: List[float]) -> Tuple[Optional[float], bool, str]:
    vals = sorted(set(regex_vals))
    candidate = sum(vals) if vals and sum(vals) <= 500 else (max(vals) if vals else None)
    if existing is None:
        return candidate, candidate is not None, "filled_from_text" if candidate is not None else "missing"
    if candidate is not None and abs(existing - candidate) > 1e-4:
        return existing, False, "text_mismatch_kept_existing"
    return existing, False, "ok"


def audit_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    out_rows: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    temp_units: Counter = Counter()
    time_units: Counter = Counter()
    actions: Counter = Counter()
    changed = 0

    for row in rows:
        text = row_text(row)
        temps, times, tu, hu = regex_conditions(text)
        temp_units.update(tu)
        time_units.update(hu)
        new = dict(row)
        temp_existing = safe_float(row.get("temperature_c"))
        time_existing = safe_float(row.get("time_h"))
        temp_new, temp_changed, temp_action = choose_temp(temp_existing, temps)
        time_new, time_changed, time_action = choose_time(time_existing, times)
        actions[f"temperature::{temp_action}"] += 1
        actions[f"time::{time_action}"] += 1
        if temp_changed:
            new["temperature_c"] = temp_new
        if time_changed:
            new["time_h"] = time_new
        if temp_changed or time_changed:
            changed += 1
        out_rows.append(new)

        if temp_action != "ok" or time_action != "ok":
            audit.append({
                "id": row.get("id"),
                "source_dataset": row.get("source_dataset"),
                "temperature_c_existing": temp_existing,
                "temperature_c_text_candidate": max(sorted(set(temps))) if temps else None,
                "temperature_action": temp_action,
                "time_h_existing": time_existing,
                "time_h_text_candidate": (sum(sorted(set(times))) if times and sum(sorted(set(times))) <= 500 else (max(times) if times else None)),
                "time_action": time_action,
                "text_excerpt": text[:500],
            })

    summary = {
        "n_rows": len(rows),
        "n_rows_changed": changed,
        "temperature_units_seen": dict(temp_units.most_common()),
        "time_units_seen": dict(time_units.most_common()),
        "actions": dict(actions.most_common()),
        "n_audit_rows": len(audit),
        "canonical_units": {
            "temperature_c": "degree Celsius",
            "time_h": "hour",
        },
        "policy": "Existing structured condition fields are kept when they disagree with broad regex text extraction; missing values can be filled from text.",
    }
    return out_rows, summary, audit


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit and normalize synthesis condition units to Celsius and hours.")
    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--files", nargs="*", default=[
        "route_gold.jsonl",
        "route_train_relaxed.jsonl",
        "stage2_gold.jsonl",
        "stage2_train_relaxed.jsonl",
        "stage3_gold.jsonl",
        "stage3_train_relaxed.jsonl",
    ])
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined: Dict[str, Any] = {}
    all_audits: List[Dict[str, Any]] = []
    for name in args.files:
        in_path = args.input_dir / name
        if not in_path.exists():
            continue
        rows = read_jsonl(in_path)
        out_rows, summary, audit = audit_rows(rows)
        write_jsonl(args.output_dir / name, out_rows)
        combined[name] = summary
        for rec in audit:
            rec["file"] = name
        all_audits.extend(audit)

    write_json(args.output_dir / "condition_unit_audit_summary.json", combined)
    write_jsonl(args.output_dir / "condition_unit_audit_records.jsonl", all_audits)
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
