from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    task: str
    split: str
    protocol: str
    evaluated_fields: tuple[str, ...]
    denominator: str
    match_rule: str
    tolerance: dict[str, Any]
    k: int | None
    candidate_budget: str
    missing_value_policy: str
    canonicalization_version: str
    higher_is_better: bool
    implementation_function: str

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "MetricDefinition":
        fields = row.get("evaluated_fields", [])
        return cls(
            metric_id=str(row["metric_id"]),
            task=str(row["task"]),
            split=str(row.get("split", "any")),
            protocol=str(row["protocol"]),
            evaluated_fields=tuple(str(x) for x in fields),
            denominator=str(row["denominator"]),
            match_rule=str(row["match_rule"]),
            tolerance=dict(row.get("tolerance", {})),
            k=row.get("k"),
            candidate_budget=str(row.get("candidate_budget", "candidate_budget_v1")),
            missing_value_policy=str(row["missing_value_policy"]),
            canonicalization_version=str(row.get("canonicalization_version", "canonicalization_v1")),
            higher_is_better=bool(row.get("higher_is_better", True)),
            implementation_function=str(row["implementation_function"]),
        )


def _read_yaml_like(path: Path) -> list[dict[str, Any]]:
    """Load registry YAML using PyYAML when present, JSON fallback otherwise."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore

            payload = yaml.safe_load(text)
        except Exception:
            payload = _parse_metric_registry_yaml_subset(text)
    if isinstance(payload, dict):
        payload = payload.get("metrics", [])
    if not isinstance(payload, list):
        raise ValueError(f"Metric registry must contain a list, got {type(payload)!r}")
    return payload


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [x.strip().strip('"').strip("'") for x in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        out: dict[str, Any] = {}
        if not inner:
            return out
        for part in inner.split(","):
            k, _, v = part.partition(":")
            out[k.strip().strip('"').strip("'")] = _parse_scalar(v.strip())
        return out
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _parse_metric_registry_yaml_subset(text: str) -> dict[str, Any]:
    """Very small YAML subset parser for research/specs/metric_registry_v1.yaml.

    It supports a top-level `metrics:` sequence of flat mappings and inline
    list/dict scalars. This keeps the research runner dependency-free.
    """

    metrics: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_metrics = False
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("metrics:"):
            in_metrics = True
            continue
        if not in_metrics:
            continue
        stripped = raw.strip()
        if stripped.startswith("- "):
            if current:
                metrics.append(current)
            current = {}
            stripped = stripped[2:]
            if stripped:
                key, _, value = stripped.partition(":")
                current[key.strip()] = _parse_scalar(value)
            continue
        if current is not None and ":" in stripped:
            key, _, value = stripped.partition(":")
            current[key.strip()] = _parse_scalar(value)
    if current:
        metrics.append(current)
    return {"metrics": metrics}


class MetricRegistry:
    def __init__(self, metrics: Iterable[MetricDefinition]):
        metric_list = list(metrics)
        self._metrics = {m.metric_id: m for m in metric_list}
        if len(self._metrics) != len(metric_list):
            raise ValueError("Duplicate metric_id detected")

    @classmethod
    def load(cls, path: str | Path) -> "MetricRegistry":
        rows = _read_yaml_like(Path(path))
        return cls(MetricDefinition.from_mapping(row) for row in rows)

    def get(self, metric_id: str) -> MetricDefinition:
        return self._metrics[metric_id]

    def ids(self) -> list[str]:
        return sorted(self._metrics)

    def validate(self) -> list[str]:
        errors: list[str] = []
        seen: set[str] = set()
        required = {
            "metric_id",
            "task",
            "protocol",
            "evaluated_fields",
            "denominator",
            "match_rule",
            "missing_value_policy",
            "implementation_function",
        }
        for mid, metric in self._metrics.items():
            if mid in seen:
                errors.append(f"duplicate metric_id: {mid}")
            seen.add(mid)
            raw = metric.__dict__
            for key in required:
                if not raw.get(key):
                    errors.append(f"{mid}: missing {key}")
            if any(token in mid for token in ["v3", "v4", "v12", "run", "2026", "stage35"]):
                errors.append(f"{mid}: metric_id mixes model/run/version naming")
            if "operational" in metric.protocol and "accuracy" in mid:
                errors.append(f"{mid}: operational metrics must not be named accuracy")
        return errors


def load_default_registry(project_root: str | Path = ".") -> MetricRegistry:
    return MetricRegistry.load(Path(project_root) / "research/specs/metric_registry_v1.yaml")
