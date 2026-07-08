#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def deep_get(cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve_templates(obj: Any, cfg: Dict[str, Any], max_iter: int = 5) -> Any:
    """
    Resolve strings like:
      {project_root}
      {infer_name}
      {stage2.gflownet_run_dir}
    """

    def lookup(name: str) -> str:
        if "." in name:
            val = deep_get(cfg, name, "")
        else:
            val = cfg.get(name, "")
        return str(val)

    def resolve_str(s: str) -> str:
        out = s
        for _ in range(max_iter):
            old = out
            for token in set(part.split("}")[0] for part in out.split("{")[1:]):
                key = token.strip()
                out = out.replace("{" + key + "}", lookup(key))
            if out == old:
                break
        return out

    if isinstance(obj, dict):
        return {k: resolve_templates(v, cfg, max_iter=max_iter) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_templates(v, cfg, max_iter=max_iter) for v in obj]
    if isinstance(obj, str):
        return resolve_str(obj)
    return obj


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = load_yaml(path)

    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})

    # iterative template resolution over whole config
    for _ in range(5):
        new_cfg = resolve_templates(cfg, cfg)
        if new_cfg == cfg:
            break
        cfg = new_cfg

    cfg["_config_path"] = str(Path(path).resolve())
    return cfg
