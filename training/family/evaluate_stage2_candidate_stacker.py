#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.family.evaluate_stage2_candidate_fusion import load_source  # noqa: E402
from training.family.evaluate_stage2_oof_chemistry_rescore import (  # noqa: E402
    family_length_modes,
    json_set,
    label_chemistry,
)
from training.family.train_stage2_oof_candidate_stacker import (  # noqa: E402
    MatSciFeatureBuilder,
    MULTILABEL_FEATURE_DIM,
    append_matsci_features,
    build_row_candidates_and_features,
    exact_metrics,
    load_matsci_views,
    load_matsci_multilabel_scores,
    merge_preserved_base_prefix,
    multilabel_candidate_features,
    rank_query_rows,
)


SetKey = Tuple[int, ...]


def parse_named_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"expert must be NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    return name.strip(), path.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a validation-frozen candidate LambdaRank stacker to validation or test."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--expert", action="append", default=[], help="Repeat NAME=candidates.jsonl")
    parser.add_argument(
        "--matsci_embeddings", default="",
        help="Cache used by the frozen stacker when MatSciBERT features are enabled.",
    )
    parser.add_argument(
        "--matsci_multilabel_scores", default="",
        help="Fine-tuned label-logit cache required by compatible frozen stackers.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_candidates_jsonl", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    state = joblib.load(Path(args.model).resolve())
    expert_paths = dict(parse_named_source(value) for value in args.expert)
    expected_names = [str(value) for value in state["expert_names"]]
    source_agnostic = bool(state.get("source_agnostic", False))
    base_aware = bool(state.get("base_aware", False))
    if not source_agnostic and list(expert_paths) != expected_names:
        raise ValueError(
            f"expert order/name mismatch: expected {expected_names!r}, got {list(expert_paths)!r}"
        )
    if source_agnostic and len(expert_paths) < 2:
        raise ValueError("a source-agnostic stacker still requires at least two inference experts")

    pack = np.load(input_dir / f"{args.split}.npz", allow_pickle=True)
    y = np.asarray(pack["y_multi_hot"], dtype=np.float32)
    targets: List[SetKey] = [tuple(np.flatnonzero(row > 0.5).tolist()) for row in y]
    train_y = np.asarray(
        np.load(input_dir / "train.npz", allow_pickle=True)["y_multi_hot"], dtype=np.float32
    )
    meta = pd.read_csv(input_dir / f"{args.split}_meta.csv", low_memory=False)
    train_meta = pd.read_csv(input_dir / "train_meta.csv", low_memory=False)
    experts = [
        load_source(path, len(targets), int(state["source_limit"]))
        for path in expert_paths.values()
    ]
    names = [
        str(value)
        for value in json.loads((input_dir / "precursor_names.json").read_text(encoding="utf-8"))
    ]
    matsci_builder = state.get("matsci_builder")
    matsci_direct = None
    matsci_projected = None
    if matsci_builder is not None:
        if not str(args.matsci_embeddings).strip():
            raise ValueError("the frozen model requires --matsci_embeddings")
        _, _, query_views = load_matsci_views(
            Path(args.matsci_embeddings).resolve(), input_dir, str(args.split), names
        )
        matsci_direct, matsci_projected = matsci_builder.transform_queries(query_views)
    matsci_multilabel_scores = None
    if bool(state.get("matsci_multilabel_enabled", False)):
        if not str(args.matsci_multilabel_scores).strip():
            raise ValueError("the frozen model requires --matsci_multilabel_scores")
        matsci_multilabel_scores = load_matsci_multilabel_scores(
            Path(args.matsci_multilabel_scores).resolve(), input_dir, str(args.split), names
        )
    prior_builder = state.get("candidate_prior_builder")
    template_prior_builder = state.get("template_prior_builder")
    label_elements, label_groups, label_metals = label_chemistry(names)
    train_seen = np.asarray(train_y.sum(axis=0) > 0)
    length_modes = family_length_modes(train_meta, train_y)
    families = meta["family_signature_primary"].fillna("UNK").astype(str).to_numpy()
    selected_families = {str(value) for value in state.get("families", [])}
    cations = [json_set(value) for value in meta["target_cation_elements"]]
    anions = [json_set(value) for value in meta["target_anion_elements"]]

    candidate_rows: List[List[SetKey]] = []
    feature_rows: List[np.ndarray] = []
    for row_index in range(len(targets)):
        row_candidates, row_features = build_row_candidates_and_features(
            [expert[row_index] for expert in experts],
            cations[row_index],
            anions[row_index],
            label_elements,
            label_groups,
            label_metals,
            train_seen,
            int(length_modes.get(str(families[row_index]), length_modes["__GLOBAL__"])),
            int(state["union_limit"]),
            prior_builder=prior_builder,
            template_prior_builder=template_prior_builder,
            family=str(families[row_index]),
            source_agnostic=source_agnostic,
            base_aware=base_aware,
        )
        row_features = append_matsci_features(
            row_features,
            row_candidates,
            matsci_builder,
            None if matsci_direct is None else matsci_direct[row_index],
            None if matsci_projected is None else matsci_projected[row_index],
        )
        if matsci_multilabel_scores is not None:
            row_features = np.concatenate([
                row_features,
                multilabel_candidate_features(
                    row_candidates, matsci_multilabel_scores[row_index]
                ),
            ], axis=1)
        if row_features.shape[1] != int(state["feature_dim"]):
            raise ValueError(
                f"feature dimension mismatch: expected {state['feature_dim']}, got {row_features.shape[1]}"
            )
        candidate_rows.append(row_candidates)
        feature_rows.append(row_features)

    groups = [int(len(values)) for values in feature_rows]
    matrix = np.vstack(feature_rows)
    row_indices = list(range(len(targets)))
    ranked_map = rank_query_rows(state["model"], matrix, groups, row_indices, candidate_rows)
    ranked = [ranked_map.get(index, candidate_rows[index]) for index in row_indices]
    preserve_base_prefix = state.get("preserve_base_prefix")
    if preserve_base_prefix is not None:
        ranked = [
            merge_preserved_base_prefix(experts[0][index], row, int(preserve_base_prefix))
            for index, row in enumerate(ranked)
        ]
    if selected_families:
        ranked = [
            row if str(families[index]) in selected_families else list(experts[0][index])
            for index, row in enumerate(ranked)
        ]
    report = {
        "protocol": f"{args.split}_formula_disjoint_frozen_candidate_lambdarank_stacker",
        "model": str(Path(args.model).resolve()),
        "expert_paths": expert_paths,
        "feature_dim": int(state["feature_dim"]),
        "matscibert_enabled": matsci_builder is not None,
        "matscibert_multilabel_enabled": matsci_multilabel_scores is not None,
        "candidate_priors_enabled": prior_builder is not None,
        "template_priors_enabled": template_prior_builder is not None,
        "source_agnostic": source_agnostic,
        "base_aware": base_aware,
        "preserve_base_prefix": preserve_base_prefix,
        "families": sorted(selected_families),
        "mean_union_candidates": float(np.mean([len(row) for row in candidate_rows])),
        "evaluation": exact_metrics(targets, ranked),
    }
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with Path(args.output_candidates_jsonl).resolve().open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(ranked):
            handle.write(json.dumps({
                "row_index": row_index,
                "candidate_label_ids": [list(candidate) for candidate in row],
            }) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
