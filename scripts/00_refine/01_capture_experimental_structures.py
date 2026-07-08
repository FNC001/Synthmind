import os
import re
import json
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# =========================================================
# Config
# =========================================================
CONFIG = {
    "MP_API_KEY": "IOrGB7l1UrUnL6B7ZbZxHqlXxbDvBqyD",
    "OUTPUT_DIR": "mp_full_archive_export",

    # 输出
    "SAVE_POSCAR": True,
    "SAVE_SUMMARY_JSON": True,
    "SAVE_DOI_JSON": True,
    "SAVE_PROVENANCE_JSON": True,

    # 查询条件：四选一，也可以都留空
    "MATERIAL_IDS": [],      # e.g. ["mp-149", "mp-13"]
    "FORMULAE": [],          # e.g. ["SiO2", "Fe2O3"]
    "CHEMSYS": [],           # e.g. ["Li-Fe-O"]
    "ELEMENTS": [],          # e.g. ["Si", "O"]

    # 过滤
    "EXPERIMENTAL_ONLY": True,   # True -> theoretical=False
    "DEPRECATED": False,

    # 批量参数
    "CHUNK_SIZE": 500,
    "NUM_CHUNKS": None,
    "BATCH_SIZE_META": 200,

    # 节流
    "SLEEP_EVERY": 20,
    "SLEEP_SECONDS": 0.2,

    # 调试
    "DEBUG_EVERY": 500,
    "DEBUG_SHOW_LAST": 5,
    "DEBUG_LOG_FILE": "debug_every_500.log",

    # 是否在 CSV 里放完整 JSON 字符串
    # 数据量很大时建议 False，因为 CSV 会非常大
    "EMBED_FULL_JSON_IN_CSV": False,
}

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


# =========================================================
# Helpers
# =========================================================
def ensure_api_key():
    if not CONFIG["MP_API_KEY"] or CONFIG["MP_API_KEY"] == "YOUR_MP_API_KEY_HERE":
        raise ValueError(
            "请先设置 MP_API_KEY。"
            "可以直接改 CONFIG['MP_API_KEY']，或先 export MP_API_KEY。"
        )


def batches(seq: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def shorten_text(s: Any, max_len: int = 160) -> str:
    if s is None:
        return "None"
    s = str(s).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def append_log(msg: str):
    log_file = CONFIG.get("DEBUG_LOG_FILE")
    if not log_file:
        return
    outdir = Path(CONFIG["OUTPUT_DIR"])
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / log_file
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def emit(msg: str):
    print(msg, flush=True)
    append_log(msg)


def extract_dois_in_obj(obj: Any) -> list[str]:
    found = set()

    def walk(x: Any):
        if x is None:
            return
        if isinstance(x, str):
            for m in DOI_PATTERN.findall(x):
                found.add(m.strip().rstrip(".,;]})"))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for item in x:
                walk(item)

    walk(obj)
    return sorted(found)


def collect_values_for_keys(obj: Any, key_substrings: tuple[str, ...]) -> list[Any]:
    results = []

    def walk(x: Any):
        if isinstance(x, dict):
            for k, v in x.items():
                k_lower = str(k).lower()
                if any(substr in k_lower for substr in key_substrings):
                    results.append(v)
                walk(v)
        elif isinstance(x, (list, tuple, set)):
            for item in x:
                walk(item)

    walk(obj)
    return results


def flatten_texts(values: list[Any]) -> list[str]:
    out = []
    for v in values:
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())

    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def get_rester(mpr: MPRester, candidates: list[tuple[str, ...]]):
    for path in candidates:
        cur = mpr
        ok = True
        for name in path:
            if not hasattr(cur, name):
                ok = False
                break
            cur = getattr(cur, name)
        if ok:
            return cur
    return None


def safe_json_dump(obj: Any, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_debug_rows(rows: list[dict], stage: str, total_count: int):
    if not rows:
        return

    emit("\n" + "=" * 120)
    emit(f"[DEBUG] {stage} | processed = {total_count}")
    emit("=" * 120)

    tail = rows[-CONFIG["DEBUG_SHOW_LAST"]:]
    for r in tail:
        emit(
            f"material_id={r.get('material_id')} | "
            f"formula={r.get('formula_pretty')} | "
            f"mp_doi={shorten_text(r.get('mp_doi'), 80)} | "
            f"literature_dois={shorten_text(r.get('literature_dois_found'), 120)} | "
            f"title={shorten_text(r.get('title_candidates_from_provenance'), 180)}"
        )

    emit("=" * 120 + "\n")


def print_debug_single(row: dict, stage: str, total_count: int):
    emit("\n" + "-" * 120)
    emit(f"[DEBUG] {stage} | processed = {total_count}")
    emit(
        f"material_id={row.get('material_id')} | "
        f"formula={row.get('formula_pretty')} | "
        f"mp_doi={shorten_text(row.get('mp_doi'), 80)} | "
        f"literature_dois={shorten_text(row.get('literature_dois_found'), 120)} | "
        f"title={shorten_text(row.get('title_candidates_from_provenance'), 180)}"
    )
    emit("-" * 120 + "\n")


# =========================================================
# Fetch
# =========================================================
def fetch_summary_docs(mpr: MPRester):
    search_kwargs = {
        "all_fields": True,
        "chunk_size": CONFIG["CHUNK_SIZE"],
        "num_chunks": CONFIG["NUM_CHUNKS"],
        "deprecated": CONFIG["DEPRECATED"],
    }

    if CONFIG["EXPERIMENTAL_ONLY"]:
        search_kwargs["theoretical"] = False

    if CONFIG["MATERIAL_IDS"]:
        search_kwargs["material_ids"] = CONFIG["MATERIAL_IDS"]
    elif CONFIG["FORMULAE"]:
        search_kwargs["formula"] = CONFIG["FORMULAE"]
    elif CONFIG["CHEMSYS"]:
        search_kwargs["chemsys"] = CONFIG["CHEMSYS"]
    elif CONFIG["ELEMENTS"]:
        search_kwargs["elements"] = CONFIG["ELEMENTS"]

    return list(mpr.materials.summary.search(**search_kwargs))


def fetch_meta_docs(mpr: MPRester, material_ids: list[str], mode: str) -> dict[str, dict]:
    if mode == "doi":
        rester = get_rester(mpr, [("doi",), ("materials", "doi")])
        desc = "Fetching DOI"
    elif mode == "provenance":
        rester = get_rester(mpr, [("materials", "provenance"), ("provenance",)])
        desc = "Fetching provenance"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if rester is None:
        emit(f"[WARN] 未找到 {mode} endpoint，跳过。")
        return {}

    result = {}
    batch_list = list(batches(material_ids, CONFIG["BATCH_SIZE_META"]))

    for i, batch in enumerate(tqdm(batch_list, desc=desc, unit="batch", ncols=100), start=1):
        try:
            docs = rester.search(material_ids=batch)
        except TypeError:
            docs = rester.search(material_ids=batch, all_fields=True)
        except Exception as e:
            emit(f"[WARN] {mode} batch {i} 抓取失败: {e}")
            continue

        for doc in docs:
            if not isinstance(doc, dict):
                continue
            mpid = doc.get("material_id")
            if mpid:
                result[mpid] = doc

        if i % CONFIG["SLEEP_EVERY"] == 0:
            time.sleep(CONFIG["SLEEP_SECONDS"])

    return result


# =========================================================
# Main
# =========================================================
def main():
    ensure_api_key()

    outdir = Path(CONFIG["OUTPUT_DIR"])
    outdir.mkdir(parents=True, exist_ok=True)

    if CONFIG.get("DEBUG_LOG_FILE"):
        log_path = outdir / CONFIG["DEBUG_LOG_FILE"]
        if log_path.exists():
            log_path.unlink()

    poscar_dir = outdir / "poscar"
    summary_dir = outdir / "summary_json"
    doi_dir = outdir / "doi_json"
    provenance_dir = outdir / "provenance_json"

    if CONFIG["SAVE_POSCAR"]:
        poscar_dir.mkdir(exist_ok=True)
    if CONFIG["SAVE_SUMMARY_JSON"]:
        summary_dir.mkdir(exist_ok=True)
    if CONFIG["SAVE_DOI_JSON"]:
        doi_dir.mkdir(exist_ok=True)
    if CONFIG["SAVE_PROVENANCE_JSON"]:
        provenance_dir.mkdir(exist_ok=True)

    emit("[INFO] Start running...")

    # use_document_model=False 便于原样保存 dict
    with MPRester(CONFIG["MP_API_KEY"], use_document_model=False) as mpr:
        emit("[INFO] 正在抓取 summary 全字段数据...")
        summary_docs = fetch_summary_docs(mpr)
        emit(f"[INFO] 共获取 {len(summary_docs)} 条 summary 文档")

        rows = []
        summary_map = {}
        structures = {}
        material_ids = []

        for idx, doc in enumerate(
            tqdm(summary_docs, desc="Processing summary", unit="doc", ncols=100), start=1
        ):
            if not isinstance(doc, dict):
                continue

            mpid = doc.get("material_id")
            formula = doc.get("formula_pretty")
            structure_dict = doc.get("structure")

            if mpid:
                material_ids.append(mpid)
                summary_map[mpid] = doc

            if structure_dict:
                structures[mpid] = structure_dict

            rows.append(
                {
                    "material_id": mpid,
                    "formula_pretty": formula,
                    "theoretical": doc.get("theoretical"),
                    "deprecated": doc.get("deprecated"),
                }
            )

            if CONFIG["SAVE_SUMMARY_JSON"] and mpid:
                safe_json_dump(doc, summary_dir / f"{sanitize_filename(mpid)}.json")

            if idx % CONFIG["DEBUG_EVERY"] == 0:
                emit(
                    f"[DEBUG] Summary processed={idx} | "
                    f"latest material_id={mpid} | formula={formula}"
                )

        emit("[INFO] 正在抓取 DOI...")
        doi_map = fetch_meta_docs(mpr, material_ids, mode="doi")

        emit("[INFO] 正在抓取 provenance...")
        prov_map = fetch_meta_docs(mpr, material_ids, mode="provenance")

    final_rows = []
    for idx, row in enumerate(
        tqdm(rows, desc="Merging metadata", unit="row", ncols=100), start=1
    ):
        mpid = row["material_id"]
        doi_doc = doi_map.get(mpid, {})
        prov_doc = prov_map.get(mpid, {})
        summary_doc = summary_map.get(mpid, {})

        if CONFIG["SAVE_DOI_JSON"] and mpid and doi_doc:
            safe_json_dump(doi_doc, doi_dir / f"{sanitize_filename(mpid)}.json")

        if CONFIG["SAVE_PROVENANCE_JSON"] and mpid and prov_doc:
            safe_json_dump(prov_doc, provenance_dir / f"{sanitize_filename(mpid)}.json")

        title_candidates = flatten_texts(
            collect_values_for_keys(prov_doc, ("title", "citation", "reference"))
        )
        literature_dois = extract_dois_in_obj(prov_doc)

        database_ids = None
        if isinstance(prov_doc, dict):
            if "database_IDs" in prov_doc:
                database_ids = prov_doc["database_IDs"]
            elif "database_ids" in prov_doc:
                database_ids = prov_doc["database_ids"]
            else:
                dbid_candidates = collect_values_for_keys(
                    prov_doc, ("database_ids", "database_id", "database_ids")
                )
                if dbid_candidates:
                    database_ids = dbid_candidates[0]

        merged_row = {
            **row,
            "mp_doi": doi_doc.get("doi"),
            "mp_bibtex": doi_doc.get("bibtex"),
            "title_candidates_from_provenance": " | ".join(title_candidates[:10]) if title_candidates else None,
            "literature_dois_found": " | ".join(literature_dois) if literature_dois else None,
            "database_ids_json": json.dumps(database_ids, ensure_ascii=False) if database_ids is not None else None,
            "summary_json_path": str(summary_dir / f"{sanitize_filename(mpid)}.json") if mpid and CONFIG["SAVE_SUMMARY_JSON"] else None,
            "doi_json_path": str(doi_dir / f"{sanitize_filename(mpid)}.json") if mpid and CONFIG["SAVE_DOI_JSON"] and doi_doc else None,
            "provenance_json_path": str(provenance_dir / f"{sanitize_filename(mpid)}.json") if mpid and CONFIG["SAVE_PROVENANCE_JSON"] and prov_doc else None,
        }

        if CONFIG["EMBED_FULL_JSON_IN_CSV"]:
            merged_row["summary_json"] = json.dumps(summary_doc, ensure_ascii=False)
            merged_row["doi_json"] = json.dumps(doi_doc, ensure_ascii=False)
            merged_row["provenance_json"] = json.dumps(prov_doc, ensure_ascii=False)

        final_rows.append(merged_row)

        if idx % CONFIG["DEBUG_EVERY"] == 0:
            print_debug_rows(final_rows, stage="Merged metadata", total_count=idx)

    df = pd.DataFrame(final_rows)
    csv_path = outdir / "mp_full_archive_metadata.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    emit(f"[INFO] 总表 CSV 已保存到: {csv_path}")

    saved_poscar = 0
    for idx, (mpid, structure_dict) in enumerate(
        tqdm(structures.items(), desc="Saving POSCAR", unit="structure", ncols=100), start=1
    ):
        row_match = df[df["material_id"] == mpid]
        if row_match.empty:
            formula = "unknown"
            debug_row = {
                "material_id": mpid,
                "formula_pretty": "unknown",
                "mp_doi": None,
                "literature_dois_found": None,
                "title_candidates_from_provenance": None,
            }
        else:
            debug_row = row_match.iloc[0].to_dict()
            formula = debug_row.get("formula_pretty", "unknown")

        if CONFIG["SAVE_POSCAR"] and structure_dict:
            try:
                structure = Structure.from_dict(structure_dict)
                stem = sanitize_filename(f"{mpid}_{formula}")
                poscar_path = poscar_dir / f"{stem}.vasp"
                Poscar(structure).write_file(str(poscar_path))
                saved_poscar += 1
            except Exception as e:
                emit(f"[WARN] 保存 POSCAR 失败 {mpid}: {e}")

        if idx % CONFIG["DEBUG_EVERY"] == 0:
            print_debug_single(debug_row, stage="Saved POSCAR", total_count=idx)

        if idx % CONFIG["SLEEP_EVERY"] == 0:
            time.sleep(CONFIG["SLEEP_SECONDS"])

    emit(f"[INFO] POSCAR 保存完成: {saved_poscar}")
    emit("[INFO] 全部完成。")


if __name__ == "__main__":
    main()
