import json
import pandas as pd

rows = []
with open("/Users/wyc/SynPred/data/raw/mp_synth_direct_aligned/direct_aligned_dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

print("total:", len(df))
print(df["match_level"].value_counts(dropna=False))

exact_df = df[df["match_level"] == "exact"].copy()
parent_df = df[df["match_level"] == "parent_strict"].copy()

print("exact:", len(exact_df))
print("parent_strict:", len(parent_df))
exact_df.to_json("/Users/wyc/SynPred/data/strict_exact_only.jsonl", orient="records", lines=True, force_ascii=False)
parent_df.to_json("/Users/wyc/SynPred/data/strict_parent_aug.jsonl", orient="records", lines=True, force_ascii=False)
