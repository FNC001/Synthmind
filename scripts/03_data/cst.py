import pandas as pd

paths = [
    "/Users/wyc/MP_exp_doi/data/interim/features/structdesc_features_stage3_v2/stage3_train_raw.csv",
    "/Users/wyc/MP_exp_doi/data/interim/features/structdesc_features_stage3_v2/stage3_val_raw.csv",
    "/Users/wyc/MP_exp_doi/data/interim/features/structdesc_features_stage3_v2/stage3_test_raw.csv",
]

dfs = [pd.read_csv(p) for p in paths]
full = pd.concat(dfs, ignore_index=True)
out = "/Users/wyc/MP_exp_doi/data/interim/features/structdesc_features_stage3_v2/stage3_mixed_source_v1.csv"
full.to_csv(out, index=False)
print(out, full.shape)

