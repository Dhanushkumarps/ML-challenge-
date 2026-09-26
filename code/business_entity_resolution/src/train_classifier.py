import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FEATURES = OUTPUT_DIR / "features.csv"
MODEL_OUT = OUTPUT_DIR / "matcher_model.txt"

FEATURE_COLS = [
    "name_jaccard", "name_fuzz_ratio", "name_token_sort_ratio",
    "addr_jaccard", "addr_fuzz_ratio", "country_match",
]

print("Loading features (chunked)...")

numeric_dtypes = {
    "name_jaccard": "float32",
    "name_fuzz_ratio": "float32",
    "name_token_sort_ratio": "float32",
    "addr_jaccard": "float32",
    "addr_fuzz_ratio": "float32",
    "country_match": "int8",
    "label": "int8",
}

chunks = []
for i, chunk in enumerate(pd.read_csv(FEATURES, chunksize=500_000)):
    for col, dt in numeric_dtypes.items():
        chunk[col] = chunk[col].astype(dt)
    chunk["source1_entity_id"] = chunk["source1_entity_id"].astype("string")
    chunk["candidate_entity_id"] = chunk["candidate_entity_id"].astype("string")
    chunks.append(chunk)
    print(f"  loaded chunk {i+1} ({(i+1)*500_000:,} rows so far)")

df = pd.concat(chunks, ignore_index=True)
del chunks

df["source1_entity_id"] = df["source1_entity_id"].astype("category")
df["candidate_entity_id"] = df["candidate_entity_id"].astype("category")

print(f"Total rows: {len(df):,}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
print(f"Total pairs: {len(df):,}  Positive: {df['label'].sum():,}")

# Split by source1_entity_id so no entity leaks between train/val
unique_s1 = df["source1_entity_id"].unique()
train_ids, val_ids = train_test_split(unique_s1, test_size=0.2, random_state=42)

train_df = df[df["source1_entity_id"].isin(train_ids)]
val_df = df[df["source1_entity_id"].isin(val_ids)]

print(f"Train pairs: {len(train_df):,}  Val pairs: {len(val_df):,}")

X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
X_val, y_val = val_df[FEATURE_COLS], val_df["label"]

train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "is_unbalance": True,  # handles the fact that most pairs are negative
}

print("Training LightGBM...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=500,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)],
)

model.save_model(str(MODEL_OUT))
print(f"Model saved -> {MODEL_OUT}")

# Save validation predictions for threshold tuning next step
val_df = val_df.copy()
val_df["pred_prob"] = model.predict(X_val, num_iteration=model.best_iteration)
val_df[["source1_entity_id", "candidate_entity_id", "label", "pred_prob"]].to_csv(
    OUTPUT_DIR / "val_predictions.csv", index=False
)
print(f"Validation predictions saved -> {OUTPUT_DIR / 'val_predictions.csv'}")

# feature importance, useful for your methodology doc
importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
print("\nFeature importance (gain):")
print(importance.to_string(index=False))