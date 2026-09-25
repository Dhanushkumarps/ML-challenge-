import pandas as pd
import re
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.parent.parent / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Normalize business names
# ============================================================

def normalize_name(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


# ============================================================
# Build Source 2/3 lookup
# ============================================================

print("Building Source 2/3 name lookup...")

lookup_parts = []

for source_file in ["train_source2.tsv", "train_source3.tsv"]:

    print(f"Reading {source_file}...")

    path = DATA_DIR / source_file

    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["entity_id", "business_name"],
        dtype=str
    )

    df["normalized_name"] = normalize_name(df["business_name"])

    df = df[df["normalized_name"] != ""]

    lookup_parts.append(
        df[["normalized_name", "entity_id"]]
    )

    print(f"  Rows loaded: {len(df):,}")


# Combine Source 2 and Source 3
lookup = pd.concat(
    lookup_parts,
    ignore_index=True
)

print(f"\nTotal Source 2/3 records: {len(lookup):,}")


# ============================================================
# Load Source 1
# ============================================================

print("\nReading train_source1.tsv...")

source1 = pd.read_csv(
    DATA_DIR / "train_source1.tsv",
    sep="\t",
    usecols=["entity_id", "business_name"],
    dtype=str
)

source1["normalized_name"] = normalize_name(
    source1["business_name"]
)

source1 = source1[
    source1["normalized_name"] != ""
]

print(f"Source 1 rows: {len(source1):,}")


# ============================================================
# Exact normalized-name matching
# ============================================================

print("\nFinding exact normalized-name matches...")

results = source1.merge(
    lookup,
    on="normalized_name",
    how="inner",
    suffixes=("_source1", "_matched")
)


# ============================================================
# Format output
# ============================================================

results = results.rename(
    columns={
        "entity_id_source1": "source1_entity_id",
        "entity_id_matched": "matched_entity_id"
    }
)

results = results[
    [
        "source1_entity_id",
        "matched_entity_id",
        "normalized_name"
    ]
]


# ============================================================
# Save results
# ============================================================

candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
matching_path = OUTPUT_DIR / "matching_results.tsv"

results.to_csv(
    candidate_path,
    sep="\t",
    index=False
)

results.to_csv(
    matching_path,
    sep="\t",
    index=False
)


print("\n========================================")
print("BASELINE COMPLETE")
print("========================================")

print(f"Candidate pairs: {len(results):,}")

print(f"\nSaved:")
print(candidate_path)
print(matching_path)