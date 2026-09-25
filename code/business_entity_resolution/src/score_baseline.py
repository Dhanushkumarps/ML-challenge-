import pandas as pd
from pathlib import Path
from collections import defaultdict

# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR.parent.parent / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

GROUND_TRUTH = DATA_DIR / "train_ground_truth.tsv"
PREDICTIONS = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# Per-entity F0.5
# ============================================================

def entity_f05(truth_set, pred_set):
    """
    Official rule:
    - truth empty + pred empty  -> 1.0 (correct singleton)
    - truth empty + pred nonempty -> 0.0 (false merge on singleton)
    - otherwise standard F0.5 on this entity's own sets
    """
    beta = 0.5

    if len(truth_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0

    tp = len(truth_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set)

    if precision == 0 and recall == 0:
        return 0.0

    return (
        (1 + beta ** 2) * precision * recall
        / ((beta ** 2 * precision) + recall)
    )


# ============================================================
# Load ground truth -> dict[s1_id] = set(matched_ids)
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(GROUND_TRUTH, sep="\t", dtype=str)

ground_truth = defaultdict(set)

for row in gt.itertuples(index=False):
    s1_id = row.source1_entity_id
    matched_str = row.matched_entity_ids

    # ensure every S1 entity has an entry, even if empty (singleton)
    _ = ground_truth[s1_id]

    if pd.notna(matched_str) and str(matched_str).strip():
        for matched_id in str(matched_str).split(","):
            matched_id = matched_id.strip()
            if matched_id:
                ground_truth[s1_id].add(matched_id)

all_s1 = list(ground_truth.keys())
total_gt_pairs = sum(len(v) for v in ground_truth.values())

print(f"Total Source-1 entities : {len(all_s1):,}")
print(f"Total ground-truth pairs: {total_gt_pairs:,}")


# ============================================================
# Hold-out slice: last 20% of Source-1 entities
# ============================================================

split_point = int(len(all_s1) * 0.80)
validation_s1 = set(all_s1[split_point:])

print(f"Validation Source-1 entities: {len(validation_s1):,}")
validation_gt_pairs = sum(
    len(ground_truth[s1]) for s1 in validation_s1
)
print(f"Validation ground-truth pairs: {validation_gt_pairs:,}")


# ============================================================
# Load predictions -> dict[s1_id] = set(matched_ids)
# (only for validation entities, read in chunks for memory)
# ============================================================

# ============================================================
# Load predictions -> dict[s1_id] = set(candidate_ids)
# (wide format: one row per S1 entity, comma-separated IDs)
# ============================================================

print("\nReading predictions...")

predictions = defaultdict(set)

for chunk in pd.read_csv(
    PREDICTIONS,
    sep="\t",
    dtype=str,
    chunksize=200_000
):
    chunk = chunk[chunk["source1_entity_id"].isin(validation_s1)]

    for row in chunk.itertuples(index=False):
        s1_id = row.source1_entity_id
        cand_str = row.candidate_entity_ids

        if pd.notna(cand_str) and str(cand_str).strip():
            for cand_id in str(cand_str).split(","):
                cand_id = cand_id.strip()
                if cand_id:
                    predictions[s1_id].add(cand_id)

print(f"Validation entities with predictions: {len(predictions):,}")

# ============================================================
# Per-entity macro-averaged F0.5
# (iterate over ALL validation entities, singletons included)
# ============================================================

scores = []
tp_total = 0
pred_total = 0
truth_total = 0

for s1_id in validation_s1:
    truth_set = ground_truth[s1_id]
    pred_set = predictions.get(s1_id, set())

    scores.append(entity_f05(truth_set, pred_set))

    tp_total += len(truth_set & pred_set)
    pred_total += len(pred_set)
    truth_total += len(truth_set)

macro_f05 = sum(scores) / len(scores)

# pooled numbers too, just for diagnostic comparison
pooled_precision = tp_total / pred_total if pred_total else 0.0
pooled_recall = tp_total / truth_total if truth_total else 0.0


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 60)
print("BASELINE F0.5 VALIDATION (per-entity macro-averaged)")
print("=" * 60)

print(f"Validation entities scored : {len(scores):,}")
print(f"Macro-averaged F0.5        : {macro_f05:.6f}")

print()
print("Pooled diagnostics (NOT the official metric, for reference only):")
print(f"  Pooled precision : {pooled_precision:.6f}")
print(f"  Pooled recall    : {pooled_recall:.6f}")

print("=" * 60)