import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR.parent.parent / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

GROUND_TRUTH = DATA_DIR / "train_ground_truth.tsv"
CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"


# ============================================================
# 1. Build ground-truth pair set
# ============================================================

print("Loading ground truth...")

gt = pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    dtype=str
)

ground_truth_pairs = set()

for _, row in gt.iterrows():

    s1_id = row["source1_entity_id"]

    matched_ids = str(
        row["matched_entity_ids"]
    ).split(",")

    for matched_id in matched_ids:

        matched_id = matched_id.strip()

        if matched_id:
            ground_truth_pairs.add(
                (s1_id, matched_id)
            )


print(
    f"Ground-truth pairs: "
    f"{len(ground_truth_pairs):,}"
)


# ============================================================
# 2. Read candidate pairs in chunks
# ============================================================

print("\nEvaluating candidate pairs...")

total_predictions = 0
true_positives = 0

chunksize = 200_000

for chunk in pd.read_csv(
    CANDIDATES,
    sep="\t",
    dtype=str,
    chunksize=chunksize
):

    total_predictions += len(chunk)

    for s1_id, matched_id in zip(
        chunk["source1_entity_id"],
        chunk["matched_entity_id"]
    ):

        if (s1_id, matched_id) in ground_truth_pairs:
            true_positives += 1

    print(
        f"Processed: "
        f"{total_predictions:,} candidates",
        end="\r"
    )


# ============================================================
# 3. Calculate metrics
# ============================================================

false_positives = total_predictions - true_positives

precision = (
    true_positives / total_predictions
    if total_predictions > 0
    else 0
)

recall = (
    true_positives / len(ground_truth_pairs)
    if len(ground_truth_pairs) > 0
    else 0
)

if precision + recall > 0:
    f1 = 2 * precision * recall / (precision + recall)
else:
    f1 = 0


# ============================================================
# 4. Results
# ============================================================

print("\n\n")
print("=" * 60)
print("BASELINE EVALUATION")
print("=" * 60)

print(f"Ground-truth pairs : {len(ground_truth_pairs):,}")
print(f"Predicted pairs    : {total_predictions:,}")
print(f"True positives     : {true_positives:,}")
print(f"False positives    : {false_positives:,}")

print()
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1 Score            : {f1:.4f}")

print("=" * 60)