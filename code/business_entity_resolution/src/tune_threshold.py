import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
VAL_PREDS = OUTPUT_DIR / "val_predictions.csv"
GROUND_TRUTH = PROJECT_DIR.parent.parent / "dataset" / "train" / "train_ground_truth.tsv"

print("Loading validation predictions...")
preds = pd.read_csv(VAL_PREDS, dtype={"source1_entity_id": str, "candidate_entity_id": str})

print("Loading ground truth...")
gt = pd.read_csv(GROUND_TRUTH, sep="\t", dtype=str)
gt_map = {}
for _, row in gt.iterrows():
    s1 = row["source1_entity_id"]
    matched = row["matched_entity_ids"]
    gt_map[s1] = set(m.strip() for m in str(matched).split(",") if m.strip()) if pd.notna(matched) else set()

# All S1 entities that appear in the validation split (from preds), plus any
# that have zero candidates at all should also be scored as singletons if in val set.
val_s1_ids = set(preds["source1_entity_id"].unique())

def entity_f05(truth_set, pred_set):
    beta = 0.5
    if len(truth_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    tp = len(truth_set & pred_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(truth_set)
    if precision == 0 and recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)


def score_at_threshold(threshold):
    kept = preds[preds["pred_prob"] >= threshold]
    grouped = kept.groupby("source1_entity_id")["candidate_entity_id"].apply(set).to_dict()

    scores = []
    for s1_id in val_s1_ids:
        truth_set = gt_map.get(s1_id, set())
        pred_set = grouped.get(s1_id, set())
        scores.append(entity_f05(truth_set, pred_set))

    return sum(scores) / len(scores)


print("\nSweeping thresholds...")
best_threshold, best_score = 0, 0
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    score = score_at_threshold(t)
    print(f"  threshold={t:.2f}  macro F0.5={score:.6f}")
    if score > best_score:
        best_score = score
        best_threshold = t

print(f"\nBest threshold: {best_threshold}  (macro F0.5 = {best_score:.6f})")