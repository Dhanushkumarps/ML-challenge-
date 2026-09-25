import csv
import random
from pathlib import Path
from collections import defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.parent.parent / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

GROUND_TRUTH = DATA_DIR / "train_ground_truth.tsv"
CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"
SRC1, SRC2, SRC3 = DATA_DIR / "train_source1.tsv", DATA_DIR / "train_source2.tsv", DATA_DIR / "train_source3.tsv"

# Load ground truth: s1 -> set(matched ids)
gt = defaultdict(set)
with open(GROUND_TRUTH, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["matched_entity_ids"].strip():
            gt[row["source1_entity_id"]] = set(
                m.strip() for m in row["matched_entity_ids"].split(",") if m.strip()
            )

# Load candidates: s1 -> set(candidate ids)
cands = defaultdict(set)
with open(CANDIDATES, encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["candidate_entity_ids"].strip():
            cands[row["source1_entity_id"]] = set(row["candidate_entity_ids"].split(","))

# Find missed pairs: true matches NOT in candidates
missed = []
for s1_id, true_matches in gt.items():
    for m in true_matches:
        if m not in cands.get(s1_id, set()):
            missed.append((s1_id, m))

print(f"Total missed pairs available: {len(missed):,}")
sample = random.sample(missed, min(20, len(missed)))
sample_s1_ids = {s1 for s1, _ in sample}
sample_target_ids = {m for _, m in sample}

# Look up the actual records for the sample
def load_matching(path, ids):
    found = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["entity_id"] in ids:
                found[row["entity_id"]] = row
    return found

s1_records = load_matching(SRC1, sample_s1_ids)
s2_records = load_matching(SRC2, sample_target_ids)
s3_records = load_matching(SRC3, sample_target_ids)
target_records = {**s2_records, **s3_records}

print("\n" + "=" * 80)
for s1_id, m_id in sample:
    r1 = s1_records.get(s1_id, {})
    r2 = target_records.get(m_id, {})
    print(f"S1: {s1_id} | {r1.get('business_name','?')} | {r1.get('business_address','?')}")
    print(f"->  {m_id} | {r2.get('business_name','?')} | {r2.get('business_address','?')}")
    print("-" * 80)